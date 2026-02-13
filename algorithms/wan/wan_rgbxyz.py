import torch
import torch.nn as nn
from einops import rearrange
from .wan_i2v import WanImageToVideo
import logging
import gc
import torch
import numpy as np
import torch.distributed as dist
from einops import rearrange, repeat
from tqdm import tqdm
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from transformers import get_scheduler
import zmq
import msgpack
import io
from PIL import Image
import torchvision.transforms as transforms
from utils.video_utils import numpy_to_mp4_bytes

from .modules.model import WanRGBXYZModel, WanAttentionBlock
from .modules.t5 import umt5_xxl, T5CrossAttention, T5SelfAttention
from .modules.tokenizers import HuggingfaceTokenizer
from .modules.vae import video_vae_factory
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.distributed_utils import is_rank_zero
from .modules.clip import clip_xlm_roberta_vit_h_14


class WanRGBXYZ(WanImageToVideo):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lat_w = self.lat_w * 2
        self.max_area = self.max_area * 2
        self.max_tokens = self.max_tokens * 2
        xyz_norm_cfg = getattr(cfg, "xyz_latent_norm", None)
        self.xyz_latent_norm_enabled = bool(
            xyz_norm_cfg is not None and getattr(xyz_norm_cfg, "enabled", False)
        )
        self.xyz_latent_norm_eps = float(
            getattr(xyz_norm_cfg, "eps", 1e-6) if xyz_norm_cfg is not None else 1e-6
        )

    def _configure_modality_embedding(self):
        if not hasattr(self, "model") or not hasattr(self.model, "modality_embedding"):
            return
        modality_cfg = getattr(self.cfg, "modality_embedding", None)
        if modality_cfg is None:
            return

        module = self.model.modality_embedding
        scale = float(getattr(modality_cfg, "scale", module.scale))
        nonzero_init = bool(getattr(modality_cfg, "nonzero_init", True))
        init_std = float(getattr(modality_cfg, "init_std", 0.02))
        reinit_on_start = bool(getattr(modality_cfg, "reinit_on_start", False))
        init_std = max(init_std, 0.0)

        module.scale = scale

        with torch.no_grad():
            if not nonzero_init:
                module.rgb_embed.zero_()
                module.xyz_embed.zero_()
                logging.info(
                    "Modality embedding configured: zero init, scale=%.4f",
                    module.scale,
                )
            elif reinit_on_start:
                nn.init.normal_(module.rgb_embed, mean=0.0, std=init_std)
                nn.init.normal_(module.xyz_embed, mean=0.0, std=init_std)
                logging.info(
                    "Modality embedding configured: normal init std=%.4f, scale=%.4f",
                    init_std,
                    module.scale,
                )
            else:
                logging.info(
                    "Modality embedding configured: keep checkpoint/default init, scale=%.4f",
                    module.scale,
                )

    def configure_model(self):
        """
        This is same as WanI2V except it initializes WanRGBXYZModel rather than WanModel
        """
        logging.info("Building model...")
        # Initialize text encoder
        if not self.cfg.load_prompt_embed:
            text_encoder = (
                umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=torch.bfloat16 if self.is_inference else self.dtype,
                    device=torch.device("cpu"),
                )
                .eval()
                .requires_grad_(False)
            )
            if self.cfg.text_encoder.ckpt_path is not None:
                text_encoder.load_state_dict(
                    torch.load(
                        self.cfg.text_encoder.ckpt_path,
                        map_location="cpu",
                        weights_only=True,
                        # mmap=True,
                    )
                )
            if self.cfg.text_encoder.compile:
                text_encoder = torch.compile(text_encoder)
        else:
            text_encoder = None
        self.text_encoder = text_encoder

        # Initialize tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=self.cfg.text_encoder.name,
            seq_len=self.cfg.text_encoder.text_len,
            clean="whitespace",
        )

        # Initialize VAE
        self.vae = (
            video_vae_factory(
                pretrained_path=self.cfg.vae.ckpt_path,
                z_dim=self.cfg.vae.z_dim,
            )
            .eval()
            .requires_grad_(False)
        ).to(self.dtype)
        self.register_buffer(
            "vae_mean", torch.tensor(self.cfg.vae.mean, dtype=self.dtype)
        )
        self.register_buffer(
            "vae_inv_std", 1.0 / torch.tensor(self.cfg.vae.std, dtype=self.dtype)
        )
        self.vae_scale = [self.vae_mean, self.vae_inv_std]
        if self.xyz_latent_norm_enabled:
            xyz_mean = float(self.cfg.xyz_latent_norm.mean)
            xyz_std = float(self.cfg.xyz_latent_norm.std)
            xyz_std = max(abs(xyz_std), self.xyz_latent_norm_eps)
            self.register_buffer(
                "xyz_latent_mean",
                torch.tensor(xyz_mean, dtype=self.dtype),
            )
            self.register_buffer(
                "xyz_latent_inv_std",
                torch.tensor(1.0 / xyz_std, dtype=self.dtype),
            )
            logging.info(
                "XYZ latent normalization enabled: mean=%.6f std=%.6f",
                xyz_mean,
                xyz_std,
            )
        if self.cfg.vae.compile:
            self.vae = torch.compile(self.vae)

        # Initialize main diffusion model
        # NOTE: The only difference from WanModel
        if self.cfg.model.tuned_ckpt_path is None:
            self.model = WanRGBXYZModel.from_pretrained(self.cfg.model.ckpt_path)
        else:
            self.model = WanRGBXYZModel.from_config(
                WanRGBXYZModel._dict_from_json_file(self.cfg.model.ckpt_path + "/config.json")
            )
            if self.is_inference:
                self.model.to(torch.bfloat16)
            loaded_state_dict = self._load_tuned_state_dict()
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {}
            mismatched_keys = []
            for key, value in loaded_state_dict.items():
                if key not in model_state_dict:
                    continue
                if model_state_dict[key].shape != value.shape:
                    mismatched_keys.append(
                        (key, tuple(value.shape), tuple(model_state_dict[key].shape))
                    )
                    continue
                filtered_state_dict[key] = value

            load_result = self.model.load_state_dict(
                filtered_state_dict,
                strict=False,
                assign=not self.is_inference,
            )

            logging.info(
                "Loaded tuned checkpoint for WanRGBXYZ: matched=%d, missing=%d, unexpected=%d, shape_mismatch=%d",
                len(filtered_state_dict),
                len(load_result.missing_keys),
                len(load_result.unexpected_keys),
                len(mismatched_keys),
            )
            if mismatched_keys:
                preview = ", ".join(
                    f"{k}: ckpt{src_shape}->model{dst_shape}"
                    for k, src_shape, dst_shape in mismatched_keys[:10]
                )
                logging.warning(
                    "Skipped mismatched checkpoint keys (showing up to 10): %s",
                    preview,
                )
        if not self.is_inference:
            self.model.to(self.dtype).train()
        if self.gradient_checkpointing_rate > 0:
            self.model.gradient_checkpointing_enable(p=self.gradient_checkpointing_rate)
        self._configure_modality_embedding()
        if self.cfg.model.compile:
            self.model = torch.compile(self.model)

        self.training_scheduler, self.training_timesteps = self.build_scheduler(True)

        if self.cfg.model.tuned_ckpt_path is None:
            self.model.hack_embedding_ckpt()

        # Additionally initialize CLIP for image encoding
        clip, clip_transform = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=torch.float16 if self.is_inference else self.dtype,
            device="cpu",
        )
        if self.cfg.clip.ckpt_path is not None:
            clip.load_state_dict(
                torch.load(
                    self.cfg.clip.ckpt_path, map_location="cpu", weights_only=True
                )
            )
        if self.cfg.clip.compile:
            clip = torch.compile(clip)
        self.clip = clip
        self.clip_normalize = clip_transform.transforms[-1]

    @torch.no_grad()
    def prepare_embeds(self, batch):
        rgbs = batch["rgb"]
        xyzs = batch["xyz"]
        prompts = batch["prompts"]

        batch_size, t, _, h, w = rgbs.shape

        if t != self.n_frames:
            raise ValueError(f"Number of frames in videos must be {self.n_frames}")
        if h != self.height or w != self.width:
            raise ValueError(
                f"Height and width of videos must be {self.height} and {self.width}"
            )

        if not self.cfg.load_prompt_embed:
            prompt_embeds = self.encode_text(prompts)
        else:
            prompt_embeds = batch["prompt_embeds"].to(self.dtype)
            prompt_embed_len = batch["prompt_embed_len"]
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, prompt_embed_len)]

        rgb_lat = self.encode_video(rearrange(rgbs, "b t c h w -> b c t h w"))
        xyz_lat = self.encode_video(rearrange(xyzs, "b t c h w -> b c t h w"))
        xyz_lat = self.normalize_xyz_latent(xyz_lat)
        # video_lat ~ (b, lat_c, lat_t, lat_h, lat_w)

        batch["prompt_embeds"] = prompt_embeds
        # Concat Latent along width
        batch["video_lat"] = torch.cat([rgb_lat, xyz_lat], dim=-1)
        batch["image_embeds"] = None
        batch["clip_embeds"] = None

        # Only feed rgb first frame to CLIP
        images = rgbs[:, :1]
        has_bbox = batch["has_bbox"]  # [B, 2]
        bbox_render = batch["bbox_render"]  # [B, 2, H, W]

        batch_size, t, _, h, w = rgbs.shape
        lat_c, lat_t, lat_h, lat_w = self.lat_c, self.lat_t, self.lat_h, self.lat_w

        clip_embeds = self.clip_features(images)
        batch["clip_embeds"] = clip_embeds

        mask = torch.zeros(
            batch_size,
            self.vae_stride[0],
            lat_t,
            lat_h,
            lat_w,
            device=self.device,
            dtype=self.dtype,
        )
        # after the ckpt hack, we repurpose the 4 mask channels for bounding box conditioning
        # second last channel is indicator of bounding box
        mask[:, 2, 0] = has_bbox[..., 0, None, None]
        mask[:, 2, -1] = has_bbox[..., -1, None, None]
        # Interpolate bbox_render to match latent dimensions
        bbox_render_resized = nn.functional.interpolate(
            bbox_render,
            size=(lat_h, lat_w),
            mode="bicubic",
            align_corners=False,
        )
        # last channel is renderred bbox
        mask[:, 3, 0] = bbox_render_resized[:, 0]
        mask[:, 3, -1] = bbox_render_resized[:, -1]

        if self.diffusion_forcing.enabled:
            image_embeds = torch.zeros(
                batch_size,
                4 + lat_c,
                lat_t,
                lat_h,
                lat_w,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            padded_images = torch.zeros(batch_size, 3, t - 1, h, w, device=self.device)
            padded_images = torch.cat(
                [rearrange(images, "b 1 c h w -> b c 1 h w"), padded_images], dim=2
            )
            image_embeds = self.encode_video(
                padded_images
            )  # b, lat_c, lat_t, lat_h, lat_w
            image_embeds = torch.cat([mask, image_embeds], 1)
            mask[:, :2, 0] = 1
        batch["image_embeds"] = image_embeds

        return batch

    def decode_video(self, zs):
        rgb_lat, xyz_lat = torch.chunk(zs, dim=-1, chunks=2)
        xyz_lat = self.denormalize_xyz_latent(xyz_lat)
        rgb_lat = self.vae.decode(rgb_lat, self.vae_scale).clamp_(-1, 1)
        xyz_lat = self.vae.decode(xyz_lat, self.vae_scale).clamp_(-1, 1)
        return torch.cat([rgb_lat, xyz_lat], dim=-1)

    def normalize_xyz_latent(self, xyz_lat):
        if not self.xyz_latent_norm_enabled or not hasattr(self, "xyz_latent_mean"):
            return xyz_lat
        return (xyz_lat - self.xyz_latent_mean) * self.xyz_latent_inv_std

    def denormalize_xyz_latent(self, xyz_lat):
        if not self.xyz_latent_norm_enabled or not hasattr(self, "xyz_latent_mean"):
            return xyz_lat
        return xyz_lat / self.xyz_latent_inv_std + self.xyz_latent_mean

    def validation_step(self, batch, batch_idx=None):
        batch["videos"] = torch.cat([batch["rgb"], batch["xyz"]], dim=-1)
        return super().validation_step(batch, batch_idx)

    def visualize_local(self, video_vis, batch_idx):
        """
        Save RGB as mp4 and XYZ as npz
        """
        import os
        import imageio.v3 as iio
        output_dir = os.path.join(self.cfg.logging.save_dir, f"step_{self.global_step}")
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(video_vis)):
            if self.cfg.logging.video_type == "single":
                pred_rgb, pred_xyz = torch.chunk(video_vis[i], 2, dim=-1)
                pred_rgb = pred_rgb.numpy()
                pred_xyz = pred_xyz.numpy()
                iio.imwrite(os.path.join(output_dir, f"pred_rgb_{batch_idx}_{i}.mp4"), rearrange((pred_rgb * 255).astype(np.uint8), "t c h w -> t h w c"), fps=self.cfg.logging.fps)
                iio.imwrite(os.path.join(output_dir, f"pred_xyz_{batch_idx}_{i}.mp4"), rearrange((pred_xyz * 255).astype(np.uint8), "t c h w -> t h w c"), fps=self.cfg.logging.fps)
                # save XYZ as npz
                pred_xyz = (pred_xyz - 0.5) * 2.0
                np.savez_compressed(os.path.join(output_dir, f"pred_xyz_{batch_idx}_{i}.npz"), xyz=pred_xyz)
            else:
                pred_rgb, pred_xyz, gt_rgb, gt_xyz = torch.chunk(video_vis[i], 4, dim=-1) # (T, C, H, W) [0, 1]
                pred_rgb = pred_rgb.numpy()
                pred_xyz = pred_xyz.numpy()
                gt_rgb = gt_rgb.numpy()
                gt_xyz = gt_xyz.numpy()
                iio.imwrite(os.path.join(output_dir, f"pred_rgb_{batch_idx}_{i}.mp4"), rearrange((pred_rgb * 255).astype(np.uint8), "t c h w -> t h w c"), fps=self.cfg.logging.fps)
                iio.imwrite(os.path.join(output_dir, f"pred_xyz_{batch_idx}_{i}.mp4"), rearrange((pred_xyz * 255).astype(np.uint8), "t c h w -> t h w c"), fps=self.cfg.logging.fps)
                iio.imwrite(os.path.join(output_dir, f"gt_rgb_{batch_idx}_{i}.mp4"), rearrange((gt_rgb * 255).astype(np.uint8), "t c h w -> t h w c"), fps=self.cfg.logging.fps)
                iio.imwrite(os.path.join(output_dir, f"gt_xyz_{batch_idx}_{i}.mp4"), rearrange((gt_xyz * 255).astype(np.uint8), "t c h w -> t h w c"), fps=self.cfg.logging.fps)

                # save XYZ as npz
                pred_xyz = (pred_xyz - 0.5) * 2.0
                gt_xyz = (gt_xyz - 0.5) * 2.0
                np.savez_compressed(os.path.join(output_dir, f"pred_xyz_{batch_idx}_{i}.npz"), xyz=pred_xyz)
                np.savez_compressed(os.path.join(output_dir, f"gt_xyz_{batch_idx}_{i}.npz"), gt_xyz)

        return
