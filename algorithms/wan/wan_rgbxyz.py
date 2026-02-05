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

    def configure_model(self):
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
        if self.cfg.vae.compile:
            self.vae = torch.compile(self.vae)

        # Initialize main diffusion model
        if self.cfg.model.tuned_ckpt_path is None:
            self.model = WanRGBXYZModel.from_pretrained(self.cfg.model.ckpt_path)
        else:
            self.model = WanRGBXYZModel.from_config(
                WanRGBXYZModel._dict_from_json_file(self.cfg.model.ckpt_path + "/config.json")
            )
            if self.is_inference:
                self.model.to(torch.bfloat16)
            self.model.load_state_dict(
                self._load_tuned_state_dict(), assign=not self.is_inference
            )
        if not self.is_inference:
            self.model.to(self.dtype).train()
        if self.gradient_checkpointing_rate > 0:
            self.model.gradient_checkpointing_enable(p=self.gradient_checkpointing_rate)
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
        rgb_lat = self.vae.decode(rgb_lat, self.vae_scale).clamp_(-1, 1)
        xyz_lat = self.vae.decode(xyz_lat, self.vae_scale).clamp_(-1, 1)
        return torch.cat([rgb_lat, xyz_lat], dim=-1)