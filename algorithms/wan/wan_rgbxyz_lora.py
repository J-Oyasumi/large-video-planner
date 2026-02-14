import logging

import torch
from transformers import get_scheduler

from .lora_utils import count_trainable_parameters, inject_lora_adapters
from .wan_rgbxyz import WanRGBXYZ


class WanRGBXYZLoRA(WanRGBXYZ):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._lora_applied = False

    @staticmethod
    def _set_requires_grad(module, enabled: bool):
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    def _enable_optional_trainable_modules(self):
        lora_cfg = self.cfg.lora

        if bool(getattr(lora_cfg, "train_modality_embedding", True)) and hasattr(
            self.model, "modality_embedding"
        ):
            self._set_requires_grad(self.model.modality_embedding, True)

        if bool(getattr(lora_cfg, "train_head", False)):
            self._set_requires_grad(self.model.head, True)

        if bool(getattr(lora_cfg, "train_patch_embedding", False)):
            self._set_requires_grad(self.model.patch_embedding, True)

        if bool(getattr(lora_cfg, "train_time_embedding", False)):
            self._set_requires_grad(self.model.time_embedding, True)
            self._set_requires_grad(self.model.time_projection, True)

        if bool(getattr(lora_cfg, "train_text_embedding", False)):
            self._set_requires_grad(self.model.text_embedding, True)

        if bool(getattr(lora_cfg, "train_img_emb", False)) and hasattr(
            self.model, "img_emb"
        ):
            self._set_requires_grad(self.model.img_emb, True)

    def _apply_lora(self):
        if self._lora_applied:
            return

        lora_cfg = getattr(self.cfg, "lora", None)
        if lora_cfg is None or not bool(getattr(lora_cfg, "enabled", False)):
            logging.info("LoRA disabled. Falling back to full fine-tuning behavior.")
            self._lora_applied = True
            return

        rank = int(lora_cfg.rank)
        alpha = float(lora_cfg.alpha)
        dropout = float(lora_cfg.dropout)
        target_modules = list(lora_cfg.target_modules)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        replaced = inject_lora_adapters(
            model=self.model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if not replaced:
            raise RuntimeError(
                "No linear layers matched LoRA targets. "
                "Please check algorithm.lora.target_modules."
            )

        self._enable_optional_trainable_modules()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = count_trainable_parameters(self.model.parameters())
        logging.info(
            "LoRA enabled: replaced=%d trainable=%d total=%d (%.4f%%)",
            len(replaced),
            trainable_params,
            total_params,
            100.0 * trainable_params / max(total_params, 1),
        )
        logging.info("LoRA targets (first 20): %s", ", ".join(replaced[:20]))
        self._lora_applied = True

    def configure_model(self):
        super().configure_model()
        self._apply_lora()

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError(
                "No trainable parameters found for LoRA run. "
                "Enable LoRA or optional trainable modules."
            )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
        )
        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }
