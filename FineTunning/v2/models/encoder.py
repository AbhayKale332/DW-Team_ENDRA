"""DINOv3 ViT encoder (frozen, with a late partial-unfreeze hook).

Lifted from `FineTunning/v1/kaggle_phase0.py::DINOv3Encoder` with one addition:
`unfreeze_last_n_blocks(n)` so the trainer can un-freeze the last few transformer
blocks for the final phase of stage F (`.agents/Depth_Wizard_Plan.md`:
"optionally last 4 blocks unfrozen late").
"""

from __future__ import annotations

import torch
from torch import nn


class DINOv3Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel

        try:
            self.model = AutoModel.from_pretrained(
                cfg.encoder_model_id, token=cfg.hf_token or None,
                output_hidden_states=True, attn_implementation="sdpa",
            )
        except (ValueError, TypeError, ImportError, KeyError):
            self.model = AutoModel.from_pretrained(
                cfg.encoder_model_id, token=cfg.hf_token or None, output_hidden_states=True,
            )
        self.patch = int(getattr(self.model.config, "patch_size", 16))
        self.hidden = int(self.model.config.hidden_size)
        self.n_layers = int(getattr(self.model.config, "num_hidden_layers", 24))

        idx = tuple(cfg.encoder_feature_indices)
        if max(idx) > self.n_layers:
            idx = tuple(round(self.n_layers * q) for q in (0.25, 0.5, 0.75, 1.0))
            print(f"[model] taps exceed {self.n_layers} blocks -> {idx}")
        self.idx = idx

        self.frozen = cfg.freeze_encoder
        self._half = cfg.encoder_half
        if self.frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()
            if self._half:
                self.model.half()
        self.unfrozen_block_ids: list[int] = []
        print(f"[model] encoder={cfg.encoder_model_id} hidden={self.hidden} "
              f"patch={self.patch} taps={self.idx} frozen={self.frozen}")

    # -- late partial unfreeze ------------------------------------
    def _blocks(self):
        for attr in ("layer", "layers", "blocks"):
            enc = getattr(self.model, "encoder", self.model)
            b = getattr(enc, attr, None) or getattr(self.model, attr, None)
            if b is not None:
                return b
        raise AttributeError("cannot locate transformer blocks on the encoder")

    def unfreeze_last_n_blocks(self, n: int) -> list[nn.Parameter]:
        if n <= 0:
            return []
        if self._half:  # need fp32 for stable grads
            self.model.float()
            self._half = False
        blocks = self._blocks()
        total = len(blocks)
        newly: list[nn.Parameter] = []
        for bi in range(total - n, total):
            for p in blocks[bi].parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    newly.append(p)
            self.unfrozen_block_ids.append(bi)
        self.frozen = False
        print(f"[model] unfroze encoder blocks {self.unfrozen_block_ids} "
              f"({sum(p.numel() for p in newly)/1e6:.1f}M params)")
        return newly

    # -- forward ------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def _tokens_to_map(self, tok: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        patches = tok[:, -(hp * wp):, :]
        return patches.transpose(1, 2).reshape(tok.shape[0], self.hidden, hp, wp)

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        hp = pixel_values.shape[-2] // self.patch
        wp = pixel_values.shape[-1] // self.patch
        if self.frozen:
            self.model.eval()
            p_dtype = next(self.model.parameters()).dtype
            with torch.no_grad():
                hs = self.model(pixel_values=pixel_values.to(p_dtype)).hidden_states
                outs = [self._tokens_to_map(hs[i], hp, wp).float() for i in self.idx]
            return [o.detach() for o in outs]
        hs = self.model(pixel_values=pixel_values).hidden_states
        return [self._tokens_to_map(hs[i], hp, wp) for i in self.idx]
