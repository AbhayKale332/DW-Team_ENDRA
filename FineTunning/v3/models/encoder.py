"""DINOv3 ViT encoder with a *robust* block finder, real unfreezing and LLRD.

v2 died here.  Its `_blocks()` tried `model.layer / .layers / .blocks` on the
model and on `model.encoder`, found none of them on the HF DINOv3 module, and
raised `AttributeError: cannot locate transformer blocks on the encoder` at
epoch 21 of 30 — the exact epoch the late unfreeze was scheduled.  Consequences
in the real run: the encoder was **never** unfrozen, the process died before the
final evaluation, and no TTA / viewer / qualitative artefacts were ever written.
The whole run therefore trained 11 M decoder parameters on top of a frozen
backbone, which is v1's architecture with extra steps.

v3 finds the blocks structurally instead of by name — walk the module tree for
the `nn.ModuleList` whose length equals `config.num_hidden_layers` — so it cannot
break on a naming change.  `--freeze_epochs` warm-starts the decoder, then the
**whole** encoder is unfrozen with layer-wise LR decay, which is the standard
ViT dense-prediction recipe and the largest single lever v2 left on the table.
"""

from __future__ import annotations

import torch
from torch import nn


class DINOv3Encoder(nn.Module):
    # Overridable so tests (and any future backbone swap) can inject a module
    # without reaching the hub.  Must expose `.config` and return
    # `hidden_states` from forward.
    BUILDER = None

    def __init__(self, cfg):
        super().__init__()
        if type(self).BUILDER is not None:
            self.model = type(self).BUILDER(cfg)
        else:
            from transformers import AutoModel

            kw = dict(token=cfg.hf_token or None, output_hidden_states=True)
            try:
                self.model = AutoModel.from_pretrained(
                    cfg.encoder_model_id, attn_implementation="sdpa", **kw)
            except (ValueError, TypeError, ImportError, KeyError):
                self.model = AutoModel.from_pretrained(cfg.encoder_model_id, **kw)

        c = self.model.config
        self.patch = int(getattr(c, "patch_size", 16))
        self.hidden = int(c.hidden_size)
        self.n_layers = int(getattr(c, "num_hidden_layers", 24))

        idx = tuple(cfg.encoder_feature_indices)
        if max(idx) > self.n_layers:
            idx = tuple(max(1, round(self.n_layers * q)) for q in (0.25, 0.5, 0.75, 1.0))
            print(f"[model] taps exceed {self.n_layers} blocks -> {idx}")
        self.idx = idx

        self.blocks = self._find_blocks()
        self.frozen = True
        self.set_frozen(True)
        self._ckpt = bool(cfg.grad_checkpoint_encoder)
        print(f"[model] encoder={cfg.encoder_model_id} hidden={self.hidden} "
              f"patch={self.patch} blocks={len(self.blocks) if self.blocks else '?'} "
              f"taps={self.idx}")

    # -- block discovery ------------------------------------------------
    def _find_blocks(self) -> nn.ModuleList | None:
        """The ModuleList of transformer blocks, located by shape not by name."""
        exact, any_list = None, None
        for _name, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                if len(mod) == self.n_layers and exact is None:
                    exact = mod
                if any_list is None or len(mod) > len(any_list):
                    any_list = mod
        found = exact if exact is not None else any_list
        if found is None:
            print("[model] WARNING: no transformer block list found — "
                  "unfreeze and LLRD will fall back to whole-encoder groups")
        return found

    # -- freezing -------------------------------------------------------
    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        for p in self.model.parameters():
            p.requires_grad_(not self.frozen)
        if self.frozen:
            self.model.eval()
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[model] encoder {'FROZEN' if frozen else 'TRAINABLE'} "
              f"({n / 1e6:.1f}M grad params)")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    # -- layer-wise LR decay -------------------------------------------
    def llrd_param_groups(self, base_lr: float, decay: float, weight_decay: float):
        """One param group per block, LR decaying towards the input.

        Depth 0 = embeddings, depth L = the last block; group LR is
        `base_lr * decay ** (L - depth)`.  Norm/bias params get no weight decay.
        """
        if self.blocks is None:
            return [{"params": [p for p in self.model.parameters() if p.requires_grad],
                     "lr": base_lr, "weight_decay": weight_decay, "name": "encoder"}]

        block_ids = {id(p) for b in self.blocks for p in b.parameters()}
        depth_of: dict[int, int] = {}
        for d, b in enumerate(self.blocks, start=1):
            for p in b.parameters():
                depth_of[id(p)] = d
        L = len(self.blocks)

        buckets: dict[tuple[int, bool], list] = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            depth = depth_of.get(id(p), 0 if id(p) not in block_ids else L)
            no_decay = p.ndim <= 1 or name.endswith(".bias")
            buckets.setdefault((depth, no_decay), []).append(p)

        groups = []
        for (depth, no_decay), params in sorted(buckets.items()):
            groups.append({
                "params": params,
                "lr": base_lr * (decay ** (L - depth)),
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"enc.d{depth}{'.nd' if no_decay else ''}",
            })
        return groups

    # -- forward --------------------------------------------------------
    def _tokens_to_map(self, tok: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        # Drop CLS + register/prefix tokens: the patch tokens are always the last
        # hp*wp entries regardless of how many prefix tokens the config uses.
        patches = tok[:, -(hp * wp):, :]
        return patches.transpose(1, 2).reshape(tok.shape[0], -1, hp, wp)

    def _hidden_states(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """The tapped hidden states, robust to how the backbone is configured.

        `output_hidden_states` is passed per call rather than trusted from the
        config: setting it only via `from_pretrained` leaves `hidden_states=None`
        on transformers 5.x, which turns the whole decoder into a TypeError at
        the first forward.
        """
        out = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        if hs is None and isinstance(out, (tuple, list)):
            hs = out[-1]
        if hs is None:
            raise RuntimeError(
                f"{type(self.model).__name__} returned no hidden_states; "
                "the encoder taps cannot be built")
        return [hs[i] for i in self.idx]

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        hp = pixel_values.shape[-2] // self.patch
        wp = pixel_values.shape[-1] // self.patch
        if self.frozen:
            self.model.eval()
            with torch.no_grad():
                outs = [self._tokens_to_map(h, hp, wp).float()
                        for h in self._hidden_states(pixel_values)]
            return [o.detach() for o in outs]

        if self._ckpt and self.training:
            # HF exposes gradient checkpointing on the module itself; enabling it
            # here (rather than hand-rolling `checkpoint()` around blocks) keeps
            # hidden_states plumbing intact.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        return [self._tokens_to_map(h, hp, wp).float()
                for h in self._hidden_states(pixel_values)]
