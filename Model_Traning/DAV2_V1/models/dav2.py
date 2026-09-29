"""Depth Anything V2 backbone + its pretrained DPT neck, as one module.

This stands in for v4's `DINOv3Encoder` **and** `DPTTrunk` together, so
`DepthWizardNet.forward` keeps the same shape: image in, `(B, decoder_dim,
H/2, W/2)` out.

Why swap at all: every run so far bolted a *randomly initialised* DPT trunk
onto DINOv3.  DAv2 ships a decoder that was trained for monocular depth, and a
ViT-B/14 encoder that is 86 M parameters against ViT-L/16's 303 M.  The bet is
that task-matched pretraining beats DINOv3-SAT's domain-matched pretraining;
see `.agents/DAv2_Implementation_Plan.md` for the full argument.

`.head` is discarded — it predicts relative *inverse* depth through a sigmoid,
and this model predicts metric nDSM in metres.

Shapes, for a 518 px tile (518 = 37 x 14):

    backbone.feature_maps        4 taps, (B, 1 + 37*37, 768)
    neck(taps, 37, 37)           reassemble x(4, 2, 1, 0.5) -> 148/74/37/19,
                                 fusion upsamples x2 four times from the
                                 coarsest
    fusion[-1]                   (B, 128, 296, 296)      # 8 x 37
    interpolate -> (H/2, W/2)    (B, 128, 259, 259)
    proj 1x1                     (B, decoder_dim, 259, 259)

The interpolate is what keeps HeadB's bin logits at half resolution.  Without
it they would be (B, 96, 296, 296) — 1.3 GB at batch 8 in fp16.

`self.proj` deliberately sits *outside* `self.model`, because `encoder.model.`
is the prefix `head_state_dict()` strips: the backbone and the neck are public
pretrained weights, the projection is not.  It is also never frozen, so it
trains during the warm-start epochs along with the heads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DAV2Backbone(nn.Module):
    # Overridable so tests can inject a module without reaching the hub.  Must
    # return something with `.backbone`, `.neck` and `.config`.
    BUILDER = None

    def __init__(self, cfg):
        super().__init__()
        if type(self).BUILDER is not None:
            self.model = type(self).BUILDER(cfg)
        else:
            from transformers import DepthAnythingForDepthEstimation

            try:
                self.model = DepthAnythingForDepthEstimation.from_pretrained(
                    cfg.encoder_model_id, attn_implementation="sdpa")
            except (ValueError, TypeError, ImportError, KeyError):
                self.model = DepthAnythingForDepthEstimation.from_pretrained(
                    cfg.encoder_model_id)
        # The relative-inverse-depth head is dead weight here, and leaving it
        # attached would hand DDP a few hundred thousand parameters that no
        # gradient can reach.
        if getattr(self.model, "head", None) is not None:
            del self.model.head

        c = self.model.config
        bc = getattr(c, "backbone_config", None) or c
        self.patch = int(getattr(c, "patch_size", 14))
        self.hidden = int(getattr(bc, "hidden_size", 768))
        self.n_layers = int(getattr(bc, "num_hidden_layers", 12))
        self.fusion_hidden = int(getattr(c, "fusion_hidden_size", 128))

        self.proj = nn.Conv2d(self.fusion_hidden, int(cfg.decoder_dim), 1)

        self.blocks = self._find_blocks()
        self._ckpt = bool(cfg.grad_checkpoint_encoder)
        self._cl = bool(cfg.channels_last)
        self.frozen = True
        self.set_frozen(True)
        print(f"[model] encoder={cfg.encoder_model_id} hidden={self.hidden} "
              f"patch={self.patch} blocks={len(self.blocks) if self.blocks else '?'} "
              f"fusion={self.fusion_hidden} -> decoder_dim={cfg.decoder_dim}")

    # -- block discovery ------------------------------------------------
    def _find_blocks(self) -> nn.ModuleList | None:
        """The ModuleList of transformer blocks, located by shape not by name.

        Verbatim from v4's `models/encoder.py`, and for the same reason: v2
        looked them up by attribute name, did not find them on the HF module,
        and the encoder was therefore never unfrozen.  Searched under the
        *backbone* only, so the neck's own ModuleLists cannot win the
        "longest list" fallback.
        """
        exact, any_list = None, None
        for _name, mod in self.model.backbone.named_modules():
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
    def set_frozen(self, frozen: bool, top_blocks: int = 0) -> None:
        """Freeze backbone + neck, or unfreeze `top_blocks` of the backbone.

        The neck freezes and unfreezes *with* the encoder: it is pretrained
        depth-specific weight, not a random trunk, so training it from step 0
        against random heads is the one thing `--freeze_epochs` exists to
        prevent.  `self.proj` is never touched here — it is random and belongs
        to the decoder's optimiser group.
        """
        self.frozen = bool(frozen)
        for p in self.model.parameters():
            p.requires_grad_(not self.frozen)
        if not self.frozen and top_blocks > 0 and self.blocks is not None:
            keep = set()
            for b in list(self.blocks)[-int(top_blocks):]:
                keep.update(id(p) for p in b.parameters())
            keep.update(id(p) for p in self.model.neck.parameters())
            n_kept = 0
            for p in self.model.parameters():
                if id(p) in keep:
                    n_kept += 1
                else:
                    p.requires_grad_(False)
            print(f"[model] partial unfreeze: top {min(int(top_blocks), len(self.blocks))}"
                  f"/{len(self.blocks)} blocks + neck trainable ({n_kept} tensors); "
                  f"patch embedding and lower blocks stay frozen")
        if self.frozen:
            self.model.eval()
        else:
            # Once, here — not per forward.  v4 re-applied it every training
            # step, and `gradient_checkpointing_enable` walks every submodule
            # of the ViT and rebuilds its forwards.
            self._set_grad_checkpointing(self._ckpt)
            self._freeze_unreachable()
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[model] encoder {'FROZEN' if frozen else 'TRAINABLE'} "
              f"({n / 1e6:.1f}M grad params)")

    # -- parameters the forward cannot reach -----------------------------
    def _freeze_unreachable(self) -> list[str]:
        """Clear `requires_grad` on parameters no gradient reaches.

        Under DDP with `find_unused_parameters=False` the reducer registers a
        bucket for every `requires_grad` parameter and then waits forever for
        gradients the graph never produces; the next forward dies with
        "Expected to have finished reduction in the prior iteration".  That is
        how the 2xT4 v4 run died one step after the unfreeze.

        This backbone has at least two such places and they are structural, not
        incidental:

          * `backbone.embeddings.mask_token` — masked-image-modelling only.
          * `neck.fusion_stage.layers[0].residual_layer1` — the deepest fusion
            layer is called with no residual, so its first residual unit never
            runs.  Exactly v4's `fuse[3].rcu1`, one level down and in somebody
            else's code.

        Found by probing rather than by name, so it survives a version bump of
        `transformers`; the name list is only the fallback for when the probe
        itself cannot run.
        """
        dead = self._probe_unreachable()
        if dead is None:
            dead = self._named_unreachable()
        by_name = dict(self.model.named_parameters())
        got = []
        for nm in dead:
            p = by_name.get(nm)
            if p is not None and p.requires_grad:
                p.requires_grad_(False)
                got.append(nm)
        if got:
            print(f"[model] {len(got)} unreachable encoder parameter(s) left "
                  f"frozen (never receive gradients, would hang DDP): "
                  f"{', '.join(got)}")
        else:
            print("[model] encoder probe: every trainable parameter reaches "
                  "the output")
        return got

    def _probe_unreachable(self) -> list[str] | None:
        """Names of trainable params that get no grad from `forward`.

        `None` means the probe could not be run, not that nothing is dead.

        The input is a deterministic ramp rather than `torch.randn`: it draws
        no numbers from the global RNG, so a resumed run's RNG state stays
        where `_restore_rng` put it.  A *constant* input would make every
        LayerNorm see zero variance, hence the ramp.  Four patches a side is
        the smallest grid the x0.5 reassemble stage can halve.
        """
        ref = next(self.model.parameters(), None)
        if ref is None:
            return []
        s = self.patch * 4
        n = 3 * s * s
        x = (torch.arange(n, device=ref.device, dtype=torch.float32)
             .reshape(1, 3, s, s) / n).to(ref.dtype)
        was_training = self.model.training
        stash = [(p, p.grad) for p in self.model.parameters()]
        try:
            for p, _ in stash:
                p.grad = None
            self.model.train()
            with torch.enable_grad():
                self.forward(x).float().sum().backward()
            return [nm for nm, p in self.model.named_parameters()
                    if p.requires_grad and p.grad is None]
        except Exception as e:  # noqa: BLE001
            print(f"[model] unreachable-parameter probe failed ({e}) — "
                  "falling back to the known dead set")
            return None
        finally:
            # The probe's gradients are not training signal; drop them and put
            # back whatever was there (nothing, at every call site).
            for p, g in stash:
                p.grad = g
            self.proj.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def _named_unreachable(self) -> list[str]:
        """The dead set for this architecture, by name. Fallback only."""
        out = []
        emb = getattr(self.model.backbone, "embeddings", None)
        if isinstance(getattr(emb, "mask_token", None), nn.Parameter):
            out.append("backbone.embeddings.mask_token")
        fusion = getattr(self.model.neck, "fusion_stage", None)
        layers = getattr(fusion, "layers", None)
        if layers is not None and len(layers) > 0:
            first = layers[0]
            if getattr(first, "residual_layer1", None) is not None:
                out += [f"neck.fusion_stage.layers.0.residual_layer1.{nm}"
                        for nm, _ in first.residual_layer1.named_parameters()]
        return out

    def _set_grad_checkpointing(self, on: bool) -> None:
        # The neck is convolutional and has no checkpointing hook; the ViT is
        # where the activations are anyway.
        try:
            if on:
                self.model.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                self.model.backbone.gradient_checkpointing_disable()
            print(f"[model] encoder gradient checkpointing {'ON' if on else 'OFF'}")
        except Exception as e:  # noqa: BLE001
            print(f"[model] gradient checkpointing unavailable: {e}")

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

        The **neck is exempt from the decay** and sits at `base_lr`.  It is
        pretrained depth-specific weight, so it should neither be trained at
        the decoder's 3e-4 nor be buried at the bottom of the LLRD ladder with
        the patch embedding.
        """
        if self.blocks is None:
            return [{"params": [p for p in self.model.parameters() if p.requires_grad],
                     "lr": base_lr, "weight_decay": weight_decay, "name": "encoder"}]

        depth_of: dict[int, int] = {}
        for d, b in enumerate(self.blocks, start=1):
            for p in b.parameters():
                depth_of[id(p)] = d
        L = len(self.blocks)

        buckets: dict[tuple[int, bool], list] = {}
        neck: dict[bool, list] = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            no_decay = p.ndim <= 1 or name.endswith(".bias")
            if name.startswith("neck."):
                neck.setdefault(no_decay, []).append(p)
            else:
                buckets.setdefault((depth_of.get(id(p), 0), no_decay), []).append(p)

        groups = []
        for (depth, no_decay), params in sorted(buckets.items()):
            groups.append({
                "params": params,
                "lr": base_lr * (decay ** (L - depth)),
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"enc.d{depth}{'.nd' if no_decay else ''}",
            })
        for no_decay, params in sorted(neck.items()):
            groups.append({
                "params": params,
                "lr": base_lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"neck{'.nd' if no_decay else ''}",
            })
        return groups

    # -- forward --------------------------------------------------------
    def _neck_out(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ph = pixel_values.shape[-2] // self.patch
        pw = pixel_values.shape[-1] // self.patch
        taps = self.model.backbone(pixel_values).feature_maps
        # `fusion_stage` returns coarsest-first; DAv2's own head reads [-1]
        # (`config.head_in_index`), which is the finest map.  Each of the four
        # fusion layers upsamples x2 and the last one is unconstrained by a
        # `size=`, so the result is 8x the token grid: 37 -> 296.
        return self.model.neck(list(taps), ph, pw)[-1]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h, w = pixel_values.shape[-2:]
        if self.frozen:
            self.model.eval()
            with torch.no_grad():
                x = self._neck_out(pixel_values)
            x = x.detach()
        else:
            x = self._neck_out(pixel_values)
        # Back to the half-resolution contract every head here was written for.
        x = F.interpolate(x, size=(max(1, h // 2), max(1, w // 2)),
                          mode="bilinear", align_corners=False)
        if self._cl:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.proj(x)
