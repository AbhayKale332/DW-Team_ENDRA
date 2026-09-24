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
        self._ckpt = bool(cfg.grad_checkpoint_encoder)
        self._cl = bool(cfg.channels_last)
        self.frozen = True
        self.set_frozen(True)
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
    def set_frozen(self, frozen: bool, top_blocks: int = 0) -> None:
        """Freeze the backbone, or unfreeze `top_blocks` of it (0 = all of it).

        A partial unfreeze leaves the patch embedding and the lower blocks with
        `requires_grad=False`, which is all three of the places that matters:
        `llrd_param_groups` already skips them, DDP does not bucket them, and
        AdamW never allocates their two moment tensors.
        """
        self.frozen = bool(frozen)
        for p in self.model.parameters():
            p.requires_grad_(not self.frozen)
        if not self.frozen and top_blocks > 0 and self.blocks is not None:
            keep = set()
            for b in list(self.blocks)[-int(top_blocks):]:
                keep.update(id(p) for p in b.parameters())
            n_kept = 0
            for p in self.model.parameters():
                if id(p) in keep:
                    n_kept += 1
                else:
                    p.requires_grad_(False)
            print(f"[model] partial unfreeze: top {min(int(top_blocks), len(self.blocks))}"
                  f"/{len(self.blocks)} blocks trainable ({n_kept} tensors); "
                  f"patch embedding and lower blocks stay frozen")
        if self.frozen:
            self.model.eval()
        else:
            # Toggled here, once, instead of inside forward(): HF's
            # `gradient_checkpointing_enable` walks every submodule of the ViT
            # and rebuilds its forwards, and v4 was calling it on every
            # training step.  It is also off by default on a big card — see
            # `grad_checkpoint_encoder`, which trades ~35 % throughput for VRAM
            # there is no shortage of on an 80 GB H100.
            self._set_grad_checkpointing(self._ckpt)
            self._freeze_unreachable()
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[model] encoder {'FROZEN' if frozen else 'TRAINABLE'} "
              f"({n / 1e6:.1f}M grad params)")

    # -- parameters the tapped forward cannot reach ---------------------
    def _freeze_unreachable(self) -> list[str]:
        """Clear `requires_grad` on backbone parameters no gradient reaches.

        Unfreezing hands DDP ~415 encoder parameter tensors and three of them
        are not on any path from `pixel_values` to the four taps:

          * `embeddings.mask_token` — masked-image-modelling only.  It is read
            solely when `bool_masked_pos` is passed to the backbone, and
            `_hidden_states` never passes it.
          * `norm.weight` / `norm.bias` — on some `transformers` releases
            `DINOv3ViTModel` applies its final LayerNorm to `last_hidden_state`
            on the way out, while the taps come from `hidden_states`, the
            *pre-norm* block outputs, so that LayerNorm's result is thrown
            away.  On others the last tapped hidden state is post-norm and the
            gradient does reach it.  Which one you are on is not knowable from
            here, and it is exactly why the probe below decides rather than the
            name list — measured both ways on transformers >= 4.56.

        On one GPU those are three tensors AdamW steps with a `None` gradient,
        i.e. nothing.  Under DDP with `find_unused_parameters=False` they are a
        deadlock: the reducer registers a bucket for every `requires_grad`
        parameter at construction and then waits for gradients the graph never
        produces, the reduction never finishes, and the *next* forward dies in
        `_rebuild_buckets` with "Expected to have finished reduction in the
        prior iteration".  That is how the 2xT4 run died one step after the
        unfreeze, reporting "did not receive grad: 1 413 414" — which is
        exactly `embeddings.mask_token`, `norm.weight`, `norm.bias` in
        `named_parameters()` order (5 embedding tensors, then 24 blocks x 17,
        then the final norm).  Same failure mode as `fuse[3].rcu1` (indices
        60-65) one level down; see `models/dpt.py`.

        Found by probing, not by name: one tiny forward/backward through this
        very module, and whatever comes back with `grad is None` is by
        definition unreachable.  That survives a backbone swap or an HF
        refactor, which a hard-coded name list does not — the list is only the
        fallback for when the probe itself cannot run.
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
                  "the taps")
        return got

    def _probe_unreachable(self) -> list[str] | None:
        """Names of trainable backbone params that get no grad from `forward`.

        `None` means the probe could not be run, not that nothing is dead.

        The input is a deterministic ramp rather than `torch.randn`: it draws no
        numbers from the global RNG, so a resumed run's RNG state stays exactly
        where `_restore_rng` put it.  Values do not matter — only whether an
        autograd edge exists — but a *constant* input would make every
        LayerNorm see zero variance, so the ramp avoids the degenerate case.
        The probe runs in train mode so it exercises the same graph training
        will, gradient checkpointing included.
        """
        ref = next(self.model.parameters(), None)
        if ref is None:
            return []
        s = max(self.patch * 4, self.patch)
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
                outs = self.forward(x)
                total = outs[0].float().sum()
                for o in outs[1:]:
                    total = total + o.float().sum()
                total.backward()
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
            self.model.train(was_training)

    def _named_unreachable(self) -> list[str]:
        """The dead set for this backbone family, by name.

        Only consulted when `_probe_unreachable` raised.  Each name is checked
        against the live module, so a backbone without it is simply unaffected.
        """
        out = []
        emb = getattr(self.model, "embeddings", None)
        if isinstance(getattr(emb, "mask_token", None), nn.Parameter):
            out.append("embeddings.mask_token")
        # The ViT's own trailing LayerNorm, which sits after the last tapped
        # hidden state.  Matched as a direct attribute of the backbone so the
        # per-block `norm1` / `norm2` cannot be caught by accident.
        final_norm = getattr(self.model, "norm", None)
        if isinstance(final_norm, nn.Module):
            out += [f"norm.{nm}" for nm, _ in final_norm.named_parameters()]
        return out

    def _set_grad_checkpointing(self, on: bool) -> None:
        fn = ("gradient_checkpointing_enable" if on
              else "gradient_checkpointing_disable")
        try:
            if on:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                self.model.gradient_checkpointing_disable()
            print(f"[model] encoder gradient checkpointing {'ON' if on else 'OFF'}")
        except Exception as e:  # noqa: BLE001
            print(f"[model] {fn} unavailable: {e}")

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
        """(B, N, C) tokens -> (B, C, hp, wp) feature map.

        `transpose(1, 2).reshape(...)` cannot be a view, so it materialised a
        full copy of every tap.  Reshaping to (B, hp, wp, C) *is* a view, and the
        permute that follows leaves a tensor already laid out as channels_last —
        which is the format the DPT convs want, so the copy and the cudnn
        permute both disappear.
        """
        # Drop CLS + register/prefix tokens: the patch tokens are always the last
        # hp*wp entries regardless of how many prefix tokens the config uses.
        patches = tok[:, -(hp * wp):, :]
        m = patches.reshape(tok.shape[0], hp, wp, -1).permute(0, 3, 1, 2)
        return m if self._cl else m.contiguous()

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
        # No `.float()` here.  Under autocast the taps come out in bf16 and the
        # trunk's first op is a Conv2d that autocast casts back to bf16 anyway —
        # so the upcast was a pure round-trip, ~540 MB of fp32 allocated and
        # re-read per step at batch 32.  Without autocast the taps are already
        # fp32 and nothing changes.  Gradient checkpointing is toggled once in
        # `set_frozen`, not here: it used to be re-applied every single step.
        if self.frozen:
            self.model.eval()
            with torch.no_grad():
                outs = [self._tokens_to_map(h, hp, wp)
                        for h in self._hidden_states(pixel_values)]
            return [o.detach() for o in outs]

        return [self._tokens_to_map(h, hp, wp)
                for h in self._hidden_states(pixel_values)]
