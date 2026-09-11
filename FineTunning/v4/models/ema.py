"""Exponential moving average of the trainable weights.

Cheap, reliably worth 2-5 % RMSE on a cosine-annealed run, and it makes the final
checkpoint insensitive to which exact step training happened to stop on — which
matters here because the run is wall-clock capped and can be cut mid-epoch.
"""

from __future__ import annotations

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n += 1
        # warm up the decay so the first steps aren't dominated by the init
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            s = self.shadow.get(k)
            if s is None:
                continue
            if s.shape != v.shape:                 # a param group changed shape
                self.shadow[k] = v.detach().clone().float()
                continue
            s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    class swapped:
        """`with ema.swapped(model): ...` — evaluate on the EMA weights."""

        def __init__(self, ema: "ModelEMA", model: nn.Module):
            self.ema, self.model = ema, model
            self.backup: dict = {}

        def __enter__(self):
            sd = self.model.state_dict()
            for k, v in self.ema.shadow.items():
                if k in sd and sd[k].shape == v.shape:
                    self.backup[k] = sd[k].detach().clone()
                    sd[k].copy_(v.to(sd[k].dtype))
            return self.model

        def __exit__(self, *exc):
            sd = self.model.state_dict()
            for k, v in self.backup.items():
                sd[k].copy_(v)
            self.backup.clear()
            return False
