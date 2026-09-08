"""Test-time augmentation: 8x dihedral (flip x rot90) + multi-scale, averaged.

Nadir imagery is rotation/flip invariant, so this is ~free accuracy (the plan's
"cheap ensemble").  Eval only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# (k_rot90, flip)  — the 8 elements of the dihedral group D4
_D4 = [(k, f) for f in (False, True) for k in range(4)]


def _apply(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, k, dims=(-2, -1))
    if flip:
        x = torch.flip(x, dims=(-1,))
    return x


def _invert(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, -k, dims=(-2, -1))


@torch.no_grad()
def tta_predict(model, image: torch.Tensor, scales=(1.0,), key: str = "fused") -> torch.Tensor:
    """Return the averaged height prediction (B, 1, H, W) over D4 x scales."""
    h, w = image.shape[-2:]
    acc = torch.zeros((image.shape[0], 1, h, w), device=image.device, dtype=torch.float32)
    n = 0
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            img_s = image
        else:
            nh = int(round(h * s / 16)) * 16
            nw = int(round(w * s / 16)) * 16
            img_s = F.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=False)
        for k, flip in _D4:
            pred = model(_apply(img_s, k, flip))[key].float()
            pred = _invert(pred, k, flip)
            if pred.shape[-2:] != (h, w):
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
            acc += pred
            n += 1
    return acc / max(1, n)
