"""Test-time augmentation: the 8 elements of D4, optionally x scales.

Nadir imagery has no canonical up, so flips and 90-degree rotations are exact
symmetries of the task — averaging over them is free accuracy.  v2 implemented
this and then never ran it: the process crashed before the final eval, so no TTA
number was ever produced.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_D4 = [(k, f) for f in (False, True) for k in range(4)]


def _apply(x, k, flip):
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def _invert(x, k, flip):
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, -k, dims=(-2, -1))


@torch.no_grad()
def tta_predict(model, image, scales=(1.0,), key: str = "fused",
                patch: int = 16, amp_dtype=None) -> torch.Tensor:
    h, w = image.shape[-2:]
    acc = torch.zeros((image.shape[0], 1, h, w), device=image.device, dtype=torch.float32)
    n = 0
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            img_s = image
        else:
            nh = max(patch, int(round(h * s / patch)) * patch)
            nw = max(patch, int(round(w * s / patch)) * patch)
            img_s = F.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=False)
        for k, flip in _D4:
            ctx = (torch.autocast("cuda", dtype=amp_dtype)
                   if (amp_dtype is not None and image.is_cuda)
                   else torch.autocast("cpu", enabled=False))
            with ctx:
                pred = model(_apply(img_s, k, flip))[key]
            pred = _invert(pred.float(), k, flip)
            if pred.shape[-2:] != (h, w):
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
            acc += pred
            n += 1
    return acc / max(1, n)
