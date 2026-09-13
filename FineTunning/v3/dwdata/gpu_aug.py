"""Batched preprocessing that runs on the GPU instead of in a DataLoader worker.

Why this exists
---------------
Profiling one `TileDataset.__getitem__` on the packed GAMUS shards (512 px crop
out of a 1024 px tile) gave ~82 ms, spent like this:

    photometric_jitter      17.3 ms   pointwise float maths over 786 kB
    scene_stretch_bounds    13.4 ms   full-tile histogram, per crop
    PIL resize               7.9 ms   the actual geometric work
    normalise                4.5 ms   uint8 -> float32 CHW, /255, (x-m)/s
    assorted astype/copies  ~14 ms    float32 temporaries

At ~12 samples/s/worker, an L40S wanting ~40 samples/s needs the loader to not
be doing any of the first, second or fourth line.  So:

  * the stretch bounds are precomputed once per store and persisted
    (`PackedStore.prime_stretch_bounds`),
  * the photometric jitter and the encoder normalisation move here, onto the
    card, where the whole batch is one set of pointwise kernels (~2 ms at
    batch 32) instead of 32 x 22 ms of single-threaded numpy,
  * and the worker hands over **uint8 HWC**, which is 4x less host-to-device
    traffic than float32 CHW and permutes into `channels_last` for free.

The maths is the same as `dwdata/augment.photometric_jitter`, drawn per sample
instead of per crop: gamma, per-channel gain, saturation, contrast, brightness,
optional separable blur, gaussian noise, in that order, each gated by
`cfg.photo_p`.  Inference does not jitter, so nothing here touches the
`PreprocSpec` contract beyond `normalise`, which is byte-identical to the numpy
version up to float rounding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_LUMA = (0.299, 0.587, 0.114)


class GpuPreproc:
    """Device-resident normalisation + photometric jitter for a whole batch."""

    def __init__(self, spec, cfg, device):
        self.cfg = cfg
        self.device = device
        self.mean = torch.tensor(spec.mean, dtype=torch.float32,
                                 device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(spec.std, dtype=torch.float32,
                                device=device).view(1, 3, 1, 1)
        self.luma = torch.tensor(_LUMA, dtype=torch.float32,
                                 device=device).view(1, 3, 1, 1)

    # -- entry point ----------------------------------------------------
    def __call__(self, batch: dict, train: bool) -> dict:
        """In-place-ish: replaces `image_u8` with a normalised `image`.

        A batch that already carries `image` (the CPU path, or a val loader
        built before this existed) is passed through untouched, so the two
        pipelines stay interchangeable.
        """
        u8 = batch.pop("image_u8", None)
        if u8 is None:
            return batch
        # (B, H, W, 3) uint8 -> (B, 3, H, W) float, already channels_last.
        x = u8.permute(0, 3, 1, 2).float().div_(255.0)
        if train:
            x = self.jitter(x)
        batch["image"] = x.sub_(self.mean).div_(self.std).contiguous(
            memory_format=torch.channels_last)
        if "cls" in batch and batch["cls"].dtype != torch.long:
            batch["cls"] = batch["cls"].long()
        return batch

    # -- the jitter -----------------------------------------------------
    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] -> jittered, same shape/dtype."""
        cfg = self.cfg
        b = x.shape[0]
        dev = x.device
        on = (torch.rand(b, 1, 1, 1, device=dev) < cfg.photo_p).float()

        def gated(value, neutral):
            """`value` where this sample is being jittered, `neutral` elsewhere."""
            return on * value + (1.0 - on) * neutral

        def u(lo, hi, shape=(b, 1, 1, 1)):
            return torch.empty(shape, device=dev).uniform_(lo, hi)

        if cfg.photo_gamma > 0:
            g = gated(torch.exp(u(-cfg.photo_gamma, cfg.photo_gamma)), 1.0)
            x = x.clamp_(1e-4, 1.0).pow(g)
        if cfg.photo_channel_gain > 0:
            c = cfg.photo_channel_gain
            x = x * gated(1.0 + u(-c, c, (b, 3, 1, 1)), 1.0)
        if cfg.photo_saturation > 0:
            grey = (x * self.luma).sum(1, keepdim=True)
            f = gated(1.0 + u(-cfg.photo_saturation, cfg.photo_saturation), 1.0)
            x = (x - grey) * f + grey
        if cfg.photo_contrast > 0:
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            f = gated(1.0 + u(-cfg.photo_contrast, cfg.photo_contrast), 1.0)
            x = (x - m) * f + m
        if cfg.photo_brightness > 0:
            x = x + gated(u(-cfg.photo_brightness, cfg.photo_brightness), 0.0)
        x = x.clamp_(0.0, 1.0)

        if cfg.photo_blur_p > 0:
            do = (torch.rand(b, device=dev) < cfg.photo_blur_p) & (on.view(b) > 0)
            if bool(do.any()):
                x = _blur(x, u(0.4, 1.3, (b,)), do)
        if cfg.photo_noise_std > 0:
            sd = gated(u(0.0, cfg.photo_noise_std), 0.0)
            x = x + torch.randn_like(x) * sd
        return x.clamp_(0.0, 1.0)


def _blur(x: torch.Tensor, sigma: torch.Tensor, do: torch.Tensor,
          radius: int = 4) -> torch.Tensor:
    """Separable gaussian with a *per-sample* sigma, as one grouped conv.

    Samples with `do == False` get an identity kernel rather than a second
    kernel launch — a masked `where` on a (B, K) tensor is free next to
    branching the batch apart and stitching it back together.
    """
    b, c, h, w = x.shape
    k = 2 * radius + 1
    off = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    s = sigma.to(x.dtype).view(b, 1).clamp_min(1e-3)
    g = torch.exp(-(off.view(1, k) ** 2) / (2.0 * s * s))
    g = g / g.sum(1, keepdim=True)
    ident = torch.zeros_like(g)
    ident[:, radius] = 1.0
    g = torch.where(do.view(b, 1), g, ident)

    w_row = g.repeat_interleave(c, dim=0).view(b * c, 1, 1, k)
    w_col = w_row.view(b * c, 1, k, 1)
    y = x.reshape(1, b * c, h, w)
    y = F.conv2d(F.pad(y, (radius, radius, 0, 0), mode="replicate"),
                 w_row, groups=b * c)
    y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="replicate"),
                 w_col, groups=b * c)
    return y.reshape(b, c, h, w)
