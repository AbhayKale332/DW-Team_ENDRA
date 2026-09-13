"""The preprocessing *contract* — the single definition of "what the model eats".

This module is the answer to "how do I make inference input identical to training
input".  Training (`dwdata/dataset.py`) and inference (`infer/predict.py`) both
build their tensors here and nowhere else, and the resolved spec is serialised
into every checkpoint (`PreprocSpec.to_dict()`), so a checkpoint always carries
the recipe that produced it.

The contract, in order:

  1. **Read**  RGB uint8 (H, W, 3) + the ground sample distance in metres/pixel.
     For a GeoTIFF the GSD comes from the affine transform (converted to metres
     if the CRS is geographic); otherwise the caller supplies it.
  2. **Radiometric stretch** — per *scene* percentile stretch to full 8-bit range.
     Removes the sensor's exposure/white-balance from the input, which is what
     stops the network keying on absolute brightness.  Scene-level (not crop- or
     tile-level) so it is a single deterministic transform of the whole image.
  3. **Geometric canonicalisation** — resample so one pixel == `canonical_gsd_m`
     metres on the ground.  Heights are in metres and are NOT touched.
  4. **Tile** into `tile_size` windows (inference: overlapping + Hann-blended;
     training: one random crop, taken *before* the resample so it can never be
     padded).
  5. **Normalise** with the encoder's own mean/std (DINOv3-SAT does not use the
     ImageNet constants) -> float32 CHW.

Nothing here imports torch beyond the final tensor conversion, so the same code
runs in a DataLoader worker and in a CPU-only inference container.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# DINOv3 satellite checkpoints (SAT-493M) were pretrained with their own
# statistics, not ImageNet's.  v1/v2 used the ImageNet constants for both, which
# shifts every input by ~0.3 sigma before the encoder ever sees it.  We prefer the
# values the HF image processor reports and fall back to these.
DINOV3_SAT_MEAN = (0.430, 0.411, 0.296)
DINOV3_SAT_STD = (0.213, 0.156, 0.143)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PreprocSpec:
    """Everything needed to turn an arbitrary image into a model input."""

    encoder_model_id: str = "facebook/dinov3-vitl16-pretrain-sat493m"
    mean: tuple = DINOV3_SAT_MEAN
    std: tuple = DINOV3_SAT_STD
    canonical_gsd_m: float = 0.5
    tile_size: int = 512
    patch: int = 16
    radiometric_stretch: bool = True
    stretch_lo_pct: float = 2.0
    stretch_hi_pct: float = 98.0
    # informational: what the height output means
    target: str = "nDSM_agl_metres"
    version: str = "v3"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_config(cls, cfg, resolve_stats: bool = True) -> "PreprocSpec":
        mean, std = DINOV3_SAT_MEAN, DINOV3_SAT_STD
        if resolve_stats:
            mean, std = resolve_encoder_stats(
                cfg.encoder_model_id, getattr(cfg, "hf_token", "") or None
            )
        return cls(
            encoder_model_id=cfg.encoder_model_id,
            mean=tuple(mean), std=tuple(std),
            canonical_gsd_m=float(cfg.canonical_gsd_m),
            tile_size=int(cfg.tile_size),
            radiometric_stretch=bool(cfg.radiometric_stretch),
            stretch_lo_pct=float(cfg.stretch_lo_pct),
            stretch_hi_pct=float(cfg.stretch_hi_pct),
        )

    # -- step 5 -------------------------------------------------------
    def normalise(self, rgb_u8: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 -> (3, H, W) float32, encoder-normalised."""
        m = np.asarray(self.mean, dtype=np.float32)
        s = np.asarray(self.std, dtype=np.float32)
        x = (rgb_u8.astype(np.float32) / 255.0 - m) / s
        return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))

    def denormalise(self, chw: np.ndarray) -> np.ndarray:
        m = np.asarray(self.mean, dtype=np.float32).reshape(3, 1, 1)
        s = np.asarray(self.std, dtype=np.float32).reshape(3, 1, 1)
        x = np.clip(chw * s + m, 0, 1)
        return (np.transpose(x, (1, 2, 0)) * 255).astype(np.uint8)


def resolve_encoder_stats(model_id: str, token: str | None = None):
    """Ask the encoder's own HF image processor for its normalisation stats.

    Falls back to the published DINOv3-SAT constants (satellite checkpoints) or
    ImageNet (everything else) when the hub is unreachable.
    """
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(model_id, token=token)
        mean = tuple(float(v) for v in proc.image_mean)
        std = tuple(float(v) for v in proc.image_std)
        if len(mean) == 3 and len(std) == 3 and all(s > 0 for s in std):
            print(f"[preproc] encoder stats from processor: mean={mean} std={std}")
            return mean, std
    except Exception as e:  # noqa: BLE001
        print(f"[preproc] image-processor lookup failed ({e}); using defaults")
    if "sat" in model_id.lower():
        return DINOV3_SAT_MEAN, DINOV3_SAT_STD
    return IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------
# step 2 — radiometry
# ---------------------------------------------------------------------
def scene_stretch_bounds(
    rgb_u8: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0,
    max_samples: int = 4_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (lo, hi) byte values for a percentile stretch of one scene.

    For uint8 input the percentile comes from a 256-bin histogram, which is a
    single O(N) pass instead of `np.percentile`'s sort — ~15x faster on a 1 Mpx
    tile, and this runs once per *training crop*, so it was a real slice of the
    DataLoader budget.  Non-uint8 input falls back to `np.percentile`.
    """
    a = rgb_u8.reshape(-1, rgb_u8.shape[-1])
    if a.dtype == np.uint8:
        n, c = a.shape
        lo = np.empty(c, np.float32)
        hi = np.empty(c, np.float32)
        for ch in range(c):
            counts = np.bincount(a[:, ch], minlength=256)
            cum = np.cumsum(counts)
            lo[ch] = np.searchsorted(cum, lo_pct / 100.0 * n)
            hi[ch] = np.searchsorted(cum, hi_pct / 100.0 * n)
        return lo, np.maximum(hi, lo + 1.0)
    if a.shape[0] > max_samples:
        step = int(np.ceil(a.shape[0] / max_samples))
        a = a[::step]
    lo = np.percentile(a, lo_pct, axis=0).astype(np.float32)
    hi = np.percentile(a, hi_pct, axis=0).astype(np.float32)
    hi = np.maximum(hi, lo + 1.0)
    return lo, hi


def stretch_lut(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(256, C) uint8 lookup table for `apply_stretch` — the stretch is pointwise."""
    v = np.arange(256, dtype=np.float32)[:, None]
    x = (v - np.asarray(lo, np.float32)) / (np.asarray(hi, np.float32) - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def apply_stretch(rgb_u8: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if rgb_u8.dtype == np.uint8 and rgb_u8.ndim == 3:
        return apply_stretch_lut(rgb_u8, stretch_lut(lo, hi))
    x = (rgb_u8.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def apply_stretch_lut(rgb_u8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a `stretch_lut` table — a gather over uint8, no float image at all."""
    out = np.empty_like(rgb_u8)
    for ch in range(rgb_u8.shape[-1]):
        np.take(lut[:, ch], rgb_u8[..., ch], out=out[..., ch])
    return out


def stretch_scene(rgb_u8: np.ndarray, spec: PreprocSpec) -> np.ndarray:
    if not spec.radiometric_stretch:
        return rgb_u8
    lo, hi = scene_stretch_bounds(rgb_u8, spec.stretch_lo_pct, spec.stretch_hi_pct)
    return apply_stretch(rgb_u8, lo, hi)


# ---------------------------------------------------------------------
# step 3 — geometry
# ---------------------------------------------------------------------
def resize(arr: np.ndarray, out_hw: tuple[int, int], order: str) -> np.ndarray:
    """order: 'bilinear' | 'nearest'.  Handles uint8 RGB, float32 and int labels."""
    from PIL import Image

    h, w = int(out_hw[0]), int(out_hw[1])
    if arr.shape[:2] == (h, w):
        return arr
    resample = Image.BILINEAR if order == "bilinear" else Image.NEAREST
    if arr.ndim == 3:
        return np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((w, h), resample))
    if np.issubdtype(arr.dtype, np.integer):
        return np.asarray(
            Image.fromarray(arr.astype(np.int32), mode="I").resize((w, h), Image.NEAREST)
        ).astype(arr.dtype)
    return np.asarray(
        Image.fromarray(arr.astype(np.float32)).resize((w, h), resample), dtype=np.float32
    )


def gsd_to_shape(h: int, w: int, src_gsd_m: float, dst_gsd_m: float) -> tuple[int, int]:
    s = float(src_gsd_m) / float(dst_gsd_m)
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


def to_canonical(rgb_u8: np.ndarray, src_gsd_m: float, spec: PreprocSpec) -> np.ndarray:
    out_hw = gsd_to_shape(rgb_u8.shape[0], rgb_u8.shape[1], src_gsd_m, spec.canonical_gsd_m)
    return resize(rgb_u8, out_hw, "bilinear")


def round_to_patch(n: int, patch: int, minimum: int) -> int:
    return max(minimum, int(round(n / patch)) * patch)


# ---------------------------------------------------------------------
# step 4 — tiling / blending (inference side)
# ---------------------------------------------------------------------
def hann2d(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]
    return np.clip(np.outer(w, w), 1e-3, None).astype(np.float32)


def tile_origins(extent: int, tile: int, overlap: float) -> list[int]:
    """Start offsets covering [0, extent) with `tile`-wide windows."""
    if extent <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, extent - tile + 1, stride))
    if xs[-1] != extent - tile:
        xs.append(extent - tile)
    return xs


# ---------------------------------------------------------------------
# step 1 — readers
# ---------------------------------------------------------------------
@dataclass
class SceneMeta:
    """What we know about an input image."""

    gsd_m: float
    gsd_source: str                 # "geotiff" | "user" | "assumed"
    georeferenced: bool = False
    transform: object | None = None  # affine.Affine, when georeferenced
    crs: object | None = None
    width: int = 0
    height: int = 0
    path: str = ""

    def summary(self) -> dict:
        return {
            "path": self.path, "width": self.width, "height": self.height,
            "gsd_m": self.gsd_m, "gsd_source": self.gsd_source,
            "georeferenced": self.georeferenced,
            "crs": str(self.crs) if self.crs is not None else None,
        }


def _metres_per_unit(crs, transform, width: int, height: int) -> float:
    """Convert one transform unit to metres.  Projected CRS -> already metres."""
    if crs is None:
        return 1.0
    try:
        if crs.is_geographic:
            # degrees -> metres at the scene's centre latitude
            import math

            lat = transform.f + transform.e * (height / 2.0)
            return 111_320.0 * max(0.15, math.cos(math.radians(lat)))
        units = (crs.linear_units or "metre").lower()
        if units.startswith(("met", "m")):
            return 1.0
        if units.startswith(("foot", "ft", "us survey")):
            return 0.3048006096012192 if "us" in units else 0.3048
    except Exception:  # noqa: BLE001
        pass
    return 1.0


TIF_SUFFIXES = {".tif", ".tiff", ".gtif", ".gtiff"}


def read_scene(
    path: str | Path,
    user_gsd_m: float = 0.0,
    assumed_gsd_m: float = 0.5,
    max_side: int = 0,
) -> tuple[np.ndarray, SceneMeta]:
    """Load any of PNG / JPG / (Geo)TIFF as (H, W, 3) uint8 + SceneMeta.

    GSD precedence: `user_gsd_m` > GeoTIFF transform > `assumed_gsd_m`.
    `max_side` (optional) downsamples enormous scenes before anything else; the
    reported GSD is scaled to match so metres stay correct.
    """
    path = Path(path)
    rgb, meta = None, None

    if path.suffix.lower() in TIF_SUFFIXES:
        try:
            import rasterio

            with rasterio.open(path) as src:
                idx = [1, 2, 3] if src.count >= 3 else [1] * 3
                rgb = src.read(indexes=idx).transpose(1, 2, 0)
                mpu = _metres_per_unit(src.crs, src.transform, src.width, src.height)
                native = float(abs(src.transform.a)) * mpu
                georef = src.crs is not None and abs(src.transform.a) > 0
                gsd = user_gsd_m if user_gsd_m > 0 else (native if georef else assumed_gsd_m)
                meta = SceneMeta(
                    gsd_m=float(gsd),
                    gsd_source="user" if user_gsd_m > 0 else ("geotiff" if georef else "assumed"),
                    georeferenced=bool(georef), transform=src.transform, crs=src.crs,
                    width=src.width, height=src.height, path=str(path),
                )
        except Exception as e:  # noqa: BLE001
            print(f"[preproc] rasterio read failed ({e}); falling back to PIL")

    if rgb is None:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(path).convert("RGB")
        rgb = np.asarray(im)
        meta = SceneMeta(
            gsd_m=float(user_gsd_m if user_gsd_m > 0 else assumed_gsd_m),
            gsd_source="user" if user_gsd_m > 0 else "assumed",
            georeferenced=False, width=im.width, height=im.height, path=str(path),
        )

    rgb = _to_uint8_rgb(rgb)
    meta.height, meta.width = rgb.shape[0], rgb.shape[1]

    if max_side and max(rgb.shape[:2]) > max_side:
        sc = max_side / max(rgb.shape[:2])
        rgb = resize(rgb, (int(rgb.shape[0] * sc), int(rgb.shape[1] * sc)), "bilinear")
        meta.gsd_m /= sc
        meta.height, meta.width = rgb.shape[0], rgb.shape[1]
        print(f"[preproc] downsampled scene to {rgb.shape[1]}x{rgb.shape[0]} "
              f"(gsd now {meta.gsd_m:.3f} m)")
    return rgb, meta


def _to_uint8_rgb(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = a[..., :3]
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    a = a.astype(np.float32)
    if a.max() <= 1.5:                       # float 0..1
        a = a * 255.0
    elif a.max() > 255.0:                    # uint16 / radiance
        lo, hi = scene_stretch_bounds(a.astype(np.float32), 1.0, 99.0)
        a = (a - lo) / (hi - lo) * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def save_spec(spec: PreprocSpec, path: str | Path) -> None:
    Path(path).write_text(json.dumps(spec.to_dict(), indent=2))
