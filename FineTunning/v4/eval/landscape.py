"""Assign each val tile a landscape class, so the rubric's own axis can be scored.

The evaluation criteria say the DSM must "demonstrate performance stability
across urban, sparse, hilly, and forested landscapes".  No dataset here ships
that label, and v1-v3 never reported it — the closest thing was a per-*class*
table whose class ids are known to be wrong (README §1.6).  So we derive the
label from the **ground-truth height field** itself, which works on any dataset
and needs no extra annotation:

    relief      std of a heavily blurred GT  -> terrain-scale undulation
    frac_tall   fraction of pixels above 3 m -> is there structure at all
    roughness   mean |Laplacian| over tall pixels, normalised by their height
                -> canopy (rough, fractal) vs roofs (smooth, planar)

and then

    relief >= relief_m                      -> hilly
    frac_tall < sparse_frac                 -> sparse
    roughness >= forest_rough               -> forested
    otherwise                               -> urban

These are heuristics with published thresholds, not ground truth, and the
descriptors are written into `metrics.json` next to every result so anyone can
check the assignment rather than take it on faith.  The point is not a perfect
taxonomy — it is that "RMSE 3.1 m overall, 2.4 m urban, 5.8 m hilly" is an
answerable claim and "RMSE 3.1 m" alone is not.

Caveat worth stating in the deck: our target is nDSM (height *above ground*), so
terrain has largely been differenced out and `relief` mostly fires on datasets
whose nDSM is a proxy (GeoNRW) or on genuine large-scale structure.  A tile-level
`hilly` count near zero on GAMUS is expected and is itself the finding — GAMUS is
flat US urban, which is exactly why the plan brings in other sources.
"""

from __future__ import annotations

import numpy as np

from config import LANDSCAPE_NAMES


def _box_blur(a: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur via a summed-area table — no scipy dependency."""
    if k < 2 or min(a.shape) < 2:
        return a
    k = min(k, min(a.shape) // 2 * 2 + 1)
    pad = k // 2
    b = np.pad(a.astype(np.float64), pad, mode="edge")
    c = b.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = a.shape
    s = (c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w])
    return (s / (k * k)).astype(np.float32)


def descriptors(gt: np.ndarray, valid: np.ndarray | None = None,
                gsd_m: float = 0.5) -> dict:
    """Scale-aware shape descriptors of one ground-truth height tile."""
    gt = np.asarray(gt, np.float32)
    if gt.ndim > 2:
        gt = gt.reshape(gt.shape[-2], gt.shape[-1])
    if valid is not None:
        v = np.asarray(valid).reshape(gt.shape).astype(bool)
        if v.mean() < 0.05:
            return {"relief_m": 0.0, "frac_tall": 0.0, "roughness": 0.0, "n_valid": 0}
        gt = np.where(v, gt, float(np.median(gt[v])))
    else:
        v = np.ones_like(gt, bool)

    # ~60 m of ground, so the window means the same thing at any GSD
    k = max(3, int(round(60.0 / max(gsd_m, 1e-3))) | 1)
    relief = float(_box_blur(gt, min(k, min(gt.shape) - 1)).std())

    tall = gt > 3.0
    frac_tall = float((tall & v).mean())

    lap = np.zeros_like(gt)
    lap[1:-1, 1:-1] = (4.0 * gt[1:-1, 1:-1] - gt[:-2, 1:-1] - gt[2:, 1:-1]
                       - gt[1:-1, :-2] - gt[1:-1, 2:])
    sel = tall & v
    sel[0, :] = sel[-1, :] = sel[:, 0] = sel[:, -1] = False
    if sel.sum() > 32:
        # curvature per metre of height: a tree canopy is rough relative to its
        # own height, a flat roof of the same height is not
        rough = float(np.abs(lap[sel]).mean() / max(float(gt[sel].mean()), 1e-3))
    else:
        rough = 0.0
    return {"relief_m": relief, "frac_tall": frac_tall, "roughness": rough,
            "n_valid": int(v.sum())}


def classify(gt: np.ndarray, valid: np.ndarray | None = None, gsd_m: float = 0.5,
             *, relief_m: float = 6.0, sparse_frac: float = 0.12,
             forest_rough: float = 0.22) -> tuple[str, dict]:
    d = descriptors(gt, valid, gsd_m)
    if d["n_valid"] == 0:
        return "sparse", d
    if d["relief_m"] >= relief_m:
        name = "hilly"
    elif d["frac_tall"] < sparse_frac:
        name = "sparse"
    elif d["roughness"] >= forest_rough:
        name = "forested"
    else:
        name = "urban"
    assert name in LANDSCAPE_NAMES
    return name, d
