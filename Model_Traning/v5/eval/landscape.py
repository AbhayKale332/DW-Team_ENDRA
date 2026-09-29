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

Caveat worth stating in the deck, and stronger than it first looked: our target
is nDSM, so terrain is subtracted out **by construction** and hilliness is not
recoverable from the label at all.  Measured on a synthetic hillside town whose
DTM has 3.44 m of std, the nDSM's ground-masked relief is 0.000 m.

The v4 run is what forced this.  `relief` was the std of the *raw* blurred nDSM,
so a cluster of towers survived the 60 m window and read as a hill: all eight
tiles the rule called "hilly" had `frac_tall > 0.30` (median 0.73, one at 0.92).
They were downtown cores.  "Worst landscape = hilly, 4.96 m" was tall-structure
compression — the same failure as `tall_bias` and the 20 m+ stratum — counted a
third time under a terrain name, while the rubric's actual hilly axis went
unmeasured.

So: `relief` now comes from a **DTM** when one is supplied, and otherwise from
the ground surface of the nDSM, which reads ~0 on any true nDSM.  A `hilly`
count of zero on GAMUS is the correct and honest output — GAMUS is flat US
urban. Report that axis as *unmeasured*, not as evidence of stability, until a
source with terrain (DFC2019, GeoNRW) is in the mix.
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
                gsd_m: float = 0.5, terrain: np.ndarray | None = None) -> dict:
    """Scale-aware shape descriptors of one ground-truth height tile.

    `terrain` is an optional DTM for the same tile.  It matters more than it
    looks: our target is **nDSM**, so the terrain has been subtracted out by
    construction and a genuinely hilly town is identically ~0 m on the ground.
    Measured on a synthetic hillside whose DTM has 3.44 m of std, the nDSM's
    ground-masked relief is 0.000 m — the hill is not in the label at all.
    `relief` is therefore taken from `terrain` when one is supplied, and is
    otherwise a ground-surface statistic that will read ~0 on any true nDSM.
    """
    gt = np.asarray(gt, np.float32)
    if gt.ndim > 2:
        gt = gt.reshape(gt.shape[-2], gt.shape[-1])
    if valid is not None:
        v = np.asarray(valid).reshape(gt.shape).astype(bool)
        if v.mean() < 0.05:
            return {"relief_m": 0.0, "relief_raw_m": 0.0,
                    "relief_from": "empty", "frac_tall": 0.0,
                    "roughness": 0.0, "n_valid": 0}
        gt = np.where(v, gt, float(np.median(gt[v])))
    else:
        v = np.ones_like(gt, bool)

    # ~60 m of ground, so the window means the same thing at any GSD
    k = max(3, int(round(60.0 / max(gsd_m, 1e-3))) | 1)
    kk = min(k, min(gt.shape) - 1)

    tall = gt > 3.0
    frac_tall = float((tall & v).mean())

    # `relief` must describe the *terrain*, not the buildings standing on it.
    # Blurring the raw nDSM let a cluster of towers survive the 60 m window and
    # read as a hill: on the v4 run every tile the rule called "hilly" had
    # frac_tall > 0.30 (median 0.73, one at 0.92) — they were downtown cores, so
    # the rubric's hilly axis was measuring tall-structure compression a second
    # time instead of terrain.  Filling tall pixels with the ground median
    # before the blur makes relief a property of the ground surface alone, which
    # on an nDSM is ~0 unless there is genuine large-scale undulation — and a
    # near-zero hilly count on flat US urban GAMUS is then the honest finding
    # this module's own docstring predicts.
    if terrain is not None:
        t = np.asarray(terrain, np.float32).reshape(gt.shape)
        relief = float(_box_blur(t, kk).std())
        relief_from = "dtm"
    else:
        ground = v & ~tall
        base = (np.where(ground, gt, float(np.median(gt[ground])))
                if ground.sum() > 64 else gt)
        relief = float(_box_blur(base, kk).std())
        relief_from = "ndsm_ground"
    # kept so an old classification stays re-derivable from metrics.json
    relief_raw_m = float(_box_blur(gt, kk).std())

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
    return {"relief_m": relief, "relief_raw_m": relief_raw_m,
            "relief_from": relief_from,
            "frac_tall": frac_tall, "roughness": rough,
            "n_valid": int(v.sum())}


def classify(gt: np.ndarray, valid: np.ndarray | None = None, gsd_m: float = 0.5,
             *, relief_m: float = 6.0, sparse_frac: float = 0.12,
             forest_rough: float = 0.22,
             terrain: np.ndarray | None = None,
             no_urban: bool = False) -> tuple[str, dict]:
    """`no_urban`: the source's labels hold no buildings (NEON CHM), so what the
    roughness rule calls urban is a closed canopy — see
    `Config.landscape_no_urban_sources`."""
    d = descriptors(gt, valid, gsd_m, terrain)
    if d["n_valid"] == 0:
        return "sparse", d
    if d["relief_m"] >= relief_m:
        name = "hilly"
    elif d["frac_tall"] < sparse_frac:
        name = "sparse"
    elif d["roughness"] >= forest_rough:
        name = "forested"
    else:
        name = "forested" if no_urban else "urban"
    assert name in LANDSCAPE_NAMES
    return name, d
