"""Does Head B's spread (σ, `b_std` / `ndsm_std_m`) track the error?  Sparsification, AUSE, AURG.

The standard check for a per-pixel uncertainty on dense depth regression, after
Poggi et al., "On the uncertainty of self-supervised monocular depth
estimation", CVPR 2020 (arXiv 2005.06209), with the maths of their reference
code (github.com/mattpoggi/mono-uncertainty, `compute_aucs`).  Per image:

  * **sparsification curve** — remove the most uncertain pixels first, a
    fraction 0, 1/K, ..., (K-1)/K at a time (pixels kept: σ at or below the
    matching percentile), and take the RMSE of what is left; a final 0 at 1.
  * **oracle** — the same, ordered by the true squared error: the best curve any
    σ could give.
  * **AUSE** = area(σ curve) - area(oracle).  0 when σ ranks the pixels exactly as
    their error does; the larger, the worse.
  * **AURG** = RMSE(all) - area(σ curve).  Removing pixels at random leaves the
    RMSE flat, so > 0 means σ is better than chance, < 0 worse.

Both are unnormalised, in metres of RMSE, as the reference code.  Scores are
per image (here per tile) and averaged over images, which is that paper's
protocol; the curves reported are the per-tile curves averaged.

Also the cut `infer/predict.py` calls "confident" (σ <= max(1 m, the tile's
median σ)) and the RMSE on those pixels, so the web app's "RMSE on confident
pixels" has a measured counterpart.

numpy only (torch tensors are accepted by `add_batch`), so `eval_test.py` can
load it by file path under `--model_code ../v3`.
"""

from __future__ import annotations

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

#: `infer/predict.py` `uncertainty_summary`: the confident cut is never below 1 m.
CONFIDENT_FLOOR_M = 1.0


def _rmse_kept(err2: np.ndarray, u: np.ndarray, quants: np.ndarray) -> list[float]:
    """RMSE of the pixels left after removing the highest-`u` ones, per quantile.

    `compute_aucs` negates the uncertainty and keeps `-u >= percentile(-u, q)`;
    the same selection is written here directly.
    """
    neg = -u
    thr = np.percentile(neg, quants)
    return [float(np.sqrt(err2[neg >= t].mean())) for t in thr] + [0.0]


def sparsification(pred, gt, sigma, valid=None, intervals: int = 50,
                   min_px: int = 64) -> dict | None:
    """Sparsification curves, AUSE and AURG of one image, on RMSE.

    `pred`, `gt`, `sigma` are same-shape arrays in metres; pixels where any is
    non-finite, or `valid` is False, are left out.  None with fewer than
    `min_px` usable pixels.
    """
    p = np.asarray(pred, np.float64).ravel()
    g = np.asarray(gt, np.float64).ravel()
    s = np.asarray(sigma, np.float64).ravel()
    m = np.isfinite(p) & np.isfinite(g) & np.isfinite(s)
    if valid is not None:
        m &= np.asarray(valid, bool).ravel()
    if int(m.sum()) < max(min_px, 1):
        return None
    err2 = (p[m] - g[m]) ** 2
    s = s[m]

    quants = np.arange(intervals) * (100.0 / intervals)
    x = np.arange(intervals + 1) / intervals
    curve = _rmse_kept(err2, s, quants)
    oracle = _rmse_kept(err2, err2, quants)
    rmse = float(np.sqrt(err2.mean()))
    area = float(_trapz(curve, x=x))

    thr = max(CONFIDENT_FLOOR_M, float(np.median(s)))
    conf = s <= thr
    return {
        "n": int(err2.size),
        "rmse_m": rmse,
        "ause_rmse_m": area - float(_trapz(oracle, x=x)),
        "aurg_rmse_m": rmse - area,
        "removed": x.tolist(),
        "curve_rmse_m": curve,
        "oracle_rmse_m": oracle,
        "confident_threshold_m": thr,
        "confident_frac": float(conf.mean()),
        "confident_rmse_m": float(np.sqrt(err2[conf].mean())) if conf.any() else float("nan"),
    }


def _np(t) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().float().cpu().numpy()
    return np.asarray(t)


class SparsificationMeter:
    """Per-tile `sparsification`, averaged over tiles (the reference protocol)."""

    def __init__(self, intervals: int = 50, min_px: int = 64):
        self.intervals, self.min_px = intervals, min_px
        self.n_tiles = 0
        self.n_px = 0
        self._sum: dict[str, float] = {}
        self._curve = np.zeros(intervals + 1)
        self._oracle = np.zeros(intervals + 1)
        self._conf_n = 0          # tiles with any confident pixel

    def add(self, pred, gt, sigma, valid=None) -> dict | None:
        """One image (H, W)."""
        r = sparsification(pred, gt, sigma, valid, self.intervals, self.min_px)
        if r is None:
            return None
        self.n_tiles += 1
        self.n_px += r["n"]
        for k in ("rmse_m", "ause_rmse_m", "aurg_rmse_m", "confident_threshold_m",
                  "confident_frac"):
            self._sum[k] = self._sum.get(k, 0.0) + r[k]
        if np.isfinite(r["confident_rmse_m"]):
            self._sum["confident_rmse_m"] = self._sum.get("confident_rmse_m", 0.0) \
                + r["confident_rmse_m"]
            self._conf_n += 1
        self._curve += r["curve_rmse_m"]
        self._oracle += r["oracle_rmse_m"]
        return r

    def add_batch(self, pred, gt, sigma, valid=None) -> None:
        """A batch (B, [1,] H, W) of numpy arrays or torch tensors."""
        p, g, s = _np(pred), _np(gt), _np(sigma)
        v = None if valid is None else _np(valid).astype(bool)
        b = p.shape[0]
        p, g, s = p.reshape(b, -1), g.reshape(b, -1), s.reshape(b, -1)
        v = None if v is None else v.reshape(b, -1)
        for i in range(b):
            self.add(p[i], g[i], s[i], None if v is None else v[i])

    def result(self) -> dict:
        """{} until a tile was scored."""
        n = self.n_tiles
        if not n:
            return {}
        out = {k: v / n for k, v in self._sum.items() if k != "confident_rmse_m"}
        if self._conf_n:
            out["confident_rmse_m"] = self._sum["confident_rmse_m"] / self._conf_n
        out.update({
            "n_tiles": n,
            "n_px": self.n_px,
            "intervals": self.intervals,
            "removed": (np.arange(self.intervals + 1) / self.intervals).tolist(),
            "curve_rmse_m": (self._curve / n).tolist(),
            "oracle_rmse_m": (self._oracle / n).tolist(),
            "protocol": "Poggi et al. CVPR 2020 compute_aucs on RMSE, per tile, "
                        "mean over tiles; sigma = Head B b_std, plain pass",
        })
        return out


def format_line(r: dict) -> str:
    if not r:
        return "no sigma"
    return (f"AUSE {r['ause_rmse_m']:.3f} m  AURG {r['aurg_rmse_m']:+.3f} m  "
            f"(RMSE {r['rmse_m']:.3f} m; confident {r['confident_frac'] * 100:.0f} % "
            f"-> {r.get('confident_rmse_m', float('nan')):.3f} m; {r['n_tiles']} tiles)")
