"""The run's figures.  One call, one directory of PNGs, no notebook.

Everything here reads `metrics.json` and the packed val store — nothing needs the
GPU or the checkpoint, so you can re-render the plots from a finished run's
artefacts on a laptop.

The figures, and what each is *for*:

| file | question it answers |
|---|---|
| `curves.png` | did it converge, and did the encoder unfreeze actually help |
| `per_stratum.png` | where in the height range is the error — the tall-structure tail is invisible in a global RMSE |
| `per_landscape.png` | the rubric's stability axis, urban / sparse / hilly / forested |
| `scatter.png` | predicted vs GT density + the y=x line: bias and compression at a glance |
| `error_hist.png` | signed error distribution, split flat / tall |
| `hillshade.png` | what the DSM looks like as a surface, which is what the judge sees |
| `qualitative.png` | RGB, prediction, GT, error — the contact sheet |
| `gallery_<store>.jpg` | what SynRS3D / DFC23 / India tiles look like, and how the model does on each |

Matplotlib only, Agg backend, no seaborn: one less dependency to install on a
box that has to work offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ACCENT = "#2d6cdf"
_WARN = "#e5793a"
_GREY = "#8a949e"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
    })
    return plt


def colorize(a: np.ndarray, vmin: float = 0.0, vmax: float | None = None,
             cmap: str = "turbo") -> np.ndarray:
    hi = float(vmax if vmax is not None else np.nanmax(a))
    x = np.clip((a - vmin) / max(hi - vmin, 1e-6), 0, 1)
    try:
        import matplotlib

        return (matplotlib.colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)
    except Exception:  # noqa: BLE001
        g = (x * 255).astype(np.uint8)
        return np.stack([g, g, g], -1)


def hillshade(height_m: np.ndarray, gsd_m: float = 0.5, azimuth: float = 315.0,
              altitude: float = 45.0) -> np.ndarray:
    """Standard Horn hillshade — the fastest way to see whether a DSM is *shaped*
    like terrain-with-buildings or like the crumpled mountain range v2 produced."""
    dy, dx = np.gradient(height_m.astype(np.float32), max(gsd_m, 1e-3))
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, al = np.radians(360.0 - azimuth + 90.0), np.radians(altitude)
    sh = (np.sin(al) * np.cos(slope)
          + np.cos(al) * np.sin(slope) * np.cos(az - aspect))
    return (np.clip(sh, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------
def plot_curves(metrics: dict, out: Path):
    plt = _mpl()
    hist = metrics.get("history") or []
    if not hist:
        return None
    ep = [h["epoch"] for h in hist]
    loss = [h.get("train_loss") for h in hist]
    ev = [(h["epoch"], h["val"]) for h in hist if h.get("val")]
    unfreeze = next((h["epoch"] for h in hist if h.get("encoder_frozen") is False), None)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].plot(ep, loss, color=_ACCENT, lw=1.6)
    ax[0].set(xlabel="epoch", ylabel="train loss", title="Training loss")
    if ev:
        e, m = zip(*ev)
        ax[1].plot(e, [x["global"]["rmse_m"] for x in m], "-o", ms=3,
                   color=_ACCENT, label="RMSE")
        bal = [x.get("balanced_rmse_m") for x in m]
        if any(b is not None for b in bal):
            ax[1].plot(e, bal, "-o", ms=3, color=_WARN, label="balanced RMSE")
        ax[1].legend(frameon=False)
    ax[1].set(xlabel="epoch", ylabel="metres", title="Validation error")
    for a in ax:
        if unfreeze:
            a.axvline(unfreeze - 0.5, color=_GREY, ls="--", lw=1)
            a.text(unfreeze - 0.4, a.get_ylim()[1], " encoder unfrozen",
                   color=_GREY, va="top", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _bar(ax, labels, values, counts=None, color=_ACCENT, ylabel="RMSE (m)"):
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=color, width=0.62)
    ax.set_xticks(xs)
    # the sample count belongs on the tick, not floating over it — an RMSE from
    # 200k pixels and one from 5M pixels are not equally trustworthy
    ax.set_xticklabels([f"{l}\nn={n}" for l, n in zip(labels, counts)]
                       if counts else list(labels))
    ax.set_ylabel(ylabel)
    for x, v in zip(xs, values):
        ax.text(x, v, f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)


def plot_per_stratum(result: dict, out: Path):
    plt = _mpl()
    per = result.get("per_stratum") or {}
    per = {k: v for k, v in per.items() if v.get("n")}
    if not per:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    _bar(ax[0], list(per), [v["rmse_m"] for v in per.values()],
         [f"{v['n'] / 1e6:.1f}M" for v in per.values()])
    ax[0].axhline(result["global"]["rmse_m"], color=_WARN, ls="--", lw=1.2)
    ax[0].text(0, result["global"]["rmse_m"], " global RMSE", color=_WARN,
               va="bottom", fontsize=7)
    ax[0].set_title("RMSE by height stratum")
    _bar(ax[1], list(per), [v["bias_m"] for v in per.values()], color=_WARN,
         ylabel="signed bias (m)")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_title("Bias by stratum  (negative = underestimated)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_landscape(result: dict, out: Path):
    plt = _mpl()
    per = result.get("per_landscape") or {}
    if not per:
        return None
    keys = sorted(per)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    _bar(ax, keys, [per[k]["rmse_m"] for k in keys],
         [per[k].get("tiles", 0) for k in keys])
    ax.axhline(result["global"]["rmse_m"], color=_WARN, ls="--", lw=1.2)
    spread = result.get("landscape_rmse_spread_m")
    ax.set_title("RMSE by landscape"
                 + (f"  (spread {spread:.2f} m)" if spread is not None else ""))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scatter(pred: np.ndarray, gt: np.ndarray, out: Path, max_pts: int = 300_000):
    plt = _mpl()
    p, g = pred.ravel(), gt.ravel()
    ok = np.isfinite(p) & np.isfinite(g)
    p, g = p[ok], g[ok]
    if p.size > max_pts:
        i = np.random.default_rng(0).choice(p.size, max_pts, replace=False)
        p, g = p[i], g[i]
    if p.size == 0:
        return None
    hi = float(max(np.percentile(g, 99.5), np.percentile(p, 99.5), 1.0))
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.hexbin(g, p, gridsize=70, bins="log", extent=(0, hi, 0, hi), cmap="Blues")
    ax.plot([0, hi], [0, hi], color=_WARN, lw=1.2, label="y = x")
    ax.set(xlabel="ground truth (m)", ylabel="predicted (m)", xlim=(0, hi), ylim=(0, hi),
           title="Predicted vs reference height")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_error_hist(pred: np.ndarray, gt: np.ndarray, out: Path):
    plt = _mpl()
    e = (pred - gt).ravel()
    g = gt.ravel()
    ok = np.isfinite(e) & np.isfinite(g)
    e, g = e[ok], g[ok]
    if e.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    lim = float(np.percentile(np.abs(e), 99))
    bins = np.linspace(-lim, lim, 90)
    ax.hist(e[g < 1.0], bins=bins, color=_ACCENT, alpha=0.75, density=True,
            label=f"flat GT < 1 m  (bias {np.mean(e[g < 1.0]) if (g < 1.0).any() else 0:+.2f})")
    if (g >= 15.0).any():
        ax.hist(e[g >= 15.0], bins=bins, color=_WARN, alpha=0.7, density=True,
                label=f"tall GT ≥ 15 m  (bias {np.mean(e[g >= 15.0]):+.2f})")
    ax.axvline(0, color="black", lw=0.9)
    ax.set(xlabel="predicted − reference (m)", ylabel="density",
           title="Signed error, split by regime")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_hillshade(pred: np.ndarray, gt: np.ndarray, out: Path, gsd_m: float = 0.5):
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for a, (img, t) in zip(ax, ((pred, "predicted"), (gt, "reference"))):
        a.imshow(hillshade(img, gsd_m), cmap="gray", vmin=0, vmax=255)
        a.set_title(f"{t} nDSM — hillshade")
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def strip(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RGB | pred | GT | |error| , one row, shared colour scale."""
    vmax = float(max(np.nanmax(gt), np.nanmax(pred), 1.0))
    return np.concatenate([
        rgb, colorize(pred, 0, vmax), colorize(gt, 0, vmax),
        colorize(np.abs(pred - gt), 0, max(vmax * 0.4, 1.0), "inferno"),
    ], axis=1)


def plot_contact_sheet(strips: list[np.ndarray], out: Path, max_rows: int = 6):
    plt = _mpl()
    strips = [s for s in strips if s is not None][:max_rows]
    if not strips:
        return None
    n = len(strips)
    fig, ax = plt.subplots(n, 1, figsize=(9.5, 2.4 * n), squeeze=False)
    for i, s in enumerate(strips):
        a = ax[i][0]
        a.imshow(s)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        if i == 0:
            a.set_title("RGB   |   predicted   |   reference   |   |error|", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_gallery(entry: dict, out: Path):
    """One store's tiles: RGB | reference | predicted | |error|, one row each.

    Unlike the contact sheet this carries colour bars in metres and a per-row
    label, because the rows come from different sources at different GSDs and a
    reader has no other way to tell a 3 m building from a 30 m one.  JPEG, since
    these are photographs and the report inlines every byte.
    """
    plt = _mpl()
    tiles = entry.get("tiles") or []
    if not tiles:
        return None
    n = len(tiles)
    fig, ax = plt.subplots(n, 4, figsize=(11.0, 2.9 * n), squeeze=False)
    for r, t in enumerate(tiles):
        valid = t.get("valid")
        gt = np.where(valid, t["gt"], np.nan) if valid is not None else t["gt"]
        vmax = float(max(np.nanpercentile(gt, 99.5) if np.isfinite(gt).any() else 0,
                         np.nanpercentile(t["pred"], 99.5), 1.0))
        err = np.abs(t["pred"] - gt)
        emax = max(vmax * 0.4, 1.0)
        panels = ((t["rgb"], None, None, "RGB"),
                  (gt, "turbo", (0, vmax), "reference nDSM"),
                  (t["pred"], "turbo", (0, vmax), "predicted nDSM"),
                  (err, "inferno", (0, emax), "|error|"))
        for c, (img, cmap, lim, name) in enumerate(panels):
            a = ax[r][c]
            if cmap is None:
                a.imshow(img)
            else:
                # NoData in the reference, grey rather than drawn as 0 m
                cm = plt.get_cmap(cmap).with_extremes(bad="#bfc4ca")
                im = a.imshow(img, cmap=cm, vmin=lim[0], vmax=lim[1])
                cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=7)
                cb.set_label("m", fontsize=7)
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(False)
            if r == 0:
                a.set_title(name, fontsize=9)
        rm = t.get("rmse_m")
        ax[r][0].set_ylabel(f"{t.get('stem', '')}\n"
                            + (f"RMSE {rm:.2f} m · MAE {t['mae_m']:.2f} m"
                               if rm is not None else "no valid reference"),
                            fontsize=7.5)
    fig.suptitle(f"{entry['store']} / {entry['split']}  ·  "
                 f"{entry.get('gsd_m', 0):.2f} m GSD", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=110, pil_kwargs={"quality": 85})
    plt.close(fig)
    return out


# ---------------------------------------------------------------------
def make_all(out_dir: Path, metrics: dict, samples: list | None = None,
             gsd_m: float = 0.5, gallery: list | None = None) -> dict:
    """Render every figure that the available data supports.

    `samples` is a list of (rgb_u8, pred_m, gt_m) triples; without it the
    pixel-level plots are skipped and only the metric plots are produced.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = (metrics.get("final_sliding_tta") or metrics.get("final_tta")
              or metrics.get("final_plain")
              or next((h["val"] for h in reversed(metrics.get("history") or [])
                       if h.get("val")), None) or {})
    made: dict = {}

    def keep(name, path):
        if path is not None:
            made[name] = Path(path).name

    keep("curves", plot_curves(metrics, out_dir / "curves.png"))
    if result:
        keep("per_stratum", plot_per_stratum(result, out_dir / "per_stratum.png"))
        keep("per_landscape", plot_per_landscape(result, out_dir / "per_landscape.png"))
    if samples:
        pred = np.concatenate([s[1].ravel() for s in samples])
        gt = np.concatenate([s[2].ravel() for s in samples])
        keep("scatter", plot_scatter(pred, gt, out_dir / "scatter.png"))
        keep("error_hist", plot_error_hist(pred, gt, out_dir / "error_hist.png"))
        keep("hillshade", plot_hillshade(samples[0][1], samples[0][2],
                                         out_dir / "hillshade.png", gsd_m))
        keep("qualitative", plot_contact_sheet(
            [strip(*s) for s in samples], out_dir / "qualitative.png"))
    if gallery:
        # The report reads gallery.json, not the directory, so a stale
        # gallery_*.jpg from an earlier render cannot sneak back in.
        index = []
        for e in gallery:
            p = plot_gallery(e, out_dir / f"gallery_{e['store']}.jpg")
            if p is None:
                continue
            keep(f"gallery_{e['store']}", p)
            index.append({"file": Path(p).name, "store": e["store"], "split": e["split"],
                          "seen_in_training": bool(e.get("seen_in_training")),
                          "gsd_m": e.get("gsd_m"),
                          "tiles": [{"stem": t.get("stem"), "rmse_m": t.get("rmse_m"),
                                     "mae_m": t.get("mae_m")} for t in e["tiles"]]})
        (out_dir / "gallery.json").write_text(json.dumps(index, indent=2))
    print(f"[viz] {len(made)} figures -> {out_dir}")
    return made


def from_run_dir(run_dir: str | Path) -> dict:
    """Re-render the metric figures from a finished run, without a GPU."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    samples = []
    vs = run_dir / "viewer_sample"
    if (vs / "pred_ndsm_m.npy").is_file() and (vs / "gt_ndsm_m.npy").is_file():
        from PIL import Image

        rgb = np.asarray(Image.open(vs / "rgb.png").convert("RGB"))
        samples = [(rgb, np.load(vs / "pred_ndsm_m.npy"), np.load(vs / "gt_ndsm_m.npy"))]
    return make_all(run_dir / "figures", metrics, samples)


if __name__ == "__main__":
    import sys

    print(from_run_dir(sys.argv[1] if len(sys.argv) > 1 else "outputs/v4"))
