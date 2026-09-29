"""EDA of the packed NEON store before it joins the final run.

    python tools/eda_neon.py --store /tmp/neon/stores/neon --out /tmp/neon/eda

CPU only; reads `neon/{train,val,test}` as training does (PackedStore) and
writes `eda.json` + `eda.md`.  Each section answers one question the final-run
flags depend on:

* structure    is the label the 1 m CHM replicated 2 x 2 on the 0.5 m grid
               (-> coarse_label_m 1.0), and is `cls` all unlabelled (has_seg)?
* heights      per-site / per-split strata and the tail: are the > 60 m pixels
               real trees (WREF, TEAK) or spikes?  (-> bin_max_m 120)
* landscape    what `eval.landscape.classify` calls each tile — the class that
               `landscape_sampler_boost` and `select_on` act on.  A forest the
               rule calls "urban" gets no boost and is not selected on.
* rgb          colour / brightness per site vs the stretch the loader applies;
               no-data (black) share.
* alignment    per-tile shift (px) that best aligns high-passed ExG with the
               high-passed CHM, searched over +-8 px.  NEON's camera mosaic and
               lidar are separately orthorectified; a site-wide offset would
               be taught as blur.
* leakage      stems and 32 x 32 thumbnail hashes across splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dwdata.packed import NO_LABEL, PackedStore  # noqa: E402

SPLITS = ("train", "val", "test")
EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 1e9)
STRATA = ("0-2", "2-5", "5-10", "10-20", "20-40", "40-60", ">60")
SHIFT = 8


def site_of(stem: str) -> str:
    return stem.split("_", 1)[0]


def box(a: np.ndarray, r: int) -> np.ndarray:
    """Box blur of radius r (summed-area table)."""
    k = 2 * r + 1
    p = np.pad(a.astype(np.float64), r, mode="edge")
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = a.shape
    return ((c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]) / (k * k)).astype(np.float32)


def exg(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    s = f.sum(-1) + 1e-6
    r, g, b = f[..., 0] / s, f[..., 1] / s, f[..., 2] / s
    return 2 * g - r - b


def best_shift(a: np.ndarray, b: np.ndarray, m: int = SHIFT) -> tuple[int, int, float, float]:
    """(dy, dx, peak corr, corr at 0) of b against a, over +-m px (FFT, circular on a padded crop)."""
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    cc = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape) / a.size
    win = np.roll(np.roll(cc, m, 0), m, 1)[:2 * m + 1, :2 * m + 1]
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    return int(iy - m), int(ix - m), float(win[iy, ix]), float(win[m, m])


def one_tile(store: PackedStore, i: int, stem: str) -> dict:
    from eval.landscape import classify

    rgb, hgt, cls, val = store.get(i)
    h = np.asarray(hgt, np.float32)
    v = np.asarray(val, bool) & np.isfinite(h)
    hv = h[v]
    out = {"stem": stem, "site": site_of(stem), "valid": float(v.mean())}

    # structure: 2 x 2 replication and cls
    blk = h.reshape(h.shape[0] // 2, 2, h.shape[1] // 2, 2)
    vb = v.reshape(blk.shape).all((1, 3))
    same = (blk.max((1, 3)) - blk.min((1, 3)))[vb]
    out["rep2x2"] = float((same <= 1e-2).mean()) if same.size else 1.0
    # odd-offset blocks should NOT be constant if the grid is even-aligned
    hs = h[1:-1, 1:-1]
    blk_o = hs.reshape(hs.shape[0] // 2, 2, hs.shape[1] // 2, 2)
    so = (blk_o.max((1, 3)) - blk_o.min((1, 3)))[v[1:-1, 1:-1].reshape(blk_o.shape).all((1, 3))]
    out["rep2x2_odd"] = float((so <= 1e-2).mean()) if so.size else 1.0
    out["cls_unlabelled"] = float((np.asarray(cls) == NO_LABEL).mean())

    # heights
    out["hist"] = np.histogram(hv, bins=EDGES)[0].tolist()
    out["n_valid"] = int(v.sum())
    out["h_sum"] = float(hv.sum())
    out["h_max"] = float(hv.max()) if hv.size else 0.0
    out["n_neg"] = int((hv < 0).sum())
    out["q"] = [float(x) for x in np.percentile(hv, (50, 90, 99, 99.9))] if hv.size else [0.0] * 4
    if hv.size and hv.max() > 60:
        # isolated spike (a > 60 m cell whose 5 x 5 neighbourhood median is < 20 m) vs a tall stand
        tall = v & (h > 60)
        med = box(np.where(v, h, 0), 2) / np.maximum(box(v.astype(np.float32), 2), 1e-3)
        out["n_gt60"] = int(tall.sum())
        out["n_gt60_isolated"] = int((tall & (med < 20)).sum())

    # landscape class as the loader computes it (canonical 0.5 m == store gsd -> k = 1)
    name, d = classify(np.nan_to_num(h), v, store.gsd_m)
    out["landscape"] = name
    out["frac_tall"] = d["frac_tall"]
    out["roughness"] = d["roughness"]
    # the same rule on the 1 m grid the label really lives on
    v1 = v.reshape(blk.shape).all((1, 3))
    name1, d1 = classify(np.nan_to_num(blk.mean((1, 3))), v1, 1.0)
    out["landscape_1m"] = name1
    out["roughness_1m"] = d1["roughness"]

    # rgb
    rv = rgb[v]
    out["rgb_mean"] = rv.mean(0).tolist() if rv.size else [0, 0, 0]
    out["rgb_std"] = rv.std(0).tolist() if rv.size else [0, 0, 0]
    out["black"] = float((rgb.max(-1) == 0).mean())
    lum = rgb.astype(np.float32).mean(-1)
    out["lum_p2_p98"] = [float(x) for x in np.percentile(lum, (2, 98))]
    g = exg(rgb)
    tallm = v & (h > 3)
    lowm = v & (h < 1)
    out["green_given_tall"] = float((g[tallm] > 0.05).mean()) if tallm.sum() > 100 else None
    out["green_given_low"] = float((g[lowm] > 0.05).mean()) if lowm.sum() > 100 else None

    # alignment: centre 512 px, high-passed (fine minus 8 px blur) ExG vs CHM
    c = slice(256, 768)
    hh = np.where(v, h, np.median(hv) if hv.size else 0)[c, c]
    gg = g[c, c]
    hp_h = box(hh, 1) - box(hh, 8)
    hp_g = box(gg, 1) - box(gg, 8)
    if tallm[c, c].mean() > 0.05 and hp_h.std() > 0.2:
        dy, dx, pk, z = best_shift(hp_h, hp_g)
        out["shift"] = [dy, dx, pk, z]

    # leakage
    th = rgb[::32, ::32].astype(np.uint8)
    out["thumb"] = hashlib.md5(th.tobytes()).hexdigest()
    return out


def run_split(root: Path, split: str, workers: int) -> list[dict]:
    st = PackedStore(root / split)
    stems = [s for sh in json.loads((root / split / "index.json").read_text())["shards"] for s in sh["stems"]]
    assert len(stems) == len(st), (len(stems), len(st))
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda i: one_tile(st, i, stems[i]), range(len(st))))
    for r in rows:
        r["split"] = split
    print(f"[eda] {split}: {len(rows)} tiles", flush=True)
    return rows


def summarise(rows: list[dict], root: Path) -> dict:
    rep: dict = {}
    idx = {s: json.loads((root / s / "index.json").read_text()) for s in SPLITS}
    rep["index"] = {s: {k: v for k, v in d.items() if k != "shards"} for s, d in idx.items()}
    rep["stretch_bounds"] = {s: (root / s / "stretch_bounds_2_98.npy").is_file() for s in SPLITS}

    def agg(sel: list[dict]) -> dict:
        hist = np.sum([r["hist"] for r in sel], 0)
        n = max(int(hist.sum()), 1)
        lc = Counter(r["landscape"] for r in sel)
        lc1 = Counter(r["landscape_1m"] for r in sel)
        sh = [r["shift"] for r in sel if "shift" in r and r["shift"][2] > 0.1]
        return {
            "tiles": len(sel),
            "valid": float(np.mean([r["valid"] for r in sel])),
            "h_mean": float(sum(r["h_sum"] for r in sel) / n),
            "h_max": float(max(r["h_max"] for r in sel)),
            "strata": {k: float(c / n) for k, c in zip(STRATA, hist)},
            "p99_tile_median": float(np.median([r["q"][2] for r in sel])),
            "landscape": dict(lc), "landscape_1m": dict(lc1),
            "roughness_med": float(np.median([r["roughness"] for r in sel])),
            "roughness_1m_med": float(np.median([r["roughness_1m"] for r in sel])),
            "frac_tall_med": float(np.median([r["frac_tall"] for r in sel])),
            "rgb_mean": np.mean([r["rgb_mean"] for r in sel], 0).round(1).tolist(),
            "rgb_std": np.mean([r["rgb_std"] for r in sel], 0).round(1).tolist(),
            "lum_p2_p98": np.median([r["lum_p2_p98"] for r in sel], 0).round(1).tolist(),
            "black": float(np.mean([r["black"] for r in sel])),
            "green_given_tall": _nanmean([r["green_given_tall"] for r in sel]),
            "green_given_low": _nanmean([r["green_given_low"] for r in sel]),
            "shift_n": len(sh),
            "shift_med": (np.median([s[:2] for s in sh], 0).tolist() if sh else None),
            "shift_abs_gt2": (float(np.mean([max(abs(s[0]), abs(s[1])) > 2 for s in sh])) if sh else None),
            "shift_gain": (float(np.median([s[2] - s[3] for s in sh])) if sh else None),
            "gt60": int(sum(r.get("n_gt60", 0) for r in sel)),
            "gt60_isolated": int(sum(r.get("n_gt60_isolated", 0) for r in sel)),
            "neg": int(sum(r["n_neg"] for r in sel)),
        }

    rep["split"] = {s: agg([r for r in rows if r["split"] == s]) for s in SPLITS}
    by_site = defaultdict(list)
    for r in rows:
        by_site[r["site"]].append(r)
    rep["site"] = {k: agg(v) for k, v in sorted(by_site.items())}
    rep["structure"] = {
        "rep2x2_min": float(min(r["rep2x2"] for r in rows)),
        "rep2x2_mean": float(np.mean([r["rep2x2"] for r in rows])),
        "rep2x2_odd_mean": float(np.mean([r["rep2x2_odd"] for r in rows])),
        "cls_unlabelled_min": float(min(r["cls_unlabelled"] for r in rows)),
    }
    # leakage
    st = {s: {r["stem"] for r in rows if r["split"] == s} for s in SPLITS}
    km = {s: {r["stem"].rsplit("_r", 1)[0] for r in rows if r["split"] == s} for s in SPLITS}
    th = defaultdict(set)
    for r in rows:
        th[r["thumb"]].add(r["split"])
    rep["leakage"] = {
        "stem_overlap": {f"{a}/{b}": len(st[a] & st[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        "km_overlap": {f"{a}/{b}": len(km[a] & km[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        "thumb_cross_split": sum(1 for v in th.values() if len(v) > 1),
    }
    # tiles to look at
    rows_s = sorted(rows, key=lambda r: -r["h_max"])
    rep["tallest_tiles"] = [(r["stem"], r["split"], round(r["h_max"], 1), r.get("n_gt60", 0),
                             r.get("n_gt60_isolated", 0)) for r in rows_s[:12]]
    forest_urban = [r for r in rows if r["landscape"] == "urban" and r["frac_tall"] > 0.3]
    rep["urban_called_tiles"] = {"n": len(forest_urban),
                                 "sites": dict(Counter(r["site"] for r in forest_urban)),
                                 "roughness_med": (float(np.median([r["roughness"] for r in forest_urban]))
                                                   if forest_urban else None)}
    return rep


def _nanmean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def to_md(rep: dict) -> str:
    L = ["# NEON store EDA", ""]
    L += ["## Splits", "", "| split | tiles | valid | h mean | max | " + " | ".join(STRATA) + " | urban/sparse/hilly/forested |",
          "|---|" + "---|" * (5 + len(STRATA))]
    for s, a in rep["split"].items():
        lc = a["landscape"]
        L.append(f"| {s} | {a['tiles']} | {a['valid']:.1%} | {a['h_mean']:.2f} | {a['h_max']:.0f} | "
                 + " | ".join(f"{a['strata'][k]:.1%}" for k in STRATA)
                 + f" | {lc.get('urban', 0)}/{lc.get('sparse', 0)}/{lc.get('hilly', 0)}/{lc.get('forested', 0)} |")
    L += ["", "## Sites", "",
          "| site | tiles | h mean | >20 m | >40 m | landscape @0.5 m | landscape @1 m | rough 0.5/1 m | green|tall | green|low | shift med (n, >2px) | rgb mean |",
          "|---|" + "---|" * 11]
    for k, a in rep["site"].items():
        f = lambda d: ",".join(f"{n[0]}{c}" for n, c in sorted(d.items()))  # noqa: E731
        gt = a["green_given_tall"]
        gl = a["green_given_low"]
        sh = (f"{a['shift_med']} ({a['shift_n']}, {a['shift_abs_gt2']:.0%})" if a["shift_med"] is not None else "-")
        L.append(f"| {k} | {a['tiles']} | {a['h_mean']:.1f} | {a['strata']['20-40'] + a['strata']['40-60'] + a['strata']['>60']:.1%} | "
                 f"{a['strata']['40-60'] + a['strata']['>60']:.1%} | {f(a['landscape'])} | {f(a['landscape_1m'])} | "
                 f"{a['roughness_med']:.2f}/{a['roughness_1m_med']:.2f} | "
                 f"{'-' if gt is None else f'{gt:.0%}'} | {'-' if gl is None else f'{gl:.0%}'} | {sh} | {a['rgb_mean']} |")
    L += ["", "## Structure / leakage / tail", "", "```", json.dumps(
        {k: rep[k] for k in ("index", "stretch_bounds", "structure", "leakage", "tallest_tiles", "urban_called_tiles")},
        indent=1), "```"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="the neon/ dir holding train/ val/ test/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    root, out = Path(a.store), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for s in SPLITS for r in run_split(root, s, a.workers)]
    rep = summarise(rows, root)
    (out / "tiles.json").write_text(json.dumps(rows))
    (out / "eda.json").write_text(json.dumps(rep, indent=1))
    (out / "eda.md").write_text(to_md(rep))
    print(to_md(rep))


if __name__ == "__main__":
    main()
