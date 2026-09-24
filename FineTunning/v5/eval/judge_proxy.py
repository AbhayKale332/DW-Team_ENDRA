"""A benchmark that scores the product the way the judges will.

Host FAQ (`CompetitionContext/FAQs.md`): the accuracy half (50 %) is RMSE, MAE
and correlation of the **absolute DSM against SRTM or Copernicus 30 m**, on
Cartosat-2S 0.6 m GeoTIFFs, across urban / sparse / hilly / forested scenes.
GAMUS RMSE does not measure that; this does.

    scene (GeoTIFF / NRSC product / PAN+MX)
      -> infer.predict.run_windowed   (the deliverable path, RAM-bounded)
      -> dsm_m.tif anchored to --anchor
      -> scored against every --refs DEM, both
           * per pixel   — the reference bilinearly resampled onto our grid
           * per 30 m cell — our DSM block-averaged onto the reference's scale
         since the hosts have not said which (Host_Questions_Draft.md, Q1)

Also reported: the datum sanity number — median(DSM - reference) over pixels the
model calls ground (nDSM < 1 m).  With the right vertical datum it sits near 0;
an ellipsoid/geoid mix-up shows up as tens of metres (−62 m at Bhubaneswar).

    python -m eval.judge_proxy cartosat_2S_Sample/Cartosat-2E/247677521 \\
        --landscape sparse --ckpt outputs/v5/best.pt --out outputs/judge_proxy

`--stub` runs a random-initialised tiny model instead of a checkpoint: it checks
the pipeline, the RAM bound and the datum, not accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Acc:
    """Streaming RMSE / MAE / bias / Pearson r."""

    def __init__(self):
        self.n = 0
        self.s = np.zeros(6)            # sum p, t, p2, t2, pt, |d|

    def add(self, p, t):
        m = np.isfinite(p) & np.isfinite(t)
        if not m.any():
            return
        p, t = p[m].astype(np.float64), t[m].astype(np.float64)
        self.n += p.size
        self.s += [p.sum(), t.sum(), (p * p).sum(), (t * t).sum(), (p * t).sum(),
                   np.abs(p - t).sum()]

    def result(self) -> dict:
        if not self.n:
            return {"n": 0}
        n = self.n
        sp, st, spp, stt, spt, sad = self.s
        mse = (spp - 2 * spt + stt) / n
        cov = spt / n - (sp / n) * (st / n)
        vp, vt = spp / n - (sp / n) ** 2, stt / n - (st / n) ** 2
        return {"n": int(n), "rmse_m": float(np.sqrt(max(mse, 0))), "mae_m": float(sad / n),
                "bias_m": float((sp - st) / n),
                "pearson_r": float(cov / np.sqrt(max(vp * vt, 1e-12)))}


def _hist_median(h: np.ndarray, lo: float, step: float) -> float | None:
    c = np.cumsum(h)
    if not c.size or c[-1] == 0:
        return None
    return float(lo + step * (np.searchsorted(c, c[-1] / 2) + 0.5))


def score_scene(out_dir: Path, meta, k: int, refs: list[str], *, dem_cache: str = "",
                band_rows: int = 2048) -> dict:
    """Score `dsm_m.tif` in `out_dir` against each reference DEM."""
    import rasterio
    from affine import Affine
    from rasterio.windows import Window

    from geo.calibrate import block_mean, upsample_rows
    from geo.dem import fetch_dem

    dsm_p, nd_p = out_dir / "dsm_m.tif", out_dir / "ndsm_m.tif"
    with rasterio.open(dsm_p) as d:
        H, W = d.height, d.width
    hb, wb = -(-H // k), -(-W // k)
    tr_c = meta.transform * Affine.scale(k)
    out = {}
    # our DSM on the cell grid, once
    cell_sum = np.zeros((hb, wb))
    cell_n = np.zeros((hb, wb))
    with rasterio.open(dsm_p) as d:
        for r0 in range(0, H, band_rows):
            r1 = min(H, r0 + band_rows)
            a = d.read(1, window=Window(0, r0, W, r1 - r0))
            fin = np.isfinite(a)
            rb = np.arange(r0, r1) // k
            cb = np.arange(W) // k
            for b in np.unique(rb):
                m = rb == b
                cell_sum[b] += np.bincount(cb, np.where(fin[m], a[m], 0).sum(0), wb)[:wb]
                cell_n[b] += np.bincount(cb, fin[m].sum(0), wb)[:wb]
    with np.errstate(invalid="ignore"):
        ours_c = np.where(cell_n > 0, cell_sum / np.maximum(cell_n, 1), np.nan)

    for ref in refs:
        dem = fetch_dem(tr_c, meta.crs, wb, hb, source=ref, cache_dir=dem_cache)
        if dem.coverage <= 0:
            out[ref] = {"error": dem.note}
            continue
        ref_c = dem.array.astype(np.float64)
        px, lo_step = _Acc(), (-100.0, 0.05)
        hist = np.zeros(4000)
        with rasterio.open(dsm_p) as d, rasterio.open(nd_p) as nd:
            for r0 in range(0, H, band_rows):
                r1 = min(H, r0 + band_rows)
                w = Window(0, r0, W, r1 - r0)
                a = d.read(1, window=w)
                t = upsample_rows(np.nan_to_num(ref_c, nan=np.nanmean(ref_c)), k, r0, r1, W)
                px.add(a, t)
                g = nd.read(1, window=w) < 1.0
                diff = (a - t)[g & np.isfinite(a)]
                idx = np.clip(((diff - lo_step[0]) / lo_step[1]).astype(int), 0, hist.size - 1)
                hist += np.bincount(idx, minlength=hist.size)[:hist.size]
        cells = _Acc()
        cells.add(ours_c, ref_c)
        out[ref] = {"datum": dem.datum, "per_pixel": px.result(), "per_30m_cell": cells.result(),
                    "median_dsm_minus_ref_on_ground_m": _hist_median(hist, *lo_step),
                    "dem": dem.summary()}
    return out


def _stub_model(cfg_tile: int = 512):
    import torch

    from config import Config
    from models.heads import DepthWizardNet
    from tests.stub_encoder import use_stub

    use_stub(hidden=32, patch=16, layers=8)
    c = Config()
    c.tile_size, c.decoder_dim, c.n_bins = cfg_tile, 32, 8
    c.grad_checkpoint_encoder = False
    torch.manual_seed(0)
    from dwdata.preprocess import PreprocSpec

    return DepthWizardNet(c).eval(), PreprocSpec(tile_size=cfg_tile), c


def main(argv=None) -> None:
    import torch

    from dwdata.preprocess import _meta_from_source, open_scene
    from geo.calibrate import anchor_cell_px
    from infer.predict import run_windowed

    ap = argparse.ArgumentParser(description="score the absolute DSM like the judges")
    ap.add_argument("scene", nargs="+", help="scene(s); a PAN+MX pair is joined with '+'")
    ap.add_argument("--landscape", default="", help="label per scene, comma list "
                    "(urban/sparse/hilly/forested)")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--stub", action="store_true", help="random tiny model (pipeline check)")
    ap.add_argument("--anchor", default="copernicus30")
    ap.add_argument("--refs", default="copernicus30,srtmgl1")
    ap.add_argument("--detail_gain", type=float, default=1.0)
    ap.add_argument("--band_rows", type=int, default=2048)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--dem_cache", default="")
    ap.add_argument("--out", default="outputs/judge_proxy")
    ap.add_argument("--device", default="")
    a = ap.parse_args(argv)

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if a.stub:
        model, spec, cfg = _stub_model()
        tag = "stub"
    else:
        from infer.predict import load_model

        model, spec, cfg = load_model(a.ckpt, device)
        tag = Path(a.ckpt).parent.name or "ckpt"
    labels = a.landscape.split(",") if a.landscape else []
    refs = [r.strip() for r in a.refs.split(",") if r.strip()]
    report = {"model": tag, "anchor": a.anchor, "detail_gain": a.detail_gain,
              "refs": refs, "scenes": []}
    for i, sc in enumerate(a.scene):
        inputs = sc.split("+") if "+" in sc else sc
        src = open_scene(inputs)
        meta = _meta_from_source(src, inputs)
        name = Path(sc.split("+")[0]).name
        od = Path(a.out) / tag / name
        t0 = time.time()
        payload = run_windowed(model, spec, src, meta, device, od, absolute=True,
                               dem_source=a.anchor, detail_gain=a.detail_gain,
                               dem_cache=a.dem_cache, tta=a.tta,
                               band_rows=a.band_rows, mesh=False)
        k = anchor_cell_px(meta.gsd_m)
        sc_res = score_scene(od, meta, k, refs, dem_cache=a.dem_cache,
                             band_rows=a.band_rows) if (od / "dsm_m.tif").is_file() else {}
        rec = {"scene": sc, "landscape": labels[i] if i < len(labels) else "",
               "size_px": [meta.height, meta.width], "gsd_m": meta.gsd_m,
               "product": (src.product.summary() if src.product else None),
               "seconds": round(time.time() - t0, 1),
               "sun": payload.get("sun"), "scores": sc_res}
        report["scenes"].append(rec)
        print(json.dumps({k2: v for k2, v in rec.items() if k2 != "product"}, indent=1,
                         default=str)[:3000])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"judge_proxy_{tag}.json").write_text(json.dumps(report, indent=2,
                                                                     default=str))
    # a flat table
    lines = ["| scene | landscape | ref | per-px RMSE | MAE | r | 30 m RMSE | 30 m r | "
             "median(DSM-ref) on ground |", "|---|---|---|---|---|---|---|---|---|"]
    for s in report["scenes"]:
        for ref, r in s["scores"].items():
            if "per_pixel" not in r:
                lines.append(f"| {Path(s['scene']).name} | {s['landscape']} | {ref} | "
                             f"{r.get('error', '')} |||||")
                continue
            p, c = r["per_pixel"], r["per_30m_cell"]
            med = r["median_dsm_minus_ref_on_ground_m"]
            lines.append(f"| {Path(s['scene']).name} | {s['landscape']} | {ref} ({r['datum']}) "
                         f"| {p['rmse_m']:.2f} | {p['mae_m']:.2f} | {p['pearson_r']:.3f} | "
                         f"{c.get('rmse_m', float('nan')):.2f} | "
                         f"{c.get('pearson_r', float('nan')):.3f} | "
                         f"{'n/a' if med is None else f'{med:+.2f}'} |")
    (Path(a.out) / f"judge_proxy_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
