"""Cartosat inference and coarse DEM agreement, without high-resolution truth."""
from pathlib import Path
import argparse
import html
import json
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def scenes(root):
    return sorted(p for p in Path(root).rglob("*.tif")
                  if p.parent.name in ("testing_crops", "testing_crops_hills"))


def main():
    import numpy as np
    import torch
    from PIL import Image
    from config import Config
    from dwdata.preprocess import PreprocSpec, open_scene, _meta_from_source
    from eval.judge_proxy import score_scene
    from eval_test import load_checkpoint
    from geo.calibrate import anchor_cell_px
    from infer.predict import run_windowed
    from train import resolve_hf_token

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", default="copernicus30,srtmgl1,nasadem")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import pyproj
    pyproj.datadir.append_data_dir(str(out / "proj"))
    cfg = Config()
    cfg.hf_token = resolve_hf_token(cfg)
    device = torch.device("cuda")
    model, ck = load_checkpoint(cfg, a.ckpt, "", [], device)
    spec = PreprocSpec.from_dict(ck["preproc"]) if "preproc" in ck else PreprocSpec.from_config(cfg)
    del ck
    inputs = scenes(a.root)
    prepared = {r["path"]: r for r in json.loads((out / "inputs.json").read_text())}
    if len(inputs) != 67:
        raise RuntimeError(f"Expected 67 Cartosat crops, found {len(inputs)}")
    report = {"checkpoint": a.ckpt, "protocol": "Baseline frozen inference; Copernicus30 anchoring. "
              "Copernicus agreement is self-consistency. SRTM/NASADEM are coarse historical "
              "DEM comparisons, not building/tree truth. PAN is replicated into RGB.", "scenes": {}}
    report_path = out / "metrics.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    for i, p in enumerate(inputs):
        key = str(p.relative_to(a.root))
        if report["scenes"].get(key, {}).get("status") == "complete":
            continue
        dest = out / "scenes" / p.stem
        start = time.monotonic()
        print(f"[{i+1}/{len(inputs)}] {key}", flush=True)
        try:
            source = open_scene(str(p))
            meta = _meta_from_source(source, str(p))
            if not ((dest / "ndsm_m.tif").exists() and (dest / "meta.json").exists()):
                run_windowed(model, spec, source, meta, device, dest, absolute=True,
                             dem_source="copernicus30", dem_cache=str(out / "dem_cache"),
                             amp_dtype=torch.bfloat16, batch_tiles=2, band_rows=512,
                             mesh=False, overview_max=1024)
            h = np.load(dest / "ndsm_m.npy")
            finite = h[np.isfinite(h)]
            if not finite.size:
                raise RuntimeError("No finite predicted heights")
            # Shared 0..40 m colour scale makes crops visually comparable.
            v = np.clip(np.nan_to_num(h) / 40., 0, 1)
            rgb = np.stack([np.clip(2*v, 0, 1), np.clip(2*v-0.5, 0, 1), 1-v], axis=-1)
            rgb[~np.isfinite(h)] = 0
            Image.fromarray((rgb*255).astype(np.uint8)).save(dest / "height_0_40m.png")
            scores = score_scene(dest, meta, anchor_cell_px(meta.gsd_m), a.refs.split(","),
                                 dem_cache=str(out / "dem_cache"), band_rows=512,
                                 ref_paths=prepared[key]["ref_paths"]) if (dest / "dsm_m.tif").exists() else {"error": "No absolute DSM: anchor unavailable"}
            report["scenes"][key] = {"status": "complete", "seconds": time.monotonic()-start,
                "gsd_m": meta.gsd_m, "crs": str(meta.crs), "size": [meta.height, meta.width],
                "finite_fraction": float(np.isfinite(h).mean()),
                "ndsm_percentiles_m": np.percentile(finite, [0, 25, 50, 75, 95, 99, 100]).tolist(),
                "predicted_fraction_above_20m": float((finite > 20).mean()), "dem_agreement": scores}
        except Exception as e:
            traceback.print_exc()
            report["scenes"][key] = {"status": "failed", "error": str(e)}
        tmp = report_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(report_path)
    rows = ["<h1>Cartosat inference</h1><p>No scene DSM ground truth. Heights below are predictions. "
            "Copernicus agreement is enforced by anchoring, not independent validation. "
            "Height colours use a shared 0–40 m scale.</p>"]
    for key, r in report["scenes"].items():
        stem = Path(key).stem
        rows.append(f'<h2>{html.escape(key)}</h2><p>{html.escape(r["status"])}</p>')
        if r["status"] == "complete":
            rows.append(f'<img width="450" src="scenes/{stem}/rgb.png"><img width="450" src="scenes/{stem}/height_0_40m.png">')
            rows.append('<pre>'+html.escape(json.dumps(r, indent=2))+'</pre>')
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8">'+''.join(rows))
    failures = [k for k, r in report["scenes"].items() if r["status"] != "complete"]
    print(f"Finished: {len(inputs)-len(failures)}/{len(inputs)} crops", flush=True)
    if failures:
        raise RuntimeError(f"Failed crops: {failures}")


if __name__ == "__main__":
    main()
