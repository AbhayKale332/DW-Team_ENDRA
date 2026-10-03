"""Modal Cartosat input preparation and frozen L4 inference."""
from pathlib import Path
import os
import subprocess
import sys
import threading
import modal
from final_modal import DATA, RESULTS, V5, image, hf, data_vol, res_vol, _every

app = modal.App("dw-cartosat", image=image)
INPUT = f"{DATA}/cartosat_test"
OUTPUT = f"{RESULTS}/cartosat_benchmark"


@app.function(cpu=2, memory=8192, timeout=3600,
              secrets=[modal.Secret.from_name("dw-cartosat-hf"), modal.Secret.from_name("dw-opentopo")],
              volumes={DATA: data_vol, RESULTS: res_vol})
def prep():
    import json
    from huggingface_hub import snapshot_download
    sys.path.insert(0, V5)
    from dwdata.preprocess import scene_geometry
    from geo.calibrate import anchor_cell_px
    from geo.dem import fetch_dem
    from affine import Affine
    snapshot_download("Abhay-Kale/C2-C3-Test", repo_type="dataset",
                      token=os.environ["HF_DATA_TOKEN"], local_dir=INPUT,
                      allow_patterns=["*/testing_crops/*.tif", "*/testing_crops_hills/*.tif", "upload_manifest.json"])
    paths = sorted(p for p in Path(INPUT).rglob("*.tif")
                   if p.parent.name in ("testing_crops", "testing_crops_hills"))
    if len(paths) != 67:
        raise RuntimeError(f"Expected 67 crops, found {len(paths)}")
    out = Path(OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    import urllib.request
    grids = out / "proj"
    grids.mkdir(exist_ok=True)
    for name in ("us_nga_egm96_15.tif", "us_nga_egm08_25.tif"):
        target = grids / name
        if not target.exists():
            with urllib.request.urlopen(f"https://cdn.proj.org/{name}", timeout=120) as response:
                tmp = target.with_suffix(".part")
                with tmp.open("wb") as f:
                    import shutil
                    shutil.copyfileobj(response, f)
                tmp.replace(target)
    res_vol.commit()
    # Ten regional API requests cover all 67 crops and preserve the daily quota.
    import math
    from collections import defaultdict
    from rasterio.warp import transform_bounds
    from rasterio.transform import array_bounds
    groups = defaultdict(list)
    geometries = {}
    for p in paths:
        m = scene_geometry(str(p))
        bounds = transform_bounds(m.crs, "EPSG:4326", *array_bounds(m.height, m.width, m.transform))
        group = (math.floor((bounds[0]+bounds[2])/2), math.floor((bounds[1]+bounds[3])/2))
        geometries[str(p)] = (m, group)
        groups[group].append(bounds)
    regional = {}
    for group, bounds in groups.items():
        w, s = min(b[0] for b in bounds)-.01, min(b[1] for b in bounds)-.01
        e, n = max(b[2] for b in bounds)+.01, max(b[3] for b in bounds)+.01
        tr = Affine(1/3600, 0, w, 0, -1/3600, n)
        refs = {}
        for ref in ("srtmgl1", "nasadem"):
            dem = fetch_dem(tr, "EPSG:4326", math.ceil((e-w)*3600), math.ceil((n-s)*3600),
                            source=ref, cache_dir=str(out / "dem_cache"))
            if dem.coverage > 0 and dem.files:
                refs[ref] = str(dem.files[0])
            print({"region": group, "ref": ref, "result": dem.summary()}, flush=True)
        regional[group] = refs
    rows = []
    for p in paths:
        meta, group = geometries[str(p)]
        k = anchor_cell_px(meta.gsd_m)
        refs = {}
        for ref in ("copernicus30",):
            dem = fetch_dem(meta.transform*Affine.scale(k), meta.crs,
                            -(-meta.width//k), -(-meta.height//k), source=ref,
                            cache_dir=str(out / "dem_cache"))
            refs[ref] = dem.summary()
        rows.append({"path": str(p.relative_to(INPUT)), "anchor": refs["copernicus30"],
                     "ref_paths": regional[group]})
        print(rows[-1], flush=True)
    (out / "inputs.json").write_text(json.dumps(rows, indent=2))
    data_vol.commit()
    res_vol.commit()
    return {"crops": len(paths), "anchor_available": sum(r["anchor"].get("coverage", 0)>0 for r in rows)}


@app.function(gpu="L4", cpu=2, memory=8192, timeout=5400,
              secrets=[hf, modal.Secret.from_name("dw-opentopo")],
              volumes={DATA: data_vol.with_mount_options(read_only=True), RESULTS: res_vol})
def benchmark():
    out = Path(OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "inputs.json").exists():
        raise RuntimeError("Run CPU prep successfully before GPU inference")
    stop = threading.Event()
    _every(60, res_vol.commit, stop)
    try:
        with (out / "benchmark.log").open("w", buffering=1) as log:
            proc = subprocess.Popen([sys.executable, "-u", "tools/cartosat_benchmark.py",
                "--root", INPUT, "--out", OUTPUT, "--ckpt", f"{RESULTS}/v5_final_forest/best.pt"],
                cwd=V5, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if proc.wait():
                raise RuntimeError(f"Cartosat benchmark exited {proc.returncode}")
    finally:
        stop.set()
        res_vol.commit()
    return f"{OUTPUT}/metrics.json"


@app.function(cpu=2, memory=8192, timeout=600,
              volumes={DATA: data_vol.with_mount_options(read_only=True), RESULTS: res_vol})
def package():
    """Audit output grids and make a portable gallery; keep full rasters on Modal."""
    import json
    import tarfile
    import rasterio
    from pyproj import CRS
    out = Path(OUTPUT)
    report = json.loads((out / "metrics.json").read_text())
    if len(report["scenes"]) != 67 or any(r["status"] != "complete" for r in report["scenes"].values()):
        raise RuntimeError("All 67 crops must finish before packaging")
    for key, r in report["scenes"].items():
        dest = out / "scenes" / Path(key).stem
        meta = json.loads((dest / "meta.json").read_text())
        r["height_statistics_grid"] = "overview, maximum dimension 1024 px"
        r["full_resolution_stats"] = meta.get("full_resolution_stats", {})
        grids = []
        for name in ("ndsm_m.tif", "dsm_m.tif", "dtm_m.tif", "seg.tif"):
            with rasterio.open(dest / name) as d:
                if [d.height, d.width] != r["size"]:
                    raise RuntimeError(f"Output grid size mismatch: {key}/{name}")
                grids.append((d.transform, d.crs))
                if name in ("dsm_m.tif", "dtm_m.tif") and d.tags(1).get("VERTICAL_DATUM") != "EGM2008":
                    raise RuntimeError(f"Output datum mismatch: {key}/{name}")
        if any(g[0] != grids[0][0] for g in grids):
            raise RuntimeError(f"Output transforms differ: {key}")
        if abs(grids[0][0].a - r["gsd_m"]) > 1e-6:
            raise RuntimeError(f"Output GSD mismatch: {key}")
        with rasterio.open(Path(INPUT, key)) as src:
            for transform, crs in grids:
                horizontal = CRS(crs)
                if horizontal.is_compound:
                    horizontal = horizontal.sub_crs_list[0]
                if transform != src.transform or horizontal != CRS(src.crs):
                    raise RuntimeError(f"Output georeferencing differs from input: {key}")
        r["output_grid_audit"] = "passed: native size, input transform and horizontal CRS, GSD, EGM2008 DSM/DTM"
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    page = out / "index.html"
    legend = ('<style>body{font-family:system-ui;max-width:1000px;margin:24px auto;padding:16px}'
              'img{max-width:48%;height:auto}pre{white-space:pre-wrap}</style>'
              '<p>Height legend (above ground; values above 40 m saturate):</p>'
              '<div style="height:20px;max-width:600px;background:linear-gradient(to right,'
              '#0000ff,#8000bf,#ff8080,#ffff40,#ffff00)"></div>'
              '<p>0 m &nbsp;&nbsp;&nbsp; 10 m &nbsp;&nbsp;&nbsp; 20 m &nbsp;&nbsp;&nbsp; 30 m &nbsp;&nbsp;&nbsp; 40 m+</p>'
              '<p>Predicted height percentiles use the overview grid. '
              'Full-resolution statistics and output grid audits are in <a href="metrics.json">metrics.json</a>.</p>')
    text = page.read_text()
    if 'Height legend (above ground' not in text:
        page.write_text(text.replace('<h1>', legend+'<h1>', 1))
    archive = out / "cartosat_gallery.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("index.html", "metrics.json", "inputs.json", "benchmark.log"):
            tar.add(out / name, arcname=name)
        for p in sorted((out / "scenes").rglob("*")):
            if p.is_file() and p.suffix in (".png", ".json"):
                tar.add(p, arcname=str(p.relative_to(out)))
    res_vol.commit()
    return {"archive": str(archive), "bytes": archive.stat().st_size, "grid_audits": 67}
