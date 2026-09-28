"""FastAPI service: upload an image -> DSM products -> the 3D viewer.

    uvicorn serve.app:app --host 0.0.0.0 --port 8000
    # or
    python -m serve.app --ckpt outputs/v4/best.pt --port 8000

Then open http://localhost:8000/ — the viewer is served from the same origin, so
a finished job's URL (`/view/<job>`) loads the result with no CORS, no file
picker and no manual copying.

Design notes:

* **One inference path, again.**  This service calls `infer.engine.predict_scene`,
  the same function the final evaluation and the CLI call.  If a number on the
  report is right, the number the demo shows is right, because it came out of the
  same code.

* **Two runtimes.**  `--runtime torch` loads the checkpoint; `--runtime onnx`
  loads `depthwizard.onnx` and needs neither CUDA nor `transformers` nor an HF
  token.  The ONNX path is what ships in the standalone bundle: a judge's laptop
  runs it on CPU.

* **Jobs are files on disk**, one directory per job, with exactly the layout the
  viewer and `infer/predict.py` already agree on.  No database, no queue, no
  session state — which is the cheapest way to be stable under a demo, and it
  means a crashed process loses nothing.
"""

# NOTE: no `from __future__ import annotations` here on purpose.  FastAPI resolves
# endpoint signatures at definition time, and postponed annotations turn
# `file: UploadFile = File(...)` into the string "UploadFile", which pydantic
# cannot resolve because the import is local to create_app().  The failure is a
# 500 at request time, not an import error, so it would have shipped.
import argparse
import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JOBS = Path(os.environ.get("DW_JOBS_DIR", ROOT / "outputs" / "jobs"))
VIEWER = ROOT / "viewer"

_state: dict = {"model": None, "spec": None, "cfg": None, "runtime": "torch",
                "session": None, "device": None, "ckpt": "", "lock": threading.Lock()}
_jobs: dict = {}


# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------
def load_backend(ckpt: str = "", onnx: str = "", device: str = "") -> dict:
    """Bring up whichever runtime was asked for.  Called once at startup."""
    import torch

    if onnx:
        import onnxruntime as ort

        from dwdata.preprocess import PreprocSpec

        meta_path = Path(str(onnx) + ".json")
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        _state.update(
            runtime="onnx",
            session=ort.InferenceSession(
                str(onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]),
            spec=PreprocSpec.from_dict(meta.get("preproc", {})),
            device=torch.device("cpu"), ckpt=str(onnx))
        print(f"[serve] onnx runtime: {onnx}  providers="
              f"{_state['session'].get_providers()}")
        return _state

    from infer.predict import load_model

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, spec, cfg = load_model(ckpt, dev)
    _state.update(runtime="torch", model=model, spec=spec, cfg=cfg, device=dev,
                  ckpt=str(ckpt))
    print(f"[serve] torch runtime on {dev}: {ckpt}")
    return _state


class _OnnxModel:
    """Adapter so `infer.engine` can drive an ONNX session unchanged.

    The engine only ever calls `model(tensor)["fused"]` / `["seg"]`, so matching
    that shape here is enough to reuse the whole tiling, blending and TTA path
    rather than writing a second, subtly different one for deployment.
    """

    def __init__(self, session):
        self.session = session

    def eval(self):
        return self

    def __call__(self, t):
        import torch

        outs = self.session.run(None, {"image": t.detach().cpu().numpy()})
        d = {"fused": torch.from_numpy(outs[0]), "seg": _onehot(torch.from_numpy(outs[1]))}
        if len(outs) > 2:                   # v5 graphs carry Head B's spread
            d["b_std"] = torch.from_numpy(outs[2])
        return d


def _onehot(seg):
    """(B,1,H,W) class ids -> (B,C,H,W) logits, because engine.py argmaxes."""
    import torch
    import torch.nn.functional as F

    from config import N_SEG_CLASSES

    return F.one_hot(seg[:, 0].long().clamp(0, N_SEG_CLASSES - 1),
                     N_SEG_CLASSES).permute(0, 3, 1, 2).float() * 10.0 \
        if torch.is_tensor(seg) else seg


def _model():
    if _state["runtime"] == "onnx":
        return _OnnxModel(_state["session"])
    return _state["model"]


# ---------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------
def run_job(job_id: str, image_path: Path, opts: dict) -> None:
    """One upload -> one product.  `image_path` may be a file, a zip, or a folder
    of the uploaded files (an NRSC product: BAND1..4.tif + BAND_META.txt).
    Scenes above `windowed_mp` megapixels stream through `run_windowed`."""
    import torch

    from dwdata.preprocess import _meta_from_source, open_scene, read_scene
    from infer.engine import predict_scene
    from infer.predict import run_windowed, write_outputs

    job = _jobs[job_id]
    out_dir = JOBS / job_id
    try:
        spec, device = _state["spec"], _state["device"]
        job.update(stage="reading", progress=0.05)
        bands = ([int(b) for b in str(opts["bands"]).split(",")]
                 if opts.get("bands") else None)
        gsd = float(opts.get("gsd") or 0)
        source = open_scene(image_path, bands, gsd, spec.canonical_gsd_m)
        amp = (torch.bfloat16 if device.type == "cuda" else None)
        absolute = bool(opts.get("absolute"))
        mode = opts.get("dsm_mode") or "dem_anchored"
        gain = float(opts.get("detail_gain", 1.0))

        def prog(done, total):
            job.update(progress=0.15 + 0.7 * done / max(total, 1),
                       detail=f"{done}/{total}")

        max_side = int(opts.get("max_side") or 0)
        if not max_side and source.width * source.height > float(
                opts.get("windowed_mp") or 40.0) * 1e6:
            meta = _meta_from_source(source, image_path)
            job.update(stage="predicting (windowed)", progress=0.1,
                       scene={"w": meta.width, "h": meta.height, "gsd_m": meta.gsd_m,
                              "georeferenced": meta.georeferenced})
            with _state["lock"]:
                payload = run_windowed(
                    _model(), spec, source, meta, device, out_dir, absolute=absolute,
                    dem_source=opts.get("dem_source", "copernicus30"),
                    dem_path=opts.get("dem", ""), detail_gain=gain,
                    tta=bool(opts.get("tta")), amp_dtype=amp,
                    batch_tiles=int(opts.get("batch_tiles") or 4),
                    mesh=bool(opts.get("mesh", True)), progress=prog)
            job.update(stage="done", progress=1.0, result=payload,
                       files=sorted(p.name for p in out_dir.iterdir()),
                       finished=time.time())
            return

        rgb, meta = read_scene(image_path, user_gsd_m=gsd,
                               assumed_gsd_m=spec.canonical_gsd_m,
                               max_side=max_side, bands=bands)
        job.update(stage="predicting", progress=0.15,
                   scene={"w": meta.width, "h": meta.height, "gsd_m": meta.gsd_m,
                          "georeferenced": meta.georeferenced})

        # the same call the CLI and the final evaluation make
        with _state["lock"]:
            height, seg, std = predict_scene(
                _model(), rgb, meta.gsd_m, spec, device,
                tta=bool(opts.get("tta")), amp_dtype=amp,
                overlap=0.25, batch_tiles=int(opts.get("batch_tiles") or 4),
                want_seg=True, progress=prog, valid=meta.valid, return_std=True)

        dsm_abs, dtm, extra, datum = None, None, {}, ""
        if absolute and meta.georeferenced:
            job.update(stage="terrain", progress=0.88)
            from infer.predict import to_absolute

            dsm_abs, extra = to_absolute(
                height, seg, meta, dem_source=opts.get("dem_source", "copernicus30"),
                dem_path=opts.get("dem", ""), mode=mode, detail_gain=gain)
            dtm = extra.pop("dtm", None)
            extra.pop("ground_mask", None)
            datum = extra.get("vertical_datum", "")

        job.update(stage="writing", progress=0.93)
        payload = write_outputs(out_dir, Path(image_path).stem, rgb, height, meta,
                                spec, dsm_abs, extra, seg=seg, dtm=dtm,
                                mesh=bool(opts.get("mesh", True)), datum=datum, std=std)
        job.update(stage="done", progress=1.0, result=payload,
                   files=sorted(p.name for p in out_dir.iterdir()),
                   finished=time.time())
    except Exception as e:  # noqa: BLE001
        import traceback

        job.update(stage="error", error=str(e), traceback=traceback.format_exc(),
                   finished=time.time())
        print(f"[serve] job {job_id} failed:\n{job['traceback']}")


# ---------------------------------------------------------------------
# app
# ---------------------------------------------------------------------
def create_app():
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="DepthWizard", version="5.0",
                  description="Single-view RGB -> metric DSM -> 3D flythrough")
    JOBS.mkdir(parents=True, exist_ok=True)

    @app.get("/api/health")
    def health():
        return {
            "ok": _state["spec"] is not None,
            "runtime": _state["runtime"],
            "checkpoint": _state["ckpt"],
            "device": str(_state["device"]),
            "preproc": _state["spec"].to_dict() if _state["spec"] else None,
            "jobs": len(_jobs),
        }

    @app.post("/api/predict")
    async def predict(background: BackgroundTasks,
                      file: list[UploadFile] = File(...),
                      gsd: float = Form(0.0),
                      tta: bool = Form(False),
                      absolute: bool = Form(False),
                      dem_source: str = Form("copernicus30"),
                      dsm_mode: str = Form("dem_anchored"),
                      detail_gain: float = Form(1.0),
                      bands: str = Form(""),
                      max_side: int = Form(0)):
        """One image, a product .zip, or several files at once (an NRSC MERGED
        product arrives as BAND1..4.tif + BAND_META.txt — send them together)."""
        if _state["spec"] is None:
            raise HTTPException(503, "no model loaded — start with --ckpt or --onnx")
        job_id = uuid.uuid4().hex[:12]
        d = JOBS / job_id
        d.mkdir(parents=True, exist_ok=True)
        files = [f for f in file if f is not None]
        if len(files) == 1:
            f0 = files[0]
            dst = d / f"input{Path(f0.filename or 'scene.png').suffix or '.png'}"
            with open(dst, "wb") as fh:
                shutil.copyfileobj(f0.file, fh)
        else:
            dst = d / "input"
            dst.mkdir(exist_ok=True)
            for f in files:
                name = Path(f.filename or "band.tif").name   # no directory traversal
                with open(dst / name, "wb") as fh:
                    shutil.copyfileobj(f.file, fh)
        _jobs[job_id] = {"id": job_id, "stage": "queued", "progress": 0.0,
                         "filename": ", ".join(f.filename or "" for f in files),
                         "started": time.time()}
        background.add_task(run_job, job_id, dst,
                            {"gsd": gsd, "tta": tta, "absolute": absolute,
                             "dem_source": dem_source, "max_side": max_side,
                             "dsm_mode": dsm_mode, "detail_gain": detail_gain,
                             "bands": bands})
        return {"job": job_id, "status_url": f"/api/job/{job_id}",
                "view_url": f"/view/{job_id}"}

    @app.get("/api/job/{job_id}")
    def job_status(job_id: str):
        j = _jobs.get(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        return {k: v for k, v in j.items() if k != "traceback"}

    @app.get("/api/jobs")
    def list_jobs():
        return sorted(({"id": k, "stage": v["stage"], "started": v["started"],
                        "filename": v.get("filename")} for k, v in _jobs.items()),
                      key=lambda r: -r["started"])

    def _job_dir(job_id: str) -> Path:
        d = (JOBS / job_id).resolve()
        if d.parent != JOBS.resolve() or not (d / "meta.json").is_file():
            raise HTTPException(404, "unknown job or no result yet")
        return d

    @app.post("/api/reference/{job_id}")
    async def reference(job_id: str, file: UploadFile = File(...),
                        kind: str = Form("auto"), datum: str = Form("")):
        """Validate a finished job against an uploaded reference DSM / nDSM
        (any GeoTIFF): reprojected onto the prediction grid, datum-converted,
        scored per pixel, on confident pixels and per 30 m cell."""
        from serve.validate import validate_reference

        d = _job_dir(job_id)
        if kind not in ("auto", "dsm", "ndsm"):
            raise HTTPException(422, "kind must be auto, dsm or ndsm")
        if datum and datum not in ("EGM96", "EGM2008", "WGS84"):
            raise HTTPException(422, "datum must be EGM96, EGM2008 or WGS84")
        dst = d / f"reference{Path(file.filename or 'ref.tif').suffix or '.tif'}"
        with open(dst, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        try:
            return validate_reference(d, str(dst), kind=kind, ref_datum=datum)
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        except Exception as e:  # noqa: BLE001  rasterio errors are not ValueErrors
            raise HTTPException(422, f"could not read the reference: {e}") from e

    @app.post("/api/aoi/{job_id}")
    def aoi(job_id: str, row: int = Form(...), col: int = Form(...),
            h: int = Form(...), w: int = Form(...), max_px: int = Form(2048)):
        """A full-resolution window (given in the job's viewer pixels) as its own
        product directory; the viewer then loads `base` like any result."""
        from serve.validate import extract_aoi

        d = _job_dir(job_id)
        try:
            with _state["lock"]:
                r = extract_aoi(d, row, col, h, w, max_px=min(max(64, max_px), 4096))
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        return {"base": f"/api/result/{job_id}/{r['dir']}", "shape": r["shape"],
                "decimation": r["decimation"], "window_full_res": r["window_full_res"]}

    @app.get("/api/result/{job_id}/{name:path}")
    def result_file(job_id: str, name: str):
        root = JOBS.resolve()
        p = (JOBS / job_id / name).resolve()
        # containment check: a job id or name from a URL must never escape JOBS
        # (`name` may hold an AOI subdirectory; a string-prefix test would also
        # accept a sibling such as `jobs2/`)
        if not p.is_relative_to(root) or p == root or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(p)

    @app.get("/view/{job_id}")
    def view(job_id: str):
        return RedirectResponse(f"/viewer/index.html?result=/api/result/{job_id}")

    @app.get("/api/report/{job_id}")
    def report(job_id: str):
        """Per-scene HTML report — the same generator the training run uses."""
        d = JOBS / job_id
        if not (d / "meta.json").is_file():
            raise HTTPException(404, "job has no result yet")
        meta = json.loads((d / "meta.json").read_text())
        pred = np.load(d / "ndsm_m.npy")
        gt_p = d / "gt_ndsm_m.npy"
        from PIL import Image

        from viz.figures import make_all
        from viz.report_html import build

        rgb = np.asarray(Image.open(d / "rgb.png").convert("RGB"))
        samples = [(rgb, pred, np.load(gt_p))] if gt_p.is_file() else None
        synth = {"config": {"datasets": "inference"}, "preproc": meta.get("preproc"),
                 "history": [], "final_plain": {"global": {}}}
        make_all(d / "figures", synth, samples, float(meta["scene"]["gsd_m"]))
        (d / "metrics.json").write_text(json.dumps(synth))
        return FileResponse(build(d, d / "report.html",
                                  title=f"DepthWizard — {meta['stem']}"))

    if VIEWER.is_dir():
        app.mount("/viewer", StaticFiles(directory=str(VIEWER), html=True),
                  name="viewer")

    @app.get("/")
    def index():
        p = HERE / "static" / "index.html"
        if p.is_file():
            return FileResponse(p)
        return JSONResponse({"service": "DepthWizard v4",
                             "viewer": "/viewer/index.html",
                             "health": "/api/health"})

    return app


app = None
if os.environ.get("DW_CKPT") or os.environ.get("DW_ONNX"):
    load_backend(os.environ.get("DW_CKPT", ""), os.environ.get("DW_ONNX", ""))
try:
    app = create_app()
except ImportError:                    # fastapi absent -> importable for tests
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description="DepthWizard inference service")
    ap.add_argument("--ckpt", default=str(ROOT / "outputs" / "v4" / "best.pt"))
    ap.add_argument("--onnx", default="", help="use an ONNX graph instead (CPU-friendly)")
    ap.add_argument("--device", default="")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    import uvicorn

    load_backend("" if a.onnx else a.ckpt, a.onnx, a.device)
    global app
    app = app or create_app()
    print(f"[serve] http://{a.host}:{a.port}/   viewer at /viewer/index.html")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
