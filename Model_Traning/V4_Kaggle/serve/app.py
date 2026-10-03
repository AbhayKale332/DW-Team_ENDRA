"""FastAPI inference service.

Start with `python -m serve.app --ckpt <checkpoint>`.
Use /docs to upload images through the API and /api/report/{job_id}
to view generated reports. Jobs and inference products are stored on disk.
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

        h, s = self.session.run(None, {"image": t.detach().cpu().numpy()})
        return {"fused": torch.from_numpy(h), "seg": _onehot(torch.from_numpy(s))}


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
    import torch

    from dwdata.preprocess import read_scene
    from infer.engine import predict_scene
    from infer.predict import write_outputs

    job = _jobs[job_id]
    out_dir = JOBS / job_id
    try:
        spec, device = _state["spec"], _state["device"]
        job.update(stage="reading", progress=0.05)
        rgb, meta = read_scene(image_path, user_gsd_m=float(opts.get("gsd") or 0),
                               assumed_gsd_m=spec.canonical_gsd_m,
                               max_side=int(opts.get("max_side") or 0))
        job.update(stage="predicting", progress=0.15,
                   scene={"w": meta.width, "h": meta.height, "gsd_m": meta.gsd_m,
                          "georeferenced": meta.georeferenced})

        amp = (torch.bfloat16 if device.type == "cuda" else None)

        def prog(done, total):
            job.update(progress=0.15 + 0.7 * done / max(total, 1),
                       detail=f"tile {done}/{total}")

        # the same call the CLI and the final evaluation make
        with _state["lock"]:
            height, seg = predict_scene(
                _model(), rgb, meta.gsd_m, spec, device,
                tta=bool(opts.get("tta")), amp_dtype=amp,
                overlap=0.25, batch_tiles=int(opts.get("batch_tiles") or 4),
                want_seg=True, progress=prog)

        dsm_abs, dtm, extra = None, None, {}
        if opts.get("absolute") and meta.georeferenced:
            job.update(stage="terrain", progress=0.88)
            from infer.predict import to_absolute

            dsm_abs, extra = to_absolute(
                height, seg, meta, dem_source=opts.get("dem_source", "copernicus30"),
                dem_path=opts.get("dem", ""))
            dtm = extra.pop("dtm", None)
            extra.pop("ground_mask", None)

        job.update(stage="writing", progress=0.93)
        payload = write_outputs(out_dir, Path(image_path).stem, rgb, height, meta,
                                spec, dsm_abs, extra, seg=seg, dtm=dtm,
                                mesh=bool(opts.get("mesh", True)))
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

    app = FastAPI(title="DepthWizard", version="4.0",
                  description="Single-view RGB -> metric DSM inference API")
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
                      file: UploadFile = File(...),
                      gsd: float = Form(0.0),
                      tta: bool = Form(False),
                      absolute: bool = Form(False),
                      dem_source: str = Form("copernicus30"),
                      max_side: int = Form(0)):
        if _state["spec"] is None:
            raise HTTPException(503, "no model loaded — start with --ckpt or --onnx")
        job_id = uuid.uuid4().hex[:12]
        d = JOBS / job_id
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"input{Path(file.filename or 'scene.png').suffix or '.png'}"
        with open(dst, "wb") as f:
            shutil.copyfileobj(file.file, f)
        _jobs[job_id] = {"id": job_id, "stage": "queued", "progress": 0.0,
                         "filename": file.filename, "started": time.time()}
        background.add_task(run_job, job_id, dst,
                            {"gsd": gsd, "tta": tta, "absolute": absolute,
                             "dem_source": dem_source, "max_side": max_side})
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

    @app.get("/api/result/{job_id}/{name}")
    def result_file(job_id: str, name: str):
        p = (JOBS / job_id / name).resolve()
        # containment check: a job id or name from a URL must never escape JOBS
        if not str(p).startswith(str(JOBS.resolve())) or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(p)

    @app.get("/view/{job_id}")
    def view(job_id: str):
        return RedirectResponse(f"/api/report/{job_id}")

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

    @app.get("/")
    def index():
        return JSONResponse({"service": "DepthWizard V4_Kaggle",
                             "docs": "/docs",
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
    print(f"[serve] http://{a.host}:{a.port}/   API docs at /docs")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
