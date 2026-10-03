"""The FastAPI service, driven with a stub model.

Skipped when fastapi is absent so the offline suite still runs on a bare image.
The one non-obvious assertion is the path-traversal check on `/api/result`: the
job id and filename both come from a URL, and this service is meant to be run on
a demo machine on somebody else's network.
"""
import io
import json
from pathlib import Path

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from config import Config  # noqa: E402
from dwdata.preprocess import PreprocSpec  # noqa: E402
from tests.stub_encoder import use_stub  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    import torch

    import serve.app as sa

    undo = use_stub(hidden=32, patch=16, layers=8)
    monkeypatch.setattr(sa, "JOBS", tmp_path / "jobs")

    from models.heads import DepthWizardNet

    cfg = Config()
    cfg.tile_size, cfg.decoder_dim, cfg.n_bins = 64, 32, 8
    cfg.grad_checkpoint_encoder = False
    net = DepthWizardNet(cfg).eval()
    sa._state.update(runtime="torch", model=net, cfg=cfg,
                     spec=PreprocSpec(tile_size=64, canonical_gsd_m=0.5),
                     device=torch.device("cpu"), ckpt="stub")
    sa._jobs.clear()
    try:
        yield TestClient(sa.create_app()), sa
    finally:
        undo()


def _png(n=200):
    from PIL import Image

    buf = io.BytesIO()
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((n, n, 3)) * 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_health_reports_the_contract(client):
    c, _ = client
    h = c.get("/api/health").json()
    assert h["ok"] and h["runtime"] == "torch"
    assert h["preproc"]["canonical_gsd_m"] == 0.5
    assert h["preproc"]["target"] == "nDSM_agl_metres"


def test_upload_predict_and_fetch(client):
    c, sa = client
    r = c.post("/api/predict", files={"file": ("scene.png", _png(), "image/png")},
               data={"gsd": "0.5"})
    assert r.status_code == 200
    job = r.json()
    st = c.get(job["status_url"]).json()
    assert st["stage"] == "done", st.get("error")
    m = st["result"]
    assert m["size_px"] == [200, 200]
    assert m["units"] == "metres_above_ground"
    assert m["product"] == "rDSM"                 # a PNG carries no georeference

    npy = c.get(f"/api/result/{job['job']}/ndsm_m.npy")
    assert npy.status_code == 200
    arr = np.load(io.BytesIO(npy.content))
    assert arr.shape == (200, 200) and arr.dtype == np.float32
    assert np.isfinite(arr).all() and (arr >= 0).all()

    # the mesh the viewer loads must be produced too
    assert c.get(f"/api/result/{job['job']}/terrain.glb").status_code == 200
    meta = c.get(f"/api/result/{job['job']}/meta.json").json()
    assert meta["preproc"]["tile_size"] == 64


def test_view_redirects_to_the_report(client):
    c, _ = client
    r = c.post("/api/predict", files={"file": ("s.png", _png(), "image/png")})
    job = r.json()
    v = c.get(job["view_url"], follow_redirects=False)
    assert v.status_code in (302, 307)
    assert v.headers["location"] == f"/api/report/{job['job']}"


def test_result_path_cannot_escape_the_jobs_dir(client, tmp_path):
    c, sa = client
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")
    for probe in ("../secret.txt", "..%2Fsecret.txt", "....//secret.txt"):
        r = c.get(f"/api/result/anyjob/{probe}")
        assert r.status_code == 404, f"{probe} was served"
    assert c.get("/api/result/nope/meta.json").status_code == 404


def test_unknown_job_is_404(client):
    c, _ = client
    assert c.get("/api/job/deadbeef").status_code == 404


def test_index_describes_the_api(client):
    c, _ = client
    assert c.get("/").status_code == 200
    assert "DepthWizard" in c.get("/").text
    assert c.get("/").json()["docs"] == "/docs"
    assert c.get("/viewer/index.html").status_code == 404
