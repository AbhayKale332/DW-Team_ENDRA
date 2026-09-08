import json
import zipfile

from package_results import build_zip, cloudflared_argv, share_via_cloudflared


def _fake_outputs(d):
    (d / "best.pt").write_bytes(b"x" * 10)
    (d / "metrics.json").write_text(json.dumps({"best_val_rmse_m": 3.0}))
    (d / "run.log").write_text("log\n")
    vs = d / "viewer_sample"
    vs.mkdir()
    (vs / "rgb.png").write_bytes(b"p")
    (vs / "meta.json").write_text("{}")


def test_build_zip_members(tmp_path):
    _fake_outputs(tmp_path)
    zp = build_zip(str(tmp_path))
    with zipfile.ZipFile(zp) as z:
        names = set(z.namelist())
    assert "best.pt" in names
    assert "metrics.json" in names
    assert "viewer_sample/rgb.png" in names
    assert "stageP_last.pt" not in names  # absent inputs skipped, not errored


def test_cloudflared_argv():
    a = cloudflared_argv("/bin/cloudflared", 8000)
    assert a[:2] == ["/bin/cloudflared", "tunnel"]
    assert "http://localhost:8000" in a


def test_share_noop_without_binary(tmp_path, monkeypatch):
    _fake_outputs(tmp_path)
    zp = build_zip(str(tmp_path))
    monkeypatch.setattr("package_results._ensure_cloudflared", lambda: None)
    url, procs = share_via_cloudflared(str(zp), keep_alive_minutes=0, port=8757)
    for p in procs:
        p.terminate()
    assert url is None  # gracefully degraded; local zip still exists
    assert zp.exists()
