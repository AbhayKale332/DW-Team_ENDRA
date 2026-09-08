import os

import pytest

from dwdata.streaming import BoundedCacheHF


def test_lru_eviction(tmp_path):
    c = BoundedCacheHF("dummy/repo", str(tmp_path), max_bytes=3000)
    d = c.root
    for i in range(6):
        (d / f"f{i}.bin").write_bytes(b"x" * 1000)
        os.utime(d / f"f{i}.bin", (i, i))  # ascending mtime
    c._maybe_evict()
    size = c._dir_size()
    assert size <= 3000
    # newest files survive, oldest evicted
    assert (d / "f5.bin").exists()
    assert not (d / "f0.bin").exists()


def test_cache_hit_touches_mtime(tmp_path):
    c = BoundedCacheHF("dummy/repo", str(tmp_path), max_bytes=10**9)
    p = c.local_path("a/b.bin")
    p.parent.mkdir(parents=True)
    p.write_bytes(b"hello")
    before = p.stat().st_mtime
    os.utime(p, (1, 1))
    got = c.get("a/b.bin")
    assert got == p and got.stat().st_mtime >= before - 1


def test_archive_argv_and_extract(tmp_path):
    import zipfile

    arc = tmp_path / "sample.zip"
    with zipfile.ZipFile(arc, "w") as z:
        z.writestr("opt/x.tif", b"1")
        z.writestr("gt_nDSM/x.tif", b"2")
    out = tmp_path / "out"
    BoundedCacheHF._extract(arc, out, "")
    assert (out / "opt" / "x.tif").exists()


def _fake_archive(root, name, n_files, size):
    d = root / "_extracted" / name
    (d / "opt").mkdir(parents=True)
    for i in range(n_files):
        (d / "opt" / f"{i}.tif").write_bytes(b"x" * size)
    (d / ".extracted_ok").write_text("ok")
    return d


def test_evict_never_touches_pinned_archive(tmp_path):
    c = BoundedCacheHF("dummy/repo", str(tmp_path), max_bytes=5000)
    old = _fake_archive(c.root, "old.zip", 4, 1000)   # 4000 B, extracted first
    new = _fake_archive(c.root, "new.zip", 4, 1000)   # 4000 B, current epoch
    c.pin(["new.zip"])
    c._maybe_evict()
    # the pinned archive keeps every tile; the unpinned one is dropped wholesale
    assert not old.exists()
    assert sorted(p.name for p in (new / "opt").glob("*.tif")) == [
        "0.tif", "1.tif", "2.tif", "3.tif"
    ]
    assert c._dir_size() <= 5000


def test_stale_extracted_flag_triggers_reextract(tmp_path, monkeypatch):
    c = BoundedCacheHF("dummy/repo", str(tmp_path), max_bytes=10**9)
    d = c.root / "_extracted" / "a.zip"
    d.mkdir(parents=True)
    (d / ".extracted_ok").write_text("ok")  # sentinel present, tiles gone

    calls = {"n": 0}

    def fake_download(rel):
        calls["n"] += 1
        (d / "opt").mkdir(parents=True, exist_ok=True)
        (d / "opt" / "x.tif").write_bytes(b"1")
        return c.root / rel

    monkeypatch.setattr(c, "_raw_download", fake_download)
    monkeypatch.setattr(BoundedCacheHF, "_extract", staticmethod(lambda *a, **k: None))
    out = c.ensure_archive("a.zip")
    assert calls["n"] == 1 and (out / "opt" / "x.tif").exists()


@pytest.mark.hf
def test_hf_list_and_fetch_gamus(tmp_path):
    tok = os.environ.get("HF_TOKEN")
    c = BoundedCacheHF("earthflow/GAMUS", str(tmp_path), max_bytes=10**9, token=tok)
    stems = c.list_stems("images/val/", "_RGB.h5")
    assert len(stems) > 10
    p = c.get(f"images/val/{stems[0]}_RGB.h5")
    assert p.is_file() and p.stat().st_size > 0
