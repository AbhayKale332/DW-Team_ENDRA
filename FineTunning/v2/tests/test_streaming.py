import os

import pytest

from data.streaming import BoundedCacheHF


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


@pytest.mark.hf
def test_hf_list_and_fetch_gamus(tmp_path):
    tok = os.environ.get("HF_TOKEN")
    c = BoundedCacheHF("earthflow/GAMUS", str(tmp_path), max_bytes=10**9, token=tok)
    stems = c.list_stems("images/val/", "_RGB.h5")
    assert len(stems) > 10
    p = c.get(f"images/val/{stems[0]}_RGB.h5")
    assert p.is_file() and p.stat().st_size > 0
