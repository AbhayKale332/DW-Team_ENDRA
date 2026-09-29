"""The final Lightning run's glue: flags, Kaggle fetch helpers, sweep pick rule."""

import json
import sys
from pathlib import Path

import pytest

V5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V5 / "lightning"))


def _root(tmp_path, stores, with_test=()):
    for s in stores:
        (tmp_path / s / "train").mkdir(parents=True)
        (tmp_path / s / "train" / "index.json").write_text("{}")
    for s in with_test:
        (tmp_path / s / "test").mkdir(parents=True, exist_ok=True)
        (tmp_path / s / "test" / "index.json").write_text("{}")
    return tmp_path


BASE = ["mvs3dm", "gamus", "us3d", "synrs3d_g05", "synrs3d_g1"]


def test_flags_without_neon_parse_and_drop_dfc(tmp_path):
    from final_flags import build, check

    f = build(_root(tmp_path, BASE), output_dir=str(tmp_path / "o"))
    check(f)
    assert f["datasets"] == ",".join(BASE)
    assert "dfc23" not in f["datasets"] and "india" not in f["datasets"]
    assert f["coarse_label_sources"] == ""
    assert f["select_on"] == "mvs3dm"
    assert "neon" not in f["test_sources"]
    assert "landscape_no_urban_sources" not in f


def test_flags_with_neon(tmp_path):
    from final_flags import build, check

    f = build(_root(tmp_path, BASE + ["neon"], with_test=["neon"]))
    check(f)
    assert f["datasets"].split(",")[:2] == ["mvs3dm", "neon"]   # mvs3dm stays primary val
    assert f["coarse_label_sources"] == "neon"
    assert f["landscape_no_urban_sources"] == "neon"
    assert f["select_on"] == "neon,mvs3dm"
    assert f["test_sources"].startswith("neon:test,")
    assert f["landscape_gallery_source"] == "neon"


def test_missing_store_stops(tmp_path):
    from final_flags import build

    with pytest.raises(SystemExit):
        build(_root(tmp_path, BASE[:2]))


def test_sizing_applies_and_scales_lr(tmp_path):
    from final_flags import OVERRIDES, build, check

    sz = {"batch_size": 48, "grad_accum": 1, "compile_model": True, "num_workers": 20,
          "prefetch_factor": 4, "eval_batch_mult": 3}
    f = build(_root(tmp_path, BASE), sz)
    check(f)
    assert (f["batch_size"], f["compile_model"], f["eval_batch_mult"]) == ("48", "true", "3")
    s = (48 / 32) ** 0.5
    assert float(f["learning_rate"]) == pytest.approx(float(OVERRIDES["learning_rate"]) * s, rel=1e-2)
    # the reference effective batch leaves the LR alone
    f2 = build(tmp_path, {**sz, "batch_size": 16, "grad_accum": 2})
    assert f2["learning_rate"] == OVERRIDES["learning_rate"]


def test_fetch_spec_and_split_filter():
    sys.path.insert(0, str(V5 / "tools"))
    from kaggle_fetch import parse_spec, wanted

    assert parse_spec("o/d:train,val") == ("o/d", ("train", "val"))
    assert parse_spec("o/d") == ("o/d", ("train", "val", "test"))
    with pytest.raises(ValueError):
        parse_spec("o/d:trian")
    assert wanted("train/train/shard_000_rgb.npy", ("train",))
    assert not wanted("test/test/shard_000_rgb.npy", ("train", "val"))
    assert wanted("val/index.json", ("val",))
    assert wanted("README.md", ("train",))


def test_sweep_pick_rule_and_bench_parse():
    sys.path.insert(0, str(V5 / "tools"))
    from h100_sweep import accum_for, parse_bench, pick

    rows = [
        {"batch_size": 16, "img_s": 45.0, "status": "ok"},
        {"batch_size": 24, "img_s": 47.5, "status": "ok"},
        {"batch_size": 32, "img_s": 48.0, "status": "ok"},
        {"batch_size": 40, "img_s": 49.0, "status": "over"},     # past the VRAM ceiling
        {"batch_size": 48, "status": "oom"},
    ]
    # 47.5 and 48.0 are within 2 %: the larger batch wins; 40 never counts
    assert pick(rows)["batch_size"] == 32
    assert pick([{"batch_size": 16, "status": "oom"}]) is None
    assert [accum_for(b) for b in (16, 24, 32, 48)] == [2, 1, 1, 1]
    log = "x\n[bench] window start 100.5\n[bench] " + json.dumps({"img_s": 3.0}) + "\n"
    rep, start = parse_bench(log)
    assert rep == {"img_s": 3.0} and start == 100.5


def test_shell_script_parses():
    import subprocess

    r = subprocess.run(["bash", "-n", str(V5 / "lightning" / "final_h100.sh")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_neon_is_linked_and_has_val_split():
    from dwdata.loaders import val_split_of

    assert val_split_of("neon") == "val"
    assert "|neon)" in (V5 / "run_kaggle.sh").read_text()
