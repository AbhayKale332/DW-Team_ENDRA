"""Self-check for the argv/flag glue.  `python test_modal_app.py`

Needs no Modal account and no GPU — it only exercises the string plumbing,
which is where a silent mistake costs a GPU-hour.
"""
import json
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "V4_Kaggle"))
from config import Config

import modal_app as m


def last(a, k):
    return a[len(a) - 1 - a[::-1].index(k) + 1]


def test_every_flag_is_a_real_config_field():
    # config.py's parser ends in parse_known_args, so an unknown flag is
    # SILENTLY DROPPED and the run uses the default you thought you overrode.
    emitted = set(m.FLAGS) | {"smoke", "freeze_epochs", "resume"}
    assert not emitted - {f.name for f in fields(Config)}


def test_callers_value_always_wins():
    a = m.argv(epochs=26, extra="--batch_size 32 --epochs 99")
    assert last(a, "--batch_size") == "32"   # extra beats the profile
    assert last(a, "--epochs") == "99"       # extra beats a named arg


def test_gamus_stays_first():
    # loaders.py:31-43 selects best.pt on the first source with a val split.
    assert m.argv()[m.argv().index("--datasets") + 1].startswith("gamus,")


def test_empty_args_do_not_emit_junk():
    a = m.argv(batch_size=0, datasets="", resume="")
    assert "--resume" not in a
    assert a[a.index("--batch_size") + 1] == "24"


def test_smoke_cannot_land_on_a_real_run():
    a = m.argv(output_dir=f"{m.RESULTS}/smoke_v4m", extra="--smoke")
    assert a[a.index("--output_dir") + 1].endswith("smoke_v4m")
    assert a[-1] == "--smoke"


def test_finalize_runs_no_training_loop():
    a = m.argv(epochs="0", freeze_epochs="0")
    assert a[a.index("--epochs") + 1] == "0"
    assert a[a.index("--freeze_epochs") + 1] == "0"
    assert a[a.index("--output_dir") + 1] == f"{m.RESULTS}/v4m"


def test_max_minutes_tracks_epochs():
    # It SIZES the LR cosine (train.py:626-632); moving one without the other
    # is how v4-2 stopped annealing at 81%.
    assert round(m.cost(26)[0] * 60 * 0.9) < int(m.FLAGS["max_minutes"])


def test_store_name_is_the_store_not_the_split():
    # .parent is "train"; .parents[1] is "dfc23_g050".  That is the ba78dd8 bug.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "dfc23_g050" / "train"
        p.mkdir(parents=True)
        (p / "index.json").write_text(json.dumps(
            {"n": 1506, "tile_px": 512, "gsd_m": 0.5, "has_seg": True}))
        assert m.stores(d) == 1
        idx = p / "index.json"
        assert idx.parents[1].name == "dfc23_g050"


def _import_with(env: dict) -> dict:
    """FLAGS / _PROFILE_ENV as a fresh import sees them under `env`."""
    import os
    import subprocess

    code = ("import json, modal_app as m; "
            "print(json.dumps({'flags': m.FLAGS, 'env': m._PROFILE_ENV}))")
    e = {k: v for k, v in os.environ.items() if k not in ("DW_CODE", "DW_PROFILE", "DW_AUDIT")}
    r = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent,
                       env={**e, **env}, capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_v5_profile_reaches_the_container():
    # Local import with DW_CODE=v5: the v5 profile, and the same dict baked into
    # the image env -- the container re-imports this module WITHOUT DW_CODE.
    loc = _import_with({"DW_CODE": "v5"})
    f = loc["flags"]
    assert f["output_dir"] == "/results/v5" and f["detail_branch"] == "true"
    assert f["data_root"] == m.SCRATCH and f["batch_size"] == "16"
    assert json.loads(loc["env"]["DW_PROFILE"]) == f
    # ...and inside the container the baked profile wins over the v4 default
    rem = _import_with({"DW_PROFILE": loc["env"]["DW_PROFILE"]})
    assert rem["flags"] == f and rem["env"] == {}
    # v4 stays byte-identical: no env, no profile
    assert _import_with({})["env"] == {} and m.FLAGS["output_dir"].endswith("/v4m")


def test_v5_audit_overrides_only_step0_keys():
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "audit.json"
        a.write_text(json.dumps({"flags": {"coarse_pool": "8", "epochs": "99"}}))
        f = _import_with({"DW_CODE": "v5", "DW_AUDIT": str(a)})["flags"]
    assert f["coarse_pool"] == "8" and f["epochs"] == "26"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall passed")
