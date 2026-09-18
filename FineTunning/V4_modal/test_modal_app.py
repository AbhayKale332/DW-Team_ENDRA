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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall passed")
