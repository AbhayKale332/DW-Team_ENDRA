"""GAMUS ingest: stage once, then pack offline.

The old path called `hf_hub_download` once per file -- ~13 200 requests for a
full prepare -- earned a CAS 429 partway through, and swallowed each failure as
a printed `skip: ...`.  Nothing summed those lines, so v3 wrote a `gamus/val`
store holding 129 of 400 tiles and reported success; the run that trained on it
looked fine until the val number was traced back.

So two things are load-bearing here and neither is visible from the store:

First, `classes/` has to actually arrive.  The Kaggle PNG mirror that replaced
this path has no class rasters, which made `seg_ce_loss` exactly 0.0 on every
step while `w_seg` still read 0.2, trained Head C on synthetic SynRS3D alone,
and left `geo/calibrate.py` without the ground mask its DTM fit wants.  `has_seg`
in `index.json` is the one bit that records whether any of that works.

Second, a short pack has to be an error, not a log line.
"""
import json
import sys
import types

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

TILE = 1024          # prepare_gamus hardcodes it; the fixture must match
_SUBS = (("images", "RGB", np.uint8, 7), ("heights", "AGL", np.float32, 3.5),
         ("classes", "CLS", np.int32, 2))


@pytest.fixture
def staged(monkeypatch):
    """Stub the Hub: `snapshot_download` is a no-op, listing falls back to disk."""
    box = {}
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *a, **k: box.get("dir", "")

    class _Api:                      # forces _gamus_file_list onto from_disk()
        def list_repo_files(self, *a, **k):
            raise RuntimeError("offline")

    hub.HfApi = _Api
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return box


def _write(tmp, split, stems, subs=_SUBS):
    for sub, tag, dt, v in subs:
        (tmp / sub / split).mkdir(parents=True, exist_ok=True)
        for s in stems:
            shape = (TILE, TILE, 3) if sub == "images" else (TILE, TILE)
            with h5py.File(tmp / sub / split / f"{s}_{tag}.h5", "w") as f:
                f["image"] = np.full(shape, v, dt)


def test_staged_pack_keeps_classes(tmp_path, staged):
    from prepare_data import prepare_gamus

    stems = [f"DC_{i:02d}_10" for i in range(3)]
    tmp = tmp_path / "_dl" / "gamus_val"
    _write(tmp, "val", stems)
    staged["dir"] = str(tmp)

    prepare_gamus(tmp_path, "val", 0, None, False, "earthflow/GAMUS", workers=2)

    idx = json.loads((tmp_path / "gamus/val/index.json").read_text())
    assert idx["n"] == 3 and idx["tile_px"] == TILE
    assert idx["has_seg"] is True, "classes/ was staged; the store must record it"
    assert (np.load(tmp_path / "gamus/val/shard_000_cls.npy") == 2).all()
    assert np.allclose(np.load(tmp_path / "gamus/val/shard_000_hgt.npy"), 3.5)
    assert not tmp.exists(), "staging is consumed, not left on a billed disk"


def test_no_classes_still_packs(tmp_path, staged):
    from prepare_data import prepare_gamus

    stems = [f"DC_{i:02d}_10" for i in range(3)]
    tmp = tmp_path / "_dl" / "gamus_val"
    _write(tmp, "val", stems, subs=_SUBS[:2])        # images + heights only
    staged["dir"] = str(tmp)

    prepare_gamus(tmp_path, "val", 0, None, False, "earthflow/GAMUS", workers=2)
    idx = json.loads((tmp_path / "gamus/val/index.json").read_text())
    assert idx["n"] == 3 and idx["has_seg"] is False


def test_short_pack_raises(tmp_path, staged):
    """The v3 failure: most tiles unusable, store written anyway."""
    from prepare_data import prepare_gamus

    stems = [f"DC_{i:02d}_10" for i in range(5)]
    tmp = tmp_path / "_dl" / "gamus_val"
    _write(tmp, "val", stems, subs=_SUBS[:1])                 # every RGB
    _write(tmp, "val", stems[:2], subs=_SUBS[1:2])            # heights for 2 of 5
    staged["dir"] = str(tmp)

    with pytest.raises(RuntimeError, match="of 5 staged tiles"):
        prepare_gamus(tmp_path, "val", 0, None, False, "earthflow/GAMUS", workers=2)


def test_stage_resumes_after_rate_limit(monkeypatch, staged, tmp_path):
    """A 429 here is about the shared egress IP, not the request.

    HF's Xet backend mints a per-file `/xet-read-token/<hash>` before each
    transfer, and on a Modal container that endpoint limits the whole IP while
    the token is perfectly valid -- the 429 body even talks about anonymous
    callers, so it reads as an auth failure.  Waiting is the only useful
    response, and `snapshot_download` resumes into `local_dir`, so a retry must
    not restart the ~51 GiB from zero.
    """
    import prepare_data

    calls, slept = [], []
    def flaky(*a, **k):
        calls.append(k.get("max_workers"))
        if len(calls) < 3:
            raise RuntimeError("429 Client Error: Too Many Requests")
        return "ok"

    hub = sys.modules["huggingface_hub"]
    monkeypatch.setattr(hub, "snapshot_download", flaky, raising=False)
    monkeypatch.setattr(prepare_data.time, "sleep", lambda s: slept.append(s))

    prepare_data._stage("r", "tok", tmp_path, ["images/val/*"], 4)
    assert len(calls) == 3, "must retry until it succeeds"
    assert slept == [60, 120], f"exponential backoff, got {slept}"


def test_stage_gives_up_and_raises(monkeypatch, staged, tmp_path):
    import prepare_data

    hub = sys.modules["huggingface_hub"]
    monkeypatch.setattr(hub, "snapshot_download",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429")),
                        raising=False)
    monkeypatch.setattr(prepare_data.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="429"):
        prepare_data._stage("r", "tok", tmp_path, ["images/val/*"], 4, tries=3)
