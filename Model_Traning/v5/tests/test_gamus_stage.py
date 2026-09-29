"""GAMUS ingest: stage once, then pack offline.

The old path called `hf_hub_download` once per file -- ~13 200 requests for a
full prepare -- earned a CAS 429 partway through, and swallowed each failure as
a printed `skip: ...`.  Nothing summed those lines, so v3 wrote a `gamus/val`
store holding 129 of 400 tiles and reported success; the run that trained on it
looked fine until the val number was traced back.

So two things are load-bearing here and neither is visible from the store:

First, `classes/` has to actually arrive.  The Kaggle PNG mirror that once replaced
this path (since removed) had no class rasters, which made `seg_ce_loss` exactly 0.0 on every
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
    """Stub the Hub: the tree listing fails, so everything comes off disk."""
    box = {}
    hub = types.ModuleType("huggingface_hub")

    class _Api:                      # forces _repo_tree onto from_disk()
        def __init__(self, *a, **k):
            pass

        def list_repo_tree(self, *a, **k):
            raise RuntimeError("offline")

        def repo_info(self, *a, **k):
            raise RuntimeError("offline")

    hub.HfApi = _Api
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return box


def _write(tmp, split, stems, subs=_SUBS, img_tag="RGB"):
    for sub, tag, dt, v in subs:
        (tmp / sub / split).mkdir(parents=True, exist_ok=True)
        for s in stems:
            shape = (TILE, TILE, 3) if sub == "images" else (TILE, TILE)
            name = f"{s}_{img_tag if sub == 'images' else tag}.h5"
            with h5py.File(tmp / sub / split / name, "w") as f:
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


def _fake_requests(monkeypatch, handler):
    """A `requests` module whose GET is `handler(url, headers)`."""
    class _Resp:
        def __init__(self, status, body=b"", retry_after=None):
            self.status_code = status
            self._body = body
            self.headers = {"Retry-After": str(retry_after)} if retry_after else {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def iter_content(self, n):
            yield self._body

    class _Session:
        def get(self, url, headers=None, **k):
            return _Resp(*handler(url, headers or {}))

        def mount(self, *a, **k):
            pass

        def close(self):
            pass

    mod = types.ModuleType("requests")
    mod.Session = _Session
    mod.adapters = types.SimpleNamespace(HTTPAdapter=lambda **k: None)
    utils = types.ModuleType("huggingface_hub.utils")
    utils.build_hf_headers = lambda token=None: {"Authorization": f"Bearer {token}"}
    monkeypatch.setitem(sys.modules, "requests", mod)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return _Resp


def _fake_clock(monkeypatch, prepare_data):
    """A clock that only moves when the code sleeps, so backoff is instant."""
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    monkeypatch.setattr(prepare_data.time, "sleep", sleep)
    monkeypatch.setattr(prepare_data.time, "monotonic", lambda: now[0])
    return slept


def test_stage_skips_what_is_already_on_disk(monkeypatch, staged, tmp_path):
    """The bug that made every retry pointless.

    `snapshot_download` HEADs all ~15 000 files before transferring a byte, and
    does it again on each resume -- so a run that died at file 300 re-spent the
    whole request budget, met the 429 again, and never got further.  A file that
    is already the right size must cost zero requests.
    """
    import prepare_data

    got = []
    _fake_requests(monkeypatch, lambda u, h: got.append(u) or (200, b"xy"))
    (tmp_path / "images/val").mkdir(parents=True)
    (tmp_path / "images/val/a_RGB.h5").write_bytes(b"xy")      # complete
    (tmp_path / "images/val/b_RGB.h5").write_bytes(b"x")       # truncated

    rels = ["images/val/a_RGB.h5", "images/val/b_RGB.h5", "images/val/c_RGB.h5"]
    prepare_data._stage("r", "tok", tmp_path, rels, dict.fromkeys(rels, 2),
                        "deadbeef", workers=1)

    assert [u.rsplit("/", 1)[1] for u in got] == ["b_RGB.h5", "c_RGB.h5"]
    assert all(u.startswith("https://huggingface.co/datasets/r/resolve/deadbeef/")
               for u in got), "the revision has to be pinned, not 'main'"
    assert (tmp_path / "images/val/c_RGB.h5").read_bytes() == b"xy"


def test_stage_waits_out_a_429_then_resumes(monkeypatch, staged, tmp_path):
    """A 429 here is about the shared egress IP, not the request, so the only
    useful response is to park every worker on one deadline and carry on."""
    import prepare_data

    calls = []

    def handler(url, headers):
        calls.append(url)
        return (429, b"", 7) if len(calls) < 3 else (200, b"xy")

    _fake_requests(monkeypatch, handler)
    slept = _fake_clock(monkeypatch, prepare_data)

    rels = ["images/val/a_RGB.h5"]
    prepare_data._stage("r", "tok", tmp_path, rels, {rels[0]: 2}, "sha", workers=1)

    assert len(calls) == 3, "must retry until it succeeds"
    assert slept and all(s <= 7 for s in slept), f"honours Retry-After, got {slept}"
    assert (tmp_path / rels[0]).read_bytes() == b"xy"


def test_stage_gives_up_and_raises(monkeypatch, staged, tmp_path):
    """A file that never lands is an error.  Packing a short store is worse."""
    import prepare_data

    _fake_requests(monkeypatch, lambda u, h: (429, b"", 1))
    _fake_clock(monkeypatch, prepare_data)

    rels = ["images/val/a_RGB.h5"]
    with pytest.raises(RuntimeError, match="gave up"):
        prepare_data._stage("r", "tok", tmp_path, rels, {rels[0]: 2}, "sha",
                            workers=1, tries=3)
    assert not (tmp_path / rels[0]).exists()


def test_test_split_packs_to_its_own_store(tmp_path, staged):
    """GAMUS's third split, packed but inert.

    It exists because `gamus/val` is doing double duty -- `best.pt` is selected
    on it (train.py:901) and `final_plain`/`final_tta` are then reported on the
    same tiles (train.py:957), so the headline number is measured on the set the
    checkpoint was picked on.  2861 held-out tiles with height and class ground
    truth fix that, but only while nothing trains or selects on them: loaders.py
    opens `root/<name>/train` for training and `val_split_of("gamus") == "val"`
    for validation, and neither globs the store directory.
    """
    from prepare_data import prepare_gamus

    stems = [f"JAX_{i:02d}_10" for i in range(3)]
    tmp = tmp_path / "_dl" / "gamus_test"
    _write(tmp, "test", stems)
    staged["dir"] = str(tmp)

    prepare_gamus(tmp_path, "test", 0, None, False, "earthflow/GAMUS", workers=2)

    idx = json.loads((tmp_path / "gamus/test/index.json").read_text())
    assert idx["n"] == 3 and idx["has_seg"] is True
    assert not (tmp_path / "gamus/val").exists(), "must not land in the val store"
    assert not (tmp_path / "gamus/train").exists(), "must never reach training"


def test_test_split_is_opt_in(monkeypatch, tmp_path):
    """Default must be skip: adding this flag cannot make an existing prepare
    command start pulling 23 GiB more."""
    import prepare_data

    seen = []
    monkeypatch.setattr(prepare_data, "prepare_gamus",
                        lambda root, split, n, *a, **k: seen.append((split, n)))

    argv = ["prepare_data.py", "--data_root", str(tmp_path), "--datasets", "gamus"]
    monkeypatch.setattr(sys, "argv", argv)
    prepare_data.main()
    assert [sp for sp, _ in seen] == ["val", "train"], seen

    seen.clear()
    monkeypatch.setattr(sys, "argv", argv + ["--gamus_test", "0"])
    prepare_data.main()
    assert seen[-1] == ("test", 0), seen


def test_img_and_rgb_tiles_both_pack(tmp_path, staged):
    """GAMUS names its images two ways and only one of them was matched.

    `suf = "_RGB.h5"` dropped every `_IMG.h5` tile: 1167 of 5004 in train and
    1000 of 2861 in test, all NYC.  It hid for so long because val has no `_IMG`
    files at all, so the split everyone checks read a clean 859/859 -- and the
    `< 0.99` yield guard could not see it either, because the dropped tiles
    never entered `stems` and the store looked complete against a short target.
    """
    from prepare_data import prepare_gamus

    tmp = tmp_path / "_dl" / "gamus_train"
    _write(tmp, "train", ["JAX_01_10", "JAX_02_10"], img_tag="RGB")
    _write(tmp, "train", ["NYC_22835", "NYC_22836", "NYC_22837"], img_tag="IMG")
    staged["dir"] = str(tmp)

    prepare_gamus(tmp_path, "train", 0, None, False, "earthflow/GAMUS", workers=2)

    idx = json.loads((tmp_path / "gamus/train/index.json").read_text())
    assert idx["n"] == 5, "both naming conventions must reach the store"
    assert idx["has_seg"] is True


def test_unknown_image_tag_is_fatal(tmp_path, staged):
    """A third convention must stop the run, not quietly shrink the dataset."""
    from prepare_data import prepare_gamus

    tmp = tmp_path / "_dl" / "gamus_train"
    _write(tmp, "train", ["JAX_01_10"], img_tag="RGB")
    _write(tmp, "train", ["JAX_09_10"], img_tag="PAN")     # invented
    staged["dir"] = str(tmp)

    with pytest.raises(RuntimeError, match="match none of"):
        prepare_gamus(tmp_path, "train", 0, None, False, "earthflow/GAMUS", workers=2)
