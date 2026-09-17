"""One-time data materialisation:  Hugging Face  ->  local memmap shards.

Run this once per Lightning studio (the studio disk is persistent, so a later
training run starts instantly).  Training never touches the network afterwards.

    python prepare_data.py --datasets gamus --gamus_train 4000 --gamus_val 400
    python prepare_data.py --datasets synrs3d --synrs3d_archives 3
    python prepare_data.py --datasets geonrw            # 32 GB tar, slowest

    # Indian imagery (see dwdata/india.py for why this is a local-directory
    # ingest rather than a hub download):
    python prepare_data.py --datasets india_labeled   --india_dir ~/dfc23_newdelhi
    python prepare_data.py --datasets india_unlabeled --india_dir ~/bhuvan_tiles
    python prepare_data.py --datasets india_unlabeled --india_tile_url 'https://…/{z}/{x}/{y}.png'

Disk cost, packed (uint8 RGB + fp16 height + uint8 class + bool valid):
    GAMUS   1024 px tile  ~ 6.3 MB/tile   -> 4000 tiles ~ 25 GB
    SynRS3D  512 px tile  ~ 1.6 MB/tile   -> 1 archive  ~ 6-10 GB
    GeoNRW  1000 px tile  ~ 6.0 MB/tile   -> ~7300 tiles ~ 44 GB (use --geonrw_max)

Everything is resumable: a split whose `index.json` already exists is skipped
unless `--force` is passed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from dwdata.packed import NO_LABEL, ShardWriter, store_exists  # noqa: E402

GAMUS_GSD_M = 0.33
GEONRW_GSD_M = 1.0

# SynRS3D GSD families.  The `gNNN` token in a folder name is a *range*, not a
# value, and the ranges below are the ones the dataset card publishes
# (https://huggingface.co/datasets/JTRNEO/SynRS3D):
#
#     g005  0.05 - 0.30 m      g05  0.30 - 0.60 m      g1  0.60 - 1.00 m
#
# The previous table here read {"g005": 0.3, "g05": 0.7}.  Both entries were
# wrong: 0.3 is the *ceiling* of g005 rather than anything representative, and
# 0.7 is outside g05 altogether.  Nominals are now the midpoint of the published
# range, which is the best a single scalar can do — SynRS3D publishes GSD at the
# folder level only, there is no per-image GSD anywhere in the archives, and
# `index.json` carries one `gsd_m` per store (dwdata/packed.py).  So a g05 tile
# labelled 0.45 m is still up to ±33 % off its own true GSD.  That residual is
# inherent to the dataset and is the reason the families are packed into
# *separate stores* below rather than pooled: pooling g005 (0.05-0.30) with g05
# (0.30-0.60) under one nominal is a 3-10x scale lie, and `dataset.py` turns
# `store.gsd_m` straight into the crop's effective GSD, which is what ties
# apparent object size to metric height.
_SYN_GSD = {"g005": 0.175, "g05": 0.45, "g1": 0.8}

# g1 first, and it is the only family that serves the reason this source exists.
# GAMUS's 1024 px / 0.33 m tiles cap at 1024*0.33/512 = 0.66 m
# (dwdata/augment.py:achievable_gsd_range), so anything coarser has to come from
# somewhere else — and the list here used to start with g05, which tops out at
# 0.60 m, while carrying a comment claiming it supplied the ">0.66 m scale
# range".  No g1 archive was in the list at all.  These three are the 5,537
# images that actually reach past GAMUS's ceiling.
# The whole g1 family is only 5,537 images across three archives, so the first
# four entries are "all of g1, plus the g05 archive v4-2 actually trained on" —
# `--synrs3d_archives 4` gets both stores and keeps continuity with that run.
SYNRS3D_ARCHIVES = [
    "SynRS3D/terrain_g1_low_v1.zip",    # 4,285
    "SynRS3D/terrain_g1_high_v1.zip",   #   904
    "SynRS3D/terrain_g1_mid_v1.zip",    #   348
    "SynRS3D/grid_g05_mid_v1.zip", "SynRS3D/terrain_g05_mid_v1.zip",
    "SynRS3D/grid_g005_mid_v1.zip", "SynRS3D/terrain_g005_mid_v1.zip",
    "SynRS3D/grid_g05_high_v1.zip", "SynRS3D/grid_g05_low_v1.zip",
    "SynRS3D/grid_g005_high_v1.zip", "SynRS3D/grid_g005_low_v1.zip",
    "SynRS3D/terrain_g05_low_v1.zip", "SynRS3D/terrain_g005_low_v1.zip",
]

# Store name per family, e.g. "synrs3d_g05".  These are the names that go in
# --datasets / --sampler_weights / _VAL_SPLIT / run_kaggle.sh's link whitelist.
SYNRS3D_STORES = tuple(f"synrs3d_{k}" for k in _SYN_GSD)


def _syn_family(rel: str) -> str:
    """Which GSD family an archive path belongs to.  Longest token first, so
    `_g005_` is never matched by the `g05` key."""
    for k in sorted(_SYN_GSD, key=len, reverse=True):
        if f"_{k}_" in rel:
            return k
    return "g05"

# Class-id remaps into a shared 0..6 space.  These are best-effort (see README
# §1.6 — GAMUS's own id order is not what v1/v2 assumed), which is exactly why
# the semantic head is only an 0.2-weight auxiliary and why `--dump_class_stats`
# prints the measured histogram instead of trusting a table.
SYN_TO_SHARED = {0: NO_LABEL, 1: 0, 2: 1, 3: 0, 4: 4, 5: 1, 6: 3, 7: 1, 8: 2}
GEONRW_TO_SHARED = {0: NO_LABEL, 1: 1, 2: 3, 3: 1, 4: 2, 5: 1, 6: 4, 7: 4, 8: 0, 9: 4, 10: 2}

GEONRW_TEST_CITIES = ("duesseldorf", "herne", "neuss")


def _remap(a: np.ndarray, table: dict) -> np.ndarray:
    """Map source ids through `table`; anything unlisted becomes NO_LABEL.

    The clip used to be `np.clip(a, 0, len(lut) - 1)`, which silently folded
    *every* out-of-range id onto the highest class instead of discarding it — so
    a void/ignore id (GAMUS-style 255) would have been packed as a real class and
    trained on as ground truth.  Out-of-range now routes to NO_LABEL, which
    `dataset.py` turns into SEG_IGNORE_INDEX and the CE loss skips.
    """
    lut = np.full(int(max(table) + 1), NO_LABEL, np.uint8)
    for k, v in table.items():
        lut[k] = v
    a = np.asarray(a).astype(np.int32)
    out = np.full(a.shape, NO_LABEL, np.uint8)
    ok = (a >= 0) & (a < len(lut))
    out[ok] = lut[a[ok]]
    return out


def _token(explicit: str = "") -> str | None:
    for v in (explicit, os.environ.get("HF_TOKEN"),
              os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if v:
            return v
    return None


# ---------------------------------------------------------------------
# GAMUS
# ---------------------------------------------------------------------
def _gamus_file_list(repo: str, token, staged: Path, pre: str, suf: str) -> list[str]:
    """Repo file listing, from the Hub or — failing that — from the staged copy.

    `list_repo_files` is an API call, so with `HF_HUB_OFFLINE=1` it raises and
    takes the whole offline pack down with it.  That matters because offline is
    the *point* of the two-stage runbook: staging every tile with one bulk
    `snapshot_download` and then packing without touching the network is what
    avoids ~13 000 individual requests, the 429s they earn partway through, and
    the "skip: No such file" lines that let v3 write a gamus/val store holding
    129 of 400 tiles and still call it done.

    So when the Hub is unreachable — offline, rate-limited, or simply down — fall
    back to listing what the staging directory actually holds.  A pack built
    from that is a pack of exactly the tiles on disk, which is the honest answer
    to "what do we have", and `prepare_data.sh` gates the READY stamp on the
    resulting yield.
    """
    from huggingface_hub import HfApi

    def from_disk() -> list[str]:
        rels = (str(q.relative_to(staged)) for q in staged.rglob(f"*{suf}"))
        return sorted(r for r in rels if r.startswith(pre))

    try:
        return list(HfApi(token=token).list_repo_files(repo, repo_type="dataset"))
    except Exception as e:  # noqa: BLE001
        local = from_disk()
        if not local:
            raise RuntimeError(
                f"cannot list {repo} ({type(e).__name__}: {e}) and nothing is staged "
                f"under {staged}. Run the fetch stage first, or unset "
                f"HF_HUB_OFFLINE to list the repo online."
            ) from e
        print(f"[gamus] hub listing unavailable ({type(e).__name__}) — using the "
              f"{len(local)} tiles staged under {staged}")
        return local


def prepare_gamus(root: Path, split: str, n_tiles: int, token, force: bool,
                  repo: str, workers: int = 12) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    out = root / "gamus" / split
    if store_exists(out) and not force:
        print(f"[gamus/{split}] already prepared -> {out}")
        return

    import h5py

    tmp = root / "_dl" / f"gamus_{split}"
    tmp.mkdir(parents=True, exist_ok=True)

    pre, suf = f"images/{split}/", "_RGB.h5"
    files = _gamus_file_list(repo, token, tmp, pre, suf)
    stems = sorted(f[len(pre):-len(suf)] for f in files
                   if f.startswith(pre) and f.endswith(suf))
    if not stems:
        raise RuntimeError(f"no GAMUS tiles under {pre} in {repo}")
    if n_tiles and n_tiles < len(stems):
        rng = np.random.default_rng(0)
        stems = sorted(np.asarray(stems)[rng.permutation(len(stems))[:n_tiles]].tolist())
    print(f"[gamus/{split}] packing {len(stems)} tiles -> {out}")

    def fetch(stem: str):
        paths = {}
        for key, (sub, tag) in {"rgb": ("images", "RGB"), "hgt": ("heights", "AGL"),
                                "cls": ("classes", "CLS")}.items():
            rel = f"{sub}/{split}/{stem}_{tag}.h5"
            try:
                paths[key] = hf_hub_download(repo, rel, repo_type="dataset",
                                             local_dir=str(tmp), token=token)
            except Exception:  # noqa: BLE001 — classes/ is optional
                if key != "cls":
                    raise
                paths[key] = None
        return stem, paths

    def read_h5(p):
        with h5py.File(p, "r") as f:
            k = "image" if "image" in f else list(f.keys())[0]
            return np.asarray(f[k][()])

    # `classes/` is fetched best-effort (the except below swallows a 404 so a
    # repo without semantics still packs).  That silence is exactly how every v4
    # run so far trained with a 0.2-weight semantic head that contributed
    # nothing: the PNG mirror has no classes, and this path would not have said
    # so either.  Count the yield and stamp it into the index.
    writer = ShardWriter(out, tile_px=1024, gsd_m=GAMUS_GSD_M, shard_tiles=128)
    done = 0
    n_cls = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for stem, paths in ex.map(_safe(fetch), stems):
            if paths is None:
                continue
            try:
                rgb = read_h5(paths["rgb"]).astype(np.uint8)
                agl = read_h5(paths["hgt"]).astype(np.float32)
                cls = _remap(read_h5(paths["cls"]).astype(np.int32), {i: i for i in range(7)}) \
                    if paths["cls"] else None
                valid = np.isfinite(agl) & (agl > -2.0) & (agl < 500.0)
                agl = np.where(valid, np.clip(agl, 0.0, None), 0.0)
                writer.add(stem, rgb, agl, cls, valid)
                done += 1
                n_cls += cls is not None
            except Exception as e:  # noqa: BLE001
                print(f"[gamus/{split}] skip {stem}: {e}")
            finally:
                for p in paths.values():
                    if p:
                        Path(p).unlink(missing_ok=True)
            if done % 200 == 0:
                print(f"  {done}/{len(stems)}", flush=True)
    writer.has_seg = n_cls > 0
    idx = writer.finalise()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[gamus/{split}] packed {idx['n']} tiles into {len(idx['shards'])} shards")
    if n_cls == done and done:
        print(f"[gamus/{split}] semantic labels: {n_cls}/{done} tiles OK")
    elif n_cls:
        print(f"[gamus/{split}] !! semantic labels on only {n_cls}/{done} tiles — "
              f"the rest train with seg ignored")
    else:
        print(f"[gamus/{split}] !! NO semantic labels fetched (0/{done}). "
              f"seg_ce_loss will be exactly 0.0 and w_seg buys nothing; "
              f"flatness_loss loses its ground/road/water restriction. "
              f"Check the token's access to {repo}:classes/{split}/, or pass --w_seg 0.")


def _safe(fn):
    def inner(x):
        try:
            return fn(x)
        except Exception as e:  # noqa: BLE001
            print(f"  fetch failed for {x}: {e}")
            return x, None
    return inner


# ---------------------------------------------------------------------
# SynRS3D
# ---------------------------------------------------------------------
def prepare_synrs3d(root: Path, n_archives: int, token, force: bool, repo: str,
                    val_frac: float = 0.0) -> None:
    """SynRS3D archives -> one packed store *per GSD family*.

    Not one store for everything.  `index.json` carries a single `gsd_m`
    (dwdata/packed.py), `dataset.py` reads it straight into the crop's effective
    GSD, and that is what ties apparent object size to metric height — so
    pooling g005 (0.05-0.30 m) and g05 (0.30-0.60 m) under one nominal tells the
    model a 3-10x scale lie about most of its own training set.  The previous
    version did exactly that: one `ShardWriter(..., gsd_m=0.5)` built outside the
    archive loop, while the per-archive GSD was computed and then only printed.
    """
    from huggingface_hub import hf_hub_download

    import zipfile

    import tifffile

    rels = SYNRS3D_ARCHIVES[:n_archives]
    fams = sorted({_syn_family(r) for r in rels})
    outs = {f: root / f"synrs3d_{f}" / "train" for f in fams}
    todo = [f for f in fams if force or not store_exists(outs[f])]
    for f in fams:
        if f not in todo:
            print(f"[synrs3d] already prepared -> {outs[f]}")
    if not todo:
        return

    writers = {f: ShardWriter(outs[f], tile_px=512, gsd_m=_SYN_GSD[f],
                              shard_tiles=512) for f in todo}
    work = root / "_dl" / "synrs3d"
    total = dict.fromkeys(todo, 0)
    for rel in rels:
        fam = _syn_family(rel)
        if fam not in writers:
            continue
        ex_dir = work / rel.replace("/", "__")
        try:
            print(f"[synrs3d] downloading {rel} …", flush=True)
            arc = hf_hub_download(repo, rel, repo_type="dataset",
                                  local_dir=str(work), token=token)
            ex_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(arc) as z:
                z.extractall(ex_dir)
            Path(arc).unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"[synrs3d] skip {rel}: {e}")
            shutil.rmtree(ex_dir, ignore_errors=True)
            continue

        opts = sorted(ex_dir.rglob("opt/*.tif"))
        print(f"[synrs3d] {rel}: {len(opts)} tiles -> {fam} "
              f"(nominal {_SYN_GSD[fam]} m)")
        w = writers[fam]
        for opt in opts:
            nd = _sibling(opt, ("gt_nDSM", "gt_ndsm", "nDSM", "ndsm"))
            if nd is None:
                continue
            sg = _sibling(opt, ("gt_ss_mask", "gt_ssmask", "ss_mask"))
            try:
                rgb = np.asarray(tifffile.imread(opt))[..., :3].astype(np.uint8)
                h = np.asarray(tifffile.imread(nd)).astype(np.float32)
                if rgb.shape[:2] != (512, 512) or h.shape != (512, 512):
                    continue
                cls = _remap(np.asarray(tifffile.imread(sg)).astype(np.int32),
                             SYN_TO_SHARED) if sg else None
                valid = np.isfinite(h) & (h > -2.0) & (h < 500.0)
                # Stems repeat across archives (P_0001.tif is in all of them),
                # and `stems` is what PackedStore keys on, so namespace them.
                w.add(f"{ex_dir.name}__{opt.stem}", rgb,
                      np.where(valid, np.clip(h, 0, None), 0.0), cls, valid)
                total[fam] += 1
            except Exception:  # noqa: BLE001
                continue
        shutil.rmtree(ex_dir, ignore_errors=True)
        print(f"[synrs3d] running total {dict(total)}", flush=True)

    for f in todo:
        idx = writers[f].finalise()
        print(f"[synrs3d] packed {idx['n']} tiles -> synrs3d_{f} "
              f"@ 512px / {_SYN_GSD[f]} m")
    shutil.rmtree(work, ignore_errors=True)
    print(f"[synrs3d] add to --datasets: "
          f"{','.join('synrs3d_' + f for f in fams)}")


def _sibling(opt: Path, subdirs: tuple[str, ...]) -> Path | None:
    root = opt.parent.parent
    for sub in subdirs:
        c = root / sub / opt.name
        if c.exists():
            return c
        d = root / sub
        if d.is_dir():
            hits = sorted(d.glob(f"{opt.stem}.*"))
            if hits:
                return hits[0]
    return None


# ---------------------------------------------------------------------
# GeoNRW
# ---------------------------------------------------------------------
def prepare_geonrw(root: Path, max_tiles: int, token, force: bool, repo: str) -> None:
    from huggingface_hub import hf_hub_download

    out_tr = root / "geonrw" / "train"
    if store_exists(out_tr) and not force:
        print(f"[geonrw] already prepared -> {out_tr}")
        return
    import tarfile

    work = root / "_dl" / "geonrw"
    work.mkdir(parents=True, exist_ok=True)
    print("[geonrw] downloading nrw_dataset.tar.gz (~32 GB) …", flush=True)
    arc = hf_hub_download(repo, "nrw_dataset.tar.gz", repo_type="dataset",
                          local_dir=str(work), token=token)
    ex = work / "x"
    ex.mkdir(exist_ok=True)
    with tarfile.open(arc, "r:gz") as t:
        t.extractall(ex, filter="data")
    Path(arc).unlink(missing_ok=True)

    writer = ShardWriter(out_tr, tile_px=1000, gsd_m=GEONRW_GSD_M, shard_tiles=128)
    n = 0
    for rp in sorted(ex.rglob("*_rgb.jp2")):
        if rp.parent.name.lower().startswith(GEONRW_TEST_CITIES):
            continue
        try:
            rgb = _read_raster(rp)[..., :3].astype(np.uint8)
            dem = _read_raster(rp.with_name(rp.name.replace("_rgb.jp2", "_dem.tif")))
            ndsm = _ndsm_from_dem(np.asarray(dem, np.float32))
            sp = rp.with_name(rp.name.replace("_rgb.jp2", "_seg.tif"))
            cls = _remap(np.asarray(_read_raster(sp), np.int32), GEONRW_TO_SHARED) \
                if sp.exists() else None
            if rgb.shape[:2] != (1000, 1000):
                continue
            valid = np.isfinite(ndsm) & (ndsm < 500.0)
            writer.add(rp.name[:-8], rgb, np.where(valid, np.clip(ndsm, 0, None), 0.0),
                       cls, valid)
            n += 1
        except Exception:  # noqa: BLE001
            continue
        if max_tiles and n >= max_tiles:
            break
        if n % 200 == 0:
            print(f"  {n} tiles", flush=True)
    idx = writer.finalise()
    shutil.rmtree(work, ignore_errors=True)
    print(f"[geonrw] packed {idx['n']} tiles")


def _read_raster(p: Path) -> np.ndarray:
    try:
        import rasterio

        with rasterio.open(p) as ds:
            a = ds.read()
        return np.transpose(a, (1, 2, 0)) if a.shape[0] > 1 else a[0]
    except Exception:  # noqa: BLE001
        from PIL import Image

        return np.asarray(Image.open(p))


def _ndsm_from_dem(dem: np.ndarray, win: int = 120) -> np.ndarray:
    """GeoNRW ships an absolute DEM, not an nDSM — subtract a morphological
    ground estimate.  Coarse, which is why GeoNRW is auxiliary-only."""
    dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    try:
        from scipy.ndimage import gaussian_filter, minimum_filter

        g = gaussian_filter(minimum_filter(dem, size=win, mode="nearest"), sigma=win / 3.0)
    except Exception:  # noqa: BLE001
        g = np.full_like(dem, float(np.percentile(dem, 5)))
    return np.clip(dem - g, 0.0, None)


# ---------------------------------------------------------------------
# India  (see dwdata/india.py for why this is a directory ingest)
# ---------------------------------------------------------------------
def prepare_india_labeled(root: Path, a) -> None:
    """Paired Indian rasters -> `india_labeled/{train,val}`.

    The val split is carved out by *scene*, not by tile: tiles cut from the same
    source raster are near-duplicates of each other, so splitting them at random
    would leak the val set into training and produce a val RMSE that means
    nothing.
    """
    from dwdata.india import pack_labeled, pair_rasters, raster_gsd_m

    src = Path(a.india_dir).expanduser()
    if not a.india_dir or not src.is_dir():
        print("[india_labeled] needs --india_dir pointing at paired rasters "
              "(e.g. the DFC2023 New Delhi tiles); skipping")
        return
    recs = [r for r in pair_rasters(src) if r["hgt"] is not None]
    if not recs:
        print(f"[india_labeled] no rgb+height pairs under {src}; skipping")
        return
    gsd = raster_gsd_m(recs[0]["rgb"], a.india_gsd)
    print(f"[india_labeled] {len(recs)} scenes, gsd {gsd:.3f} m")

    n_val = max(1, int(round(len(recs) * a.india_val_frac))) if len(recs) > 4 else 0
    order = np.random.default_rng(0).permutation(len(recs))
    groups = {"val": [recs[i] for i in order[:n_val]],
              "train": [recs[i] for i in order[n_val:]]}
    for split, group in groups.items():
        if not group:
            continue
        stage = root / "_india_stage" / split
        stage.mkdir(parents=True, exist_ok=True)
        _link_group(group, stage)
        pack_labeled(stage, root / "india_labeled" / split, a.india_tile, gsd,
                     max_tiles=a.india_max, force=a.force,
                     dsm_is_absolute=a.india_absolute_dsm)
    shutil.rmtree(root / "_india_stage", ignore_errors=True)


def _link_group(recs, stage: Path) -> None:
    """Symlink one split's files into a staging dir so the packer sees only them."""
    for r in recs:
        for key in ("rgb", "hgt", "seg"):
            p = r.get(key)
            if p is None:
                continue
            dst = stage / p.name
            if not dst.exists():
                try:
                    dst.symlink_to(p.resolve())
                except OSError:
                    shutil.copy2(p, dst)


# ---------------------------------------------------------------------
# DFC23 Track 2  (2023 IEEE GRSS Data Fusion Contest)
# ---------------------------------------------------------------------
# Real 0.5 m satellite optical with a real nDSM reference — the only source in
# the mix that is actually the deployment domain.  GAMUS is US aerial, SynRS3D
# is synthetic, GeoNRW is a DEM proxy.
#
# SAR is deliberately ignored.  The packed store is 3-channel
# (dwdata/packed.py) and DINOv3's patch embedding is Conv2d(3, …), so fusing the
# co-registered SAR band means a store-format change *and* encoder surgery — for
# a signal a Cartosat-class single-view optical deployment will not have at
# inference anyway.
#
# `track2/val` and `track2_test_data` ship no reference nDSM (the contest
# withheld it), so the only usable labels are under `track2/train`, and the val
# split has to be carved out of those scenes here.
_DFC23_RGB_DIRS = ("rgb", "opt", "optical", "image", "images")
_DFC23_HGT_DIRS = ("dsm", "ndsm", "nDSM", "height", "agl", "gt_nDSM")


def _dfc23_pairs(src: Path) -> list[dict]:
    """Pair `<src>/rgb/NAME.tif` with `<src>/dsm/NAME.tif`.

    DFC23 discriminates the two rasters by **parent directory** and gives them
    identical filenames, which is exactly what `dwdata.india.pair_rasters` cannot
    see: it keys on filename *suffix*, so both files would be claimed as the RGB
    and every scene would come out unlabelled.  Pair here, then hand the rest to
    `pack_labeled`, which already does the tiling, the height-raster resize onto
    the RGB grid (DFC23 nDSMs come from ~2 m stereo), the valid/clip convention
    and the mostly-nodata drop.
    """
    def pick(names: tuple[str, ...]) -> Path | None:
        for n in names:
            d = src / n
            if d.is_dir():
                return d
        return None

    rgb_d, hgt_d = pick(_DFC23_RGB_DIRS), pick(_DFC23_HGT_DIRS)
    if rgb_d is None or hgt_d is None:
        have = sorted(x.name for x in src.iterdir() if x.is_dir()) if src.is_dir() else []
        print(f"[dfc23] need an rgb-like and a dsm-like subdirectory under {src}; "
              f"found {have}")
        return []
    hgt = {p.stem: p for p in hgt_d.iterdir()
           if p.suffix.lower() in (".tif", ".tiff")}
    recs = []
    for r in sorted(rgb_d.iterdir()):
        if r.suffix.lower() not in (".tif", ".tiff"):
            continue
        h = hgt.get(r.stem)
        if h is not None:
            recs.append({"stem": r.stem, "rgb": r, "hgt": h, "seg": None})
    print(f"[dfc23] {len(recs)} rgb+nDSM pairs under {src} "
          f"({rgb_d.name}/ + {hgt_d.name}/)")
    return recs


def _dfc23_stage(recs: list[dict], stage: Path) -> None:
    """Symlink a scene group into the `<stem>_rgb` / `<stem>_ndsm` convention
    `dwdata.india.pair_rasters` understands."""
    stage.mkdir(parents=True, exist_ok=True)
    for r in recs:
        for key, suf in (("rgb", "_rgb"), ("hgt", "_ndsm")):
            p = r[key]
            dst = stage / f"{r['stem']}{suf}{p.suffix}"
            if dst.exists():
                continue
            try:
                dst.symlink_to(p.resolve())
            except OSError:
                shutil.copy2(p, dst)


def prepare_dfc23(root: Path, a) -> None:
    """DFC23 Track 2 rgb+nDSM -> one packed store per GSD family, per split.

    Split is by **scene, not tile**, for the same reason `prepare_india_labeled`
    is: tiles cut from one scene are near-duplicates, so a random split leaks the
    val set into training and the val number stops meaning anything.

    Grouped by GSD for the same reason SynRS3D is: DFC23 optical is SuperView-1
    at 0.5 m *and* Gaofen-2 at 0.8 m, and `index.json` carries one `gsd_m` per
    store.
    """
    from dwdata.india import pack_labeled, raster_gsd_m

    src = Path(a.dfc23_dir).expanduser()
    if not a.dfc23_dir or not src.is_dir():
        print("[dfc23] needs --dfc23_dir pointing at a DFC23 Track 2 split "
              "holding rgb/ and dsm/ (e.g. .../track2/train); skipping")
        return
    recs = _dfc23_pairs(src)
    if not recs:
        return

    # One store per rounded GSD family.  Read per scene from the GeoTIFF
    # transform; --dfc23_gsd is only the fallback for a scene with no CRS.
    fams: dict[str, list[dict]] = {}
    for r in recs:
        g = raster_gsd_m(r["rgb"], a.dfc23_gsd)
        fams.setdefault(f"g{round(g, 2):.2f}".replace(".", ""), []).append(
            dict(r, gsd=g))
    for fam, group in sorted(fams.items()):
        gsd = float(np.median([r["gsd"] for r in group]))
        name = f"dfc23_{fam}"
        n_val = (max(1, int(round(len(group) * a.dfc23_val_frac)))
                 if len(group) > 4 else 0)
        order = np.random.default_rng(0).permutation(len(group))
        splits = {"val": [group[i] for i in order[:n_val]],
                  "train": [group[i] for i in order[n_val:]]}
        print(f"[dfc23] {name}: {len(group)} scenes @ {gsd:.3f} m "
              f"-> {len(splits['train'])} train / {n_val} val (split by scene)")
        for split, grp in splits.items():
            if not grp:
                continue
            out = root / name / split
            if store_exists(out) and not a.force:
                print(f"[dfc23] already prepared -> {out}")
                continue
            stage = root / "_dfc23_stage" / f"{name}_{split}"
            shutil.rmtree(stage, ignore_errors=True)
            _dfc23_stage(grp, stage)
            idx = pack_labeled(stage, out, a.dfc23_tile, gsd,
                               max_tiles=a.dfc23_max, force=a.force,
                               dsm_is_absolute=a.dfc23_absolute_dsm,
                               label=f"dfc23/{name}/{split}",
                               max_height_m=a.dfc23_max_height_m)
            if idx:
                _dfc23_height_check(out, f"{name}/{split}")
    shutil.rmtree(root / "_dfc23_stage", ignore_errors=True)


def _dfc23_height_check(out: Path, label: str, tall_m: float = 100.0,
                        edge_px: int = 32) -> None:
    """Say what was actually packed, and check the two things that bite.

    **Is it really an nDSM?**  DFC23 Track 2 ships height above ground, so the
    median belongs near 0.  A median metres above zero is an absolute DSM, which
    would poison the target silently.

    **Are the very tall pixels real?**  The DFC23 nDSMs are stereo-derived and
    carry blunders at tile edges.  Measured on a New Delhi training scene, every
    single pixel above 100 m — 2,615 of them, up to 183.2 m — sat in rows 0-31, a
    ribbon glued to the top border, over RGB that is ordinary city (luminance
    119.5 vs 133.5 for the rest of the tile, no structure).  Nothing downstream
    catches that: `pack_labeled` filters at 500 m and `dataset.py` clamps at
    `--max_valid_height_m` 200, so a phantom 183 m target on a normal rooftop is
    trained as fact — and `StratumBalancer` (beta 0.7, clip 8) hands the tallest
    stratum the *largest* loss weight in the batch, on a model whose measured
    problem is already tall-structure bias.

    So this reports the tall mass split by border vs interior.  A tall fraction
    concentrated in the border band is the artefact signature; use
    `--dfc23_max_height_m` to drop it.  It is a report, not a silent edit: real
    buildings do sit at tile edges, because DFC23's tiles are cut from larger
    scenes.
    """
    from dwdata.packed import PackedStore

    st = PackedStore(out)
    # The border/interior line is a *prevalence* claim, so it has to be read off
    # the whole store.  Reading the first 64 tiles is how this check reported
    # `[!] concentrated at the tile border` on the real 1506-tile train store:
    # that front slice caught GF2_NewDelhi_28.5557_77.1194 and almost none of
    # the Rio / New York high-rise scenes, whereas over all 1773 scenes >100 m
    # mass is 0.64x as *dense* in the border band as in the interior.  Sample
    # the whole store, strided, and say how many tiles the numbers came from.
    # Exact for any store this packer builds (DFC23 train is 1506 tiles, ~30 s
    # to read back); the stride only guards a pathologically large store, and a
    # strided sample still misses the handful of scenes that carry most of the
    # tall mass — over the real split the five tallest Rio tiles hold 137k of
    # the 214k pixels above 100 m.
    cap = 4096
    step = max(1, len(st) // cap)
    idx = list(range(0, len(st), step))[:cap]
    n = len(idx)
    vals, n_edge, n_inner, px_edge, px_inner = [], 0, 0, 0, 0
    for i in idx:
        _, h, _, v = st.get(i)
        h = np.asarray(h, np.float32)
        v = np.asarray(v, bool)
        if not v.any():
            continue
        vals.append(h[v])
        e = np.zeros(h.shape, bool)
        k = min(edge_px, min(h.shape) // 2)
        e[:k] = e[-k:] = True
        e[:, :k] = e[:, -k:] = True
        tall = (h > tall_m) & v
        n_edge += int((tall & e).sum());   px_edge += int((v & e).sum())
        n_inner += int((tall & ~e).sum()); px_inner += int((v & ~e).sum())
    if not vals:
        print(f"[dfc23] HEIGHT CHECK {label}: no valid pixels")
        return
    h = np.concatenate(vals)
    med, p99, mx = (float(np.median(h)), float(np.percentile(h, 99)),
                    float(h.max()))
    zero = float((h == 0.0).mean()) * 100.0
    note = ""
    if med > 3.0:
        note = ("\n        [!] median is metres above ground — this looks like an "
                "ABSOLUTE DSM, not an nDSM.  Re-run with --dfc23_absolute_dsm.")
    print(f"[dfc23] HEIGHT CHECK {label}: median {med:.2f} m  p99 {p99:.1f} m  "
          f"max {mx:.1f} m  exactly-0 {zero:.1f} %  (over {n} tiles){note}")
    if n_edge or n_inner:
        f_e = 100.0 * n_edge / max(1, px_edge)
        f_i = 100.0 * n_inner / max(1, px_inner)
        flag = ""
        # Uniform structure would put roughly equal *rates* in both bands.
        if f_e > 4.0 * max(f_i, 1e-6):
            flag = ("   [!] concentrated at the tile border — stereo blunders, "
                    "not buildings.  Consider --dfc23_max_height_m.")
        print(f"[dfc23]   >{tall_m:.0f} m: {f_e:.3f} % of border-{edge_px}px "
              f"pixels vs {f_i:.3f} % of interior{flag}")


def prepare_india_unlabeled(root: Path, a) -> None:
    """RGB-only Indian tiles -> `india_unlabeled/train` for the mean-teacher branch."""
    from dwdata.india import INDIA_AOIS, fetch_xyz_tiles, pack_unlabeled, tile_gsd_m

    out = root / "india_unlabeled" / "train"
    if store_exists(out) and not a.force:
        print(f"[india_unlabeled] already prepared -> {out}")
        return

    src = Path(a.india_dir).expanduser() if a.india_dir else None
    gsd = a.india_gsd
    if a.india_tile_url:
        src = root / "_dl" / "india_xyz"
        lat = float(np.mean([(b[1] + b[3]) / 2 for b in INDIA_AOIS.values()]))
        gsd = tile_gsd_m(a.india_zoom, lat)
        n = fetch_xyz_tiles(a.india_tile_url, src, zoom=a.india_zoom,
                            per_aoi=a.india_per_aoi)
        print(f"[india_unlabeled] fetched {n} scenes @ ~{gsd:.2f} m/px")
    if src is None or not src.is_dir():
        print("[india_unlabeled] needs --india_dir or --india_tile_url; skipping")
        return
    pack_unlabeled(src, out, a.india_tile, gsd, max_tiles=a.india_max, force=a.force)
    if a.india_tile_url:
        shutil.rmtree(root / "_dl" / "india_xyz", ignore_errors=True)


# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="materialise datasets into memmap shards")
    ap.add_argument("--datasets", default="gamus")
    ap.add_argument("--data_root", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--gamus_train", type=int, default=4000, help="0 = all")
    ap.add_argument("--gamus_val", type=int, default=400)
    ap.add_argument("--synrs3d_archives", type=int, default=2)
    ap.add_argument("--geonrw_max", type=int, default=2500)
    ap.add_argument("--gamus_repo", default="earthflow/GAMUS")
    ap.add_argument("--synrs3d_repo", default="JTRNEO/SynRS3D")
    ap.add_argument("--geonrw_repo", default="torchgeo/geonrw")
    ap.add_argument("--india_dir", default="",
                    help="local directory of Indian rasters to ingest "
                         "(see dwdata/india.py for the pairing convention)")
    ap.add_argument("--india_gsd", type=float, default=0.5,
                    help="metres/pixel of --india_dir imagery when it carries no "
                         "GeoTIFF transform")
    ap.add_argument("--india_tile", type=int, default=512)
    ap.add_argument("--india_max", type=int, default=0, help="0 = all")
    ap.add_argument("--india_val_frac", type=float, default=0.15,
                    help="fraction of labeled Indian tiles held out for validation")
    ap.add_argument("--india_absolute_dsm", action="store_true",
                    help="the height rasters are elevation ASL, not height AGL")
    ap.add_argument("--india_tile_url", default="",
                    help="XYZ template ({z}/{x}/{y}) to build the unlabeled set "
                         "from; you must supply one you are licensed to use")
    ap.add_argument("--india_zoom", type=int, default=18)
    ap.add_argument("--india_per_aoi", type=int, default=60)
    ap.add_argument("--dfc23_dir", default="",
                    help="DFC23 Track 2 directory holding rgb/ and dsm/ "
                         "subdirectories with matching filenames "
                         "(e.g. /kaggle/input/<ds>/track2/train)")
    ap.add_argument("--dfc23_tile", type=int, default=512)
    ap.add_argument("--dfc23_max", type=int, default=0, help="0 = all")
    ap.add_argument("--dfc23_val_frac", type=float, default=0.15,
                    help="fraction of DFC23 *scenes* (not tiles) held out for "
                         "validation")
    ap.add_argument("--dfc23_gsd", type=float, default=0.5,
                    help="fallback metres/pixel when a scene carries no GeoTIFF "
                         "transform")
    ap.add_argument("--dfc23_max_height_m", type=float, default=0.0,
                    help="mark nDSM pixels above this INVALID (0 = off). The "
                         "DFC23 stereo nDSMs carry >100 m blunders in a ribbon "
                         "along tile borders; the HEIGHT CHECK printed after "
                         "packing says whether yours do")
    ap.add_argument("--dfc23_absolute_dsm", action="store_true",
                    help="the height rasters are elevation ASL, not height AGL "
                         "(DFC23 Track 2 ships nDSM, so normally leave this off)")
    ap.add_argument("--hf_token", default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    root = Path(a.data_root)
    root.mkdir(parents=True, exist_ok=True)
    tok = _token(a.hf_token)
    names = [d.strip() for d in a.datasets.split(",") if d.strip()]
    if not tok and "gamus" in names:
        print("[warn] no HF token — GAMUS is gated and will 401")

    if "gamus" in names:
        prepare_gamus(root, "val", a.gamus_val, tok, a.force, a.gamus_repo, a.workers)
        prepare_gamus(root, "train", a.gamus_train, tok, a.force, a.gamus_repo, a.workers)
    # `synrs3d` prepares every family the requested archives cover; the
    # per-family store names (synrs3d_g05, …) are what --datasets takes at
    # *train* time, and naming one of them here means the same thing.
    if any(n == "synrs3d" or n.startswith("synrs3d_") for n in names):
        prepare_synrs3d(root, a.synrs3d_archives, tok, a.force, a.synrs3d_repo)
    if "dfc23" in names or any(n.startswith("dfc23_") for n in names):
        prepare_dfc23(root, a)
    if "geonrw" in names:
        prepare_geonrw(root, a.geonrw_max, tok, a.force, a.geonrw_repo)
    if "india_labeled" in names:
        prepare_india_labeled(root, a)
    if "india_unlabeled" in names:
        prepare_india_unlabeled(root, a)

    print("\nprepared stores:")
    for p in sorted(root.rglob("index.json")):
        import json

        i = json.loads(p.read_text())
        gib = sum(f.stat().st_size for f in p.parent.glob("*.npy")) / 1024 ** 3
        print(f"  {p.parent.relative_to(root)}: {i['n']} tiles "
              f"@ {i['tile_px']}px / {i['gsd_m']} m  ({gib:.1f} GiB)")


if __name__ == "__main__":
    main()
