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

# SynRS3D archives, coarse-GSD family first (they give the model the >0.66 m
# scale range that GAMUS's 1024 px tiles physically cannot reach).
SYNRS3D_ARCHIVES = [
    "SynRS3D/grid_g05_mid_v1.zip", "SynRS3D/terrain_g05_mid_v1.zip",
    "SynRS3D/grid_g005_mid_v1.zip", "SynRS3D/terrain_g005_mid_v1.zip",
    "SynRS3D/grid_g05_high_v1.zip", "SynRS3D/grid_g05_low_v1.zip",
    "SynRS3D/grid_g005_high_v1.zip", "SynRS3D/grid_g005_low_v1.zip",
    "SynRS3D/terrain_g05_low_v1.zip", "SynRS3D/terrain_g005_low_v1.zip",
]
_SYN_GSD = {"g005": 0.3, "g05": 0.7}

# Class-id remaps into a shared 0..6 space.  These are best-effort (see README
# §1.6 — GAMUS's own id order is not what v1/v2 assumed), which is exactly why
# the semantic head is only an 0.2-weight auxiliary and why `--dump_class_stats`
# prints the measured histogram instead of trusting a table.
SYN_TO_SHARED = {0: NO_LABEL, 1: 0, 2: 1, 3: 0, 4: 4, 5: 1, 6: 3, 7: 1, 8: 2}
GEONRW_TO_SHARED = {0: NO_LABEL, 1: 1, 2: 3, 3: 1, 4: 2, 5: 1, 6: 4, 7: 4, 8: 0, 9: 4, 10: 2}

GEONRW_TEST_CITIES = ("duesseldorf", "herne", "neuss")


def _remap(a: np.ndarray, table: dict) -> np.ndarray:
    lut = np.full(int(max(table) + 1), NO_LABEL, np.uint8)
    for k, v in table.items():
        lut[k] = v
    return lut[np.clip(a, 0, len(lut) - 1).astype(np.int32)]


def _token(explicit: str = "") -> str | None:
    for v in (explicit, os.environ.get("HF_TOKEN"),
              os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if v:
            return v
    return None


# ---------------------------------------------------------------------
# GAMUS
# ---------------------------------------------------------------------
def prepare_gamus(root: Path, split: str, n_tiles: int, token, force: bool,
                  repo: str, workers: int = 12) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    out = root / "gamus" / split
    if store_exists(out) and not force:
        print(f"[gamus/{split}] already prepared -> {out}")
        return

    import h5py

    api = HfApi(token=token)
    files = api.list_repo_files(repo, repo_type="dataset")
    pre, suf = f"images/{split}/", "_RGB.h5"
    stems = sorted(f[len(pre):-len(suf)] for f in files
                   if f.startswith(pre) and f.endswith(suf))
    if not stems:
        raise RuntimeError(f"no GAMUS tiles under {pre} in {repo}")
    if n_tiles and n_tiles < len(stems):
        rng = np.random.default_rng(0)
        stems = sorted(np.asarray(stems)[rng.permutation(len(stems))[:n_tiles]].tolist())
    print(f"[gamus/{split}] packing {len(stems)} tiles -> {out}")

    tmp = root / "_dl" / f"gamus_{split}"
    tmp.mkdir(parents=True, exist_ok=True)

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

    writer = ShardWriter(out, tile_px=1024, gsd_m=GAMUS_GSD_M, shard_tiles=128)
    done = 0
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
            except Exception as e:  # noqa: BLE001
                print(f"[gamus/{split}] skip {stem}: {e}")
            finally:
                for p in paths.values():
                    if p:
                        Path(p).unlink(missing_ok=True)
            if done % 200 == 0:
                print(f"  {done}/{len(stems)}", flush=True)
    idx = writer.finalise()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[gamus/{split}] packed {idx['n']} tiles into {len(idx['shards'])} shards")


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
    from huggingface_hub import hf_hub_download

    out = root / "synrs3d" / "train"
    if store_exists(out) and not force:
        print(f"[synrs3d] already prepared -> {out}")
        return
    import zipfile

    import tifffile

    writer = ShardWriter(out, tile_px=512, gsd_m=0.5, shard_tiles=512)
    work = root / "_dl" / "synrs3d"
    total = 0
    for rel in SYNRS3D_ARCHIVES[:n_archives]:
        gsd = next((v for k, v in _SYN_GSD.items() if f"_{k}_" in rel), 0.5)
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
        print(f"[synrs3d] {rel}: {len(opts)} tiles @ ~{gsd} m")
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
                writer.add(opt.stem, rgb, np.where(valid, np.clip(h, 0, None), 0.0),
                           cls, valid)
                total += 1
            except Exception:  # noqa: BLE001
                continue
        shutil.rmtree(ex_dir, ignore_errors=True)
        print(f"[synrs3d] running total {total} tiles", flush=True)

    # SynRS3D GSD varies per archive; the index stores the nominal value and the
    # dataset clamps the achievable range per tile anyway.
    idx = writer.finalise()
    shutil.rmtree(work, ignore_errors=True)
    print(f"[synrs3d] packed {idx['n']} tiles")


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
    if "synrs3d" in names:
        prepare_synrs3d(root, a.synrs3d_archives, tok, a.force, a.synrs3d_repo)
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
