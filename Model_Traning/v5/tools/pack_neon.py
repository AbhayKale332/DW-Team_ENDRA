"""NEON AOP (forest / savanna / scrub) -> packed store `neon/{train,val,test}` (0.5 m, 1024 px).

    export NEON_TOKEN=...            # data.neonscience.org -> My Account -> API token
    python tools/pack_neon.py plan    --work /tmp/neon
    python tools/pack_neon.py pack    --work /tmp/neon --data_root /tmp/neon/stores
    python tools/pack_neon.py clean   --data_root /tmp/neon/stores
    python tools/pack_neon.py preview --data_root /tmp/neon/stores --out /tmp/neon/preview.png
    python tools/pack_neon.py kaggle  --work /tmp/neon --data_root /tmp/neon/stores

The point of this store is **real tree heights**: DFC23 / india_labeled put
trees at 0 m, GAMUS / US3D / MVS3DM are mostly urban.  NEON flies its sites
with a lidar and a camera together, and publishes both on one 1 km UTM grid:

* DP3.30015.001 "Ecosystem structure": canopy height model (CHM), 1 m, metres
  above the lidar ground, trees *and* any other object (few at these sites);
* DP3.30010.001 camera mosaic, 0.1 m RGB, orthorectified;
* DP3.30024.001 lidar DSM + DTM, 1 m.

The CHM leaves **buildings out** (a barn roof reads 0 m) and keeps **wires**
(a power line reads 15 m over bare ground, invisible in the image).  Both are
found against DSM - DTM, which has the roofs but not the wires, and get no
label (`object_mask`): building = DSM - DTM - CHM > 2 m, wire = a CHM
ridge < 3 cells wide, > 2 m above DSM - DTM, >= 10 m long; each dilated by
one 1 m cell.

**plan**  For each site in `SITES` the newest *released* flight month that has
all three products (inside the site's leaf-on months) is picked, the two file
lists are read from the NEON data API, every CHM tile of the site is fetched
(~1 MB each) and summarised (valid share, canopy cover > 2 m, mean, p95).
Tiles are then split **by location** in 2 x 2 km blocks: `test` and `val`
blocks first, then `train` from what is left, minus the 8-neighbours of any
held-out 1 km tile, so no train tile touches a val/test tile.  Within each
split tiles are drawn stratified by canopy cover, so each site gives dense,
open and bare ground in proportion.  Writes `<work>/plan.json`.

**pack**  Per planned 1 km tile: RGB 10000^2 -> 2000^2 by 5 x 5 block mean
(a pixel with any no-data source pixel is no-data), CHM 1000^2 -> 2000^2 by
2 x 2 replication, so every 2 x 2 block of the label is exactly one lidar
cell.  That is what `coarse_label_m 1.0` scores on (`models/losses.py`), and
the 1024 px tiles start on even pixels (0, 976: a 2 x 2 grid that overlaps
by 48 px) so the blocks stay on the 1 m grid.  1024 px, not 512, because a
512 px training crop at up to 1.0 m GSD (`gsd_jitter_hi_m`, clamped by
`achievable_gsd_range`) needs a 1024 px source at 0.5 m, as GAMUS has.  A tile is kept when
>= `min_valid` of it has image and label; heights above `max_h` are invalid.
Downloads are deleted once packed, so the peak disk is a few RGB files.

**clean**  Per-site height cap: 1.5 x the median per-tile p99.9 of the
site's train tiles + 10 m.  Above it, and 2 px around, the label is invalid.
The EDA (`tools/eda_neon.py`) found > 60 m pixels that are not trees: a
canyon wall at MOAB (desert scrub, 74 m), wire remnants at LAJA (113 m), the
SERC flux tower, single-cell spikes at GUAN / CLBJ / JORN.  Only 0.01 % of
pixels, but on a ~3 m RMSE, one 95 m miss per 10^4 pixels adds ~0.15 m.
The cap keeps WREF's 80 m firs and TEAK / SOAP's 60-70 m pines.  The caps
are stored in build_info.json; a second run reuses them, so it is idempotent.

**kaggle**  `<split>.zip` with members at the zip root (Kaggle extracts it to
`<split>/`), README.md, LICENSE.txt (NEON data: CC BY 4.0), build_info.json,
dataset-metadata.json.  `kaggle datasets create -p <work>/kaggle` then uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API = "https://data.neonscience.org/api/v0"
CHM_DP, RGB_DP, DEM_DP = "DP3.30015.001", "DP3.30010.001", "DP3.30024.001"
DOI = {CHM_DP: "https://doi.org/10.48443/8qst-0w84", RGB_DP: "https://doi.org/10.48443/8vq2-s021",
       DEM_DP: "https://doi.org/10.48443/sxrt-ne87"}
GSD = 0.5
TILE = 1024
KM_PX = 2000                                   # one 1 km NEON tile at 0.5 m
OFFSETS = (0, 976)                             # even: 2x2 label blocks stay on the 1 m grid
LEAF_ON = tuple(range(4, 11))                  # Apr-Oct, northern hemisphere
ALL_YEAR = tuple(range(1, 13))

# site -> (landscape, flight months allowed, 1 km tiles to pack).  Chosen for the
# land covers Cartosat sees over India that the other stores lack: closed and
# open forest, tree savanna, dry scrub, tropical dry / wet forest.
SITES = {
    # closed forest
    "HARV": ("forest", LEAF_ON, 18), "BART": ("forest", LEAF_ON, 16),
    "SCBI": ("forest", LEAF_ON, 18), "SERC": ("forest", LEAF_ON, 18),
    "MLBS": ("forest", LEAF_ON, 16), "GRSM": ("forest", LEAF_ON, 18),
    "TALL": ("forest", LEAF_ON, 20), "DELA": ("forest", LEAF_ON, 18),
    "LENO": ("forest", LEAF_ON, 16), "UKFS": ("forest", LEAF_ON, 18),
    "UNDE": ("forest", LEAF_ON, 14), "WREF": ("forest", LEAF_ON, 18),
    "ABBY": ("forest", LEAF_ON, 16), "TEAK": ("forest", LEAF_ON, 18),
    "RMNP": ("forest", LEAF_ON, 14), "GUAN": ("forest", ALL_YEAR, 22),
    "PUUM": ("forest", ALL_YEAR, 18),
    # open pine / oak woodland and tree savanna
    "OSBS": ("savanna", LEAF_ON, 20), "JERC": ("savanna", LEAF_ON, 20),
    "SOAP": ("savanna", LEAF_ON, 20), "SJER": ("savanna", (3, 4, 5, 6), 24),
    "CLBJ": ("savanna", LEAF_ON, 24), "OAES": ("savanna", LEAF_ON, 20),
    "SRER": ("savanna", (7, 8, 9, 10), 24), "KONZ": ("savanna", LEAF_ON, 16),
    # shrubland / dry scrub
    "JORN": ("scrub", (7, 8, 9, 10), 24), "ONAQ": ("scrub", LEAF_ON, 20),
    "MOAB": ("scrub", LEAF_ON, 20), "DSNY": ("scrub", ALL_YEAR, 18),
    "LAJA": ("scrub", ALL_YEAR, 16),
}


# ---------------------------------------------------------------------
# NEON API
# ---------------------------------------------------------------------
def _session():
    import requests

    s = requests.Session()
    tok = os.environ.get("NEON_TOKEN", "").strip()
    if not tok and Path("/tmp/neon_token").is_file():
        tok = Path("/tmp/neon_token").read_text().strip()
    if tok:
        s.headers["X-API-Token"] = tok
    return s


def _get_json(s, url: str, retries: int = 5) -> dict:
    err = None
    for i in range(retries):
        try:
            r = s.get(url, timeout=60)
            if r.status_code == 429:
                raise RuntimeError("rate limited")
            d = r.json()
            if d.get("error"):
                raise RuntimeError(f"{url}: {d['error']}")
            return d["data"]
        except Exception as e:  # noqa: BLE001 — network, 429, bad JSON: retried
            err = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"{url}: {err}")


def released_months(s, dp: str) -> dict[str, set[str]]:
    out = {}
    for sc in _get_json(s, f"{API}/products/{dp}")["siteCodes"]:
        rel = [r for r in sc.get("availableReleases", []) if r["release"] != "PROVISIONAL"]
        out[sc["siteCode"]] = {m for r in rel for m in r["availableMonths"]}
    return out


TILE_RE = re.compile(r"_(\d{5,7})_(\d{6,8})_(CHM|image|DSM|DTM)\.tif$")


def tile_files(s, dp: str, site: str, month: str, kind: str | None = None
               ) -> dict[tuple[int, int], dict]:
    """{(easting, northing): {"name", "size", "url", ...}} of one site-month's tiles
    (`kind` picks DSM or DTM out of DP3.30024.001)."""
    out = {}
    for f in _get_json(s, f"{API}/data/{dp}/{site}/{month}")["files"]:
        m = TILE_RE.search(f["name"])
        if m and (kind is None or m[3] == kind):
            out[(int(m[1]), int(m[2]))] = {k: f.get(k) for k in ("name", "size", "url", "md5", "crc32c")}
    return out


def download(s, url: str, dst: Path, size: int | None = None, retries: int = 5) -> Path:
    if dst.is_file() and (not size or dst.stat().st_size == int(size)):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    err = None
    for i in range(retries):
        try:
            with s.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fo:
                    for chunk in r.iter_content(8 << 20):
                        fo.write(chunk)
            if size and tmp.stat().st_size != int(size):
                raise RuntimeError(f"size {tmp.stat().st_size} != {size}")
            tmp.replace(dst)
            return dst
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(5 * (i + 1))
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"{dst.name}: {err}")


# ---------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------
def chm_stats(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as d:
        h = d.read(1).astype(np.float32)
        nd = d.nodata
    ok = np.isfinite(h) & (h >= 0) & (h < 150)
    if nd is not None:
        ok &= h != nd
    v = h[ok]
    if v.size == 0:
        return {"valid": 0.0}
    return {"valid": round(float(ok.mean()), 4), "cover2": round(float((v > 2).mean()), 4),
            "mean": round(float(v.mean()), 2), "p95": round(float(np.percentile(v, 95)), 2),
            "low_0_2": round(float(((v > 0) & (v <= 2)).mean()), 4)}


def _stratified(tiles: list, n: int, rng) -> list:
    """`n` of `tiles` spread over their canopy-cover quantiles."""
    if n >= len(tiles):
        return list(tiles)
    t = sorted(tiles, key=lambda x: x["cover2"])
    bins = np.array_split(np.arange(len(t)), n)
    return [t[int(rng.choice(b))] for b in bins if len(b)]


def split_site(stats: dict, n: int, n_val: int, n_test: int, seed: int,
               min_valid: float = 0.6) -> dict:
    """{split: [tile dict]} by 2 x 2 km blocks; train never touches val/test."""
    rng = np.random.default_rng(seed)
    cand = [dict(e=e, n=nn, **st) for (e, nn), st in stats.items() if st.get("valid", 0) >= min_valid]
    blocks: dict = {}
    for t in cand:
        blocks.setdefault((t["e"] // 2000, t["n"] // 2000), []).append(t)
    keys = sorted(blocks)
    order = [keys[i] for i in rng.permutation(len(keys))]
    out, held = {"test": [], "val": []}, set()
    for sp, want in (("test", n_test), ("val", n_val)):
        pool = []
        while order and len(pool) < 2 * want:
            pool += blocks[order.pop()]
        out[sp] = _stratified(pool, want, rng)
        held |= {(t["e"], t["n"]) for t in pool}
    near = {(e + dx * 1000, nn + dy * 1000) for e, nn in held
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    rest = [t for t in cand if (t["e"], t["n"]) not in near]
    out["train"] = _stratified(rest, n - len(out["test"]) - len(out["val"]), rng)
    return out


def plan(work: Path, sites: list[str], val_n: int, test_n: int, workers: int) -> None:
    s = _session()
    chm_m, rgb_m, dem_m = (released_months(s, dp) for dp in (CHM_DP, RGB_DP, DEM_DP))
    out = {"sites": {}}
    for site in sites:
        land, months, n = SITES[site]
        both = sorted(m for m in chm_m.get(site, set()) & rgb_m.get(site, set())
                      & dem_m.get(site, set()) if int(m[5:7]) in months)
        if not both:
            print(f"[plan] {site}: no released month with CHM + RGB + DSM in {months} — skipped", flush=True)
            continue
        month = both[-1]
        chm, rgb = tile_files(s, CHM_DP, site, month), tile_files(s, RGB_DP, site, month)
        dsm, dtm = (tile_files(s, DEM_DP, site, month, k) for k in ("DSM", "DTM"))
        common = sorted(set(chm) & set(rgb) & set(dsm) & set(dtm))
        d = work / "chm" / f"{site}_{month}"

        def one(k):
            p = download(s, chm[k]["url"], d / chm[k]["name"], chm[k]["size"])
            return k, chm_stats(p)

        with ThreadPoolExecutor(workers) as ex:
            stats = dict(ex.map(one, common))
        seed = int(hashlib.md5(site.encode()).hexdigest()[:8], 16)
        sp = split_site(stats, n, val_n, test_n, seed)
        allv = [v for v in stats.values() if v.get("valid", 0) > 0]
        summ = {k: round(float(np.mean([v[k] for v in allv])), 3) for k in ("cover2", "mean", "low_0_2")}
        out["sites"][site] = {
            "landscape": land, "month": month, "n_tiles_available": len(common), "site_stats": summ,
            "tiles": {k: [{"e": t["e"], "n": t["n"], "chm": chm[(t["e"], t["n"])],
                           "rgb": rgb[(t["e"], t["n"])],
                           "dsm": dsm[(t["e"], t["n"])], "dtm": dtm[(t["e"], t["n"])],
                           **{x: t[x] for x in ("valid", "cover2", "mean", "p95")}} for t in v]
                      for k, v in sp.items()}}
        print(f"[plan] {site} {month} {land}: {len(common)} tiles, cover>2m {summ['cover2']:.0%}, "
              f"mean {summ['mean']:.1f} m, 0-2 m share {summ['low_0_2']:.1%} -> "
              + ", ".join(f"{k} {len(v)}" for k, v in sp.items()), flush=True)
        (work / "plan.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------
def block_mean_rgb(rgb: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """(3, H, W) uint8 -> (H/k, W/k, 3) uint8 mean + bool ok (no no-data source pixel)."""
    c, H, W = rgb.shape
    bad = (rgb == 0).all(0)
    r = rgb.reshape(c, H // k, k, W // k, k).astype(np.float32).mean((2, 4))
    ok = ~bad.reshape(H // k, k, W // k, k).any((1, 3))
    return np.clip(np.rint(r), 0, 255).astype(np.uint8).transpose(1, 2, 0), ok


def object_mask(chm: np.ndarray, dsm: np.ndarray, dtm: np.ndarray, nodata,
                parts: bool = False, min_wire_m: float = 10.0):
    """1 m cells whose CHM is wrong: buildings the CHM dropped, wires it kept.

    A wire is a CHM ridge narrower than 3 cells (> 2 m above the CHM's own 3 x 3
    opening) standing > 2 m above DSM - DTM, in a connected run spanning
    >= `min_wire_m`.  CHM vs DSM - DTM alone is no wire test: the DSM also sits low under sparse crowns
    and scatters ground cells through closed canopy (17-47 % of the canopy on
    GUAN / SJER tiles), and those are real trees.
    """
    from scipy import ndimage

    ok = (np.isfinite(dsm) & np.isfinite(dtm) & (dsm != nodata) & (dtm != nodata)
          & np.isfinite(chm) & (chm != nodata) & (chm >= 0))
    nd = np.where(ok, dsm - dtm, np.nan)
    with np.errstate(invalid="ignore"):
        bldg = ok & (nd - chm > 2.0)
        ridge = chm - ndimage.grey_opening(np.where(ok, chm, 0), size=(3, 3)) > 2.0
        cand = ok & ridge & (chm - nd > 2.0)
    lab, n = ndimage.label(cand, structure=np.ones((3, 3), bool))
    wire = np.zeros_like(cand)
    if n:
        sl = ndimage.find_objects(lab)
        span = np.array([np.hypot(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in sl])
        wire = np.isin(lab, 1 + np.flatnonzero(span >= min_wire_m))
    bad = ndimage.binary_dilation(bldg | wire, iterations=1)
    return (bad, bldg, wire) if parts else bad


def label_2x(chm: np.ndarray, nodata, bad: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(chm) & (chm >= 0)
    if nodata is not None:
        ok &= chm != nodata
    if bad is not None:
        ok &= ~bad
    h = np.where(ok, chm, 0).astype(np.float32)
    return np.repeat(np.repeat(h, 2, 0), 2, 1), np.repeat(np.repeat(ok, 2, 0), 2, 1)


def cut(rgb: np.ndarray, rgb_ok: np.ndarray, h: np.ndarray, h_ok: np.ndarray,
        min_valid: float, max_h: float):
    """Yield (r, c, rgb, hgt, valid) of the 2 x 2 grid of 1024 px tiles."""
    for r, y in enumerate(OFFSETS):
        for c, x in enumerate(OFFSETS):
            sl = (slice(y, y + TILE), slice(x, x + TILE))
            hh = h[sl]
            valid = rgb_ok[sl] & h_ok[sl] & (hh <= max_h)
            if valid.mean() < min_valid:
                continue
            img = np.where(rgb_ok[sl][..., None], rgb[sl], 0).astype(np.uint8)
            yield r, c, img, np.where(valid, hh, 0).astype(np.float32), valid


def pack_one(job: dict) -> dict:
    """One 1 km tile -> its 1024 px tiles (runs in a worker process)."""
    import rasterio

    s = _session()
    tmp = Path(job["tmp"])
    t = job["tile"]
    chm_p = download(s, t["chm"]["url"], tmp / t["chm"]["name"], t["chm"]["size"])
    dem_p = [download(s, t[k]["url"], tmp / t[k]["name"], t[k]["size"]) for k in ("dsm", "dtm")]
    rgb_p = download(s, t["rgb"]["url"], tmp / t["rgb"]["name"], t["rgb"]["size"])
    try:
        with rasterio.open(chm_p) as d:
            chm, nd, cb, ccrs = d.read(1).astype(np.float32), d.nodata, d.bounds, d.crs
        dem = []
        for p in dem_p:
            with rasterio.open(p) as d:
                if tuple(np.round(d.bounds)) != tuple(np.round(cb)) or d.shape != (1000, 1000):
                    return {"error": f"{p.name} grid {tuple(d.bounds)} {d.shape}", **job["key"]}
                dem.append(d.read(1).astype(np.float32))
        with rasterio.open(rgb_p) as d:
            rb, rcrs, rshape, rres = d.bounds, d.crs, (d.height, d.width), d.res
            rgb = d.read([1, 2, 3])
        if (tuple(np.round(cb)) != tuple(np.round(rb)) or chm.shape != (1000, 1000)
                or rshape != (10000, 10000) or ccrs != rcrs or rgb.dtype != np.uint8):
            return {"error": f"grid mismatch chm {tuple(cb)} {chm.shape} {ccrs} / rgb {tuple(rb)} "
                             f"{rshape} {rres} {rcrs} {rgb.dtype}", **job["key"]}
        img, img_ok = block_mean_rgb(rgb)
        del rgb
        bad = object_mask(chm, dem[0], dem[1], -9999.0)
        h, h_ok = label_2x(chm, nd, bad)
        out = [(f"{job['stem']}_r{r}_c{c}", a, b, v)
               for r, c, a, b, v in cut(img, img_ok, h, h_ok, job["min_valid"], job["max_h"])]
        return {"tiles": out, "crs": str(ccrs), "object_masked": float(bad.mean()), **job["key"]}
    finally:
        for p in (rgb_p, chm_p, *dem_p):
            p.unlink(missing_ok=True)


def _bounded(ex, fn, jobs: list, window: int):
    """Results of `fn(job)` in completion order, at most `window` in flight.

    Submitting everything up front kept every finished future — and its ~25 MB
    of tiles — alive until the end: the first full run was OOM-killed at
    370 / 562 km tiles on a 14 GB box.
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    it, live = iter(jobs), set()
    while True:
        for j in it:
            live.add(ex.submit(fn, j))
            if len(live) >= window:
                break
        if not live:
            return
        done, live = wait(live, return_when=FIRST_COMPLETED)
        for f in done:
            try:
                yield f.result()
            except Exception as e:  # noqa: BLE001
                yield e


def pack(work: Path, data_root: Path, min_valid: float, max_h: float, workers: int) -> None:
    from dwdata.packed import ShardWriter

    pl = json.loads((work / "plan.json").read_text())
    tmp = work / "dl"
    writers = {sp: ShardWriter(data_root / "neon" / sp, tile_px=TILE, gsd_m=GSD,
                               shard_tiles=48, has_seg=False) for sp in ("train", "val", "test")}
    jobs = []
    for site, si in pl["sites"].items():
        yr = si["month"][:4]
        for sp, tiles in si["tiles"].items():
            for t in tiles:
                jobs.append({"tile": t, "tmp": str(tmp), "min_valid": min_valid, "max_h": max_h,
                             "stem": f"{site}_{yr}_{t['e']}_{t['n']}",
                             "key": {"site": site, "split": sp, "e": t["e"], "n": t["n"]}})
    print(f"[pack] {len(jobs)} x 1 km tiles, "
          f"{sum(int(j['tile']['rgb']['size']) for j in jobs) / 1e9:.0f} GB of RGB to fetch", flush=True)
    counts = {sp: 0 for sp in writers}
    per_site: dict = {}
    failed, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(_bounded(ex, pack_one, jobs, 2 * workers), 1):
            if isinstance(r, Exception):        # one bad download must not end the run
                failed.append({"error": f"{type(r).__name__}: {r}"})
                print(f"[pack] FAILED {r}", flush=True)
                continue
            if "error" in r:
                failed.append(r)
                print(f"[pack] {r['site']} {r['e']} {r['n']}: {r['error']} — skipped", flush=True)
                continue
            sp = r["split"]
            for stem, rgb, hgt, val in r["tiles"]:
                writers[sp].add(stem, rgb, hgt, None, val)
            counts[sp] += len(r["tiles"])
            ps = per_site.setdefault(r["site"], {"train": 0, "val": 0, "test": 0, "crs": r["crs"],
                                                 "object_masked": []})
            ps[sp] += len(r["tiles"])
            ps["object_masked"].append(r["object_masked"])
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"[pack] {i}/{len(jobs)} km tiles  {counts}  {el / 60:.0f} min, "
                      f"eta {el / i * (len(jobs) - i) / 60:.0f} min", flush=True)
    from dwdata.packed import PackedStore

    for sp, w in writers.items():
        print(f"[pack] neon/{sp}: {w.finalise()['n']} tiles", flush=True)
        # ship the loader's per-tile stretch table (config stretch_lo/hi_pct 2 / 98),
        # so a read-only Kaggle mount never has to write it
        PackedStore(data_root / "neon" / sp).prime_stretch_bounds(2.0, 98.0, workers=workers)
    for ps in per_site.values():
        ps["object_masked"] = round(float(np.mean(ps["object_masked"])), 5)
    info = {"gsd_m": GSD, "tile_px": TILE, "tile_offsets_px": OFFSETS, "min_valid": min_valid,
            "max_valid_height_m": max_h, "counts": counts, "per_site": per_site, "failed": failed,
            "products": DOI,
            "sites": {k: {x: v[x] for x in ("landscape", "month", "n_tiles_available", "site_stats")}
                      | {"km_tiles": {sp: [(t["e"], t["n"]) for t in ts] for sp, ts in v["tiles"].items()}}
                      for k, v in pl["sites"].items()}}
    (data_root / "neon" / "build_info.json").write_text(json.dumps(info, indent=1))


# ---------------------------------------------------------------------
# clean: per-site height cap
# ---------------------------------------------------------------------
CAP_K, CAP_ADD_M, CAP_DILATE_PX = 1.5, 10.0, 2


def site_height_caps(p999_by_site: dict[str, list[float]]) -> dict[str, float]:
    """site -> cap (m): CAP_K x median per-tile p99.9 + CAP_ADD_M."""
    return {k: round(CAP_K * float(np.median(v)) + CAP_ADD_M, 1)
            for k, v in sorted(p999_by_site.items()) if len(v)}


def cap_mask(h: np.ndarray, v: np.ndarray, cap: float, dilate: int = CAP_DILATE_PX) -> np.ndarray:
    """Valid pixels to drop: above `cap`, grown by `dilate` px (the ramp of a spike / cliff)."""
    bad = v & (h > cap)
    if bad.any() and dilate:
        from scipy import ndimage

        bad = ndimage.binary_dilation(bad, np.ones((3, 3), bool), iterations=dilate)
    return bad & v


def _shards(d: Path):
    for sh in json.loads((d / "index.json").read_text())["shards"]:
        yield sh["file"], sh["stems"]


def clean(data_root: Path) -> None:
    store = data_root / "neon"
    # absent on a `run_kaggle.sh link`ed data root (only the splits are linked)
    bi = store / "build_info.json"
    info = json.loads(bi.read_text()) if bi.is_file() else {}
    caps = info.get("height_caps")
    if not caps:
        p999 = {}
        for f, stems in _shards(store / "train"):
            hg = np.load(store / "train" / f"{f}_hgt.npy", mmap_mode="r")
            va = np.load(store / "train" / f"{f}_val.npy", mmap_mode="r")
            for j, stem in enumerate(stems):
                hv = np.asarray(hg[j], np.float32)[va[j]]
                if hv.size:
                    p999.setdefault(stem.split("_", 1)[0], []).append(float(np.percentile(hv, 99.9)))
        caps = site_height_caps(p999)
    dropped = {}
    for sp in ("train", "val", "test"):
        d = store / sp
        if not (d / "index.json").is_file():
            continue
        n_sp = {}
        for f, stems in _shards(d):
            hg = np.load(d / f"{f}_hgt.npy", mmap_mode="r")
            va = np.load(d / f"{f}_val.npy", mmap_mode="r+")
            for j, stem in enumerate(stems):
                site = stem.split("_", 1)[0]
                bad = cap_mask(np.asarray(hg[j], np.float32), np.asarray(va[j]), caps[site])
                if bad.any():
                    va[j] = np.asarray(va[j]) & ~bad
                    n_sp[site] = n_sp.get(site, 0) + int(bad.sum())
            va.flush()
            del va
        for c in d.glob("landscape_v1*.npy"):           # classes were cut from the old labels
            c.unlink()
        dropped[sp] = n_sp
        print(f"[clean] neon/{sp}: {sum(n_sp.values())} px above the site cap dropped  {n_sp}", flush=True)
    info["height_caps"] = caps
    info["height_cap_rule"] = f"{CAP_K} x median per-tile p99.9 (train) + {CAP_ADD_M} m, dilated {CAP_DILATE_PX} px"
    if any(dropped.values()) or "height_capped_px" not in info:   # a re-run drops nothing: keep the first count
        info["height_capped_px"] = dropped
    (store / "build_info.json").write_text(json.dumps(info, indent=1))
    print(f"[clean] caps (m): {caps}")


# ---------------------------------------------------------------------
# preview / verify
# ---------------------------------------------------------------------
def _turbo(x: np.ndarray) -> np.ndarray:
    """0..1 -> RGB uint8, a cheap perceptual ramp (dark blue -> green -> yellow -> red)."""
    stops = np.array([[48, 18, 59], [40, 120, 220], [30, 200, 140], [180, 230, 50],
                      [250, 170, 30], [180, 20, 10]], np.float32)
    p = np.clip(x, 0, 1) * (len(stops) - 1)
    i = np.minimum(p.astype(int), len(stops) - 2)
    f = (p - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def preview(data_root: Path, out: Path, per_split: int = 6, vmax: float = 40.0) -> None:
    from PIL import Image

    from dwdata.packed import PackedStore

    rows, rng = [], np.random.default_rng(0)
    for sp in ("train", "val", "test"):
        st = PackedStore(data_root / "neon" / sp)
        n = len(st)
        v = np.concatenate([np.asarray(st._arr(i, "val")).ravel()[::97] for i in range(len(st.index["shards"]))])
        h = np.concatenate([np.asarray(st._arr(i, "hgt"), np.float32).ravel()[::97]
                            for i in range(len(st.index["shards"]))])
        h = h[v]
        print(f"[verify] neon/{sp}: {n} tiles, valid {v.mean():.1%}, h mean {h.mean():.2f} m, "
              f"p50 {np.percentile(h, 50):.1f} p95 {np.percentile(h, 95):.1f} max {h.max():.1f}, "
              f"> 2 m {(h > 2).mean():.1%}, (0, 2] {((h > 0) & (h <= 2)).mean():.1%}, "
              f"finite {np.isfinite(h).all()}", flush=True)
        site_px: dict = {}
        for i in range(n):
            _, hh, _, vv = st.get(i)
            e = site_px.setdefault(st.stems[i].split("_")[0], np.zeros(3))
            e += (vv.sum(), (vv & (hh > 3)).sum(), vv.size)
        for site, (nv, n3, nall) in sorted(site_px.items()):
            print(f"[verify]   {sp:5} {site}: labelled {nv / nall:.0%}, > 3 m {n3 / max(nv, 1):.1%}",
                  flush=True)
        for i in rng.choice(n, min(per_split, n), replace=False):
            rgb, hgt, _, val = st.get(int(i))
            col = _turbo(hgt / vmax)
            col[~val] = 0
            rows.append(np.concatenate([rgb[::2, ::2], col[::2, ::2]], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print(f"[verify] preview -> {out} (left RGB, right height 0-{vmax:g} m; black = no label)")


# ---------------------------------------------------------------------
# kaggle
# ---------------------------------------------------------------------
def kaggle(work: Path, data_root: Path, slug: str) -> None:
    import zipfile

    store, up = data_root / "neon", work / "kaggle"
    up.mkdir(parents=True, exist_ok=True)
    info = json.loads((store / "build_info.json").read_text())
    sizes = {}
    for sp in ("train", "val", "test"):
        d = store / sp
        zp = up / f"{sp}.zip"
        files = sorted(p for p in d.iterdir() if p.is_file())
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            for p in files:
                z.write(p, p.name)                      # flat: Kaggle extracts to <split>/<file>
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
        assert all("/" not in n for n in names) and "index.json" in names, names[:5]
        sizes[sp] = zp.stat().st_size / 1e9
        print(f"[kaggle] {zp.name}: {len(names)} members, {sizes[sp]:.2f} GB", flush=True)

    c = info["counts"]
    rows = []
    for site, si in sorted(info["sites"].items(), key=lambda kv: (kv[1]["landscape"], kv[0])):
        ps = info["per_site"].get(site, {})
        ss = si["site_stats"]
        rows.append(f"| {site} | {si['landscape']} | {si['month']} | {ss['cover2']:.0%} | "
                    f"{ss['mean']:.1f} | {ps.get('train', 0)} / {ps.get('val', 0)} / {ps.get('test', 0)} |")
    readme = f"""# DepthWizard neon

Height-above-ground tiles for single-image height estimation with **real tree
heights**: RGB at 0.5 m and a lidar canopy height model from the NSF NEON
Airborne Observation Platform (AOP), over {len(info['sites'])} forest, savanna and
scrub sites in the USA (incl. Puerto Rico and Hawaii).  Packed in the
DepthWizard store layout (`dwdata/packed.py`).

## Splits

| split | tiles | what |
|---|---|---|
| `train` | {c['train']} | per site, the 1 km tiles left after val/test, minus every tile touching a val/test tile |
| `val` | {c['val']} | per site, 2 x 2 km blocks held out by location |
| `test` | {c['test']} | per site, other 2 x 2 km blocks held out by location — never trained or selected on |

Each 1 km NEON tile gives up to 4 tiles of 1024 px (a 2 x 2 grid at offsets
0 / 976 px, so neighbours overlap by 48 px); all 4 belong to the split of
their 1 km tile.  Within a split, tiles are drawn stratified by canopy
cover, so dense forest, open woodland and bare ground are all present.

## Files

```
<split>/index.json          {{"tile_px": 1024, "gsd_m": 0.5, "n": N, "has_seg": false,
                              "shards": [{{"file": "shard_000", "n": 48, "stems": [...]}}, ...]}}
<split>/shard_XXX_rgb.npy   (n, 1024, 1024, 3) uint8   R, G, B
<split>/shard_XXX_hgt.npy   (n, 1024, 1024)    float16 metres above ground (trees included)
<split>/shard_XXX_val.npy   (n, 1024, 1024)    bool    pixel has both image and label
<split>/shard_XXX_cls.npy   (n, 1024, 1024)    uint8   all 255 (no classes)
<split>/stretch_bounds_2_98.npy  (n, 2, 3) float32  per-tile 2 / 98 % RGB bounds (loader cache)
build_info.json             sites, flight months, 1 km tile list per split, tile counts
```

Stems are `<SITE>_<year>_<easting>_<northing>_r<row>_c<col>`: the NEON 1 km
tile's lower-left UTM corner (WGS84, the site's zone; `per_site.crs` in
`build_info.json`) and the 1024 px tile's position in the 2 x 2 grid.

```python
import json, numpy as np
d = "/kaggle/input/datasets/{slug}/train"
idx = json.load(open(f"{{d}}/index.json"))
rgb = np.load(f"{{d}}/shard_000_rgb.npy", mmap_mode="r")
hgt = np.load(f"{{d}}/shard_000_hgt.npy", mmap_mode="r")
val = np.load(f"{{d}}/shard_000_val.npy", mmap_mode="r")
```

With the DepthWizard code, `run_kaggle.sh link` maps `depthwizard-neon/<split>`
to the store name `neon`; `lightning/final_flags.py` adds it to the run with
`coarse_label_sources neon`, `coarse_label_m 1.0`.

## How it was built (`tools/pack_neon.py`)

* **Flight.** Per site, the newest flight month in a NEON release that has
  the camera mosaic, the CHM and the lidar DSM / DTM inside the site's leaf-on
  season; all come from that one flight and share NEON's 1 km UTM tile grid.
* **Image.** DP3.30010.001 camera mosaic, 0.1 m RGB, reduced to 0.5 m by a
  5 x 5 block mean.  A 0.5 m pixel with any no-data (0, 0, 0) source pixel is
  no-data.  Colours are NEON's as delivered (no stretch).
* **Height.** DP3.30015.001 canopy height model, 1 m, replicated 2 x 2 onto the
  0.5 m grid, so every 2 x 2 block is one lidar cell: score it on 2 x 2 block
  means (DepthWizard: `coarse_label_m 1.0`), not per pixel.  Heights above
  {info['max_valid_height_m']:.0f} m and CHM no-data are marked invalid.  A tile is
  kept when >= {info['min_valid']:.0%} of it is valid.
* **Per-site height cap.** {info.get('height_cap_rule', 'none')}; above it the
  label is invalid (a canyon wall, wire remnants, a flux tower and single-cell
  spikes, ~0.01 % of pixels).  Caps per site are in `build_info.json`.
* **Objects the CHM gets wrong** have no label.  NEON's CHM leaves buildings
  out (a barn roof reads 0 m) and keeps wires (a power line reads ~15 m over
  bare ground).  Against DP3.30024.001 DSM - DTM: building = DSM - DTM - CHM
  > 2 m, wire = a CHM ridge < 3 m wide, > 2 m above DSM - DTM, >= 10 m
  long (crowns the DSM misses keep their label), both dilated by 1 m.

| site | landscape | flight | canopy > 2 m | mean height m | tiles train / val / test |
|---|---|---|---|---|---|
{chr(10).join(rows)}

(canopy share and mean height over all of the site's CHM tiles.)

## Known limitations

* The CHM is a 1 m product: crowns narrower than ~2 m and thin gaps are
  blurred; the camera is 10x sharper than the label.
* Buildings carry no label (see above), so this store teaches trees and
  ground, not roofs; the other DepthWizard stores cover buildings.
* Camera and lidar are from the same flight, but the image is a mosaic of
  oblique frames: tall crowns can lean a little near frame edges, and
  mosaic seams show as small colour steps.
* Aerial camera colour and sharpness differ from Cartosat-2S; pair with
  degradation augmentation when the target is Cartosat.

## Source, licence and citation

NEON data are licensed under **CC BY 4.0**
(https://creativecommons.org/licenses/by/4.0/).  This dataset is a derived
work: resampled, tiled and repackaged.  Please cite the source products:

> NEON (National Ecological Observatory Network). Ecosystem structure
> (DP3.30015.001). {DOI[CHM_DP]}

> NEON (National Ecological Observatory Network). High-resolution orthorectified
> camera imagery mosaic (DP3.30010.001). {DOI[RGB_DP]}

> NEON (National Ecological Observatory Network). Elevation - LiDAR
> (DP3.30024.001). {DOI[DEM_DP]}

The National Ecological Observatory Network is a program sponsored by the U.S.
National Science Foundation and operated under cooperative agreement by
Battelle.  This material uses data from the NEON program.
"""
    lic = """DepthWizard neon — licence
==========================

Derived from NSF NEON Airborne Observation Platform data products
DP3.30015.001 (Ecosystem structure / canopy height model), DP3.30010.001
(High-resolution orthorectified camera imagery mosaic) and DP3.30024.001
(Elevation - LiDAR, used only to mask buildings and wires).

NEON data are licensed under the Creative Commons Attribution 4.0
International licence (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

Changes made: the camera mosaic was resampled from 0.1 m to 0.5 m (5 x 5 block
mean); the canopy height model was replicated from 1 m to 0.5 m; both were cut
into 1024 x 1024 px tiles, with buildings and wires found from the lidar DSM / DTM
marked as unlabelled, and stored as NumPy arrays with a validity mask.

Attribution: "This material uses data from the National Ecological Observatory
Network (NEON), a program sponsored by the U.S. National Science Foundation and
operated under cooperative agreement by Battelle."  NEON does not endorse this
dataset or its uses.
"""
    (up / "README.md").write_text(readme)
    (up / "LICENSE.txt").write_text(lic)
    (up / "build_info.json").write_text(json.dumps(info, indent=1))
    meta = {"title": "DepthWizard neon", "id": slug,
            "subtitle": "NEON AOP RGB + lidar canopy height tiles, forest/savanna/scrub",
            "description": readme, "licenses": [{"name": "other"}]}
    (up / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print("[kaggle] ready:", sorted(p.name for p in up.iterdir()), {k: round(v, 2) for k, v in sizes.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["plan", "pack", "clean", "preview", "kaggle"])
    ap.add_argument("--work", type=Path, default=Path("/tmp/neon"))
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--val_tiles", type=int, default=1, help="1 km tiles per site")
    ap.add_argument("--test_tiles", type=int, default=2, help="1 km tiles per site")
    ap.add_argument("--min_valid", type=float, default=0.5)
    ap.add_argument("--max_valid_height_m", type=float, default=150.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--slug", default="abhaydkale232/depthwizard-neon")
    a = ap.parse_args()
    work = a.work.expanduser()
    work.mkdir(parents=True, exist_ok=True)
    data_root = (a.data_root or work / "stores").expanduser()
    if a.stage == "plan":
        plan(work, a.sites, a.val_tiles, a.test_tiles, max(8, a.workers))
    elif a.stage == "pack":
        pack(work, data_root, a.min_valid, a.max_valid_height_m, a.workers)
    elif a.stage == "clean":
        clean(data_root)
    elif a.stage == "preview":
        preview(data_root, a.out or work / "preview.png")
    else:
        kaggle(work, data_root, a.slug)


if __name__ == "__main__":
    main()
