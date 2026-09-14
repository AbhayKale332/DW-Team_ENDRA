#!/usr/bin/env python3
"""Bulk-stage every dataset file DepthWizard v4 needs, in one snapshot per repo.

Why this exists
---------------
`prepare_data.py` fetches GAMUS with one `hf_hub_download` per file — three
files a tile, so ~13 200 individual requests for a 4 000 + 400 tile pack.  That
earns a CAS 429 partway through, and a tile whose download failed is caught as
`skip: ...`: the v3 run wrote a `gamus/val` store holding **129 of 400 tiles**
and still reported success.

`snapshot_download` asks for the whole repo subtree once, resumes, and is Xet-
accelerated.  Files land under `_dl/<subdir>/<repo-relative path>`, which is
exactly where `hf_hub_download(..., local_dir=_dl/<subdir>)` looks — so the pack
step afterwards runs with `HF_HUB_OFFLINE=1`, never opens a socket, and a
missing file becomes a hard error instead of a quiet skip.

    python dw_fetch.py --repo-root FineTunning/v4 --data-root ~/DepthWizard-data \
                       --datasets gamus,synrs3d --workers 8
    python dw_fetch.py --datasets gamus --list      # sizes only, download nothing

Exit codes: 0 every requested repo is staged, 1 at least one failed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# (dataset, subdir under _dl, default repo, env override, allow_patterns builder)
#
# `allow_patterns` is the whole point of staging into a per-split subdir: GAMUS
# train and val are separate stores packed by separate calls, each with its own
# `local_dir`, so each gets only its own split's files.
SPECS: dict[str, dict] = {
    "gamus_train": {
        "repo": "earthflow/GAMUS", "env": "DW_GAMUS_REPO", "subdir": "gamus_train",
        "patterns": ["images/train/*", "heights/train/*", "classes/train/*"],
        "note": "GAMUS train tiles (RGB + AGL + classes)",
    },
    "gamus_val": {
        "repo": "earthflow/GAMUS", "env": "DW_GAMUS_REPO", "subdir": "gamus_val",
        "patterns": ["images/val/*", "heights/val/*", "classes/val/*"],
        "note": "GAMUS val tiles",
    },
    "synrs3d": {
        "repo": "JTRNEO/SynRS3D", "env": "DW_SYNRS3D_REPO", "subdir": "synrs3d",
        "patterns": None,           # filled from the module's archive list
        "note": "SynRS3D archives",
    },
    "geonrw": {
        "repo": "torchgeo/geonrw", "env": "DW_GEONRW_REPO", "subdir": "geonrw",
        "patterns": ["nrw_dataset.tar.gz"],
        "note": "GeoNRW tarball (~32 GB — slowest source by far)",
    },
}

# `--datasets gamus` means both of its splits.
EXPAND = {"gamus": ["gamus_train", "gamus_val"],
          "synrs3d": ["synrs3d"],
          "geonrw": ["geonrw"],
          "all": ["gamus_train", "gamus_val", "synrs3d", "geonrw"]}


def synrs3d_patterns(repo_root: Path, n_archives: int) -> list[str]:
    """The archives `prepare_synrs3d` will actually open, and no others.

    Read out of `prepare_data.py` rather than duplicated here: staging an
    archive the packer does not want is 7 GB of wasted transfer, and staging one
    it does want under a stale name is a silent fallback to the online path.
    """
    sys.path.insert(0, str(repo_root))
    try:
        from prepare_data import SYNRS3D_ARCHIVES
    except Exception as e:  # noqa: BLE001
        print(f"[fetch] cannot read SYNRS3D_ARCHIVES from {repo_root} ({e}) — "
              f"staging the whole repo")
        return None
    return list(SYNRS3D_ARCHIVES[:n_archives])


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def plan(keys: list[str], repo_root: Path, n_archives: int) -> list[dict]:
    out = []
    for k in keys:
        spec = dict(SPECS[k])
        spec["key"] = k
        spec["repo"] = os.environ.get(spec["env"]) or spec["repo"]
        if k == "synrs3d":
            spec["patterns"] = synrs3d_patterns(repo_root, n_archives)
        out.append(spec)
    return out


def repo_size(repo: str, patterns, token) -> tuple[int, int]:
    """(files, bytes) the snapshot would pull.  Best-effort — listing can fail."""
    from huggingface_hub import HfApi

    try:
        info = HfApi(token=token).repo_info(repo, repo_type="dataset", files_metadata=True)
    except Exception as e:  # noqa: BLE001
        print(f"[fetch] cannot size {repo}: {type(e).__name__}: {e}")
        return 0, 0
    import fnmatch

    n = total = 0
    for s in info.siblings or []:
        if patterns and not any(fnmatch.fnmatch(s.rfilename, p) for p in patterns):
            continue
        n += 1
        total += s.size or 0
    return n, total


def fetch_one(spec: dict, dl_root: Path, workers: int, token, max_gib: float) -> bool:
    from huggingface_hub import snapshot_download

    dest = dl_root / spec["subdir"]
    dest.mkdir(parents=True, exist_ok=True)
    n, size = repo_size(spec["repo"], spec["patterns"], token)
    if max_gib and size / 1024 ** 3 > max_gib:
        print(f"[fetch] {spec['key']}: {human(size)} exceeds DW_FETCH_MAX_GIB={max_gib:g}"
              f" — skipping.  Raise the cap or narrow --datasets.")
        return False

    print(f"[fetch] {spec['key']:<12} {spec['repo']:<24} "
          f"{n or '?'} files, {human(size) if size else 'size unknown'}  -> {dest}",
          flush=True)
    t0 = time.time()
    try:
        snapshot_download(
            repo_id=spec["repo"], repo_type="dataset", local_dir=str(dest),
            allow_patterns=spec["patterns"], token=token, max_workers=workers,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[fetch] {spec['key']}: FAILED after {time.time() - t0:.0f}s — "
              f"{type(e).__name__}: {e}")
        print(f"[fetch]   partial files are kept; re-running resumes them")
        return False
    have = sum(1 for p in dest.rglob("*") if p.is_file()
               and ".cache" not in p.parts)
    mins = (time.time() - t0) / 60
    print(f"[fetch] {spec['key']:<12} done — {have} files in {mins:.1f} min", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="bulk-stage DepthWizard datasets")
    ap.add_argument("--repo-root", default="FineTunning/v4",
                    help="where prepare_data.py lives (for the SynRS3D archive list)")
    ap.add_argument("--data-root", required=True, help="the DepthWizard-data directory")
    ap.add_argument("--datasets", default="gamus,synrs3d")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--synrs3d-archives", type=int,
                    default=int(os.environ.get("DW_SYNRS3D_ARCHIVES", "2")))
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    a = ap.parse_args()

    keys: list[str] = []
    for d in (x.strip() for x in a.datasets.split(",")):
        if not d:
            continue
        if d not in EXPAND:
            print(f"[fetch] unknown dataset {d!r} — known: {', '.join(EXPAND)}")
            return 2
        for k in EXPAND[d]:
            if k not in keys:
                keys.append(k)

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN") or None)
    if not token:
        print("[fetch] no HF_TOKEN — GAMUS and DINOv3-SAT are gated and will 401")

    repo_root = Path(a.repo_root).expanduser().resolve()
    dl_root = Path(a.data_root).expanduser().resolve() / "_dl"
    specs = plan(keys, repo_root, a.synrs3d_archives)

    if a.list:
        grand = 0
        for spec in specs:
            n, size = repo_size(spec["repo"], spec["patterns"], token)
            grand += size
            print(f"  {spec['key']:<12} {spec['repo']:<24} {n:>6} files  "
                  f"{human(size):>10}   {spec['note']}")
        print(f"  {'TOTAL':<12} {'':<24} {'':>6}        {human(grand):>10}")
        print(f"\n  staging dir: {dl_root}")
        print("  packed shards are roughly 60-70 % of the staged size")
        return 0

    max_gib = float(os.environ.get("DW_FETCH_MAX_GIB", "250"))
    ok = True
    for spec in specs:
        ok &= fetch_one(spec, dl_root, a.workers, token, max_gib)
    if not ok:
        print("[fetch] at least one repo did not finish — re-run to resume")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
