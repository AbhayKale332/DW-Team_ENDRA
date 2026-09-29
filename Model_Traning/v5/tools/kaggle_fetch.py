"""Pull DepthWizard's Kaggle inputs onto a non-Kaggle box (Lightning H100).

    python tools/kaggle_fetch.py datasets --dest /tmp/kin \
        --dataset abhaydkale232/depthwizard-gamus:train,val,test \
        --dataset abhaydkale232/depthwizard-neon:train,val,test --optional neon
    python tools/kaggle_fetch.py checkpoint --kernel abhaydkale232/<slug> \
        --run v5_probe_v4init_mvs3dm --dest /tmp/prev

**datasets**  Per-file and resumable, not one whole-dataset zip: the ~130 GB of
stores would need twice that while the zip and its extraction coexist, and a
dropped connection would restart from zero.  Each file lands at
`<dest>/datasets/<owner>/<slug>/<its path in the dataset>`, i.e. exactly the
layout Kaggle mounts under /kaggle/input, so `run_kaggle.sh link` (with
`DW_KAGGLE_INPUT=<dest>`) finds it unchanged.  A file whose local size already
equals the listed size is skipped, so re-running after a failure only fetches
what is missing.  Only the requested splits are fetched (a path component
equal to `train` / `val` / `test`).

**checkpoint**  `kernels_output` with a file pattern, so only `best.pt` and
`metrics.json` of `outputs/<run>/` come down — not the 5 GB `last_full.pt`.
The file is checked to be a warm-start checkpoint (no optimiser state).

Needs `~/.kaggle/kaggle.json` (or KAGGLE_USERNAME / KAGGLE_KEY) and
`pip install kaggle`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SPLITS = ("train", "val", "test")


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def parse_spec(spec: str) -> tuple[str, tuple[str, ...]]:
    """"owner/slug:train,val" -> ("owner/slug", ("train", "val")); no ":" = all splits."""
    ds, _, sp = spec.partition(":")
    splits = tuple(s.strip() for s in sp.split(",") if s.strip()) or SPLITS
    bad = set(splits) - set(SPLITS)
    if "/" not in ds or bad:
        raise ValueError(f"bad --dataset {spec!r} (want owner/slug[:train,val,test])")
    return ds, splits


def wanted(name: str, splits: tuple[str, ...]) -> bool:
    """Keep a dataset file if one of its directory components is a wanted split.

    Files outside any split directory (README, manifest.json) are kept too:
    they are small and a store never lives there.
    """
    parts = Path(name).parts[:-1]
    in_split = [p for p in parts if p in SPLITS]
    return not in_split or any(p in splits for p in in_split)


def list_files(api, dataset: str) -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        r = api.dataset_list_files(dataset, page_token=token, page_size=200)
        if r is None:
            break
        if getattr(r, "error_message", None):
            raise RuntimeError(f"{dataset}: {r.error_message}")
        for f in r.files or []:
            out.append((f.name, int(getattr(f, "total_bytes", 0) or 0)))
        token = getattr(r, "next_page_token", None)
        if not token:
            break
    return out


def _fetch_one(dataset: str, name: str, size: int, root: Path, retries: int = 4) -> str:
    dst = root / name
    if dst.is_file() and (size <= 0 or dst.stat().st_size == size):
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for attempt in range(retries):
        tmp = Path(tempfile.mkdtemp(prefix=".part_", dir=root))
        try:
            _api().dataset_download_file(dataset, name, path=str(tmp), force=True, quiet=True)
            got = [p for p in tmp.iterdir() if p.is_file() and not p.name.endswith(".kaggle-partial")]
            if len(got) != 1:
                raise RuntimeError(f"expected one file in {tmp}, got {[p.name for p in got]}")
            src = got[0]
            # Large files come back zipped as <name>.zip; the store wants the member.
            if src.suffix == ".zip" and not name.endswith(".zip"):
                with zipfile.ZipFile(src) as z:
                    members = [m for m in z.namelist() if not m.endswith("/")]
                    if len(members) != 1:
                        raise RuntimeError(f"{src.name}: {len(members)} members")
                    with z.open(members[0]) as fi, open(tmp / "unz", "wb") as fo:
                        shutil.copyfileobj(fi, fo, 16 << 20)
                src.unlink()
                src = tmp / "unz"
            if size > 0 and src.stat().st_size != size:
                raise RuntimeError(f"size {src.stat().st_size} != listed {size}")
            src.replace(dst)
            return "ok"
        except Exception as e:  # noqa: BLE001 — network, auth, zip: all retried
            err = e
            time.sleep(5 * (attempt + 1))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    raise RuntimeError(f"{dataset}/{name}: {err}")


def cmd_datasets(a) -> None:
    api = _api()
    dest = Path(a.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    optional = {s.strip() for s in (a.optional or "").split(",") if s.strip()}
    plan = []
    for spec in a.dataset:
        ds, splits = parse_spec(spec)
        slug = ds.split("/")[1]
        try:
            files = [(n, s) for n, s in list_files(api, ds) if wanted(n, splits)]
        except Exception as e:  # noqa: BLE001
            if any(o in slug for o in optional):
                print(f"[fetch] {ds}: not available ({e}) — optional, skipped")
                continue
            raise
        root = dest / "datasets" / ds
        need = sum(s for n, s in files
                   if not ((root / n).is_file() and (root / n).stat().st_size == s))
        print(f"[fetch] {ds} [{','.join(splits)}]: {len(files)} files, "
              f"{sum(s for _, s in files) / 1e9:.1f} GB, {need / 1e9:.1f} GB to fetch")
        plan.append((ds, root, files, need))
    total = sum(p[3] for p in plan)
    free = shutil.disk_usage(dest).free
    print(f"[fetch] total to fetch {total / 1e9:.1f} GB; free on {dest}: {free / 1e9:.1f} GB")
    if total > free * 0.97 and not a.force:
        raise SystemExit("[fetch] not enough disk — free space or drop a split (--force to try anyway)")
    if a.dry_run:
        return
    t0 = time.time()
    done_b = 0
    for ds, root, files, _ in plan:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_fetch_one, ds, n, s, root): (n, s) for n, s in files}
            for f in as_completed(futs):
                n, s = futs[f]
                st = f.result()
                if st == "ok":
                    done_b += s
                    el = max(1e-6, time.time() - t0)
                    print(f"[fetch] {ds.split('/')[1]}/{n}  {s / 1e9:.2f} GB  "
                          f"({done_b / 1e9:.1f}/{total / 1e9:.1f} GB, {done_b / el / 1e6:.0f} MB/s)",
                          flush=True)
    print(f"[fetch] done in {(time.time() - t0) / 60:.1f} min")


def check_warm_start(path: Path) -> dict:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise SystemExit(f"{path}: not a checkpoint dict")
    if "opt" in ck:
        raise SystemExit(f"{path}: has optimiser state (a full-state resume file), "
                         f"expected a warm-start best.pt")
    info = {k: ck.get(k) for k in ("epoch", "encoder_included") if k in ck}
    m = ck.get("metrics") or {}
    if isinstance(m, dict) and "global" in m:
        info["val_rmse_m"] = m["global"].get("rmse_m")
    cfg = ck.get("config") or {}
    for k in ("bin_max_m", "n_bins", "detail_branch", "datasets"):
        if k in cfg:
            info[k] = cfg[k]
    if ck.get("encoder_included") is False:
        print(f"[ckpt] !! {path.name} has no encoder weights — the fine-tuned encoder "
              f"would be replaced by the hub's.  Is this the right file?")
    return info


def cmd_checkpoint(a) -> None:
    dest = Path(a.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "best.pt"
    if not target.is_file():
        pat = rf"(^|/)outputs/{re.escape(a.run)}/(best\.pt|metrics\.json)$"
        files, _ = _api().kernels_output(a.kernel, str(dest / "kernel_output"),
                                         file_pattern=pat, force=True, quiet=False)
        got = {Path(f).name: Path(f) for f in files if Path(f).is_file()}
        if "best.pt" not in got:
            raise SystemExit(f"{a.kernel}: no outputs/{a.run}/best.pt in the latest version's "
                             f"output (got {sorted(got)}). Check the slug and --run, or pass "
                             f"--prev_path to final_h100.sh.")
        got["best.pt"].replace(target)
        if "metrics.json" in got:
            got["metrics.json"].replace(dest / "metrics.json")
    print(f"[ckpt] {target}  {target.stat().st_size / 1e9:.2f} GB  {check_warm_start(target)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("datasets")
    d.add_argument("--dest", required=True)
    d.add_argument("--dataset", action="append", required=True,
                   help="owner/slug[:train,val,test]; repeatable")
    d.add_argument("--optional", default="", help="comma list of slug substrings allowed to be missing")
    d.add_argument("--workers", type=int, default=6)
    d.add_argument("--dry_run", action="store_true")
    d.add_argument("--force", action="store_true")
    c = sub.add_parser("checkpoint")
    c.add_argument("--kernel", required=True, help="owner/notebook-slug")
    c.add_argument("--run", default="v5_probe_v4init_mvs3dm")
    c.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    {"datasets": cmd_datasets, "checkpoint": cmd_checkpoint}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
