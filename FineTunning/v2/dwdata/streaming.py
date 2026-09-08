"""Bounded-LRU streaming cache over a Hugging Face dataset repo.

vast.ai has no persistent storage and ~100 GB disk; GAMUS alone is ~80 GB of
one-HDF5-per-tile.  Instead of pre-downloading a subset (v1) we pull each tile's
files on demand and keep the cache directory under a hard size cap, evicting the
least-recently-used files.

Usage
-----
    cache = BoundedCacheHF("earthflow/GAMUS", cache_dir="/tmp/dw_cache",
                           max_bytes=50 * 1024**3, token=tok)
    stems = cache.list_stems("images/train/", "_RGB.h5")
    local_path = cache.get(f"images/train/{stem}_RGB.h5")

`get()` is safe to call from many DataLoader workers at once (per-path file lock,
reused from the `filelock` dep that `huggingface_hub` already pulls in).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class BoundedCacheHF:
    def __init__(
        self,
        repo_id: str,
        cache_dir: str,
        max_bytes: int,
        repo_type: str = "dataset",
        token: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.token = token or None
        self.max_bytes = int(max_bytes)
        self.root = Path(cache_dir) / repo_id.replace("/", "__")
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._evict_guard = threading.Lock()
        self._repo_files: list[str] | None = None

    # -- repo listing ------------------------------------------------
    def list_repo_files(self) -> list[str]:
        if self._repo_files is None:
            from huggingface_hub import HfApi

            self._repo_files = HfApi(token=self.token).list_repo_files(
                self.repo_id, repo_type=self.repo_type
            )
        return self._repo_files

    def list_stems(self, prefix: str, suffix: str) -> list[str]:
        """Sorted list of `<stem>` for repo files `prefix + stem + suffix`."""
        out = [
            f[len(prefix): -len(suffix)]
            for f in self.list_repo_files()
            if f.startswith(prefix) and f.endswith(suffix)
        ]
        return sorted(out)

    # -- per-file fetch --------------------------------------------
    def _lock_for(self, rel: str) -> threading.Lock:
        with self._locks_guard:
            lk = self._locks.get(rel)
            if lk is None:
                lk = self._locks[rel] = threading.Lock()
            return lk

    def local_path(self, rel: str) -> Path:
        return self.root / rel

    def get(self, rel: str) -> Path:
        dst = self.local_path(rel)
        if dst.is_file():
            try:
                os.utime(dst, None)  # LRU touch
            except OSError:
                pass
            return dst

        with self._lock_for(rel):
            if dst.is_file():
                return dst
            self._raw_download(rel)
        self._maybe_evict()
        return dst

    # -- archive fetch (zip / tar.gz) -----------------------------
    def ensure_archive(self, rel: str, member_prefix: str = "") -> Path:
        """Download an archive from the repo, extract it under `_extracted/<name>/`,
        delete the archive, return the extracted dir.  Extracted files are subject
        to the same LRU eviction as everything else in the cache root, so a killed
        run just re-extracts on the next pass.

        `member_prefix` (optional) restricts extraction to members under a path.
        """
        name = rel.replace("/", "__")
        out_dir = self.root / "_extracted" / name
        done_flag = out_dir / ".extracted_ok"
        if done_flag.is_file():
            try:
                os.utime(done_flag, None)
            except OSError:
                pass
            return out_dir

        with self._lock_for(rel):
            if done_flag.is_file():
                return out_dir
            arc = self._raw_download(rel)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._extract(arc, out_dir, member_prefix)
            try:
                arc.unlink()
            except OSError:
                pass
            done_flag.write_text("ok")
        self._maybe_evict()
        return out_dir

    def _raw_download(self, rel: str) -> Path:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            self.repo_id, rel, repo_type=self.repo_type,
            local_dir=str(self.root), token=self.token,
        )
        p = self.local_path(rel)
        if p.is_symlink():
            real = p.resolve()
            data = real.read_bytes()
            p.unlink()
            p.write_bytes(data)
        return p

    @staticmethod
    def _extract(arc: Path, out_dir: Path, member_prefix: str) -> None:
        import tarfile
        import zipfile

        if arc.name.endswith((".zip",)):
            with zipfile.ZipFile(arc) as z:
                members = [m for m in z.namelist() if m.startswith(member_prefix)]
                z.extractall(out_dir, members=members or None)
        elif arc.name.endswith((".tar.gz", ".tgz", ".tar")):
            mode = "r:gz" if arc.name.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(arc, mode) as t:
                t.extractall(out_dir, filter="data")
        else:
            raise ValueError(f"unknown archive type: {arc.name}")

    # -- eviction --------------------------------------------------
    def _dir_size(self) -> int:
        total = 0
        for p in self.root.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def _maybe_evict(self) -> None:
        if not self._evict_guard.acquire(blocking=False):
            return
        try:
            size = self._dir_size()
            if size <= self.max_bytes:
                return
            target = int(self.max_bytes * 0.9)
            files = []
            for p in self.root.rglob("*"):
                if p.is_file() and ".cache" not in p.parts and ".locks" not in p.parts:
                    try:
                        files.append((p.stat().st_mtime, p.stat().st_size, p))
                    except OSError:
                        pass
            files.sort()  # oldest first
            for _mt, sz, p in files:
                if size <= target:
                    break
                try:
                    p.unlink()
                    size -= sz
                except OSError:
                    pass
        finally:
            self._evict_guard.release()

    # -- stats ----------------------------------------------------
    def stats(self) -> dict:
        return {
            "repo": self.repo_id,
            "cache_dir": str(self.root),
            "bytes": self._dir_size(),
            "max_bytes": self.max_bytes,
        }


def wait_for(fn, tries: int = 4, base: float = 1.5):
    """Small retry wrapper for flaky Hub reads inside DataLoader workers."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(base ** i)
    raise last
