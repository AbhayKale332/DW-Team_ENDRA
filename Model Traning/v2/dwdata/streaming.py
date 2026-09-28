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
import shutil
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
        # `_extracted/<name>` dirs that eviction must never touch — the caller
        # (e.g. the per-epoch SynRS3D rotation) pins the archives it is about to
        # read so a concurrent download's eviction pass can't delete tiles that
        # are already indexed for this epoch.
        self._pinned: set[str] = set()

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

    @staticmethod
    def _archive_name(rel: str) -> str:
        return rel.replace("/", "__")

    def pin(self, rels) -> None:
        """Protect these archives' extracted dirs from eviction until the next
        `pin()` call.  Pass the archives the current epoch will read."""
        self._pinned = {self._archive_name(r) for r in rels}

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
        name = self._archive_name(rel)
        out_dir = self.root / "_extracted" / name
        done_flag = out_dir / ".extracted_ok"
        if done_flag.is_file() and self._has_payload(out_dir):
            self._touch_tree(out_dir)  # refresh LRU recency for this epoch
            return out_dir

        with self._lock_for(rel):
            if done_flag.is_file() and self._has_payload(out_dir):
                self._touch_tree(out_dir)
                return out_dir
            # Either never extracted, or a previous eviction pass deleted tiles
            # while leaving the ``.extracted_ok`` sentinel behind — wipe and redo.
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            arc = self._raw_download(rel)
            self._extract(arc, out_dir, member_prefix)
            try:
                arc.unlink()
            except OSError:
                pass
            self._touch_tree(out_dir)
            done_flag.write_text("ok")
        self._maybe_evict()
        return out_dir

    @staticmethod
    def _has_payload(d: Path) -> bool:
        """True if `d` holds at least one real file besides the sentinel."""
        if not d.is_dir():
            return False
        for p in d.rglob("*"):
            if p.is_file() and p.name != ".extracted_ok":
                return True
        return False

    @staticmethod
    def _touch_tree(d: Path) -> None:
        now = time.time()
        for p in d.rglob("*"):
            try:
                os.utime(p, (now, now))
            except OSError:
                pass
        try:
            os.utime(d, (now, now))
        except OSError:
            pass

    @staticmethod
    def _tree_size(d: Path) -> int:
        total = 0
        for p in d.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

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

            # 1) Evict whole extracted-archive dirs, least-recently-used first,
            #    never one that is pinned for the current epoch.  Evicting a
            #    directory as a unit (with its sentinel) means a half-deleted
            #    archive can never masquerade as complete on the next pass.
            ext_root = self.root / "_extracted"
            if ext_root.is_dir():
                archives = []
                for d in ext_root.iterdir():
                    if not d.is_dir() or d.name in self._pinned:
                        continue
                    flag = d / ".extracted_ok"
                    try:
                        mt = flag.stat().st_mtime if flag.is_file() else d.stat().st_mtime
                    except OSError:
                        mt = 0.0
                    archives.append((mt, d))
                archives.sort()  # oldest first
                for _mt, d in archives:
                    if size <= target:
                        break
                    sz = self._tree_size(d)
                    shutil.rmtree(d, ignore_errors=True)
                    size -= sz

            if size <= target:
                return

            # 2) Evict loose per-file cache entries (e.g. GAMUS *.h5), oldest first.
            files = []
            for p in self.root.rglob("*"):
                if not p.is_file():
                    continue
                if ".cache" in p.parts or ".locks" in p.parts or "_extracted" in p.parts:
                    continue
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


def purge_repos(cache_dir: str, keep) -> int:
    """Delete per-repo cache subdirs under ``cache_dir`` that aren't in ``keep``.

    Every dataset repo gets its own ``<org>__<name>`` subdir under ``cache_dir``
    with an *independent* size cap (see :class:`BoundedCacheHF`), and eviction
    only ever looks inside one repo's subdir.  So data from a finished stage —
    SynRS3D after pretrain — would otherwise sit at its full cap for the rest of
    the run, making the total footprint ``n_repos * cap`` instead of one cap.
    The trainer calls this on the way into a stage to drop the previous stage's
    datasets.

    ``keep`` is an iterable of repo ids (e.g. ``"earthflow/GAMUS"``).  Returns
    the number of bytes removed.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        return 0
    keep_names = {r.replace("/", "__") for r in keep}
    freed = 0
    for d in sorted(root.iterdir()):
        if (
            not d.is_dir()
            or d.is_symlink()
            or "__" not in d.name
            or d.name in keep_names
        ):
            continue
        freed += BoundedCacheHF._tree_size(d)
        shutil.rmtree(d, ignore_errors=True)
    return freed


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
