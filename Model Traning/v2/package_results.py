"""Bundle the run's artifacts into one zip and expose it over a cloudflared
quick tunnel — vast.ai has no persistent storage, so this is how the results get
pulled elsewhere (`!wget <printed url>`).  On Lightning AI / Kaggle the zip also
just sits in ``output_dir`` on persistent disk.

Standalone use:
    python package_results.py --serve                       # uses Config's output_dir
    python package_results.py --output_dir some/dir --serve
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

try:  # keep the default in one place (config.py); fall back if imported oddly
    from config import Config as _Config

    _DEFAULT_OUTPUT_DIR = _Config.output_dir
except Exception:  # noqa: BLE001
    _DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs" / "v2")

ZIP_NAME = "results_v2.zip"
_CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

# what we try to include; missing entries are silently skipped
WANT = [
    "best.pt", "stageP_last.pt", "last.pt",
    "metrics.json", "config.json", "run.log",
    "pip_freeze.txt", "env.txt",
    "viewer_sample", "qualitative",
]


def write_env_files(output_dir: str) -> None:
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
        ).stdout
        (d / "pip_freeze.txt").write_text(freeze)
    except Exception:  # noqa: BLE001
        pass
    lines = [f"python: {sys.version}", f"platform: {platform.platform()}"]
    try:
        import torch

        lines.append(f"torch: {torch.__version__}")
        lines.append(f"cuda: {torch.version.cuda} available={torch.cuda.is_available()}")
        for i in range(torch.cuda.device_count()):
            lines.append(f"gpu{i}: {torch.cuda.get_device_name(i)}")
    except Exception:  # noqa: BLE001
        pass
    (d / "env.txt").write_text("\n".join(lines) + "\n")


def build_zip(output_dir: str, zip_path: str | None = None) -> Path:
    out = Path(output_dir)
    zp = Path(zip_path) if zip_path else out / ZIP_NAME
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in WANT:
            p = out / name
            if p.is_file():
                z.write(p, name)
            elif p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file():
                        z.write(f, str(f.relative_to(out)))
    print(f"[zip] {zp}  ({zp.stat().st_size / 1e6:.1f} MB)")
    return zp


def _ensure_cloudflared() -> str | None:
    exe = shutil.which("cloudflared")
    if exe:
        return exe
    local = Path("./bin/cloudflared")
    if local.is_file():
        return str(local)
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        print("[cf] no cloudflared and cannot auto-install for this platform")
        return None
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        print("[cf] downloading cloudflared…")
        urllib.request.urlretrieve(_CF_URL, local)
        local.chmod(local.stat().st_mode | stat.S_IEXEC)
        return str(local)
    except Exception as e:  # noqa: BLE001
        print(f"[cf] auto-install failed: {e}")
        return None


def cloudflared_argv(exe: str, port: int) -> list[str]:
    return [exe, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"]


def share_via_cloudflared(zip_path: str, keep_alive_minutes: float = 180.0, port: int = 8000):
    """Serve the zip's directory and open a trycloudflare.com tunnel.

    Returns (public_url, procs) or (None, procs). Never raises — the local path
    is always still there.
    """
    zp = Path(zip_path).resolve()
    serve_dir = zp.parent
    procs: list[subprocess.Popen] = []

    http = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=serve_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(http)

    exe = _ensure_cloudflared()
    if not exe:
        print(f"[cf] tunnel unavailable — local file: {zp}")
        return None, procs

    cf = subprocess.Popen(
        cloudflared_argv(exe, port), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    procs.append(cf)

    url = None
    t0 = time.time()
    pat = re.compile(r"https://[-\w.]+\.trycloudflare\.com")
    while time.time() - t0 < 60:
        line = cf.stdout.readline()
        if not line:
            if cf.poll() is not None:
                break
            continue
        m = pat.search(line)
        if m:
            url = m.group(0)
            break

    if url:
        print("\n" + "=" * 64)
        print(f"DOWNLOAD: {url}/{zp.name}")
        print(f"  (Kaggle:  !wget -q {url}/{zp.name} && unzip -o {zp.name})")
        print(f"  tunnel stays up ~{keep_alive_minutes:.0f} min")
        print("=" * 64 + "\n")
    else:
        print(f"[cf] could not get a tunnel URL — local file: {zp}")
    return url, procs


def keep_alive(procs, minutes: float) -> None:
    try:
        time.sleep(max(0.0, minutes) * 60)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--keep_alive_minutes", type=float, default=180.0)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    write_env_files(a.output_dir)
    zp = build_zip(a.output_dir)
    if a.serve:
        url, procs = share_via_cloudflared(str(zp), a.keep_alive_minutes, a.port)
        keep_alive(procs, a.keep_alive_minutes if url else 0.0)


if __name__ == "__main__":
    main()
