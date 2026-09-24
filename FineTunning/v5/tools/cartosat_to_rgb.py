"""Separate Cartosat PAN + MX products -> one registration-checked 0.6 m RGB GeoTIFF.

    python tools/cartosat_to_rgb.py --pan cartosat_2S_Sample/5132211 \
        --mx cartosat_2S_Sample/5132611 --out bhubaneswar_rgb.tif

Only the Bhubaneswar-style PAN + MX pair needs this: an NRSC MERGED product is
already 0.6 m and `SceneSource` stacks its BAND3,2,1 directly.

**Why not just `SceneSource([pan, mx])`?**  That path pan-sharpens on the fly and
trusts the two products' georeferencing.  The Bhubaneswar PAN is GEOREF level and
the MX is ORTHO, and their corners differ by ~45 m; any residual PAN/MX shift
shows up as colour fringes on every edge (the chroma is a whole MX pixel off
the luminance).  This tool:

1. **Measures the PAN<->MX shift** by phase correlation.  A central window of
   PAN is compared against the MX intensity (mean of all MX bands, the Brovey
   intensity) resampled onto the same PAN grid.  Both are high-passed first,
   because the two sensors' absolute radiometry differs.  The terrain is flat,
   so one global translation is the model; a peak-to-mean ratio is reported so
   a weak (untrustworthy) correlation is visible.  `--windows 3` measures it in
   three windows along the diagonal and reports the spread.
2. **Removes it** by shifting the MX geotransform, then
3. **Pan-sharpens** into a 3-band UInt16 GeoTIFF (R, G, B = MX bands 3, 2, 1),
   tagged with red/green/blue ColorInterp, NoData 0, tiled + DEFLATE, written in
   row bands so a 20k x 20k scene never sits in RAM.  `--engine gdal` runs
   `gdal_pansharpen` instead (GDAL >= 3.x on PATH) on shifted VRTs; the default
   `auto` uses it when present and the built-in weighted Brovey otherwise.  Both
   are the same maths: RGB * PAN / mean(B, G, R, NIR).

The output feeds `infer.predict` / the upload page like any 3-band GeoTIFF.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dwdata.scene_io import find_product  # noqa: E402


# ---------------------------------------------------------------------
# phase correlation
# ---------------------------------------------------------------------
def _highpass(a: np.ndarray, r: int = 8) -> np.ndarray:
    from tools.audit_labels import box_blur

    a = a.astype(np.float32)
    return a - box_blur(a, r)


def phase_correlate(ref: np.ndarray, mov: np.ndarray, valid: np.ndarray | None = None
                    ) -> tuple[float, float, float]:
    """(dy, dx, peak ratio): `mov` sampled at (y, x) shows what `ref` shows at (y+dy, x+dx).

    I.e. `mov` must be shifted by (+dy, +dx) (content moves down/right) to line
    up with `ref`.  Sub-pixel by a parabolic fit around the peak.
    """
    a, b = _highpass(ref), _highpass(mov)
    if valid is not None:
        a = np.where(valid, a, 0.0)
        b = np.where(valid, b, 0.0)
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    A, B = np.fft.fft2(a * win), np.fft.fft2(b * win)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-9)
    c = np.real(np.fft.ifft2(R))
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    peak = float(c[iy, ix])
    ratio = peak / max(float(np.abs(c).mean()), 1e-12)

    def sub(cm, c0, cp):
        d = cm - 2 * c0 + cp
        return 0.0 if abs(d) < 1e-12 else 0.5 * (cm - cp) / d

    dy = iy + sub(c[(iy - 1) % h, ix], c[iy, ix], c[(iy + 1) % h, ix])
    dx = ix + sub(c[iy, (ix - 1) % w], c[iy, ix], c[iy, (ix + 1) % w])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dy), float(dx), float(ratio)


# ---------------------------------------------------------------------
# reading PAN and MX-on-PAN-grid windows
# ---------------------------------------------------------------------
def _product_paths(p: str) -> tuple[str | None, list[str]]:
    """(PAN path, [MX band paths in band order]) for a product dir/zip or a tif."""
    pr = find_product(p)
    if pr is None:
        return None, [p]
    paths = [pr.bands[b] for b in sorted(pr.bands)]
    return (paths[0] if pr.is_pan else None), ([] if pr.is_pan else paths)


def _mx_on_grid(mx_paths, crs, transform, width, height, shift_m=(0.0, 0.0),
                bands=None):
    """MX bands warped (bilinear) onto a PAN window grid; shift_m = (east, north)
    added to the MX georeferencing first.  -> (n, height, width) float32."""
    import rasterio
    from affine import Affine
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling

    out = []
    idx = bands or list(range(1, len(mx_paths) + 1))
    for b in idx:
        with rasterio.open(mx_paths[b - 1]) as src:
            src_tr = src.transform
            if shift_m != (0.0, 0.0):
                src_tr = Affine.translation(*shift_m) * src_tr
            with WarpedVRT(src, src_crs=src.crs, src_transform=src_tr, crs=crs,
                           transform=transform, width=width, height=height,
                           resampling=Resampling.bilinear, src_nodata=src.nodata,
                           nodata=0) as v:
                out.append(v.read(1).astype(np.float32))
    return np.stack(out)


def measure_shift(pan_path: str, mx_paths: list[str], *, size: int = 2048,
                  n_windows: int = 1) -> dict:
    """PAN<->MX translation in metres (east, north), measured on PAN windows."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(pan_path) as ps:
        H, W, crs = ps.height, ps.width, ps.crs
        res = abs(ps.transform.a)
        s = min(size, H, W)
        fr = [0.5] if n_windows <= 1 else list(np.linspace(0.3, 0.7, n_windows))
        meas = []
        for f in fr:
            r0 = int(np.clip(f * H - s / 2, 0, H - s))
            c0 = int(np.clip(f * W - s / 2, 0, W - s))
            win = Window(c0, r0, s, s)
            pan = ps.read(1, window=win).astype(np.float32)
            wt = ps.window_transform(win)
            ms = _mx_on_grid(mx_paths, crs, wt, s, s)
            inten = ms.mean(0)
            valid = (pan > 0) & (inten > 0)
            if valid.mean() < 0.5:
                continue
            dy, dx, ratio = phase_correlate(pan, inten, valid)
            meas.append({"row0": r0, "col0": c0, "dy_px": dy, "dx_px": dx,
                         "peak_ratio": ratio, "valid_frac": float(valid.mean())})
    if not meas:
        return {"ok": False, "reason": "no window with >= 50 % valid overlap"}
    dy = float(np.median([m["dy_px"] for m in meas]))
    dx = float(np.median([m["dx_px"] for m in meas]))
    # MX content must move (+dy rows, +dx cols) on the PAN grid.  Rows grow
    # southward, so that is +dx*res east and -dy*res north.
    return {"ok": True, "dy_px": dy, "dx_px": dx, "pan_res_m": res,
            "shift_east_m": dx * res, "shift_north_m": -dy * res,
            "spread_px": float(np.ptp([np.hypot(m["dy_px"], m["dx_px"]) for m in meas]))
            if len(meas) > 1 else 0.0,
            "min_peak_ratio": float(min(m["peak_ratio"] for m in meas)),
            "windows": meas}


# ---------------------------------------------------------------------
# pan-sharpening
# ---------------------------------------------------------------------
def pansharpen_python(pan_path: str, mx_paths: list[str], out: str, *, shift_m=(0.0, 0.0),
                      rgb_bands=(3, 2, 1), rows: int = 1024) -> dict:
    """Weighted Brovey (equal weights, all MX bands in the intensity), row bands."""
    import rasterio
    from rasterio.windows import Window

    rgb_bands = [b for b in rgb_bands if b <= len(mx_paths)] or [1, 1, 1]
    with rasterio.open(pan_path) as ps:
        prof = {"driver": "GTiff", "height": ps.height, "width": ps.width, "count": 3,
                "dtype": "uint16", "crs": ps.crs, "transform": ps.transform, "nodata": 0,
                "tiled": True, "blockxsize": 512, "blockysize": 512,
                "compress": "deflate", "predictor": 2, "BIGTIFF": "IF_SAFER"}
        pan_nd = ps.nodata
        with rasterio.open(out, "w", **prof) as dst:
            from rasterio.enums import ColorInterp

            dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
            for r0 in range(0, ps.height, rows):
                h = min(rows, ps.height - r0)
                win = Window(0, r0, ps.width, h)
                pan = ps.read(1, window=win).astype(np.float32)
                ms = _mx_on_grid(mx_paths, ps.crs, ps.window_transform(win), ps.width, h,
                                 shift_m)
                inten = ms.mean(0)
                valid = (inten > 0) & (pan > 0)
                if pan_nd is not None:
                    valid &= pan != pan_nd
                ratio = np.where(valid, pan / np.maximum(inten, 1e-3), 0.0)
                o = np.stack([ms[b - 1] * ratio for b in rgb_bands])
                o = np.clip(np.rint(o), 0, 65535).astype(np.uint16)
                o[:, valid & np.all(o == 0, axis=0)] = 1          # 0 is NoData
                o[:, ~valid] = 0
                dst.write(o, window=win)
    return {"engine": "python", "rgb_bands": list(rgb_bands)}


def _gdal_bin() -> str | None:
    return shutil.which("gdal_pansharpen") or shutil.which("gdal_pansharpen.py")


def pansharpen_gdal(pan_path: str, mx_paths: list[str], out: str, *, shift_m=(0.0, 0.0),
                    rgb_bands=(3, 2, 1)) -> dict:
    """`gdal_pansharpen` on (optionally shifted) single-band VRTs of the MX bands."""
    import rasterio
    from affine import Affine

    tmp = Path(tempfile.mkdtemp(prefix="dw_ps_"))
    srcs = []
    try:
        for i, p in enumerate(mx_paths, 1):
            with rasterio.open(p) as s:
                tr = Affine.translation(*shift_m) * s.transform
            v = tmp / f"mx{i}.vrt"
            ul = (tr.c, tr.f, tr.c + tr.a * s.width, tr.f + tr.e * s.height)
            subprocess.run(["gdal_translate", "-q", "-of", "VRT", "-a_ullr",
                            *map(str, ul), p, str(v)], check=True)
            srcs.append(str(v))
        cmd = [_gdal_bin(), "-q", pan_path, *srcs, out, "-r", "bilinear",
               "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=IF_SAFER",
               "-nodata", "0"]
        for b in rgb_bands:
            cmd += ["-b", str(b)]
        for _ in mx_paths:                      # intensity over every MX band
            cmd += ["-w", str(1.0 / len(mx_paths))]
        subprocess.run(cmd, check=True)
        with rasterio.open(out, "r+") as ds:
            from rasterio.enums import ColorInterp

            ds.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"engine": "gdal", "rgb_bands": list(rgb_bands)}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pan", required=True, help="PAN product dir/zip, or a PAN .tif")
    p.add_argument("--mx", required=True, help="MX product dir/zip")
    p.add_argument("--out", required=True)
    p.add_argument("--engine", choices=("auto", "python", "gdal"), default="auto")
    p.add_argument("--bands", default="3,2,1", help="MX bands for R,G,B")
    p.add_argument("--window", type=int, default=2048, help="phase-correlation window (PAN px)")
    p.add_argument("--windows", type=int, default=3, help="windows along the diagonal")
    p.add_argument("--min_shift_px", type=float, default=0.25,
                   help="apply the measured shift only above this (PAN px)")
    p.add_argument("--min_peak_ratio", type=float, default=8.0,
                   help="below this the correlation is not trusted and no shift is applied")
    p.add_argument("--no_register", action="store_true", help="skip the shift measurement")
    a = p.parse_args(argv)

    pan, _ = _product_paths(a.pan)
    if pan is None:
        pan = _product_paths(a.pan)[1][0]
    _, mx = _product_paths(a.mx)
    if not mx:
        raise SystemExit(f"{a.mx} is a PAN product, not MX")
    rep: dict = {"pan": pan, "mx": mx}
    shift = (0.0, 0.0)
    if not a.no_register:
        m = measure_shift(pan, mx, size=a.window, n_windows=a.windows)
        rep["registration"] = m
        if m.get("ok"):
            mag = float(np.hypot(m["dy_px"], m["dx_px"]))
            trusted = m["min_peak_ratio"] >= a.min_peak_ratio
            print(f"[reg] MX->PAN shift {m['shift_east_m']:+.2f} m E, "
                  f"{m['shift_north_m']:+.2f} m N ({mag:.2f} PAN px), "
                  f"spread {m['spread_px']:.2f} px, peak ratio {m['min_peak_ratio']:.1f}")
            if mag >= a.min_shift_px and trusted:
                shift = (m["shift_east_m"], m["shift_north_m"])
            elif not trusted:
                print("[reg] weak correlation — georeferencing kept as delivered")
            if m["spread_px"] > 1.0:
                print("[reg] WARNING: the shift varies by > 1 px across the scene — a "
                      "single translation will leave fringes somewhere")
        else:
            print(f"[reg] {m.get('reason')}")
    rep["applied_shift_m"] = list(shift)

    bands = tuple(int(b) for b in a.bands.split(","))
    engine = a.engine
    if engine == "auto":
        engine = "gdal" if (_gdal_bin() and shutil.which("gdal_translate")) else "python"
    fn = pansharpen_gdal if engine == "gdal" else pansharpen_python
    rep.update(fn(pan, mx, a.out, shift_m=shift, rgb_bands=bands))
    rep["out"] = a.out
    Path(a.out).with_suffix(".json").write_text(json.dumps(rep, indent=2))
    print(f"[out] {a.out} ({rep['engine']})")
    return rep


if __name__ == "__main__":
    main()
