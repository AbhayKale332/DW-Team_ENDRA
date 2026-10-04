"""v5 input path: NRSC products, band order, NoData, max_side, windowed inference.

Fixtures are tiny synthetic GeoTIFFs shaped like the real Cartosat-2E MERGED
product (`cartosat_2S_Sample/Cartosat-2E/247677521`): four single-band UInt16
files, NoData 0 around a rotated-looking collar, and a `BAND_META.txt`.
"""

import zipfile

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from dwdata.preprocess import PreprocSpec, read_scene, scene_geometry  # noqa: E402
from dwdata.scene_io import SceneSource, find_product, parse_meta_txt, rgb_band_indexes  # noqa: E402

META = """ProductID=247677522
SatID=CARTOSAT-2E
Sensor=MX
ProcessingLevel=MERGED
DateOfPass=06-Jan-2022
PixelSpacingAlong=0.60
SunAzimuthAtCenter=136.04000
SunElevationAtCenter=37.67014
TiltAngle=3.21950
MeanElevation=328
"""

H, W = 96, 80
TR = from_origin(204864.0, 1737801.0, 0.6, 0.6)


def _band_values():
    """Distinct per-band levels so a swapped band order is detectable."""
    y, x = np.mgrid[0:H, 0:W]
    base = (y + x).astype(np.float32)
    vals = {1: 100 + base, 2: 300 + base, 3: 600 + base, 4: 900 + base}   # B, G, R, NIR
    valid = np.ones((H, W), bool)
    valid[:, :10] = False          # the collar
    valid[:8, :] = False
    return {b: np.where(valid, v, 0).astype(np.uint16) for b, v in vals.items()}, valid


def _write_product(d):
    d.mkdir(parents=True, exist_ok=True)
    bands, valid = _band_values()
    for b, a in bands.items():
        with rasterio.open(d / f"BAND{b}.tif", "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="uint16", crs="EPSG:32644", transform=TR,
                           nodata=0) as ds:
            ds.write(a, 1)
    (d / "BAND_META.txt").write_text(META)
    return valid


def test_meta_parse_and_product_detection(tmp_path):
    m = parse_meta_txt(META)
    assert m["ProcessingLevel"] == "MERGED" and m["SunElevationAtCenter"] == "37.67014"
    _write_product(tmp_path / "p")
    pr = find_product(tmp_path / "p")
    assert pr.level == "MERGED" and sorted(pr.bands) == [1, 2, 3, 4]
    assert pr.sun_azimuth_deg == pytest.approx(136.04)
    assert pr.sun_elevation_deg == pytest.approx(37.67014)


def test_merged_folder_stacks_rgb_as_bands_3_2_1(tmp_path):
    valid = _write_product(tmp_path / "p")
    rgb, meta = read_scene(tmp_path / "p")
    assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
    v = meta.valid
    assert v is not None and (v == valid).all()
    # band 3 (R, highest DN) must land in channel 0 only after a per-band
    # stretch — so check the ordering on the *native* reader instead
    src = SceneSource(tmp_path / "p")
    nat, _ = src._read_native(20, 21)
    assert nat[0, 40, 0] > nat[0, 40, 1] > nat[0, 40, 2]       # R(600+) > G(300+) > B(100+)
    assert meta.gsd_m == pytest.approx(0.6)
    assert meta.sun == pytest.approx((136.04, 37.67014))
    assert meta.summary()["transform"][0] == pytest.approx(0.6)


def test_zip_product_reads_identically(tmp_path):
    _write_product(tmp_path / "p")
    z = tmp_path / "p.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in (tmp_path / "p").iterdir():
            zf.write(f, f"247677521/{f.name}")
    a, _ = read_scene(tmp_path / "p")
    b, mb = read_scene(z)
    assert (a == b).all() and mb.sun is not None


def test_four_band_stack_is_not_read_as_bgr(tmp_path):
    bands, _ = _band_values()
    p = tmp_path / "stack.tif"
    with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=4,
                       dtype="uint16", crs="EPSG:32644", transform=TR, nodata=0) as ds:
        for b in range(1, 5):
            ds.write(bands[b], b)
    src = SceneSource(p)
    assert src._band_idx == [3, 2, 1]
    assert rgb_band_indexes(4, None, [1, 2, 3]) == [1, 2, 3]       # --bands override
    assert rgb_band_indexes(3) == [1, 2, 3]
    assert rgb_band_indexes(1) == [1, 1, 1]


def test_nodata_is_excluded_from_the_radiometry(tmp_path):
    _write_product(tmp_path / "p")
    src = SceneSource(tmp_path / "p")
    lo, hi = src.radiometry()
    # counting the 0-valued collar would pin lo at 0; valid R starts at 600+
    assert lo[0] > 590


def test_max_side_scales_the_transform_with_the_gsd(tmp_path):
    _write_product(tmp_path / "p")
    rgb, meta = read_scene(tmp_path / "p", max_side=40)
    assert max(rgb.shape[:2]) <= 48
    assert meta.transform.a == pytest.approx(meta.gsd_m)
    # the grid still spans the same ground
    assert meta.transform.a * meta.width == pytest.approx(0.6 * W, rel=0.05)


def test_scene_geometry_reads_no_pixels(tmp_path):
    _write_product(tmp_path / "p")
    m = scene_geometry(tmp_path / "p")
    assert (m.width, m.height) == (W, H) and m.georeferenced


class _PerPixel(torch.nn.Module):
    """Height = a per-pixel function of the input: tiling-invariant by design."""

    def forward(self, x):
        h = (x[:, :1] * 3.0 + 10.0)
        seg = torch.zeros(x.shape[0], 8, *x.shape[-2:])
        return {"fused": h, "seg": seg}


def test_windowed_equals_whole_array_across_band_seams(tmp_path):
    from infer.engine import predict_scene, predict_scene_windowed

    _write_product(tmp_path / "p")
    spec = PreprocSpec(tile_size=32, canonical_gsd_m=0.6)
    model = _PerPixel().eval()
    rgb, meta = read_scene(tmp_path / "p")
    whole, _ = predict_scene(model, rgb, meta.gsd_m, spec, torch.device("cpu"),
                             valid=meta.valid)
    src = SceneSource(tmp_path / "p")
    out = predict_scene_windowed(model, src, spec, torch.device("cpu"),
                                 tmp_path / "out", band_rows=20, block_px=16)
    with rasterio.open(out["ndsm_path"]) as ds:
        win = ds.read(1)
        assert ds.crs.to_epsg() == 32644 and ds.transform == TR
    fin = np.isfinite(whole)
    assert (np.isfinite(win) == fin).all()                  # NoData -> NaN, same pixels
    assert np.abs(win[fin] - whole[fin]).max() < 1e-3
    assert out["block_mean"].shape == (-(-H // 16), -(-W // 16))
    assert out["stats"]["valid_px"] == int(fin.sum())


def test_windowed_overviews_release_previous_full_resolution_bands(tmp_path, monkeypatch):
    import weakref
    from infer.engine import predict_scene_windowed

    _write_product(tmp_path / "p")
    source = SceneSource(tmp_path / "p")
    buffers = []
    read_rows = source.read_rows
    real_open = rasterio.open

    class Writer:
        def __init__(self, dataset):
            self.dataset = dataset

        def __enter__(self):
            self.dataset.__enter__()
            return self

        def __exit__(self, *args):
            return self.dataset.__exit__(*args)

        def close(self):
            self.dataset.close()

        def write(self, data, *args, **kwargs):
            if data.dtype == np.float32:
                owner = data.base if data.base is not None else data
                buffers.append(weakref.ref(owner))
            return self.dataset.write(data, *args, **kwargs)

    def tracked_open(path, mode="r", **kwargs):
        dataset = real_open(path, mode, **kwargs)
        return Writer(dataset) if mode == "w" else dataset

    def tracked_rows(*args):
        # The previous band's local variables can still be live; older bands must be freed.
        assert all(ref() is None for ref in buffers[:-2])
        return read_rows(*args)

    class WithStd(_PerPixel):
        def forward(self, x):
            return {**super().forward(x), "b_std": torch.full_like(x[:, :1], 0.25)}

    monkeypatch.setattr(rasterio, "open", tracked_open)
    monkeypatch.setattr(source, "read_rows", tracked_rows)
    result = predict_scene_windowed(WithStd(), source,
                                   PreprocSpec(tile_size=32, canonical_gsd_m=0.6),
                                   torch.device("cpu"), tmp_path / "out",
                                   band_rows=20, overview_max=24)
    assert len(buffers) > 4
    assert result["overview_height"].shape == result["overview_std"].shape == (24, 20)
