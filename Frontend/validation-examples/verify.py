from pathlib import Path
import io
import json
import zipfile

import numpy as np
from PIL import Image
from pyproj import CRS
import rasterio

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())


def horizontal(crs):
    parsed = CRS(crs)
    return parsed.sub_crs_list[0] if parsed.is_compound else parsed


for example in manifest['examples']:
    folder = root / example['folder']
    if example['type'] == 'nDSM':
        heights = np.load(folder / 'reference_ndsm_m.npy', allow_pickle=False)
        with Image.open(folder / 'input_rgb.png') as image:
            assert heights.shape == (image.height, image.width) == (1024, 1024)
        assert heights.dtype == np.float32 and np.isfinite(heights).any()
        assert np.isclose(np.isfinite(heights).mean(), example['valid_fraction'])
    elif example['reference_role'] == 'lidar_ground_truth':
        with rasterio.open(folder / 'input_rgb.tif') as image:
            assert image.shape == tuple(example['input_shape']) == (1024, 1024)
            assert image.count == 3 and image.res == (0.5, 0.5)
            bounds, crs = image.bounds, horizontal(image.crs)
            rgb = image.read()
        with Image.open(folder / 'input_rgb.png') as png:
            assert np.array_equal(rgb.transpose(1, 2, 0), np.asarray(png))
        arrays = []
        for name in ('reference_dsm_m.tif', 'reference_ndsm_m.tif', 'ground_dtm_m.tif'):
            with rasterio.open(folder / name) as raster:
                assert raster.shape == tuple(example['reference_shape']) == (512, 512)
                assert raster.res == (1, 1) and horizontal(raster.crs) == crs
                assert np.allclose(raster.bounds, bounds, rtol=0, atol=1e-6)
                arrays.append(raster.read(1, masked=True))
        dsm, ndsm, dtm = arrays
        valid = ~np.ma.getmaskarray(dsm) & ~np.ma.getmaskarray(ndsm) & ~np.ma.getmaskarray(dtm)
        assert valid.any() and np.isfinite(dsm.data[valid]).all()
        assert np.allclose(ndsm.data[valid], (dsm - dtm).data[valid], rtol=0, atol=0.0001)
        assert np.isclose((~np.ma.getmaskarray(dsm)).mean(), example['valid_fraction'])
        chm = np.load(folder / 'reference_packed_chm_m.npy', allow_pickle=False)
        assert chm.shape == (1024, 1024) and chm.dtype == np.float32 and np.isfinite(chm).any()
    else:
        with rasterio.open(folder / 'input_optical.tif') as optical:
            grid = (optical.shape, horizontal(optical.crs), optical.transform)
        arrays = []
        for name in ('dsm_m.tif', 'ndsm_m.tif', 'dtm_m.tif'):
            with rasterio.open(folder / name) as raster:
                assert (raster.shape, horizontal(raster.crs), raster.transform) == grid
                arrays.append(raster.read(1, masked=True))
        dsm, ndsm, dtm = arrays
        valid = ~np.ma.getmaskarray(dsm) & ~np.ma.getmaskarray(ndsm) & ~np.ma.getmaskarray(dtm)
        assert valid.any()
        assert np.allclose(dsm.data[valid], (ndsm + dtm).data[valid], rtol=0, atol=0.0001)
        with zipfile.ZipFile(folder / 'dsm_result.zip') as bundle:
            assert bundle.testzip() is None
            heights = np.load(io.BytesIO(bundle.read('dsm_m.npy')), allow_pickle=False)
            assert heights.shape == grid[0]
            assert np.allclose(heights[valid], dsm.data[valid], rtol=0, atol=0)
            assert json.loads(bundle.read('meta.json'))['product'] == 'DSM'
    print('OK', example['folder'])
