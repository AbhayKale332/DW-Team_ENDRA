"""The DEM decomposition, which is where v3 was quietly wrong.

Copernicus GLO-30 and SRTM are *surface* models: adding one to a predicted nDSM
counts every building twice.  These tests build a scene where the true terrain and
the true structure are both known, and assert the calibrated result recovers the
terrain while the naive addition does not.
"""
import numpy as np

from geo.calibrate import calibrate, fit_dtm, ground_mask, refine_with_gcps

H = W = 256


def _scene():
    y, x = np.mgrid[0:H, 0:W]
    dtm = (100 + 0.02 * x + 0.01 * y + 3 * np.sin(y / 40.0)).astype(np.float32)
    ndsm = np.zeros((H, W), np.float32)
    ndsm[60:120, 60:120] = 30.0                 # a tower
    ndsm[160:200, 40:200] = 8.0                 # a low block
    seg = np.where(ndsm > 1, 3, 1).astype(np.int32)   # building / ground (v5 ids)
    return dtm, ndsm, seg


def test_dtm_recovers_terrain_under_structures():
    dtm, ndsm, seg = _scene()
    dem_surface = dtm + ndsm                    # what GLO-30 actually contains
    cal = calibrate(ndsm, dem_surface, seg)
    err = np.abs(cal.dtm_m - dtm)
    assert err.mean() < 1.0, f"DTM mean error {err.mean():.2f} m"
    assert err[60:120, 60:120].mean() < 2.0, "terrain under the tower is wrong"


def test_calibrated_dsm_beats_the_naive_addition():
    dtm, ndsm, seg = _scene()
    dem_surface = dtm + ndsm
    cal = calibrate(ndsm, dem_surface, seg)
    truth = dtm + ndsm
    naive = dem_surface + ndsm
    at_tower = (slice(70, 110), slice(70, 110))
    assert abs(cal.dsm_m[at_tower].mean() - truth[at_tower].mean()) < 2.0
    # the naive path is off by exactly the structure height it counted twice
    assert abs(naive[at_tower].mean() - truth[at_tower].mean()) > 25.0
    assert cal.info["double_count_avoided_m"] > 5.0


def test_ground_mask_needs_both_conditions():
    """A wrong class id must not be able to poison the fit — the height test is
    the second, independent condition."""
    _dtm, ndsm, _seg = _scene()
    wrong = np.full((H, W), 5, np.int32)          # everything labelled 'road'
    m = ground_mask(ndsm, wrong)
    assert not m[80, 80], "a 30 m tower was accepted as ground"
    assert m[10, 10]


def test_ground_mask_falls_back_when_semantics_disagree():
    _dtm, ndsm, _seg = _scene()
    none_ground = np.full((H, W), 3, np.int32)    # no ground class anywhere
    m = ground_mask(ndsm, none_ground)
    assert m.mean() > 0.5, "should fall back to the height criterion"


def test_fit_survives_almost_no_ground():
    dtm, ndsm, _ = _scene()
    mask = np.zeros((H, W), bool)
    out, info = fit_dtm(dtm, mask)
    assert np.isfinite(out).all() and info["method"] == "median"


def test_gcp_ransac_rejects_an_outlier():
    _dtm, ndsm, _ = _scene()
    gcps = [(0, 0, 0.0), (90, 90, 30.0), (180, 100, 8.0), (10, 10, 99.0)]
    out, info = refine_with_gcps(ndsm, gcps)
    assert info["inliers"] == 3
    assert (10, 10, 99.0) in [tuple(r) for r in info["rejected"]]
    assert info["rmse_inliers_m"] < 0.5
    assert abs(info["scale"] - 1.0) < 0.05 and abs(info["offset_m"]) < 0.5
    assert out.shape == ndsm.shape


def test_gcp_needs_two_points():
    out, info = refine_with_gcps(np.zeros((8, 8), np.float32), [(0, 0, 1.0)])
    assert "need >= 2" in info["gcp"]


# ---------------------------------------------------------------------
# v5: dem_anchored and the decomposed GCP fit
# ---------------------------------------------------------------------
from geo.calibrate import (block_mean, calibrate_anchored,  # noqa: E402
                           refine_with_gcps_decomposed)


def test_anchored_cell_means_equal_the_dem():
    dtm, ndsm, _ = _scene()
    dem_surface = dtm + 0.5 * block_mean(ndsm, 32).repeat(32, 0).repeat(32, 1)
    cal = calibrate_anchored(ndsm, dem_surface, gsd_m=1.0, anchor_cell_m=32.0, iters=60)
    a = block_mean(cal.dsm_m, 32)
    d = block_mean(dem_surface, 32)
    assert np.abs(a - d).max() < 1e-3, np.abs(a - d).max()
    # the structure detail survives at pixel scale: the tower wall is a 30 m step
    assert abs((cal.dsm_m[90, 119] - cal.dsm_m[90, 120]) - 30.0) < 1.0


def test_anchored_needs_no_ground_pixels():
    """Closed canopy everywhere: `fit_dtm` would fall back to a flat median;
    anchoring still follows the relief."""
    y, x = np.mgrid[0:H, 0:W]
    relief = (200 + 0.8 * x).astype(np.float32)            # a 200 m slope
    canopy = np.full((H, W), 15.0, np.float32)
    cal = calibrate_anchored(canopy, relief + canopy, gsd_m=1.0, anchor_cell_m=32.0)
    err = cal.dsm_m - (relief + canopy)
    assert np.abs(err[32:-32, 32:-32]).max() < 1.0
    assert np.abs(cal.dtm_m - relief)[32:-32, 32:-32].max() < 1.0


def test_anchored_keeps_nodata():
    dtm, ndsm, _ = _scene()
    nd = ndsm.copy()
    nd[:, :20] = np.nan
    cal = calibrate_anchored(nd, dtm, gsd_m=1.0, anchor_cell_m=32.0)
    assert np.isnan(cal.dsm_m[:, :20]).all() and np.isfinite(cal.dsm_m[:, 20:]).all()


def test_detail_gain_zero_is_the_upsampled_dem():
    dtm, ndsm, _ = _scene()
    cal = calibrate_anchored(ndsm, dtm, gsd_m=1.0, anchor_cell_m=32.0, detail_gain=0.0)
    assert np.abs(cal.dsm_m - dtm)[40:-40, 40:-40].max() < 1.0


def test_decomposed_gcps_scale_structures_not_terrain():
    dtm, ndsm, _ = _scene()
    terrain = dtm + 400.0                                  # a plateau
    true = terrain + 2.0 + 1.1 * ndsm                      # offset 2 m, towers 10 % taller
    pts = [(10, 10), (90, 90), (180, 100), (240, 240), (70, 110), (200, 30)]
    gcps = [(r, c, float(true[r, c])) for r, c in pts]
    dsm, info = refine_with_gcps_decomposed(terrain, ndsm, gcps)
    assert info["gcp"] == "offset+tilt+scale"
    assert abs(info["structure_scale"] - 1.1) < 1e-3
    assert abs(info["offset_m"] - 2.0) < 1e-2
    assert np.abs(dsm - true).max() < 0.05
