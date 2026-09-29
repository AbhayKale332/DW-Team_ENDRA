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
    seg = np.where(ndsm > 1, 2, 0).astype(np.int32)
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
    wrong = np.full((H, W), 4, np.int32)          # everything labelled 'road'
    m = ground_mask(ndsm, wrong)
    assert not m[80, 80], "a 30 m tower was accepted as ground"
    assert m[10, 10]


def test_ground_mask_falls_back_when_semantics_disagree():
    _dtm, ndsm, _seg = _scene()
    none_ground = np.full((H, W), 2, np.int32)    # no ground class anywhere
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
