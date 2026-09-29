"""v5 shadow layer: image shadows, cast shadows, sun fit, scale check."""

import numpy as np
import pytest

from viz.shadow import (cast_shadows, detect_image_shadows, fit_sun, iou,
                        shadow_scale_check)

G = 0.5


def _box(h=10.0, n=96):
    z = np.zeros((n, n), np.float32)
    z[40:56, 40:56] = h
    return z


def test_box_shadow_has_the_right_length_and_side():
    el, h = 45.0, 10.0
    sh = cast_shadows(_box(h), G, az_deg=90.0, el_deg=el)      # sun due east
    length = h / np.tan(np.radians(el)) / G                     # 20 px
    row = sh[48]
    cols = np.nonzero(row)[0]
    assert cols.min() >= 40 - length - 1 and cols.max() <= 40   # west of the box
    assert abs((40 - cols.min()) - length) <= 1.0
    assert not sh[48, 60:].any()                                # nothing on the sunny side


def test_sun_from_the_north_casts_south():
    sh = cast_shadows(_box(), G, az_deg=0.0, el_deg=45.0)
    assert sh[60:70, 45:50].any() and not sh[25:35, 45:50].any()


def test_fit_sun_recovers_a_known_sun():
    z = _box(12.0)
    img = cast_shadows(z, G, 140.0, 40.0)
    r = fit_sun(z, G, img)
    assert abs(((r["azimuth_deg"] - 140.0 + 180) % 360) - 180) <= 10
    assert abs(r["elevation_deg"] - 40.0) <= 10
    assert r["shadow_iou"] > 0.7


def test_scale_check_recovers_a_known_scale():
    z = _box(10.0)
    img = cast_shadows(1.5 * z, G, 120.0, 35.0)                 # truth is 1.5x taller
    r = shadow_scale_check(z, G, 120.0, 35.0, img)
    assert r["best_scale"] == pytest.approx(1.5, abs=0.11)
    assert r["iou_best"] > r["iou_at_1"] and r["informative"]


def test_image_shadow_detection_finds_a_dark_blue_patch():
    rng = np.random.default_rng(0)
    rgb = (rng.normal(170, 8, (64, 64, 3))).clip(0, 255).astype(np.uint8)
    rgb[20:40, 20:40] = (30, 35, 60)                            # dark, bluish
    m = detect_image_shadows(rgb)
    truth = np.zeros((64, 64), bool)
    truth[20:40, 20:40] = True
    assert iou(m, truth) > 0.9
