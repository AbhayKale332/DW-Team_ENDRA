"""The landscape classifier answers the rubric's own stability axis, so it has to
put obvious archetypes in the obvious buckets."""
import numpy as np

from eval.landscape import classify, descriptors


def _flat(n=384):
    return np.zeros((n, n), np.float32)


def _urban(n=384):
    a = np.zeros((n, n), np.float32)
    for y in range(20, n - 60, 90):
        for x in range(20, n - 60, 90):
            a[y:y + 55, x:x + 55] = 14.0
    return a


def _forest(n=384, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(7 + 3 * rng.standard_normal((n, n)).astype(np.float32), 0, None)


def _hills(n=384):
    y, x = np.mgrid[0:n, 0:n]
    return (22 * np.sin(x / 110.0) + 19 * np.cos(y / 90.0) + 45).astype(np.float32)


def test_archetypes_land_in_the_right_bucket():
    assert classify(_flat())[0] == "sparse"
    assert classify(_urban())[0] == "urban"
    assert classify(_forest())[0] == "forested"
    assert classify(_hills())[0] == "hilly"


def test_roughness_separates_canopy_from_roofs():
    """Same height, different texture — this is the whole basis of the split."""
    roofs = descriptors(_urban())
    canopy = descriptors(_forest())
    assert canopy["roughness"] > 5 * roofs["roughness"]


def test_descriptors_are_gsd_aware():
    """The relief window is defined in metres, so declaring the same field at two
    GSDs must not move it across the hilly boundary."""
    h = _hills()
    a = descriptors(h, gsd_m=0.5)["relief_m"]
    b = descriptors(h, gsd_m=1.0)["relief_m"]
    assert abs(a - b) / max(a, 1e-6) < 0.35


def test_all_invalid_is_handled():
    name, d = classify(_urban(), np.zeros((384, 384), bool))
    assert name == "sparse" and d["n_valid"] == 0


def test_mask_is_respected():
    a = _urban()
    v = np.ones_like(a, bool)
    v[:, 192:] = False                       # hide half the buildings
    assert descriptors(a, v)["frac_tall"] < descriptors(a)["frac_tall"]


def test_no_urban_source_turns_smooth_tall_into_forest():
    """NEON's CHM has no buildings: a closed canopy the roughness rule calls urban is forest."""
    assert classify(_urban(), no_urban=True)[0] == "forested"
    assert classify(_flat(), no_urban=True)[0] == "sparse"
    assert classify(_forest(), no_urban=True)[0] == "forested"
