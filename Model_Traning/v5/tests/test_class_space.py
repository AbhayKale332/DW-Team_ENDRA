"""v5 class space: one id space for every source, and the terms that read it."""

import numpy as np
import torch

from config import (CLASS_NAMES, FLAT_CLASS_IDS, GROUND_LIKE_IDS,
                    SEG_IGNORE_INDEX)
from dwdata.dataset import seg_ids
from dwdata.packed import NO_LABEL
from models.losses import flatness_loss


def test_gamus_ids_pass_through():
    c = np.array([[0, 1, 2, 3], [4, 5, 6, NO_LABEL]], np.uint8)
    out = seg_ids(c, "gamus")
    assert out.tolist() == [[0, 1, 2, 3], [4, 5, 6, SEG_IGNORE_INDEX]]


def test_shared_space_sources_are_remapped_into_gamus_ids():
    # shared: 0 ground, 1 veg (tree+rangeland+agri), 2 building, 3 water, 4 road
    c = np.array([0, 1, 2, 3, 4, NO_LABEL], np.uint8)
    for src in ("synrs3d_g05", "synrs3d_g1", "geonrw"):
        out = seg_ids(c, src).tolist()
        assert out == [1, SEG_IGNORE_INDEX, 3, 4, 5, SEG_IGNORE_INDEX], src
        assert CLASS_NAMES[out[2]] == "building"
        assert CLASS_NAMES[out[3]] == "water"


def test_flat_and_ground_ids_exclude_the_tall_classes():
    for ids in (FLAT_CLASS_IDS, GROUND_LIKE_IDS):
        assert CLASS_NAMES.index("building") not in ids
        assert CLASS_NAMES.index("tree") not in ids
        assert CLASS_NAMES.index("ground") in ids
        assert CLASS_NAMES.index("road") in ids


def test_flatness_now_acts_on_ground_not_buildings():
    torch.manual_seed(0)
    tgt = torch.zeros(1, 1, 16, 16)
    pred = torch.rand(1, 1, 16, 16)            # bumpy prediction on flat GT
    valid = torch.ones_like(tgt, dtype=torch.bool)
    ground = torch.full((1, 16, 16), CLASS_NAMES.index("ground"))
    bldg = torch.full((1, 16, 16), CLASS_NAMES.index("building"))
    assert flatness_loss(pred, tgt, valid, ground) > 0
    assert flatness_loss(pred, tgt, valid, bldg) == 0
