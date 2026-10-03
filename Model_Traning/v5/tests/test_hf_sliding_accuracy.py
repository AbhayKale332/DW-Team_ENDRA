import math
import numpy as np
from tools.hf_sliding_accuracy import Accum, valid_mask


def test_pixel_weighted_metrics_and_bias():
    acc = Accum()
    acc.add([1, -2])
    acc.add([3])
    result = acc.result()
    assert result["n"] == 3 and result["sse"] == 14
    assert result["rmse_m"] == math.sqrt(14 / 3)
    assert result["mae_m"] == 2 and result["bias_m"] == 2 / 3


def test_native_validity_rejects_nodata_and_out_of_range_labels():
    target = np.array([0, 15, 150, 151, -1, np.nan, np.inf, 5], dtype=np.float32)
    source = np.array([True] * 7 + [False])
    np.testing.assert_array_equal(valid_mask(target, source),
                                  [True, True, True, False, False, False, False, False])
