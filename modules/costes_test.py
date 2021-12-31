import pytest
import numpy as np
from skimage import util

import costes as costes


def test_auto_threshold_exception_if_img1_not_numpy_array():
    with pytest.raises(BaseException):
        costes.auto_threshold([], np.narray([1, 2, 3]))


def test_auto_threshold_exception_if_img2_not_numpy_array():
    with pytest.raises(BaseException):
        costes.auto_threshold(np.array([1, 2, 3]), [])


def test_auto_threshold_exception_if_image_dimensions_different():
    img1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    img2 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

    with pytest.raises(BaseException):
        costes.auto_threshold(img1, img2)


def test_auto_threshold_returns_threshold_value():
    img1 = np.random.randint(0, 256, size=(200, 200)).astype(np.uint8)
    img2 = (util.random_noise(img1, mode="s&p", amount=0.997) * 255).astype(np.uint8)
    threshold, _ = costes.auto_threshold(img1, img2)
    print(threshold)
    # assert 1 == 0
    assert threshold >= 0 and threshold < 255
