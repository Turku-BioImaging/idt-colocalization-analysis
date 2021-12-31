import pytest
import numpy as np
from skimage import util, io, img_as_ubyte

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


def test_auto_threshold_exception_if_image_dtype_different() -> None:
    img1 = np.random.randint(0, 65535, size=(200, 200)).astype(np.uint16)
    img2 = np.random.randint(0, 256, size=(200, 200)).astype(np.uint8)

    with pytest.raises(BaseException):
        costes.auto_threshold(img1, img2)


def test_auto_threshold_returns_threshold_value():
    img1 = np.random.randint(0, 256, size=(200, 200)).astype(np.uint8)
    img2 = (util.random_noise(img1, mode="s&p", amount=0.997) * 255).astype(np.uint8)
    threshold, _, _ = costes.auto_threshold(img1, img2)
    assert threshold >= 0 and threshold < 255

    img1 = np.random.randint(0, 65536, size=(200, 200)).astype(np.uint16)
    img2 = (util.random_noise(img1, mode="s&p", amount=0.997) * 65535).astype(np.uint16)
    threshold, _, _ = costes.auto_threshold(img1, img2)
    print(threshold)

    assert threshold >= 0 and threshold < 65535
    assert 1 == 0


def test_auto_threshold_returns_threshold_for_three_dimensional_images():
    img1 = np.random.randint(0, 256, size=(200, 200, 3)).astype(np.uint8)
    img2 = (util.random_noise(img1, mode="s&p", amount=0.997) * 255).astype(np.uint8)

    threshold, _, _ = costes.auto_threshold(img1, img2)
    assert threshold >= 0 and threshold < 255


def test_auto_threshold_returns_value_for_test_images():
    img1 = img_as_ubyte(io.imread("modules/test_data/test1_ch1.png", as_gray=True))
    img2 = img_as_ubyte(io.imread("modules/test_data/test1_ch2.png", as_gray=True))

    threshold, img1_thresh, img2_thresh = costes.auto_threshold(img1, img2)

    assert threshold == 17

    assert len(np.unique(img1_thresh)) == 2
    assert len(np.unique(img2_thresh)) == 2

    assert set(np.unique(img1_thresh)) == set([0, 255])
    assert set(np.unique(img2_thresh)) == set([0, 255])
