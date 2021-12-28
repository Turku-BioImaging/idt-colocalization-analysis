import pytest
import numpy as np

from manders import manders


def test_exception_if_missing_img1():
    with pytest.raises(BaseException):
        manders(np.array([]), np.array([1, 2, 3]))


def test_exception_if_missing_img2():
    with pytest.raises(BaseException):
        manders(np.array([]), np.array([1, 2, 3]))


def test_mask_must_be_8_bit_binary():
    img1 = np.random.rand(5, 5)
    img2 = np.random.rand(5, 5)
    mask = np.random.randn(5, 5) > 0.5

    with pytest.raises(BaseException):
        manders(img1, img2, mask)


def test_both_images_must_be_8_bit_binary():
    img1 = np.random.rand(5, 5)
    img2 = (np.random.rand(5, 5) * 255).astype(np.uint8)

    with pytest.raises(BaseException):
        manders(img1, img2)

    img2 = np.random.rand(5, 5)
    img1 = (np.random.rand(5, 5) * 255).astype(np.uint8)

    with pytest.raises(BaseException):
        manders(img1, img2)


def test_returns_the_manders_coefficients_and_cooccurence_mask():
    img1 = np.array(
        [
            [255, 255, 255, 0, 0],
            [0, 255, 255, 0, 0],
            [0, 255, 255, 0, 0],
            [0, 0, 0, 0, 0],
            [255, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    img2 = np.array(
        [
            [0, 255, 255, 255, 255],
            [0, 255, 255, 255, 255],
            [0, 255, 255, 255, 255],
            [0, 255, 255, 255, 255],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    m1, m2, cooccurence = manders(img1, img2)
    assert m1 == 0.75
    assert m2 == 0.375

    assert cooccurence[1][1] == 255
    assert cooccurence[1][2] == 255
    assert cooccurence[2][1] == 255
    assert cooccurence[2][2] == 255
    assert cooccurence[0][0] == 0
    assert cooccurence[4][4] == 0


def test_accepts_a_binary_mask_for_background_subtraction():
    img1 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 255, 255, 0, 0],
            [0, 255, 255, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    img2 = np.array(
        [
            [0, 255, 255, 255, 255],
            [0, 255, 255, 255, 255],
            [0, 255, 255, 255, 255],
            [0, 255, 255, 255, 255],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 255],
            [0, 255, 255, 255, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    m1, m2, _ = manders(img1, img2, mask)

    assert m1 == 1.0
    assert m2 == 0.4