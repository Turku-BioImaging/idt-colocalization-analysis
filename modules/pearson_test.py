import pytest
import numpy as np
from pearson import pearson


def test_exception_if_missing_img1():
    with pytest.raises(BaseException):
        pearson([], [1, 2, 3])


def test_exception_if_missing_img2():
    with pytest.raises(BaseException):
        pearson([1, 2, 3], [])


def test_exception_if_unequal_image_dimensions():
    img1 = np.array([0, 255, 0, 255, 0], dtype=np.uint8)
    img2 = np.array([[0, 255, 0], [0, 255, 0]], dtype=np.uint8)

    with pytest.raises(BaseException):
        pearson(img1, img2)


def test_returns_one_dimensional_complete_correlation():
    img1 = np.array([0, 255, 0], dtype=np.uint8)
    img2 = np.array([0, 255, 0], dtype=np.uint8)

    assert pearson(img1, img2) == 1.0

    img1 = np.array([0, 127, 0], dtype=np.uint8)
    img2 = np.array([0, 127, 0], dtype=np.uint8)

    assert pearson(img1, img2) == 1.0


def test_returns_one_dimensional_complete_anticorrelation():
    img1 = np.array([255, 0, 255], dtype=np.uint8)
    img2 = np.array([0, 255, 0], dtype=np.uint8)

    assert pearson(img1, img2) == -1.0

    img1 = np.array([127, 0, 127], dtype=np.uint8)
    img2 = np.array([0, 255, 0], dtype=np.uint8)

    assert pearson(img1, img2) == -1.0


def test_returns_two_dimensional_complete_anticorrelation():
    img1 = np.array([[255, 255, 0], [255, 255, 0], [255, 255, 0]], dtype=np.uint8)
    img2 = np.array([[0, 0, 125], [0, 0, 125], [0, 0, 125]], dtype=np.uint8)

    assert pearson(img1, img2) == -1.0


def test_returns_two_dimensional_complete_correlation():
    img1 = np.array([[0, 125, 0], [0, 255, 0], [0, 100, 0]], dtype=np.uint8)
    img2 = np.array([[0, 125, 0], [0, 255, 0], [0, 100, 0]], dtype=np.uint8)

    assert pearson(img1, img2) == 1.0


def test_returns_low_correlation_on_two_dimensional_random_images():
    img1 = np.random.rand(128, 128)
    img2 = np.random.rand(128, 128)
    assert pearson(img1, img2) < 0.04 and pearson(img1, img2) > -0.04


def test_returns_three_dimensional_complete_correlation():
    img1 = np.array(
        [
            [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
            [[0, 255, 0], [255, 0, 255], [0, 255, 0]],
            [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
        ],
        dtype=np.uint8,
    )
    img2 = np.array(
        [
            [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
            [[0, 255, 0], [255, 0, 255], [0, 255, 0]],
            [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
        ],
        dtype=np.uint8,
    )

    assert pearson(img1, img2) == 1.0


def test_returns_three_dimensional_complete_anticorrelation():
    img1 = np.array(
        [
            [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
            [[0, 255, 0], [255, 0, 255], [0, 255, 0]],
            [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
        ],
        dtype=np.uint8,
    )
    img2 = np.array(
        [
            [[0, 255, 0], [255, 0, 255], [0, 255, 0]],
            [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
            [[0, 255, 0], [255, 0, 255], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )

    assert pearson(img1, img2) == -1.0


def test_it_does_not_accept_more_than_three_dimensions():
    img1 = np.array(
        [
            [
                [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
                [[0, 255, 0], [255, 0, 255], [0, 255, 0]],
                [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
            ],
            [
                [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
                [[0, 255, 0], [255, 0, 255], [0, 255, 0]],
                [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
            ],
        ],
        dtype=np.uint8,
    )

    img2 = np.array(
        [
            [
                [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
                [[0, 255, 0], [255, 0, 255], [0, 255, 0]],
                [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
            ],
            [
                [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
                [[0, 255, 0], [255, 0, 255], [0, 255, 0]],
                [[255, 0, 255], [0, 255, 0], [255, 0, 255]],
            ],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(BaseException):
        pearson(img1, img2)


def test_accepts_float_images():
    img1 = np.array(
        [[0.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0]], dtype=np.float64
    )
    img2 = np.array(
        [[0.5, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.0, 0.5]], dtype=np.float64
    )

    assert pearson(img1, img2) == -1.0


def test_mask_array_length_must_match_images():
    mask = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype=np.uint8)
    img1 = np.array([[255, 255, 255], [255, 255, 255]], dtype=np.uint8)
    img2 = np.array([[125, 125, 125], [125, 125, 125]], dtype=np.uint8)

    with pytest.raises(BaseException):
        pearson(img1, img2, mask)


def test_allows_mask_for_background_subtraction():
    mask = np.array(
        [[0, 0, 0, 0, 0], [0, 255, 255, 255, 0], [0, 0, 0, 0, 0]], dtype=np.uint8
    )

    img1 = np.array(
        [[100, 25, 50, 33, 25], [57, 255, 200, 255, 33], [237, 21, 3, 45, 57]],
        dtype=np.uint8,
    )
    img2 = np.array(
        [[23, 25, 75, 28, 121], [13, 255, 200, 255, 44], [121, 11, 25, 88, 44]],
        dtype=np.uint8,
    )

    assert pearson(img1, img2, mask) == 1.0


def test_returns_nan_when_standard_deviation_of_either_image_is_zero() -> None:
    img1 = np.array([25, 25, 25, 25], dtype=np.uint8)
    img2 = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

    assert np.isnan(pearson(img1, img2))

    img1 = np.array([10, 25, 35, 45, 55], dtype=np.uint8)
    img2 = np.array([200, 200, 200, 200, 200], dtype=np.uint8)
    assert np.isnan(pearson(img1, img2))

