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

