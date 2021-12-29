import numpy as np

from pearson import pearson


def auto_threshold(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None):
    if isinstance(img1, np.ndarray) == False or isinstance(img2, np.ndarray) == False:
        raise BaseException("Images must be a numpy arrays.")

    if img1.shape != img2.shape:
        raise BaseException("Images must have the same dimensions.")

    threshold, _ = __find_minimum_threshold(img1, img2)

    return (threshold, 5)


def __find_minimum_threshold(img1: np.ndarray, img2: np.ndarray) -> int:
    max_intensity = max([np.max(img1), np.max(img2)])
    threshold_value = max_intensity

    rho = 1.0
    while threshold_value > 0:
        rho = pearson(
            __threshold(img1, threshold_value), __threshold(img2, threshold_value)
        )

        if np.isnan(rho):
            threshold_value -= 1
            continue
        elif rho <= 0.0:
            break
        else:
            threshold_value -= 1

    return (threshold_value, 0)


def __threshold(img: np.ndarray, threshold: int) -> np.ndarray:
    thresholded = (img > threshold).astype(np.uint8)
    return thresholded
