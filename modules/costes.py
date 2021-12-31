import multiprocessing as mp
from skimage import img_as_ubyte
import numpy as np

from pearson import pearson


def auto_threshold(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None):
    if isinstance(img1, np.ndarray) == False or isinstance(img2, np.ndarray) == False:
        raise BaseException("Images must be a numpy arrays.")

    if img1.shape != img2.shape:
        raise BaseException("Images must have the same dimensions.")

    threshold = __find_minimum_threshold(img1, img2)
    img1_thresholded = __threshold(img1, threshold)
    img2_thresholded = __threshold(img2, threshold)

    return (threshold, img1_thresholded, img2_thresholded)


def __find_minimum_threshold(img1: np.ndarray, img2: np.ndarray) -> int:
    assert img1.dtype == "uint8" and img2.dtype == "uint8"

    max_intensity = max([np.max(img1), np.max(img2)])
    threshold_value = max_intensity

    candidate_thresholds = []
    while threshold_value > 0:
        if threshold_value == 255:
            threshold_value -= 1
            continue

        rho = pearson(
            __threshold(img1, threshold_value), __threshold(img2, threshold_value)
        )

        if np.isnan(rho):
            threshold_value -= 1
            continue
        elif rho <= 0.0:
            break
        else:
            item = [rho, threshold_value]
            candidate_thresholds.append(item)
            threshold_value -= 1

    minimum_threshold = __get_threshold_from_candidates(candidate_thresholds)

    return minimum_threshold


def __threshold(img: np.ndarray, threshold: int) -> np.ndarray:
    assert img.dtype == "uint8", "Image must be a numpy array of type uint8."
    thresholded = img_as_ubyte(img > threshold)
    return thresholded


def __get_threshold_from_candidates(candidates: list) -> int:
    if len(candidates) == 0:
        return 0
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
