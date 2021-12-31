import multiprocessing as mp
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
    assert img1.dtype == "uint8" and img2.dtype == "uint8"

    max_intensity = max([np.max(img1), np.max(img2)])
    threshold_value = max_intensity

    # rho = 1.0
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

    return (minimum_threshold, 0)


def __threshold(img: np.ndarray, threshold: int) -> np.ndarray:
    thresholded = (img > threshold).astype(np.uint8)
    return thresholded


def __get_threshold_from_candidates(candidates: list) -> int:
    if len(candidates) == 0:
        return 0
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
