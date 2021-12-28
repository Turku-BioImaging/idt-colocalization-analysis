import numpy as np


def pearson(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None):

    if img1.ndim != img2.ndim:
        raise BaseException("Images must have the same dimensions.")

    if img1.ndim > 3 or img2.ndim > 3:
        raise BaseException("Images must have at most 3 dimensions.")

    img1_flatten = img1.flatten()
    img2_flatten = img2.flatten()

    correlation = np.corrcoef(img1_flatten, img2_flatten)

    return correlation[0][1]
