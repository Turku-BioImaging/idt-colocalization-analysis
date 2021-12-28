import numpy as np
from skimage import img_as_bool


def pearson(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None):

    if img1.ndim != img2.ndim:
        raise BaseException("Images must have the same dimensions.")

    if img1.ndim > 3 or img2.ndim > 3:
        raise BaseException("Images must have at most 3 dimensions.")

    img1 = img1.flatten()
    img2 = img2.flatten()

    if mask is not None and isinstance(mask, np.ndarray):
        mask = mask.flatten()

        if len(mask) != len(img1):
            raise BaseException(
                "Mask must be a numpy array with the same dimensions as the images."
            )

        img1[~img_as_bool(mask)] = 0
        img2[~img_as_bool(mask)] = 0

    correlation = np.corrcoef(img1, img2)

    return correlation[0][1]
