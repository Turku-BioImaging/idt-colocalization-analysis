import numpy as np
from skimage import img_as_bool


def manders(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None):
    if img1.shape != img2.shape:
        raise BaseException("Images must have the same dimensions.")

    if img1.dtype != "uint8" or img2.dtype != "uint8":
        raise BaseException("Images must be binary 8-bit numpy arrays.")

    if mask is not None:
        if mask.dtype != "uint8" or isinstance(mask, np.ndarray) == False:
            raise BaseException("Mask must be a binary 8-bit numpy array.")

        img1[~img_as_bool(mask)] = 0
        img2[~img_as_bool(mask)] = 0

    cooccurrence_mask = img1 & img2
    m1 = np.sum(cooccurrence_mask) / np.sum(img1)
    m2 = np.sum(cooccurrence_mask) / np.sum(img2)

    return (m1, m2, cooccurrence_mask)

