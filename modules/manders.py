import numpy as np
from skimage import img_as_bool, img_as_ubyte

from .costes import get_thresholds_using_bisection


def manders(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None):
    if img1.shape != img2.shape:
        raise BaseException("Images must have the same dimensions.")

    # if img1.dtype != "uint8" or img2.dtype != "uint8":
    #     raise BaseException("Images must be binary 8-bit numpy arrays.")

    if mask is not None:
        if mask.dtype != "uint8" or isinstance(mask, np.ndarray) == False:
            raise BaseException("Mask must be a binary 8-bit numpy array.")

        img1[~img_as_bool(mask)] = 0
        img2[~img_as_bool(mask)] = 0

    # Costes' thershold calculation
    img1_thr, img2_thr = get_thresholds_using_bisection(img1, img2)

    combined_thresh_c = (img1 > img1_thr) & (img2 > img2_thr)
    img1_thresh_c = img1[combined_thresh_c]
    img2_thresh_c = img2[combined_thresh_c]
    tot_img1_thr_c = img1[(img1 > img1_thr)].sum()
    tot_img2_thr_c = img2[(img2 > img2_thr)].sum()

    # Costes' Automated Threshold
    M1 = 0
    M2 = 0
    M1 = img1_thresh_c.sum() / tot_img1_thr_c
    M2 = img2_thresh_c.sum() / tot_img2_thr_c

    return M1, M2, img_as_ubyte(img1 > img1_thr), img_as_ubyte(img2 > img2_thr)

