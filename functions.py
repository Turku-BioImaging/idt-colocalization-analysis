import numpy as np
from scipy import stats
from skimage import io, img_as_ubyte, img_as_uint, img_as_bool
from skimage.filters import threshold_otsu
import os


def pearson(img1, img2, mask=None):
    """
    Calculates Pearson's coefficient for two images.

    Parameters
    ----------
    img1: n-dimensional ndarray
        The first image
    img2: n-dimensional ndarray
        The second image
        
    Returns
    -------
    rho
        The resulting Pearson's coefficient.
    """

    img1 = img1.reshape(-1).ravel()
    img2 = img2.reshape(-1).ravel()
    mask = mask.reshape(-1).ravel()

    img1_mask_indices = np.nonzero(mask)
    img2_mask_indices = np.nonzero(mask)

    # use only image data within mask
    img1_m = img1[img1_mask_indices].reshape(-1)
    img2_m = img2[img2_mask_indices].reshape(-1)

    return np.corrcoef((img1_m, img2_m))[0][1]


def manders_otsu(img1, img2, mask=None):

    # try with otsu threshold
    img1_threshold = threshold_otsu(img1)
    img2_threshold = threshold_otsu(img2)

    if mask is not None:
        img1[~img_as_bool(mask)] = 0
        img2[~img_as_bool(mask)] = 0

    binary_otsu_img1 = img1 > img1_threshold
    binary_otsu_img2 = img2 > img2_threshold

    # if mask is not None:
    #     img1[~img_as_bool(mask)] = 0
    #     img2[~img_as_bool(mask)] = 0

    # binary_otsu_img1[~img_as_bool(mask)] = 0
    # binary_otsu_img2[~img_as_bool(mask)] = 0

    binary_combined = binary_otsu_img1 & binary_otsu_img2

    m1 = np.sum(binary_combined) / np.sum(binary_otsu_img1)
    m2 = np.sum(binary_combined) / np.sum(binary_otsu_img2)

    return (m1, m2, binary_otsu_img1, binary_otsu_img2)

