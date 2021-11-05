import numpy as np
from scipy import stats


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


def manders(img1, img2):
    return
