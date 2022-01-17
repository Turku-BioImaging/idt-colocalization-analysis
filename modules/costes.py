# import multiprocessing as mp
from skimage import img_as_ubyte, img_as_bool
import numpy as np

from scipy import stats


from pearson import pearson


def auto_threshold(
    img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None, scale_max: int = 255
):
    """
    Calculate the Costes automatic threshold for img1 and img2 in a linear fashion. This implementation is based on CellProfiler's measurecolocalization module. If a binary mask is provided, then it is used for background subtraction prior to calculating the threshold.
    """

    if img1.dtype != img2.dtype:
        raise BaseException("Image dtypes are not the same.")

    if mask is not None:
        if mask.dtype != "uint8" or isinstance(mask, np.ndarray) == False:
            raise BaseException("Mask must be a binary 8-bit numpy array.")

        img1[~img_as_bool(mask)] = 0
        img2[~img_as_bool(mask)] = 0

    i_step = 1 / scale_max
    non_zero = (img1 > 0) | (img2 > 0)
    xvar = np.var(img1[non_zero], axis=0, ddof=1)
    yvar = np.var(img2[non_zero], axis=0, ddof=1)

    xmean = np.mean(img1[non_zero], axis=0)
    ymean = np.mean(img2[non_zero], axis=0)

    z = img1[non_zero] + img2[non_zero]
    zvar = np.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + np.sqrt((yvar - xvar) * (yvar - xvar) + 4 * (covar * covar))
    a = num / denom
    b = ymean - a * xmean

    # Start at 1 step above the maximum value
    img_max = max(img1.max(), img2.max())
    i = i_step * ((img_max // i_step) + 1)

    num_true = None
    img1_max = img1.max()
    img2_max = img2.max()

    # Initialise without a threshold
    # costReg = pearson(img1, img2, mask)
    costReg, _ = stats.pearsonr(img1, img2)
    thr_img1_c = i
    thr_img2_c = (a * i) + b
    while i > img1_max and (a * i) + b > img2_max:
        i -= i_step
    while i > i_step:
        thr_img1_c = i
        thr_img2_c = (a * i) + b
        combt = (img1 < thr_img1_c) | (img2 < thr_img2_c)
        try:
            # Only run pearsonr if the input has changed.
            if (positives := np.count_nonzero(combt)) != num_true:
                # costReg = pearson(img1[combt], img2[combt])
                costReg, _ = stats.pearsonr(img1[combt], img2[combt])
                num_true = positives

            if costReg <= 0:
                break
            elif i < i_step * 10:
                i -= i_step
            elif costReg > 0.45:
                # We're way off, step down 10x
                i -= i_step * 10
            elif costReg > 0.35:
                # Still far from 0, step 5x
                i -= i_step * 5
            elif costReg > 0.25:
                # Step 2x
                i -= i_step * 2
            else:
                i -= i_step
        except ValueError:
            break

    # print(thr_img1_c, thr_img2_c)
    return thr_img1_c, thr_img2_c
