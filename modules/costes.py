from skimage import img_as_ubyte, img_as_bool
import scipy
import numpy as np

def get_thresholds_using_bisection(img1: np.ndarray, img2: np.ndarray):
    """
    Adapted from CellProfiler measurecolocalization module.

    Finds the Costes Automatic Threshold for colocalization using a bisection algorithm.
    Candidate thresholds are selected from within a window of possible intensities,
    this window is narrowed based on the R value of each tested candidate.
    We're looking for the first point below 0, and R value can become highly variable
    at lower thresholds in some samples. Therefore the candidate tested in each
    loop is 1/6th of the window size below the maximum value (as opposed to the midpoint).
    """

    # get max value of either image
    scale_max = max(img1.max(), img2.max())

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

    # Initialise variables
    left = 1
    right = scale_max
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0
    # Marks the value with the last positive R value.
    valid = 1

    while lastmid != mid:
        thr_fi_c = mid / scale_max
        thr_si_c = (a * thr_fi_c) + b
        combt = (img1 < thr_fi_c) | (img2 < thr_si_c)
        if np.count_nonzero(combt) <= 2:
            # Can't run pearson with only 2 values.
            left = mid - 1
        else:
            try:
                costReg, _ = scipy.stats.pearsonr(img1[combt], img2[combt])
                if costReg < 0:
                    left = mid - 1
                elif costReg >= 0:
                    right = mid + 1
                    valid = mid
            except ValueError:
                # Catch misc Pearson errors with low sample numbers
                left = mid - 1
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left

    thr_fi_c = (valid - 1) / scale_max
    thr_si_c = (a * thr_fi_c) + b

    return thr_fi_c, thr_si_c


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

    img1_threshold, img2_threshold = get_thresholds_using_bisection(
        img1, img2, scale_max
    )

    img1_binary = img_as_ubyte(img1 > img1_threshold)
    img2_binary = img_as_ubyte(img2 > img2_threshold)

    return img1_binary, img2_binary
