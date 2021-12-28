import numpy as np

# from numpy import typing


def pearson(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None):
    # print('test')
    if img1.ndim != img2.ndim:
        raise BaseException("Images must have the same dimensions.")

    if img1.ndim == 2:
        img1 = img1.reshape(-1).ravel()
        img2 = img2.reshape(-1).ravel()

    correlation = np.corrcoef(img1, img2)

    return correlation[0][1]
