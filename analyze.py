import os
import pandas as pd
from glob import glob
from skimage import io
from tqdm import tqdm

import functions
from modules.pearson import pearson
from modules.manders import manders

if __name__ == "__main__":
    # read images and masks
    first_img_paths = sorted(glob("data/first_images/*"))
    second_img_paths = sorted(glob("data/second_images/*"))
    mask_paths = sorted(glob("data/masks/*"))

    results = []

    # loop through images
    for idx, p in tqdm(enumerate(first_img_paths), total=len(first_img_paths)):
        first_img = io.imread(first_img_paths[idx])
        second_img = io.imread(second_img_paths[idx])
        mask = io.imread(mask_paths[idx])

        pearson_r = pearson(img1=first_img, img2=second_img, mask=mask)

        (
            otsu_manders_m1,
            otsu_manders_m2, 
            otsu_img1, 
            otsu_img2
        ) = functions.manders_otsu(first_img, second_img, mask=mask)

        (
            costes_m1, 
            costes_m2,
            costes_img1_thresholded, 
            costes_img2_thresholded
        ) = manders(img1=first_img, img2=second_img, mask=mask)


        first_img_fname = os.path.basename(p)
        second_img_fname = os.path.basename(second_img_paths[idx])
        result = {
            "first_image_fname": first_img_fname,
            "second_image_fname": second_img_fname,
            "pearson_r": pearson_r,
            "otsu_manders_m1": otsu_manders_m1,
            "otsu_manders_m2": otsu_manders_m2,
            "costes_manders_m1": costes_m1,
            "costes_manders_m2": costes_m2,
        }

        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("results/colocalization_results.csv")

