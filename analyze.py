import os
import argparse
import pandas as pd

from glob import glob
from skimage import io
from tqdm import tqdm

import functions
from modules.pearson import pearson
from modules.manders import manders

IMG_DIR = "data/images"
MASK_DIR = "data/masks"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyze colocalization in images.",
        epilog="Turku BioImaging - Image Data Team - https://bioimaging.fi",
    )

    parser.add_argument(
        "--disable-pearson",
        dest="pearson",
        action="store_false",
        help="Calculate Pearson correlation coefficient",
    )
    parser.add_argument(
        "--disable-manders-otsu",
        dest="manders_otsu",
        action="store_false",
        help="Calculate Manders using Otsu thresholding",
    )
    parser.add_argument(
        "--manders-costes",
        dest="manders_costes",
        action="store_true",
        help="Calculate Manders using Costes auto-thresholding",
    )
    parser.add_argument(
        "--disable-masks",
        dest="with_masks",
        action="store_false",
        help="Use masks to subtract background",
    )

    args = parser.parse_args()
    # read images and masks
    # first_img_paths = sorted(glob("data/first_images/*"))
    # second_img_paths = sorted(glob("data/second_images/*"))
    img_paths = sorted(glob(f"{IMG_DIR}/*"))
    mask_paths = sorted(glob(f"{MASK_DIR}/*"))

    if args.with_masks == True:
        mask_paths = sorted(glob("data/masks/*"))
    else:
        mask_paths = None

    results = []

    # loop through images
    for idx, p in tqdm(enumerate(img_paths), total=len(img_paths)):

        # split the image channels
        img = io.imread(p)
        mask = io.imread(mask_paths[idx])

        first_img = img[:, 0, :, :]
        second_img = img[:, 1, :, :]

        first_img_fname = os.path.basename(p).replace(".tif", "_c1.tif")
        second_img_fname = os.path.basename(p).replace(".tif", "_c2.tif")

        result = {
            "first_image_fname": first_img_fname,
            "second_image_fname": second_img_fname,
        }

        if mask_paths is not None:
            mask = io.imread(mask_paths[idx])
        else:
            mask = None

        if args.pearson == True:
            pearson_r = pearson(img1=first_img, img2=second_img, mask=mask)
            result["pearson_r"] = pearson_r
        if args.manders_otsu == True:
            (
                otsu_manders_m1,
                otsu_manders_m2,
                otsu_img1,
                otsu_img2,
            ) = functions.manders_otsu(first_img, second_img, mask=mask)

            result["otsu_manders_m1"] = otsu_manders_m1
            result["otsu_manders_m2"] = otsu_manders_m2

        if args.manders_costes == True:
            (
                costes_m1,
                costes_m2,
                costes_img1_thresholded,
                costes_img2_thresholded,
            ) = manders(img1=first_img, img2=second_img, mask=mask)

            result["costes_manders_m1"] = costes_m1
            result["costes_manders_m2"] = costes_m2

        results.append(result)

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/colocalization_results.csv")
