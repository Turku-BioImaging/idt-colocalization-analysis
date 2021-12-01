import os
import pandas as pd
from glob import glob
from skimage import io, img_as_ubyte
from tqdm import tqdm

import functions

# read images and masks
first_img_paths = sorted(glob("data/first_images/*"))
second_img_paths = sorted(glob("data/second_images/*"))
mask_paths = sorted(glob("data/masks/*"))

results = []

# loop through images
for idx, p in tqdm(enumerate(first_img_paths)):
    first_img = io.imread(first_img_paths[idx])
    second_img = io.imread(second_img_paths[idx])
    mask = io.imread(mask_paths[idx])

    pearson_r = functions.pearson(first_img, second_img, mask=mask)
    manders_m1, manders_m2, binary_otsu_img1, binary_otsu_img2 = functions.manders_otsu(
        first_img, second_img
    )

    manders_manual_threshold_200 = functions.manders_manual_threshold_200(
        first_img, second_img, mask
    )

    # manders_manual_threshold_300 = functions.manders_manual_threshold_300(
    #     first_img, second_img, mask
    # )

    first_img_fname = os.path.basename(p)
    second_img_fname = os.path.basename(second_img_paths[idx])
    result = {
        "first_image_fname": first_img_fname,
        "second_image_fname": second_img_fname,
        "pearson_r": pearson_r,
        "manders_m1": manders_m1,
        "manders_m2": manders_m2,
        "manual_threshold_200_m1": manders_manual_threshold_200[0],
        "manual_threshold_200_m2": manders_manual_threshold_200[1],
        # "manual_threshold_300_m1": manders_manual_threshold_300[0],
        # "manual_threshold_300_m2": manders_manual_threshold_300[1],
    }

    results.append(result)

    fname_img1 = os.path.basename(first_img_paths[idx]).replace(".tif", "_otsu.tif")
    fname_img2 = os.path.basename(second_img_paths[idx]).replace(".tif", "_otsu.tif")
    os.makedirs("results/otsu", exist_ok=True)
    os.makedirs("results/threshold_200", exist_ok=True)
    os.makedirs("results/threshold_300", exist_ok=True)

    # save threshold masks
    io.imsave(
        "results/otsu/" + fname_img1,
        img_as_ubyte(binary_otsu_img1),
        check_contrast=False,
    )
    io.imsave(
        "results/otsu/" + fname_img2,
        img_as_ubyte(binary_otsu_img2),
        check_contrast=False,
    )

    fname_img1 = fname_img1.replace("_otsu.tif", "_thresh_200.tif")
    fname_img2 = fname_img2.replace("_otsu.tif", "_thresh_200.tif")
    io.imsave(
        "results/threshold_200/" + fname_img1,
        img_as_ubyte(manders_manual_threshold_200[2]),
        check_contrast=False,
    )
    io.imsave(
        "results/threshold_200/" + fname_img2,
        img_as_ubyte(manders_manual_threshold_200[3]),
        check_contrast=False,
    )

    # fname_img1 = fname_img1.replace("thresh_200.tif", "thresh_300.tif")
    # fname_img2 = fname_img2.replace("thresh_200.tif", "thresh_300.tif")
    # io.imsave(
    #     "results/threshold_300/" + fname_img1,
    #     img_as_ubyte(manders_manual_threshold_300[2]),
    # )
    # io.imsave(
    #     "results/threshold_300/" + fname_img2,
    #     img_as_ubyte(manders_manual_threshold_300[3]),
    # )


df = pd.DataFrame(results)
df.to_csv("results/colocalization_results.csv")
