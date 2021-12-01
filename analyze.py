import os
import pandas as pd
from glob import glob
from skimage import io
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
    manders_m1, manders_m2, binary_otsu_img1, binary_otsu_img2 = functions.manders(
        first_img, second_img
    )

    first_img_fname = os.path.basename(p)
    second_img_fname = os.path.basename(second_img_paths[idx])
    result = {
        "first_image_fname": first_img_fname,
        "second_image_fname": second_img_fname,
        "pearson_r": pearson_r,
        "manders_m1": manders_m1,
        "manders_m2": manders_m2,
    }

    results.append(result)

    fname_img1 = os.path.basename(first_img_paths[idx]).replace(".tif", "_otsu.tif")
    fname_img2 = os.path.basename(second_img_paths[idx]).replace(".tif", "_otsu.tif")
    os.makedirs("results/otsu", exist_ok=True)
    io.imsave("results/otsu/" + fname_img1, binary_otsu_img1, check_contrast=False)
    io.imsave("results/otsu/" + fname_img2, binary_otsu_img2, check_contrast=False)

df = pd.DataFrame(results)
df.to_csv("results/colocalization_results.csv")
