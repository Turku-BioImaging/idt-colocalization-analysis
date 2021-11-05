import os
import pandas as pd
from glob import glob
from skimage import io

import functions

# read images and masks
first_img_paths = sorted(glob("data/first_images/*"))
second_img_paths = sorted(glob("data/second_images/*"))
mask_paths = sorted(glob("data/masks/*"))

results = []

# loop through images
for idx, p in enumerate(first_img_paths):
    first_img = io.imread(first_img_paths[idx])
    second_img = io.imread(second_img_paths[idx])
    mask = io.imread(mask_paths[idx])

    pearson_r = functions.pearson(first_img, second_img, mask=mask)
    first_img_fname = os.path.basename(p)
    second_img_fname = os.path.basename(second_img_paths[idx])
    result = {
        "first_image_fname": first_img_fname,
        "second_image_fname": second_img_fname,
        "pearson_r": pearson_r,
    }

    results.append(result)

print(results)
