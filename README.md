# IDAT Colocalization Analysis

This Docker container performs classic image colocalization analysis including Pearson correlation and Manders overlap coefficients. It accepts masks for background subtraction. Thresholding methods can be either Otsu or Costes.

## Inputs

Expects three directories:  
`first_images` should contain the first set of images (in alphabetical order) to be compared.  
`second_images` contains the set of images (in alphabetical order) against which the first set will be compared.  
`masks` (Optional) contains the masks used for background subtraction.

## Outputs

Analysis results are output to a CSV file in the `results` directory.

## Usage

The default usage calculates Pearson correlation, and Manders overlap coefficients using Otsu thresholding.

```
docker run -it \
    -v "$(pwd)<first_images_dir>:/code/data/first_images" \
    -v "$(pwd)<second_images_dir>:/code/data/second_images" \
    -v "$(pwd)<masks_dir>:/code/data/masks" \
    -v "$(pwd)<results_dir>:/code/results" \
    ghcr.io/turku-bioimaging/idt-colocalization-analysis:0.3.0
```

Disable masks for background subtraction:

```
    ...
    ghcr.io/turku-bioimaging/idt-colocalization-analysis:0.3.0 --disable-masks
```

Disable Pearson correlation:

```
    ...
    ghcr.io/turku-bioimaging/idt-colocalization-analysis:0.3.0 --disable-pearson
```

Disable Otsu thresholding for Manders overlap coefficients:

```
    ...
    ghcr.io/turku-bioimaging/idt-colocalization-analysis:0.3.0 --disable-manders-otsu
```

Enable Costes auto thresholding for Manders overlap coefficients:

```
    ...
    ghcr.io/turku-bioimaging/idt-colocalization-analysis:0.3.0 --manders-costes
```

## Credits

**Image Data Team - Turku BioImaging**  
Joanna Pylvänäinen - joanna.pylvanainen@abo.fi  
Junel Solis - junel.solis@abo.fi  
Dado Tokic - dado.tokic@abo.fi  
Pasi Kankaanpää - pasi.kankaanpaa@abo.fi

**Turku BioImaging**  
[https://bioimaging.fi](https://bioimaging.fi)
