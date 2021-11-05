# IDAT Colocalization Analysis

This container performs classic image colocalization analysis including Pearson correlation and Manders overlap coefficients.

## Inputs

Expects three directories to be provided: `first_images`, `second_images`, and `masks`.  
`first_images` should contain the first set of images (in alphabetical order) to be compared.  
`second_images` contains the set of images (in alphabetical order) against which the first set will be compared.  
`masks` contains the masks used to exclude areas from Pearson correlation.

## Outputs

Analysis results are found in the `results` directory which contains a `results.csv` with Pearson correlation and Manders overlap coefficients.

## How to Run

_This part needs instructions and shell script..._

## Credits

**Image Data Analysis Team - Turku BioImaging**  
Joanna Pylvänäinen - joanna.pylvanainen@abo.fi  
Junel Solis - junel.solis@abo.fi  
Dado Tokic - dado.tokic@abo.fi  
Pasi Kankaanpää - pasi.kankaanpaa@abo.fi

**Turku BioImaging**  
[https://bioimaging.fi](https://bioimaging.fi)
