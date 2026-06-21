This folder contains the source code and data processing logic for extracting
and mapping dominant colors from raw book cover images.

File description:

1. color_extraction.py
   This script implements the initial stage of the color analysis pipeline,
   focusing on extracting raw pixel data and mapping it to a standardized
   color space.

   Core methodology:
   - Pixel Extraction: Iterates through image files to extract all unique RGB
     pixel values and calculates their frequency distribution.
   - Perceptual Mapping: Maps each extracted RGB value to the closest standard
     CSS4 color (147 colors) using the CIELAB Delta-E 2000 formula. This ensures
     that color categorization aligns with human visual perception.
   - Data Aggregation: grouping raw pixel counts by their mapped CSS color names
     and calculating the usage ratio per image.
   - Noise Filtering: Applies a 1% usage threshold to filter out insignificant
     background colors.
   - Optimization: Utilizes multiprocessing to handle large datasets efficiently.

   *Note: This script maps colors to the 147 standard CSS colors. The further
   aggregation into 24 representative clusters occurs in downstream analysis.*

2. image_colors.csv
   The raw dataset containing extracted RGB values, pixel counts, and initial
   usage percentages for every image processed.

3. image_csccolors.csv
   An intermediate dataset where the raw RGB values from the previous step have
   been mapped to their corresponding English and Chinese CSS color names.

4. image_csccolors_group.csv
   The final processed output of this stage. It contains the aggregated counts
   and recalculated ratios for the mapped CSS colors, filtered to include only
   those exceeding the 1% coverage threshold. This file serves as the input
   for subsequent network construction.

The scripts and outputs are provided to ensure the transparency of the data
acquisition process and to support the reproducibility of the color quantification
methods described in the manuscript.
