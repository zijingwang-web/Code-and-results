This folder contains the data processing pipeline used to construct the color
co-occurrence networks from raw book cover images.

File description:

1. cooccurrence_network_build.py
   This script implements the core pipeline for extracting dominant colors and
   building weighted networks.

   Core methodology:
   - Color Extraction: Iterates through raw images to extract pixel data.
   - Color Mapping: Maps extracted RGB values to 24 representative CSS color
     clusters using the CIELAB Delta-E (CIE2000) distance metric to ensure
     perceptual accuracy.
   - Filtering: Applies a coverage ratio threshold (e.g., 1%) to remove insignificant
     background noise.
   - Edge Construction: Generates a weighted edge list where weights correspond
     to the frequency of color pairs appearing together in the same image.

2. image_csccolors_threelist.csv
   The final weighted edge list used for network construction.
   Columns: from (Color A), to (Color B), weight (Frequency).

3. image_csccolors_group.csv
   An intermediate dataset recording the grouping and coverage ratios of colors
   for each processed image.

4. image_colors.csv / image_csccolors.csv
   Raw and mapped color data tables generated during the extraction process.

The scripts and outputs are provided to demonstrate the data preprocessing
workflow and ensure the transparency of the network construction method.
