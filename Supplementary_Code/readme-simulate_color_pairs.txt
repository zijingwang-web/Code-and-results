This folder contains the simulation code and outputs used to reproduce
the color-pair selection model reported in the manuscript.

File description:

1. simulate_color_pairs.py
   This script implements a partition-based Gibbs selection model for
   simulating color-pair usage patterns. The model controls selection
   concentration through a temperature parameter (¦Ó) and a compatibility
   matrix (Gamma) between color groups.

   Key adjustable parameters include:
   - Color list and group partition
   - Compatibility matrix (Gamma)
   - Temperature parameter (¦Ó)

2. simulated_pair_probs.csv
   A complete table of simulated color-pair counts and probabilities,
   including color names and hexadecimal codes.

3. figures/
   Contains semi-log plots of the Top-60 color pairs under different
   temperature settings (¦Ó = 0.55, 0.40, 0.33), corresponding to
   Figures discussed in the Supplementary Materials.

The scripts and outputs are provided to support reproducibility and
verification of the simulation results reported in the study.
