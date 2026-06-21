This folder contains the code and resulting figures used for statistical
visualization and correlation analysis of the color networks.

File description:

1. stats_and_figures.py
   This script generates the statistical plots and figures reported in the study.
   It processes the computed network metrics and color usage data to visualize
   trends and relationships.

   Key functionalities:
   - Distribution Analysis: Plots the distribution of color usage ratios across
     different categories compared to a global baseline.
   - Correlation Analysis: Generates scatter plots to examine relationships
     between topological metrics (e.g., Degree vs. Clustering Coefficient).

2. category_color_ratios_with_global.png
   A stacked bar chart illustrating the distribution of the 24 representative
   color clusters across different book categories, including a "Global"
   aggregate for comparison.

3. correlation_{metric_x}_{metric_y}.jpg (e.g., correlation_degree_clustering.jpg)
   Scatter plots showing the statistical correlation between different network
   metrics, used to analyze the structural role of high-frequency colors.

The scripts and outputs are provided to facilitate the reproduction of the
figures and statistical findings discussed in the manuscript.
