This folder contains the computational scripts and output datasets for
quantifying the topological properties of the color co-occurrence networks.

File description:

1. network_metrics_compute.py
   This script calculates a comprehensive set of topological metrics for both
   individual nodes and the global network structure using the NetworkX library.

   Key metrics calculated:
   - Node-level: Degree, Weighted Degree, Betweenness Centrality, Closeness
     Centrality, Eigenvector Centrality, and Clustering Coefficient.
   - Network-level: Density, Transitivity, Global/Local Efficiency, Diameter,
     and Average Shortest Path Length.

   The script optimizes performance by caching metrics during batch processing
   of multiple book categories.

2. node_feature.csv
   A detailed table containing centrality and topological metrics for each
   individual color node within the network.

3. global_network_feature.csv
   A summary table presenting the macroscopic properties (e.g., small-world
   characteristics) of the entire category-specific network.

The scripts and outputs are provided to support the quantitative analysis of
network topology reported in the results section of the manuscript.
