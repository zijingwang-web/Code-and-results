This folder contains the code and data outputs for detecting community structures
and calculating spatial layouts within the color co-occurrence network.

File description:

1. community_centrality_analysis.py
   This script performs community detection and force-directed layout calculation
   for the generated color networks.

   Core algorithms and logic:
   - Community Detection: Uses the Louvain algorithm (resolution=1.0) to identify
     modular structures within the weighted graph.
   - Spatial Layout: Applies the Fruchterman-Reingold force-directed algorithm
     (Spring Layout) to determine node positions (X, Y) based on edge weights.
   - Data Aggregation: Maps raw nodes to the 24 representative color clusters
     before processing.

   Key outputs include node coordinates, community assignments, and structured
   edge lists suitable for external visualization software (e.g., Gephi).

2. nodes_with_community.csv
   A comprehensive table of node attributes required for network visualization.
   Columns include:
   - Id / Label: Color identifier.
   - Community: The modularity class ID assigned by the Louvain algorithm.
   - X / Y: Calculated spatial coordinates for network plotting.
   - Degree: Weighted degree of the node.

3. edges_with_community.csv
   A structured edge list containing Source, Target, and Weight, representing
   the strength of co-occurrence between color nodes.

The scripts and outputs are provided to support the structural analysis of
color communities and the reproducibility of the network visualizations
presented in the manuscript.
