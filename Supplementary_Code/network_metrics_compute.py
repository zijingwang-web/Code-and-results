#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
network_metrics_compute.py
Calculates topological network metrics for the generated color co-occurrence graphs.

Core idea:
- Reads the edge lists generated in the previous step.
- Computes node-level centrality metrics (Degree, Betweenness, Closeness, Eigenvector).
- Computes graph-level metrics (Density, Transitivity, Efficiency, Diameter).
- Caches results to optimize performance during batch processing.

Outputs:
- node_feature.csv: Metrics for each individual color node.
- global_network_feature.csv: Summary metrics for the entire category network.

Dependencies: pandas, networkx, tqdm
"""

import pandas as pd
import networkx as nx
import os
import multiprocessing as mp
from tqdm import tqdm

# Global variable for caching metrics during parallel processing
global_metrics = {}


def compute_node_metrics(color):
    """
    Helper to extract pre-computed metrics for a specific node.
    """
    metrics = {}
    metrics['color'] = color
    metrics['degree'] = global_metrics['degree'].get(color, 0)
    metrics['core_number'] = global_metrics['core_number'].get(color, 0)
    metrics['degree_centrality'] = global_metrics['degree_centrality'].get(color, 0)
    metrics['betweenness_centrality'] = global_metrics['betweenness_centrality'].get(color, 0)
    metrics['closeness_centrality'] = global_metrics['closeness_centrality'].get(color, 0)
    metrics['eigenvector_centrality'] = global_metrics['eigenvector_centrality'].get(color, 0)
    metrics['clustering'] = global_metrics['clustering'].get(color, 0)
    return metrics


def calculate_metrics_for_category(category_dir):
    """
    Calculates both node-level and global network metrics.
    """
    threelist_path = os.path.join(category_dir, 'image_csccolors_threelist.csv')
    if not os.path.exists(threelist_path):
        print(f"Threelist not found in {category_dir}, skipping.")
        return

    print(f"Calculating metrics for {os.path.basename(category_dir)}...")
    df = pd.read_csv(threelist_path)
    G = nx.from_pandas_edgelist(df, 'from', 'to', 'weight', create_using=nx.Graph())

    # 1. Compute Node Metrics
    global global_metrics
    try:
        global_metrics = {
            'degree': dict(nx.degree(G)),
            'core_number': nx.core_number(G),
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'closeness_centrality': nx.closeness_centrality(G),
            'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=1000),
            'clustering': nx.clustering(G)
        }
    except Exception as e:
        print(f"Error computing centrality: {e}")
        return

    all_nodes = list(G.nodes())
    results = [compute_node_metrics(node) for node in all_nodes]

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(category_dir, 'node_feature.csv'), index=False)
    print(f"Saved node features.")

    # 2. Compute Global Metrics
    global_features = {
        'number_of_nodes': G.number_of_nodes(),
        'number_of_edges': G.number_of_edges(),
        'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'transitivity': nx.transitivity(G),
        'density': nx.density(G),
        'average_clustering': nx.average_clustering(G),
        'local_efficiency': nx.local_efficiency(G),
        'global_efficiency': nx.global_efficiency(G),
        'connected_component': nx.number_connected_components(G)
    }

    try:
        global_features['diameter'] = nx.diameter(G)
        global_features['average_shortest_path_length'] = nx.average_shortest_path_length(G)
    except:
        global_features['diameter'] = -1
        global_features['average_shortest_path_length'] = -1

    df_global = pd.DataFrame([global_features])
    df_global.to_csv(os.path.join(category_dir, 'global_network_feature.csv'), index=False)
    print(f"Saved global features.")


if __name__ == '__main__':
    OUTPUT_ROOT = './output/data'

    if os.path.exists(OUTPUT_ROOT):
        categories = [os.path.join(OUTPUT_ROOT, d) for d in os.listdir(OUTPUT_ROOT)
                      if os.path.isdir(os.path.join(OUTPUT_ROOT, d))]

        for cat_dir in categories:
            calculate_metrics_for_category(cat_dir)
