#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
community_centrality_analysis.py

Function:
    Performs community detection and layout calculation for the 24-color clustered network.

Core Idea:
    1. Loads the co-occurrence network and applies the Louvain algorithm to detect communities.
    2. Aggregates colors into the 24 representative clusters defined in the study.
    3. Calculates node positions (X, Y) using force-directed layout algorithms (Spring Layout)
       weighted by edge frequency.
    4. Exports structured data for external visualization tools (e.g., Gephi) or plotting.

Outputs:
    - nodes_with_community.csv: Contains Node ID, Label, Community ID, Spatial Coordinates, and Degree.
    - edges_with_community.csv: Contains Source, Target, and Weight for the edges.

Dependencies:
    pandas, networkx, python-louvain, sklearn
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


def create_unified_color_mapping(cluster_file):
    """Creates a mapping from raw colors to 24 aggregated colors."""
    cluster_df = pd.read_csv(cluster_file)
    unified_mapping = {}

    for _, row in cluster_df.iterrows():
        cluster_hex = row['聚合颜色Hex'].strip().upper()
        if pd.notna(row['原始颜色Hex']):
            unified_mapping[row['原始颜色Hex'].strip().upper()] = cluster_hex
        if pd.notna(row['原始颜色英文']):
            unified_mapping[row['原始颜色英文'].strip().lower()] = cluster_hex

    for hex_val in set(unified_mapping.values()):
        unified_mapping[hex_val] = hex_val

    return unified_mapping


def replace_colors(data, mapping):
    if isinstance(data, str):
        key = data.strip()
        if key.upper() in mapping: return mapping[key.upper()]
        if key.lower() in mapping: return mapping[key.lower()]
    return data


def analyze_community_structure(threelist_path, group_path, mapping, output_dir):
    """Performs Louvain community detection and exports node/edge lists."""
    print(f"Analyzing community structure for: {output_dir}")

    threelist_df = pd.read_csv(threelist_path)
    group_df = pd.read_csv(group_path)

    threelist_df['from'] = threelist_df['from'].apply(lambda x: replace_colors(x, mapping))
    threelist_df['to'] = threelist_df['to'].apply(lambda x: replace_colors(x, mapping))
    group_df['css_color_en'] = group_df['css_color_en'].apply(lambda x: replace_colors(x, mapping))

    threelist_df = threelist_df[threelist_df['from'] != threelist_df['to']]
    aggregated_edges = threelist_df.groupby(['from', 'to'])['weight'].sum().reset_index()

    G = nx.Graph()
    for _, row in aggregated_edges.iterrows():
        G.add_edge(row['from'], row['to'], weight=row['weight'])

    partition = community_louvain.best_partition(G, resolution=1.0, random_state=42)
    pos = nx.spring_layout(G, k=0.5, weight='weight', seed=42)

    node_data = []
    for node in G.nodes():
        node_data.append({
            'Id': node,
            'Label': node,
            'Community': partition.get(node, 0),
            'X': pos[node][0],
            'Y': pos[node][1],
            'Degree': G.degree(node, weight='weight')
        })
    pd.DataFrame(node_data).to_csv(os.path.join(output_dir, 'nodes_with_community.csv'), index=False)

    edge_data = []
    for u, v, d in G.edges(data=True):
        edge_data.append({
            'Source': u,
            'Target': v,
            'Weight': d.get('weight', 1),
            'Type': 'Undirected'
        })
    pd.DataFrame(edge_data).to_csv(os.path.join(output_dir, 'edges_with_community.csv'), index=False)

    print(f"Exported community nodes and edges to {output_dir}")


if __name__ == '__main__':
    OUTPUT_ROOT = './output/data'
    MAPPING_FILE = './data/CSS_english_china_clustered.csv'

    if os.path.exists(MAPPING_FILE):
        mapping = create_unified_color_mapping(MAPPING_FILE)

        if os.path.exists(OUTPUT_ROOT):
            categories = [d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))]
            for cat in categories:
                cat_dir = os.path.join(OUTPUT_ROOT, cat)
                threelist = os.path.join(cat_dir, 'image_csccolors_threelist.csv')
                group = os.path.join(cat_dir, 'image_csccolors_group.csv')

                if os.path.exists(threelist) and os.path.exists(group):
                    analyze_community_structure(threelist, group, mapping, cat_dir)
