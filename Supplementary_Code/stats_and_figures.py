#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stats_and_figures.py
Generates statistical visualizations and correlation analysis for the analyzed
network data.

Core idea:
- Loads processed color and network data under ./output/data/<category>.
- Plots category color-ratio stacked bars, with and without the global baseline.
- Saves the underlying category/global ratio table.
- Plots overall histograms of the number of aggregated colors used per book cover.
- Plots per-category node-level correlation between degree and clustering.

Outputs:
- category_color_ratios.png
- category_color_ratios_with_global.png
- category_color_ratios_data_with_global.csv
- overall_color_ratio_histogram_linear.jpg
- overall_color_ratio_histogram_log.jpg
- <category>/correlation_degree_vs_clustering.jpg

Dependencies: pandas, matplotlib, numpy
"""

import os
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_unified_color_mapping(cluster_file):
    """Creates a mapping from raw colors/CSS names to 24 aggregated color Hex values."""
    cluster_df = pd.read_csv(cluster_file)
    unified_mapping = {}
    hex_to_en = {}
    hex_to_cn = {}

    for _, row in cluster_df.iterrows():
        cluster_hex = str(row['聚合颜色Hex']).strip().upper()
        cluster_en = str(row['聚合颜色英文']).strip() if pd.notna(row.get('聚合颜色英文')) else cluster_hex
        cluster_cn = str(row['聚合颜色中文']).strip() if '聚合颜色中文' in row and pd.notna(row.get('聚合颜色中文')) else cluster_en

        hex_to_en[cluster_hex] = cluster_en
        hex_to_cn[cluster_hex] = cluster_cn

        if '原始颜色Hex' in row and pd.notna(row['原始颜色Hex']):
            original_hex = str(row['原始颜色Hex']).strip().upper()
            if original_hex:
                unified_mapping[original_hex] = cluster_hex

        if '原始颜色英文' in row and pd.notna(row['原始颜色英文']):
            original_en = str(row['原始颜色英文']).strip().lower()
            if original_en:
                unified_mapping[original_en] = cluster_hex

        if '原始颜色中文' in row and pd.notna(row['原始颜色中文']):
            original_cn = str(row['原始颜色中文']).strip()
            if original_cn:
                unified_mapping[original_cn] = cluster_hex

    for hex_val in hex_to_en:
        unified_mapping[hex_val] = hex_val

    return unified_mapping, hex_to_en, hex_to_cn


def replace_colors(data, mapping):
    """Maps a color name/Hex into the unified 24-color Hex space."""
    if isinstance(data, str):
        key = data.strip()
        if key.upper() in mapping:
            return mapping[key.upper()]
        if key.lower() in mapping:
            return mapping[key.lower()]
        if key in mapping:
            return mapping[key]
    return data


def _standard_clusters(hex_to_en):
    clusters = list(hex_to_en.keys())
    if len(clusters) > 24:
        clusters = clusters[:24]
    return clusters


def collect_category_color_ratios(data_root, unified_mapping, hex_to_en):
    """Collects per-category and global color ratios from image_csccolors.csv."""
    category_data = {}
    global_data = defaultdict(float)
    total_global_ratio = 0.0

    for category_dir in sorted(os.listdir(data_root)):
        category_path = os.path.join(data_root, category_dir)
        if not os.path.isdir(category_path):
            continue

        csv_path = os.path.join(category_path, 'image_csccolors.csv')
        if not os.path.exists(csv_path):
            logger.warning(f"Missing image_csccolors.csv in {category_dir}; skipped.")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error(f"Error reading {csv_path}: {exc}")
            continue

        if 'css_color_en' not in df.columns or 'ratio (%)' not in df.columns:
            logger.warning(f"{csv_path} missing css_color_en or ratio (%) columns; skipped.")
            continue

        df = df.copy()
        df['css_color_en'] = df['css_color_en'].astype(str)
        df['cluster_color'] = df['css_color_en'].apply(lambda x: replace_colors(x, unified_mapping))
        df['ratio (%)'] = pd.to_numeric(df['ratio (%)'], errors='coerce').fillna(0)

        total_ratio_sum = float(df['ratio (%)'].sum())
        if total_ratio_sum <= 0:
            logger.warning(f"Category {category_dir} has zero total ratio; skipped.")
            continue

        grouped = df.groupby('cluster_color')['ratio (%)'].sum()
        category_data[category_dir] = (grouped / total_ratio_sum).to_dict()

        for cluster, value in grouped.items():
            global_data[cluster] += float(value)
        total_global_ratio += total_ratio_sum

    if not category_data:
        return {}, {}, []

    global_ratios = {cluster: value / total_global_ratio for cluster, value in global_data.items()}
    standard_clusters = _standard_clusters(hex_to_en)
    return category_data, global_ratios, standard_clusters


def plot_category_color_ratios(data_root, output_dir, unified_mapping, hex_to_en):
    """
    Generates category_color_ratios.png, category_color_ratios_with_global.png,
    and category_color_ratios_data_with_global.csv.
    """
    logger.info("Generating category color-ratio charts...")
    os.makedirs(output_dir, exist_ok=True)

    category_data, global_ratios, standard_clusters = collect_category_color_ratios(
        data_root, unified_mapping, hex_to_en
    )
    if not category_data:
        logger.error("No usable category color-ratio data found.")
        return False

    full_category_data = {}
    for category, ratios in category_data.items():
        full = {cluster: ratios.get(cluster, 0.0) for cluster in standard_clusters}
        full_category_data[category] = sorted(full.items(), key=lambda x: x[1], reverse=True)

    full_global = {cluster: global_ratios.get(cluster, 0.0) for cluster in standard_clusters}
    sorted_global = sorted(full_global.items(), key=lambda x: x[1], reverse=True)

    def sort_key(category):
        return tuple(-ratio for _, ratio in full_category_data[category])

    sorted_categories = sorted(full_category_data.keys(), key=sort_key)

    chart_specs = [
        (False, sorted_categories, 'category_color_ratios.png',
         'Distribution of Aggregated Color Ratios in Each Category'),
        (True, ['allbooks'] + sorted_categories, 'category_color_ratios_with_global.png',
         'Distribution of Aggregated Color Ratios in Each Category (with Global Data)'),
    ]

    for include_global, categories_list, filename, title in chart_specs:
        plt.figure(figsize=(20, len(categories_list) * 0.6 + 2), dpi=150)
        bar_height = 0.6

        for i, category in enumerate(categories_list):
            ratios = sorted_global if category == 'allbooks' else full_category_data[category]
            left = 0.0
            for cluster, ratio in ratios:
                if ratio > 0:
                    plt.barh(i, ratio, left=left, height=bar_height,
                             color=cluster, edgecolor='white', linewidth=0.5)
                    left += ratio
            plt.text(-0.02, i, category, ha='right', va='center', fontsize=12, fontweight='bold')

        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, fc=cluster, ec='white',
                          label=f"{hex_to_en.get(cluster, 'Unknown')} ({cluster})")
            for cluster in standard_clusters
        ]
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize=8, title='Aggregated Colors')
        plt.title(title, fontsize=18, pad=20)
        plt.xlabel('Color Ratio', fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(-0.5, len(categories_list) - 0.5)
        plt.yticks([])
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {out_path}")

    rows = []
    data_with_global = {'allbooks': sorted_global, **full_category_data}
    for category, ratios in data_with_global.items():
        for cluster_hex, ratio in ratios:
            rows.append({
                'category': category,
                'cluster_color_hex': cluster_hex,
                'cluster_color_en': hex_to_en.get(cluster_hex, 'Unknown'),
                'ratio': ratio,
            })

    data_path = os.path.join(output_dir, 'category_color_ratios_data_with_global.csv')
    pd.DataFrame(rows).sort_values(['category', 'ratio'], ascending=[True, False]).to_csv(
        data_path, index=False, encoding='utf-8-sig'
    )
    logger.info(f"Saved: {data_path}")
    return True


def collect_group2_data(data_root):
    """
    Collects image_csccolors_group2.csv across categories.
    If group2 is missing, derives it from image_csccolors_group.csv.
    """
    frames = []
    for category in sorted(os.listdir(data_root)):
        category_path = os.path.join(data_root, category)
        if not os.path.isdir(category_path):
            continue

        group2_path = os.path.join(category_path, 'image_csccolors_group2.csv')
        group_path = os.path.join(category_path, 'image_csccolors_group.csv')

        try:
            if os.path.exists(group2_path):
                df = pd.read_csv(group2_path)
            elif os.path.exists(group_path):
                base = pd.read_csv(group_path)
                if not {'filename', 'css_color_en'} <= set(base.columns):
                    continue
                df = (base.groupby('filename')['css_color_en']
                      .agg(lambda values: '; '.join(sorted(set(map(str, values)))))
                      .reset_index())
            else:
                continue
        except Exception as exc:
            logger.error(f"Error collecting group2 data for {category}: {exc}")
            continue

        if 'css_color_en' not in df.columns:
            continue
        df = df.copy()
        df['category'] = category
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=['filename', 'css_color_en', 'category'])
    return pd.concat(frames, ignore_index=True)


def _mapped_color_count(color_string, unified_mapping):
    if pd.isna(color_string):
        return 0
    original_colors = [c.strip() for c in str(color_string).split(';') if c.strip()]
    mapped_colors = [replace_colors(c, unified_mapping) for c in original_colors]
    return len(set(mapped_colors))


def plot_color_ratio_histograms(group2_df, output_dir, unified_mapping):
    """Generates overall_color_ratio_histogram_linear.jpg and overall_color_ratio_histogram_log.jpg."""
    if group2_df.empty:
        logger.warning("No group2 data found; skipped overall color-ratio histograms.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    df = group2_df.copy()
    df['mapped_color_count'] = df['css_color_en'].apply(lambda x: _mapped_color_count(x, unified_mapping))
    ratio_series = df['mapped_color_count'].value_counts(normalize=True).sort_index() * 100

    if ratio_series.empty:
        logger.warning("Color-count ratio series is empty; skipped histograms.")
        return False

    for use_log_scale, filename, title in [
        (False, 'overall_color_ratio_histogram_linear.jpg',
         'Distribution of Color Counts Used on Book Covers'),
        (True, 'overall_color_ratio_histogram_log.jpg',
         'Distribution of Color Counts Used on Book Covers (Log Scale)'),
    ]:
        plt.figure(figsize=(14, 8), dpi=120)
        ax = plt.gca()
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ratio_series)))
        bars = ax.bar(ratio_series.index, ratio_series.values, color=colors, edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            if height <= 0:
                continue
            y = height * 1.15 if use_log_scale else height + 0.1
            ax.text(bar.get_x() + bar.get_width() / 2., y, f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

        plt.title(title, fontsize=18, pad=20, fontweight='bold')
        plt.xlabel('Number of Colors Used', labelpad=12, fontsize=14)
        plt.ylabel('Percentage of Books (%)' + (' (Log Scale)' if use_log_scale else ''),
                   labelpad=12, fontsize=14)
        max_x = int(ratio_series.index.max())
        plt.xticks(range(0, max_x + 2, 2))
        if use_log_scale:
            ax.set_yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight', format='jpg')
        plt.close()
        logger.info(f"Saved: {out_path}")

    return True


def plot_correlation_scatter(node_metrics_file, output_dir):
    """Generates correlation_degree_vs_clustering.jpg for one category."""
    if not os.path.exists(node_metrics_file):
        return False

    df = pd.read_csv(node_metrics_file)
    if 'degree' not in df.columns or 'clustering' not in df.columns:
        logger.warning(f"{node_metrics_file} missing degree or clustering; skipped.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['degree'], df['clustering'], alpha=0.6)
    plt.title('Degree vs Clustering')
    plt.xlabel('degree')
    plt.ylabel('clustering')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'correlation_degree_vs_clustering.jpg')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', format='jpg')
    plt.close()
    logger.info(f"Saved: {out_path}")
    return True


def main():
    DATA_ROOT = './output/data'
    OUTPUT_FIGS = './output/figures'
    MAPPING_FILE = './data/CSS_english_china_clustered.csv'

    os.makedirs(OUTPUT_FIGS, exist_ok=True)

    if not os.path.exists(DATA_ROOT):
        logger.error(f"Data root not found: {DATA_ROOT}")
        return
    if not os.path.exists(MAPPING_FILE):
        logger.error(f"Mapping file not found: {MAPPING_FILE}")
        return

    mapping, hex_to_en, _ = create_unified_color_mapping(MAPPING_FILE)

    plot_category_color_ratios(DATA_ROOT, OUTPUT_FIGS, mapping, hex_to_en)

    group2_df = collect_group2_data(DATA_ROOT)
    plot_color_ratio_histograms(group2_df, OUTPUT_FIGS, mapping)

    for cat in sorted(os.listdir(DATA_ROOT)):
        cat_dir = os.path.join(DATA_ROOT, cat)
        if not os.path.isdir(cat_dir):
            continue
        node_file = os.path.join(cat_dir, 'node_feature.csv')
        cat_output = os.path.join(OUTPUT_FIGS, cat)
        plot_correlation_scatter(node_file, cat_output)


if __name__ == '__main__':
    main()
