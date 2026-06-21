#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
color_extraction.py

Function:
    Extracts dominant colors from raw book cover images and maps them to standard CSS colors.

Core Idea:
    1. Reads raw images from the input directory.
    2. Extracts all pixel RGB values and counts frequencies.
    3. Maps each extracted RGB color to the closest standard CSS4 color (147 colors)
       using the CIELAB Delta-E color difference formula to ensure perceptual accuracy.
    4. Aggregates color counts per image and calculates the usage ratio.
    5. Filters out insignificant colors (ratio < 1%).

    *Note: The aggregation from 147 CSS colors to the final 24 representative clusters
    is handled in the downstream analysis steps using the mapping files.*

Outputs:
    - image_colors.csv: Raw extracted RGB data.
    - image_colors_duplicated.csv: De-duplicated raw RGB data.
    - image_csccolors.csv: Colors mapped to CSS names.
    - image_csccolors_group.csv: Final grouped colors with ratios for each image.
    - image_csccolors_group2.csv: One row per image with semicolon-separated significant colors.

Dependencies:
    numpy, pandas, Pillow, colormath, tqdm
"""

import os
import csv
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import warnings
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="colormath")

# Fix for numpy.asscalar deprecation (compatibility for older numpy versions if needed)
if not hasattr(np, 'asscalar'):
    def asscalar(a):
        return a.item() if a.size == 1 else a


    np.asscalar = asscalar


def process_image(file_info):
    """
    Optimized image processing to extract RGB colors.
    Returns: List of [filename, R, G, B, count, ratio].
    """
    filename, image_folder, _ = file_info  # output_csv is not used directly here
    filepath = os.path.join(image_folder, filename)

    try:
        with Image.open(filepath) as img:
            img = img.convert('RGB')

            # Use numpy for efficient statistics
            arr = np.array(img)
            pixels = arr.reshape(-1, 3)
            total_pixels = pixels.shape[0]

            # Use np.unique instead of Counter
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

            results = []
            for color, count in zip(unique_colors, counts):
                r, g, b = color
                ratio = (count / total_pixels) * 100
                results.append([filename, r, g, b, count, round(ratio, 3)])

            return results
    except Exception as e:
        print(f"\nError processing file {filename}: {e}")
        return []


def find_closest_css_color(rgb, css_colors_lab):
    """
    Finds the closest CSS color for a single RGB tuple using parallel processing.
    Uses CIELAB Delta-E 2000 for perceptual distance.
    """
    r, g, b = rgb
    input_color = convert_color(sRGBColor(r, g, b, is_upscaled=True), LabColor)
    min_dist = float('inf')
    closest_name = None

    for name, css_lab in css_colors_lab.items():
        dist = delta_e_cie2000(input_color, css_lab)
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name


def process_category_extraction(category_path, output_root, css_english_china_path, pool_size=0):
    """
    Processes a single category of images: Extract -> Map -> Group.
    """
    category_name = os.path.basename(category_path)
    print(f"\n{'=' * 50}")
    print(f"Start processing category: {category_name}")
    print(f"{'=' * 50}")

    # Create output directory
    output_category = os.path.join(output_root, category_name)
    os.makedirs(output_category, exist_ok=True)

    # ---------------------------------------------------------
    # Step 1: Extract Image Colors
    # ---------------------------------------------------------
    output_csv = os.path.join(output_category, 'image_colors.csv')

    # Get image list
    image_files = [f for f in os.listdir(category_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print(f"No image files found in {category_path}, skipping.")
        return

    print(f"Found {len(image_files)} images to process.")

    # Prepare parallel tasks
    file_infos = [(f, category_path, output_csv) for f in image_files]

    # Initialize CSV with header
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'R', 'G', 'B', 'count', 'ratio (%)'])

    # Process images (Parallel or Single)
    start_time = time.time()
    if pool_size > 1:
        with mp.Pool(processes=pool_size) as pool:
            results = list(tqdm(pool.imap_unordered(process_image, file_infos),
                                total=len(image_files),
                                desc=f"Processing images for {category_name}"))
    else:
        results = []
        for file_info in tqdm(file_infos, desc=f"Processing images for {category_name}"):
            results.append(process_image(file_info))

    # Write results
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for result in results:
            if result:
                writer.writerows(result)

    duration = time.time() - start_time
    print(f"\nImage processing complete! Time: {duration:.2f}s")
    print(f"Saved raw color data to: {output_csv}")

    # ---------------------------------------------------------
    # Step 2: De-duplication
    # ---------------------------------------------------------
    print("\nReading CSV for de-duplication...")
    df = pd.read_csv(output_csv)
    print(f"Original shape: {df.shape}")

    print("De-duplicating...")
    df.drop_duplicates(subset=['filename', 'R', 'G', 'B'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Shape after de-duplication: {df.shape}")

    output_dup = os.path.join(output_category, 'image_colors_duplicated.csv')
    df.to_csv(output_dup, index=False)
    print(f"Saved de-duplicated results to: {output_dup}")

    # ---------------------------------------------------------
    # Step 3: Map to CSS Colors
    # ---------------------------------------------------------
    output_csv1 = os.path.join(output_category, 'image_csccolors.csv')

    print(f"\nReading CSS mapping file: {css_english_china_path}")
    df_english_china = pd.read_csv(css_english_china_path)

    print("Preparing CSS color space (Lab conversion)...")
    css_colors = {name: mcolors.to_rgb(hex_val) for name, hex_val in mcolors.CSS4_COLORS.items()}
    css_colors_rgb = {name: tuple(int(255 * c) for c in rgb) for name, rgb in css_colors.items()}

    # Convert standard CSS colors to Lab for distance calculation
    css_colors_lab = {}
    for name, (r, g, b) in tqdm(css_colors_rgb.items(), desc="Converting CSS colors to Lab", unit="color"):
        css_colors_lab[name] = convert_color(sRGBColor(r, g, b, is_upscaled=True), LabColor)

    css_color_zh = dict(zip(df_english_china['英文名'], df_english_china['中文名']))

    print(f"\nMapping {len(df)} extracted colors to CSS colors...")
    start_time = time.time()

    rgb_tuples = list(zip(df['R'], df['G'], df['B']))

    # Partial function to fix the css_colors_lab argument
    find_closest_partial = partial(find_closest_css_color, css_colors_lab=css_colors_lab)

    if pool_size > 1:
        with mp.Pool(processes=pool_size) as pool:
            results = list(tqdm(pool.imap(find_closest_partial, rgb_tuples),
                                total=len(rgb_tuples),
                                desc="Mapping colors"))
    else:
        results = []
        for rgb in tqdm(rgb_tuples, desc="Mapping colors"):
            results.append(find_closest_css_color(rgb, css_colors_lab))

    df['css_color_en'] = results
    df['css_color_zh'] = df['css_color_en'].apply(lambda x: css_color_zh.get(x, x))

    duration = time.time() - start_time
    print(f"Color mapping complete! Time: {duration:.2f}s")

    df.to_csv(output_csv1, index=False)
    print(f"Saved mapped colors to: {output_csv1}")

    # ---------------------------------------------------------
    # Step 4: Aggregate and Group
    # ---------------------------------------------------------
    print("\nAggregating table...")
    df = pd.read_csv(output_csv1)

    print("Calculating color ratios...")
    df_total = df.groupby(by=['filename', 'css_color_en'], as_index=False)['count'].sum()
    df_total_ = df.groupby(by=['filename'], as_index=False)['count'].sum()
    df_total_.rename(columns={'count': 'count_sum'}, inplace=True)
    df_total = df_total.merge(right=df_total_, how='left', on=['filename'])
    df_total['rate'] = df_total['count'] / df_total['count_sum']
    df_total['rate'] = df_total['rate'].apply(lambda x: round(x, 6))

    # Filter by threshold
    color_rate = 0.01
    print(f"Filtering colors with ratio >= {color_rate * 100}% ...")
    df_total = df_total[df_total['rate'] >= color_rate]

    df_total.reset_index(drop=True, inplace=True)
    df_total.sort_values(by=['filename', 'rate'], ascending=[True, False], inplace=True)
    df_total.reset_index(drop=True, inplace=True)

    output_group = os.path.join(output_category, 'image_csccolors_group.csv')
    df_total.to_csv(output_group, index=False)
    print(f"Saved grouped results to: {output_group}")

    # ---------------------------------------------------------
    # Step 5: One-row-per-image color list for downstream stats
    # ---------------------------------------------------------
    df_group2 = (
        df_total.groupby('filename')['css_color_en']
        .agg(lambda values: '; '.join(sorted(set(map(str, values)))))
        .reset_index()
    )
    df_group2['color_count'] = df_group2['css_color_en'].apply(
        lambda x: len([c for c in str(x).split(';') if c.strip()])
    )
    output_group2 = os.path.join(output_category, 'image_csccolors_group2.csv')
    df_group2.to_csv(output_group2, index=False)
    print(f"Saved per-image grouped color lists to: {output_group2}")

    print(f"\n{'=' * 50}")
    print(f"Category {category_name} extraction complete!")
    print(f"{'=' * 50}\n")
    return True


# Wrapper for outer parallelization
def process_category_wrapper(args):
    category_path, output_root, css_english_china_path = args
    try:
        # Use single process inside, as outer loop is parallelized
        success = process_category_extraction(category_path, output_root, css_english_china_path, pool_size=0)
        return category_path, success
    except Exception as e:
        print(f"\nError processing category {os.path.basename(category_path)}: {str(e)}")
        return category_path, False


def main():
    # Configure Paths (Relative paths)
    input_root = './data/images'
    output_root = './output/data'
    css_english_china_path = './data/CSS_english_china.csv'

    # Setup Multiprocessing
    num_cores = mp.cpu_count()
    print(f"Detected {num_cores} CPU cores")

    # Limit processes to avoid memory issues
    outer_pool_size = min(num_cores, 48)
    print(f"Using {outer_pool_size} processes for category parallelization\n")

    # Get categories
    categories = []
    if os.path.exists(input_root):
        for root, dirs, files in os.walk(input_root):
            if root != input_root:
                categories.append(root)
        print(f"Found {len(categories)} categories to process.")
    else:
        print(f"Input directory not found: {input_root}")
        return

    os.makedirs(output_root, exist_ok=True)

    # Prepare arguments
    args_list = [(cat, output_root, css_english_china_path) for cat in categories]

    # Run extraction
    start_time = time.time()
    with mp.Pool(processes=outer_pool_size) as outer_pool:
        results = list(tqdm(outer_pool.imap_unordered(process_category_wrapper, args_list),
                            total=len(categories),
                            desc="Processing all categories"))

    duration = time.time() - start_time
    print(f"\nAll extraction tasks complete! Total time: {duration:.2f}s")


if __name__ == '__main__':
    main()
