#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulate_color_pairs.py
Partition-based Gibbs selection model for color-pair usage simulation.

Core idea (Model A):
- Colors are partitioned into K groups (communities).
- Compatibility between groups is encoded in a KxK matrix Gamma.
- Pair selection probability follows a Gibbs distribution:
    p(i, j) ∝ exp( Gamma[g_i, g_j] / tau )
  (lower tau => more concentrated selection, steeper semi-log rank–probability decay)

Outputs:
- A CSV table of all simulated color-pair probabilities (and counts)
- A semi-log plot of Top-N pairs with a regression line on the top X% (ln(prob) vs rank)

Dependencies: numpy, pandas, matplotlib
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def hex_normalize(x: str) -> str:
    x = str(x).strip()
    if not x:
        return x
    if not x.startswith("#"):
        x = "#" + x
    return x.upper()

def safe_name(x: str) -> str:
    x = str(x)
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in x)

def upper_triangular_pairs(n: int, allow_self: bool = False) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(n):
        j0 = i if allow_self else i + 1
        for j in range(j0, n):
            pairs.append((i, j))
    return pairs

def fit_lnprob_vs_rank(probs_sorted_desc: np.ndarray, top_frac: float = 0.10) -> Tuple[float, float]:
    """
    Fit ln(prob) = a + b * rank over the top top_frac of items (rank starts from 1).
    Returns (slope b, R^2).
    """
    m = max(2, int(len(probs_sorted_desc) * top_frac))
    y = np.log(np.clip(probs_sorted_desc[:m], 1e-300, 1.0))
    x = np.arange(1, m + 1, dtype=float)
    # linear fit
    b, a = np.polyfit(x, y, 1)  # y = b*x + a
    yhat = b * x + a
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(b), float(r2)

@dataclass
class ColorItem:
    name: str
    hex: str
    group: int


# -----------------------------
# Default toy config (override via --colors_csv / --config_json)
# -----------------------------

def default_colors_24_with_groups() -> List[ColorItem]:
    """
    A reasonable 24-color palette placeholder.
    Replace with your actual 24 colors & partitions if you have them.
    """
    palette = [
        ("Black",   "#000000", 0),
        ("White",   "#FFFFFF", 0),
        ("Gray",    "#808080", 0),
        ("Brown",   "#7B4A12", 1),
        ("Beige",   "#E8D7B8", 1),
        ("Red",     "#E31A1C", 2),
        ("Orange",  "#FF7F00", 2),
        ("Yellow",  "#FFD700", 2),
        ("Green",   "#33A02C", 3),
        ("Cyan",    "#00BFC4", 3),
        ("Blue",    "#1F78B4", 4),
        ("Navy",    "#08306B", 4),
        ("Purple",  "#6A3D9A", 5),
        ("Pink",    "#FB9A99", 5),
        ("Gold",    "#D4AF37", 6),
        ("Silver",  "#C0C0C0", 6),
        ("Teal",    "#008080", 3),
        ("Olive",   "#808000", 3),
        ("Maroon",  "#800000", 2),
        ("Magenta", "#FF00FF", 5),
        ("Lavender","#B57EDC", 5),
        ("Cream",   "#FFFDD0", 1),
        ("Tan",     "#D2B48C", 1),
        ("SkyBlue", "#87CEEB", 4),
    ]
    return [ColorItem(name=n, hex=h, group=g) for (n, h, g) in palette]

def default_gamma(K: int) -> np.ndarray:
    """
    Default compatibility matrix Gamma (KxK).
    Larger => more preferred. Symmetric by default.
    You should replace with your estimated/assumed Gamma if available.
    """
    rng = np.random.default_rng(7)
    G = rng.normal(loc=0.0, scale=1.0, size=(K, K))
    G = (G + G.T) / 2.0
    # encourage within-group compatibility slightly
    for k in range(K):
        G[k, k] += 1.0
    return G


# -----------------------------
# Loading configs
# -----------------------------

def load_colors_from_csv(path: str) -> List[ColorItem]:
    """
    CSV must contain columns: name, hex, group
    """
    df = pd.read_csv(path)
    req = {"name", "hex", "group"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"colors_csv missing columns: {missing}. Required: {sorted(req)}")
    items = []
    for _, r in df.iterrows():
        items.append(ColorItem(
            name=str(r["name"]),
            hex=hex_normalize(r["hex"]),
            group=int(r["group"]),
        ))
    return items

def load_config_from_json(path: str) -> Dict:
    """
    JSON schema (minimal):
    {
      "tau": 0.55,
      "allow_self_pairs": false,
      "gamma": [[...],[...],...]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_gamma(colors: List[ColorItem], gamma_from_user: Optional[np.ndarray]) -> np.ndarray:
    groups = sorted({c.group for c in colors})
    group_to_idx = {g:i for i, g in enumerate(groups)}
    K = len(groups)

    if gamma_from_user is None:
        G = default_gamma(K)
    else:
        G = gamma_from_user
        if G.shape != (K, K):
            raise ValueError(f"Gamma shape {G.shape} does not match K={K}. Expected {(K, K)}")

    # Reindex colors' groups into 0..K-1
    for c in colors:
        c.group = group_to_idx[c.group]
    return G


# -----------------------------
# Simulation core
# -----------------------------

def build_pair_weights(colors: List[ColorItem], Gamma: np.ndarray, tau: float,
                       allow_self_pairs: bool = False) -> Tuple[List[Tuple[int,int]], np.ndarray]:
    """
    Construct unnormalized weights for each (i,j) pair.
    Weight w_ij = exp(Gamma[g_i, g_j] / tau)
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")

    n = len(colors)
    pairs = upper_triangular_pairs(n, allow_self=allow_self_pairs)
    w = np.zeros(len(pairs), dtype=float)

    for idx, (i, j) in enumerate(pairs):
        gi, gj = colors[i].group, colors[j].group
        # symmetric gamma
        w[idx] = math.exp(float(Gamma[gi, gj]) / float(tau))

    # normalize
    s = float(np.sum(w))
    if s <= 0 or not np.isfinite(s):
        raise RuntimeError("Invalid weight normalization. Check Gamma/tau.")
    w = w / s
    return pairs, w

def sample_pairs(pairs: List[Tuple[int,int]], probs: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=int(n_samples), replace=True, p=probs)
    counts = np.bincount(idx, minlength=len(pairs)).astype(int)
    return counts

def build_output_table(colors: List[ColorItem], pairs: List[Tuple[int,int]], counts: np.ndarray) -> pd.DataFrame:
    total = int(np.sum(counts))
    probs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)

    rows = []
    for k, (i, j) in enumerate(pairs):
        ci, cj = colors[i], colors[j]
        rows.append({
            "color1_name": ci.name,
            "color1_hex": ci.hex,
            "color1_group": ci.group,
            "color2_name": cj.name,
            "color2_hex": cj.hex,
            "color2_group": cj.group,
            "count": int(counts[k]),
            "prob": float(probs[k]),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["prob", "count"], ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# -----------------------------
# Plotting
# -----------------------------

def plot_topN_semilog(df: pd.DataFrame, topN: int, top_frac_fit: float,
                      out_png: str, title: str) -> Tuple[float, float]:
    """
    Plot rank vs prob on semi-log-y and fit ln(prob) vs rank on top top_frac_fit.
    """
    d = df.head(int(topN)).copy()
    x = d["rank"].to_numpy(dtype=float)
    y = d["prob"].to_numpy(dtype=float)

    # Fit on top top_frac_fit over entire df (not only topN)
    probs_sorted = df["prob"].to_numpy(dtype=float)
    slope, r2 = fit_lnprob_vs_rank(probs_sorted, top_frac=top_frac_fit)

    # regression line (in semilog-y): y_hat = exp(a + b*rank)
    m = max(2, int(len(probs_sorted) * top_frac_fit))
    x_fit = np.arange(1, min(topN, m) + 1, dtype=float)
    # To reconstruct line, refit on those points to get intercept
    y_ln = np.log(np.clip(probs_sorted[:m], 1e-300, 1.0))
    x_ln = np.arange(1, m + 1, dtype=float)
    b, a = np.polyfit(x_ln, y_ln, 1)  # y = b*x + a
    y_fit = np.exp(a + b * x_fit)

    plt.figure()
    plt.semilogy(x, y, marker="o", linestyle="None")
    plt.semilogy(x_fit, y_fit, linestyle="-")
    plt.xlabel("Rank")
    plt.ylabel("Probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return float(slope), float(r2)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate color-pair probabilities with partition-based Gibbs selection.")
    p.add_argument("--tau", type=float, default=0.55, help="Temperature parameter tau (>0). Lower => more concentrated.")
    p.add_argument("--n_samples", type=int, default=2_000_000, help="Number of pair draws (Monte Carlo samples).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--allow_self_pairs", action="store_true", help="Allow (i,i) self pairs. Default: disabled.")
    p.add_argument("--topN", type=int, default=60, help="Top-N pairs to plot.")
    p.add_argument("--top_frac_fit", type=float, default=0.10, help="Fraction of top pairs used for ln(prob)~rank regression.")
    p.add_argument("--out_dir", type=str, default="outputs", help="Output directory.")
    p.add_argument("--prefix", type=str, default="", help="Optional filename prefix.")
    p.add_argument("--colors_csv", type=str, default="", help="CSV with columns: name, hex, group. Overrides default palette.")
    p.add_argument("--config_json", type=str, default="", help="Optional JSON to override tau/allow_self_pairs/gamma.")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Load colors
    if args.colors_csv:
        colors = load_colors_from_csv(args.colors_csv)
    else:
        colors = default_colors_24_with_groups()

    # Optional config JSON (for gamma etc.)
    cfg = {}
    if args.config_json:
        cfg = load_config_from_json(args.config_json)

    tau = float(cfg.get("tau", args.tau))
    allow_self_pairs = bool(cfg.get("allow_self_pairs", args.allow_self_pairs))

    gamma_user = None
    if "gamma" in cfg:
        gamma_user = np.array(cfg["gamma"], dtype=float)

    Gamma = resolve_gamma(colors, gamma_user)

    # Build weights and sample
    pairs, probs = build_pair_weights(colors, Gamma, tau=tau, allow_self_pairs=allow_self_pairs)
    counts = sample_pairs(pairs, probs, n_samples=args.n_samples, seed=args.seed)

    # Output table
    df = build_output_table(colors, pairs, counts)

    ensure_dir(args.out_dir)
    pref = safe_name(args.prefix) + "_" if args.prefix else ""

    out_csv = os.path.join(args.out_dir, f"{pref}simulated_pair_probs_tau{tau:.3f}.csv")
    df.to_csv(out_csv, index=False)

    # Plot
    out_png = os.path.join(args.out_dir, f"{pref}sim_top{args.topN}_semilog_tau{tau:.3f}.png")
    title = f"Top-{args.topN} simulated color pairs (semi-log), tau={tau:.3f}"
    slope, r2 = plot_topN_semilog(df, topN=args.topN, top_frac_fit=args.top_frac_fit, out_png=out_png, title=title)

    # Print summary
    print("=== Simulation finished ===")
    print(f"tau = {tau:.4f}, n_samples = {args.n_samples}, seed = {args.seed}")
    print(f"Output CSV : {out_csv}")
    print(f"Output PNG : {out_png}")
    print(f"Regression (ln(prob) vs rank), top {args.top_frac_fit*100:.1f}%: slope ≈ {slope:.6f}, R² ≈ {r2:.4f}")

if __name__ == "__main__":
    main()
