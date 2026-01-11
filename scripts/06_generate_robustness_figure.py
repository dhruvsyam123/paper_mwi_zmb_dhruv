"""
Step 6: Generate robustness and additional figures for paper appendix.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
fig_dir = os.path.join(project_root, "figures")
table_dir = os.path.join(project_root, "tables")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fig_robustness_forest():
    """
    Forest plot of identification tests.
    Visualizes pre-trend tests and treatment effects with confidence intervals.
    """
    log("Creating robustness forest plot...")
    
    robustness_path = os.path.join(table_dir, "table_robustness.csv")
    if not os.path.exists(robustness_path):
        log("  ⚠ Skipping - no robustness table")
        return
    
    rob = pd.read_csv(robustness_path)
    
    # Parse numeric values - new format has "Test" and "Estimate" columns
    if "Test" in rob.columns:
        rob["label"] = rob["Test"]
        rob["ATT_num"] = rob["Estimate"].astype(float)
    else:
        rob["label"] = rob["Specification"]
        rob["ATT_num"] = rob["ATT"].astype(float)
    
    rob["SE_num"] = rob["SE"].astype(float)
    rob["ci_lo"] = rob["ATT_num"] - 1.96 * rob["SE_num"]
    rob["ci_hi"] = rob["ATT_num"] + 1.96 * rob["SE_num"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n = len(rob)
    y_pos = np.arange(n)
    
    # Color based on result type
    colors = []
    for _, row in rob.iterrows():
        result = row.get("Result", "")
        if "PASS" in str(result):
            colors.append("#2E86AB")  # Blue for pre-trend pass
        elif "SIGNIFICANT" in str(result):
            colors.append("#1a9850")  # Green for significant effects
        elif "Expected" in str(result):
            colors.append("#f0a030")  # Orange for anticipation
        else:
            colors.append("#888888")  # Gray
    
    # Plot horizontal bars for CIs
    for i, (_, row) in enumerate(rob.iterrows()):
        ax.hlines(y_pos[i], row["ci_lo"], row["ci_hi"], color=colors[i], linewidth=3)
        ax.scatter(row["ATT_num"], y_pos[i], color=colors[i], s=150, zorder=5, 
                   edgecolor="white", linewidth=2)
    
    # Add vertical line at 0
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rob["label"], fontsize=12)
    ax.set_xlabel("Coefficient Estimate (log points)", fontsize=13)
    ax.set_title("Identification Tests: Pre-Trends and Treatment Effects", 
                 fontsize=15, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', 
               markersize=12, label='Pre-trend test (PASS)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1a9850', 
               markersize=12, label='Treatment effect (significant)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f0a030', 
               markersize=12, label='Anticipation (expected)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figA2_robustness.png"))
    plt.close()
    log("  ✓ Saved robustness figure")


def fig_pretrend_test():
    """
    Focused pre-trend test visualization.
    
    Shows pre-treatment coefficients only to assess parallel trends assumption.
    Highlights the test window (k in [-6, -3]) and anticipation at k=-2.
    """
    log("Creating pre-trend test figure...")
    
    data_dir = os.path.join(project_root, "data")
    coefs_path = os.path.join(data_dir, "event_study_coefs.csv")
    if not os.path.exists(coefs_path):
        log("  ⚠ Skipping - no event study data")
        return
    
    coefs = pd.read_csv(coefs_path)
    
    # Extract pre-treatment period only
    pre = coefs[coefs["k"] < 0].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k = pre["k"].values
    beta = pre["beta"].values
    ci_lo, ci_hi = pre["ci_lo"].values, pre["ci_hi"].values
    
    # Shaded region for "core" pre-trend test window
    ax.axvspan(-6.5, -2.5, alpha=0.15, color="#2E86AB", label="Pre-trend test window (k∈[-6,-3])")
    
    # Plot CIs and points
    ax.fill_between(k, ci_lo, ci_hi, alpha=0.2, color="#2E86AB")
    ax.plot(k, beta, "o-", color="#2E86AB", markersize=10, linewidth=2.5, zorder=3)
    
    # Highlight significant pre-period coefficients (concern)
    sig_pre = pre[pre["pvalue"] < 0.05]
    if not sig_pre.empty:
        ax.scatter(sig_pre["k"], sig_pre["beta"], color="#d73027", s=180, zorder=5, 
                   marker="X", edgecolor="white", linewidth=2, label="Significant (concern)")
    
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="#d73027", linestyle="--", linewidth=2, label="Treatment (k=0)")
    
    # Annotation for k=-2 spike
    k2_row = coefs[coefs["k"] == -2].iloc[0]
    ax.annotate(f"Anticipation effect\n(k=-2): β={k2_row['beta']:.3f}***", 
                xy=(-2, k2_row["beta"]), xytext=(-6, k2_row["beta"] + 0.03),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffeb99", edgecolor="#666"))
    
    ax.set_xlabel("Months relative to road completion", fontsize=13)
    ax.set_ylabel("Effect on price dispersion (log points)", fontsize=13)
    ax.set_title("Pre-Trend Test: Coefficients Before Road Completion", 
                 fontsize=15, fontweight='bold')
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_xlim(-13, 0)
    ax.set_xticks(range(-12, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figA1_pretrend_test.png"))
    plt.close()
    log("  ✓ Saved pre-trend figure")


def fig_country_comparison():
    """Bar chart comparing effects by country."""
    log("Creating country comparison figure...")
    
    robustness_path = os.path.join(table_dir, "table_robustness.csv")
    if not os.path.exists(robustness_path):
        log("  ⚠ Skipping - no robustness table")
        return
    
    rob = pd.read_csv(robustness_path)
    
    # Get country rows
    country_rows = rob[rob["Specification"].str.contains("only")]
    if country_rows.empty:
        log("  ⚠ No country data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    countries = country_rows["Specification"].str.replace(" only", "").values
    atts = country_rows["ATT"].astype(float).values * 100  # Convert to percentage
    ses = country_rows["SE"].astype(float).values * 100
    
    colors = ["#2E86AB", "#A23B72"]
    x = range(len(countries))
    
    bars = ax.bar(x, atts, yerr=1.96 * ses, color=colors[:len(countries)], 
                  edgecolor="black", linewidth=1.5, capsize=8)
    
    ax.axhline(0, color="black", linewidth=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=14)
    ax.set_xlabel("Country", fontsize=13)
    ax.set_ylabel("Effect on price dispersion (%)", fontsize=13)
    ax.set_title("Effect Heterogeneity by Country\n(Larger Effects in Less-Connected Malawi)", 
                 fontsize=15, fontweight='bold')
    
    # Add value labels
    for bar, att, se in zip(bars, atts, ses):
        height = bar.get_height()
        sig = "***" if abs(att / se) > 2.58 else ("**" if abs(att / se) > 1.96 else ("*" if abs(att / se) > 1.645 else ""))
        ax.annotate(f'{height:.1f}%{sig}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -20 if height < 0 else 8), textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_country_comparison.png"))
    plt.close()
    log("  ✓ Saved country comparison figure")


def main():
    """
    Generate robustness/appendix figures:
    1. Forest plot of validity tests
    2. Focused pre-trend visualization
    """
    os.makedirs(fig_dir, exist_ok=True)
    
    log("=" * 60)
    log("STEP 6: GENERATE ROBUSTNESS FIGURES")
    log("=" * 60)
    
    fig_robustness_forest()
    fig_pretrend_test()
    
    log("Done!")
    log(f"Output: {fig_dir}")


if __name__ == "__main__":
    main()

