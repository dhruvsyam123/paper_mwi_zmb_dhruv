"""
Step 3: Generate all figures for the paper.
Creates: fig1-fig6 in figures/ (PNG only)
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.ops import unary_union

from config import Config, project_root, data_dir, fig_dir, ensure_dirs

warnings.filterwarnings('ignore')

# Plotting style - clean and modern
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

# Helper functions for commodity classification
def is_staple(name: str) -> bool:
    """Check if commodity is a staple food (keyword-based)."""
    s = str(name).lower()
    return any(k in s for k in ["maize", "rice", "bean", "sorghum", "millet", "cassava", "wheat", "flour"])


def is_perishable(name: str) -> bool:
    """Check if commodity is perishable (keyword-based)."""
    s = str(name).lower()
    return any(k in s for k in ["tomato", "onion", "potato", "cabbage", "banana", "fish", "meat", "milk", "egg"])


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fig1_event_study_main():
    """
    Figure 1: Main event study plot.
    Shows treatment effect on price dispersion over time relative to road completion.
    """
    log("Figure 1: Main event study...")
    
    coefs = pd.read_csv(os.path.join(data_dir(), "event_study_coefs.csv"))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract coefficients and confidence intervals
    k = coefs["k"].values
    beta = coefs["beta"].values
    ci_lo, ci_hi = coefs["ci_lo"].values, coefs["ci_hi"].values
    
    # Plot confidence interval band and point estimates
    ax.fill_between(k, ci_lo, ci_hi, alpha=0.15, color="#2E86AB")
    ax.plot(k, beta, "o-", color="#2E86AB", markersize=8, linewidth=2.5, zorder=3)
    
    # Highlight statistically significant post-treatment effects
    sig = coefs[(coefs["k"] >= 0) & (coefs["pvalue"] < 0.05)]
    if not sig.empty:
        ax.scatter(sig["k"], sig["beta"], color="#1a9850", s=150, zorder=5, marker="o", 
                   edgecolor="white", linewidth=2, label="Significant (p<0.05)")
    
    # Reference lines
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="#d73027", linestyle="--", linewidth=2, label="Road completion")
    
    ax.set_xlabel("Months relative to road completion", fontsize=13)
    ax.set_ylabel("Effect on price dispersion (log points)", fontsize=13)
    ax.set_title("Road Infrastructure and Food Price Dispersion\nMalawi & Zambia", 
                 fontsize=15, fontweight='bold')
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax.set_xlim(-13, 13)
    ax.set_xticks(range(-12, 13, 2))
    
    # Add text box showing average short-run effect
    post_avg = coefs[(coefs["k"] >= 0) & (coefs["k"] <= 6)]["beta"].mean()
    ax.text(
        0.70,
        0.72,
        f"Avg. effect (0–6): {post_avg:.1%}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4", edgecolor="#666", alpha=0.9),
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir(), "fig1_event_study.png"))
    plt.close()
    log("  ✓ Saved")


def fig2_by_commodity():
    """
    Figure 2: Event study by commodity type (all commodities).
    Shows heterogeneity in treatment effects across different food products.
    """
    log("Figure 2: By commodity...")
    
    coefs_path = os.path.join(data_dir(), "event_study_by_commodity.csv")
    if not os.path.exists(coefs_path):
        log("  ⚠ Skipping - no commodity data")
        return
    
    coefs = pd.read_csv(coefs_path)
    commodities = coefs["commodity"].unique()
    n_comm = len(commodities)
    
    if n_comm == 0:
        log("  ⚠ No commodities")
        return
    
    # Colors for different commodities
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_comm, 10)))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, comm in enumerate(commodities):
        c = coefs[coefs["commodity"] == comm]
        if c.empty:
            continue
        ax.plot(c["k"], c["beta"], "o-", color=colors[i], markersize=5, 
                linewidth=2, label=comm, alpha=0.8)
    
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="#d73027", linestyle="--", linewidth=2)
    
    ax.set_xlabel("Months relative to road completion", fontsize=13)
    ax.set_ylabel("Effect on price dispersion (log points)", fontsize=13)
    ax.set_title("Effect Heterogeneity by Commodity Type", fontsize=15, fontweight='bold')
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9, ncol=2)
    ax.set_xlim(-13, 13)
    ax.set_xticks(range(-12, 13, 2))
    
    plt.tight_layout()
    # Move to appendix numbering to keep main figures sequential
    plt.savefig(os.path.join(fig_dir(), "figA3_by_commodity.png"))
    plt.close()
    log("  ✓ Saved")


def _bar_from_food_class_file(groups, title, out_file):
    """
    Helper: Create bar chart comparing food groups.
    Loads pre-computed event study results and computes average treatment effects.
    """
    path = os.path.join(data_dir(), "event_study_by_food_class.csv")
    if not os.path.exists(path):
        log("  ⚠ Skipping - no event_study_by_food_class.csv (run Step 2)")
        return
    df = pd.read_csv(path)
    df = df[df["group"].isin(groups)].copy()
    if df.empty:
        log("  ⚠ No rows for requested groups")
        return

    # Compute average treatment effect (0-6 months) for each group
    rows = []
    for g in groups:
        sub = df[df["group"] == g]
        post = sub[(sub["k"] >= 0) & (sub["k"] <= 6)]
        if post.empty:
            log(f"  ⚠ Missing post-period coefficients for group={g}; skipping figure {out_file}")
            return
        att = post["beta"].mean()
        se = np.sqrt((post["se"] ** 2).mean()) if not post["se"].isna().all() else np.nan
        rows.append({"group": g.replace("_", " "), "att": att, "se": se})

    if len(rows) != len(groups):
        log(f"  ⚠ Incomplete groups for {out_file}; skipping")
        return

    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"staple": "#2E86AB", "non staple": "#888888", "perishable": "#1a9850", "non perishable": "#888888"}
    x = range(len(rdf))
    ax.bar(
        x,
        rdf["att"] * 100,
        yerr=1.96 * rdf["se"] * 100,
        color=[colors.get(g, "#888888") for g in rdf["group"]],
        edgecolor="black",
        linewidth=1.5,
        capsize=6,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([g.title() for g in rdf["group"]], fontsize=12)
    ax.set_xlabel("Food group", fontsize=13)
    ax.set_ylabel("Effect on dispersion (%)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir(), out_file))
    plt.close()
    log(f"  ✓ Saved {out_file}")


def fig2b_staple_vs_nonstaple():
    """Figure 2b: staples vs non-staples (short-run 0–6 months)."""
    log("Figure 2b: Staples vs non-staples...")
    _bar_from_food_class_file(
        groups=["staple", "non_staple"],
        title="Effect by Food Group: Staples vs Non‑Staples (0–6 months)",
        out_file="fig3_staple_vs_nonstaple.png",
    )


def fig2c_perishable_vs_nonperishable():
    """Figure 2c: perishable vs non-perishable (short-run 0–6 months)."""
    log("Figure 2c: Perishable vs non-perishable...")
    _bar_from_food_class_file(
        groups=["perishable", "non_perishable"],
        title="Effect by Food Group: Perishable vs Non‑Perishable (0–6 months)",
        out_file="fig2c_perishable_vs_nonperishable.png",
    )


def fig3_by_distance():
    """
    Figure 3: Heterogeneity by distance - bar chart.
    Shows how treatment effects vary with distance between market pairs.
    Theory predicts larger effects at intermediate distances.
    """
    log("Figure 3: By distance...")
    
    coefs_path = os.path.join(data_dir(), "event_study_by_distance.csv")
    if not os.path.exists(coefs_path):
        log("  ⚠ Skipping - no distance data")
        return
    
    coefs = pd.read_csv(coefs_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = ["0-50km", "50-100km", "100-200km", "200-300km"]
    colors = ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b"]
    
    # Compute average treatment effect for each distance bin
    results = []
    for dist_bin in bins:
        c = coefs[coefs["dist_bin"] == dist_bin]
        if c.empty:
            continue
        post = c[(c["k"] >= 0) & (c["k"] <= 6)]
        if post.empty:
            continue
        att = post["beta"].mean()
        se = np.sqrt((post["se"] ** 2).mean()) if not post["se"].isna().all() else np.nan
        results.append({"dist_bin": dist_bin, "att": att, "se": se})
    
    if not results:
        log("  ⚠ No data")
        return
    
    rdf = pd.DataFrame(results)
    x = range(len(rdf))
    
    bars = ax.bar(x, rdf["att"] * 100, yerr=1.96 * rdf["se"] * 100, 
                  color=colors[:len(rdf)], edgecolor="black", linewidth=1.5, capsize=6)
    ax.axhline(0, color="black", linewidth=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(rdf["dist_bin"], fontsize=12)
    ax.set_xlabel("Distance between market pairs", fontsize=13)
    ax.set_ylabel("Effect on dispersion (%)", fontsize=13)
    ax.set_title("Largest Effects at Intermediate Distances", fontsize=15, fontweight='bold')
    
    # Add value labels
    for bar, row in zip(bars, rdf.itertuples()):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -15 if height < 0 else 5), textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir(), "fig2_by_distance.png"))
    plt.close()
    log("  ✓ Saved")


def fig5_map():
    """
    Figure 5: Combined map of both countries with roads and markets.
    Visualizes spatial distribution of treatment: road projects and nearby markets.
    """
    log("Figure 5: Map...")
    
    cfg = Config()
    markets = pd.read_csv(os.path.join(data_dir(), "markets.csv"))
    
    # Load road project geometries
    roads_path = os.path.join(project_root(), "all_roads_filtered.gpkg")
    roads = gpd.read_file(roads_path)
    roads = roads[roads["Recipient.ISO-3"].isin(cfg.countries)].to_crs("EPSG:4326")
    
    # Load country boundaries for context
    try:
        ne_path = os.path.join(data_dir(), "ne_countries", "ne_110m_admin_0_countries.shp")
        world = gpd.read_file(ne_path)
        countries = world[world['ISO_A3'].isin(['MWI', 'ZMB', 'TZA', 'MOZ', 'ZWE', 'BWA', 'COD', 'AGO', 'NAM'])]
        mwi_zmb = world[world['ISO_A3'].isin(['MWI', 'ZMB'])]
    except Exception as e:
        log(f"  ⚠ Could not load country boundaries: {e}")
        countries = None
        mwi_zmb = None
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Set map bounds based on market locations
    all_lons = markets["longitude"]
    all_lats = markets["latitude"]
    pad = 2.0
    xlim = (all_lons.min() - pad, all_lons.max() + pad)
    ylim = (all_lats.min() - pad, all_lats.max() + pad)
    
    # Plot neighboring countries for geographic context
    if countries is not None and not countries.empty:
        countries.plot(ax=ax, color='#e8e8e8', edgecolor='#666666', linewidth=1.5, zorder=1)
    
    # Highlight study countries (Malawi and Zambia)
    if mwi_zmb is not None and not mwi_zmb.empty:
        mwi_zmb.plot(ax=ax, color='#f5f5dc', edgecolor='#333333', linewidth=2.5, zorder=2)
    
    # Plot road project corridors (50km buffer zones)
    if not roads.empty:
        # Plot the road polygons (buffered corridors)
        roads.plot(ax=ax, color='#d73027', edgecolor='#8b0000', linewidth=2, 
                   alpha=0.8, zorder=4)
        
        # Add outline for better visibility
        for idx, row in roads.iterrows():
            if row.geometry is not None:
                try:
                    if row.geometry.geom_type == 'MultiPolygon':
                        for poly in row.geometry.geoms:
                            x, y = poly.exterior.xy
                            ax.plot(x, y, color='#8b0000', linewidth=3, zorder=5)
                    elif row.geometry.geom_type == 'Polygon':
                        x, y = row.geometry.exterior.xy
                        ax.plot(x, y, color='#8b0000', linewidth=3, zorder=5)
                except:
                    pass
    
    # Create GeoDataFrame for markets
    m_gdf = gpd.GeoDataFrame(markets, 
                              geometry=gpd.points_from_xy(markets["longitude"], markets["latitude"]), 
                              crs="EPSG:4326")
    
    # Plot control markets
    control = m_gdf[m_gdf["T_i"].isna()]
    if not control.empty:
        control.plot(ax=ax, color="#888888", markersize=50, alpha=0.6, 
                     label="Control markets", zorder=6, edgecolor='white', linewidth=0.5)
    
    # Plot treated markets
    treated = m_gdf[m_gdf["T_i"].notna()]
    if not treated.empty:
        treated.plot(ax=ax, color="#1a9850", markersize=120, marker="^", 
                     edgecolor="black", linewidth=1.5, label="Treated markets", zorder=7)
    
    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Labels
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Study Area: Malawi & Zambia\nMarkets and Chinese-Funded Road Projects", 
                 fontsize=15, fontweight='bold')
    
    # Add country labels
    if mwi_zmb is not None:
        for idx, row in mwi_zmb.iterrows():
            centroid = row.geometry.centroid
            name = row.get('NAME', row.get('ADMIN', row.get('name', '')))
            ax.annotate(name, xy=(centroid.x, centroid.y), fontsize=16, 
                        fontweight='bold', ha='center', va='center', color='#333333',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='#666'))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#d73027', edgecolor='#8b0000', linewidth=2, label='Road project corridors'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#1a9850', 
                   markersize=14, markeredgecolor='black', markeredgewidth=1.5, label='Treated markets'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', 
                   markersize=10, label='Control markets'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=12, framealpha=0.95)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir(), "fig4_map.png"))
    plt.close()
    log("  ✓ Saved")


def main():
    """
    Generate all paper figures:
    1. Main event study plot
    2. Food group heterogeneity (staples vs non-staples)
    3. Distance heterogeneity
    4. Geographic map of treatment
    """
    ensure_dirs()
    
    log("=" * 60)
    log("STEP 3: GENERATE FIGURES")
    log("=" * 60)
    
    t0 = time.time()
    
    # Check that previous steps have been run
    if not os.path.exists(os.path.join(data_dir(), "event_study_coefs.csv")):
        log("ERROR: Run 01 and 02 scripts first!")
        sys.exit(1)
    
    # Generate each figure
    fig1_event_study_main()
    fig2b_staple_vs_nonstaple()
    fig3_by_distance()
    fig5_map()
    
    log(f"Done in {time.time() - t0:.1f}s")
    log(f"Output: {fig_dir()}")


if __name__ == "__main__":
    main()
