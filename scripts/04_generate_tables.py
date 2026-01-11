"""
Step 4: Generate all tables for the paper.
Creates: table1-table3 in tables/
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

from config import Config, project_root, data_dir, table_dir, ensure_dirs

warnings.filterwarnings('ignore')


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def month_int_to_date(m: int) -> str:
    """Convert month integer back to YYYY-MM format."""
    return f"{m // 12}-{m % 12:02d}"


def table1_summary_stats():
    """
    Table 1: Summary statistics.
    Shows sample sizes by country: markets, pairs, roads, observations.
    """
    log("Table 1: Summary statistics...")
    
    cfg = Config()
    prices = pd.read_csv(os.path.join(data_dir(), "prices.csv"))
    markets = pd.read_csv(os.path.join(data_dir(), "markets.csv"))
    pairs = pd.read_csv(os.path.join(data_dir(), "pairs.csv"))
    
    roads = gpd.read_file(os.path.join(project_root(), "all_roads_filtered.gpkg"))
    roads = roads[roads["Recipient.ISO-3"].isin(cfg.countries)]
    
    # Build summary statistics by country
    rows = []
    for iso, name in cfg.country_names.items():
        p = prices[prices["countryiso3"] == iso]
        m = markets[markets["countryiso3"] == iso]
        pr = pairs[pairs["countryiso3"] == iso]
        r = roads[roads["Recipient.ISO-3"] == iso]
        
        rows.append({
            "Country": name,
            "Markets": len(m),
            "Treated markets": int(m["T_i"].notna().sum()),
            "Market pairs": len(pr),
            "Treated pairs": int(pr["T_ij"].notna().sum()),
            "Road projects": len(r),
            "Commodities": int(p["commodity_id"].nunique()) if "commodity_id" in p.columns else "N/A",
            "Price observations": len(p),
        })
    
    # Total row
    rows.append({
        "Country": "Total",
        "Markets": len(markets),
        "Treated markets": int(markets["T_i"].notna().sum()),
        "Market pairs": len(pairs),
        "Treated pairs": int(pairs["T_ij"].notna().sum()),
        "Road projects": len(roads),
        "Commodities": int(prices["commodity_id"].nunique()) if "commodity_id" in prices.columns else "N/A",
        "Price observations": len(prices),
    })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(table_dir(), "table1_summary.csv"), index=False)
    log("  ✓ Saved")
    return df


def table2_main_results():
    """
    Table 2: Main regression results.
    Reports average treatment effects for different time windows
    and tests for parallel pre-trends.
    """
    log("Table 2: Main results...")
    
    coefs = pd.read_csv(os.path.join(data_dir(), "event_study_coefs.csv"))
    
    # Pre-trend test: joint test that all pre-period coefficients = 0
    pre = coefs[(coefs["k"] >= -6) & (coefs["k"] <= -3)].dropna(subset=["beta", "se"])
    if not pre.empty and len(pre) > 0:
        wald = ((pre["beta"] / pre["se"]) ** 2).sum()
        pretrend_f = wald / len(pre)
        pretrend_p = 1 - stats.chi2.cdf(wald, len(pre))
    else:
        pretrend_f, pretrend_p = np.nan, np.nan
    
    # Helper: compute average effect over a window of months
    def _window(k_min, k_max):
        sub = coefs[(coefs["k"] >= k_min) & (coefs["k"] <= k_max)].dropna(subset=["beta"])
        if sub.empty:
            return np.nan, np.nan, np.nan
        avg = sub["beta"].mean()
        se = np.sqrt((sub["se"] ** 2).mean()) if not sub["se"].isna().all() else np.nan
        pval = 2 * (1 - stats.norm.cdf(abs(avg / se))) if se > 0 else np.nan
        return avg, se, pval
    
    pre_avg, pre_se, pre_p = _window(-6, -3)
    short_avg, short_se, short_p = _window(0, 6)
    med_avg, med_se, med_p = _window(7, 12)
    
    results = pd.DataFrame([
        {"Window": "Pre-trend (k∈[-6,-3])", "Estimate": pre_avg, "SE": pre_se, "p-value": pre_p},
        {"Window": "Short-run (k∈[0,6])", "Estimate": short_avg, "SE": short_se, "p-value": short_p},
        {"Window": "Medium-run (k∈[7,12])", "Estimate": med_avg, "SE": med_se, "p-value": med_p},
    ])
    results.to_csv(os.path.join(table_dir(), "table2_results.csv"), index=False)
    
    # Validity tests
    validity = pd.DataFrame([{
        "Test": "Joint pre-trend F-test (k∈[-6,-3])",
        "Statistic": pretrend_f,
        "p-value": pretrend_p,
        "Passes (p>0.05)": "Yes" if pretrend_p > 0.05 else "No",
    }])
    validity.to_csv(os.path.join(table_dir(), "table2_validity.csv"), index=False)
    
    log("  ✓ Saved")
    log(f"    Pre-trend test: F={pretrend_f:.3f}, p={pretrend_p:.4f}")
    log(f"    Short-run ATT: {short_avg:.4f} (p={short_p:.4f})")
    
    return results


def table3_heterogeneity():
    """
    Table 3: Heterogeneity analysis.
    Shows how treatment effects vary by commodity type and distance.
    """
    log("Table 3: Heterogeneity...")
    
    rows = []
    panel = pd.read_csv(os.path.join(data_dir(), "panel.csv"))
    
    # Heterogeneity by commodity
    commodity_path = os.path.join(data_dir(), "event_study_by_commodity.csv")
    if os.path.exists(commodity_path):
        coefs = pd.read_csv(commodity_path)
        
        for comm in coefs["commodity"].unique():
            c = coefs[coefs["commodity"] == comm]
            if c.empty:
                continue
            # Compute short-run ATT (0-6 months)
            post = c[(c["k"] >= 0) & (c["k"] <= 6)].dropna(subset=["beta"])
            if post.empty:
                continue
            att = post["beta"].mean()
            se = np.sqrt((post["se"] ** 2).mean()) if not post["se"].isna().all() else np.nan
            n = panel[panel["commodity"] == comm]["pair_id"].nunique()
            rows.append({"Dimension": "Commodity", "Category": comm, "ATT": att, "SE": se, "N pairs": n})
    
    # By distance
    dist_path = os.path.join(data_dir(), "event_study_by_distance.csv")
    if os.path.exists(dist_path):
        coefs = pd.read_csv(dist_path)
        
        for dist in ["0-50km", "50-100km", "100-200km", "200-300km"]:
            c = coefs[coefs["dist_bin"] == dist]
            if c.empty:
                continue
            post = c[(c["k"] >= 0) & (c["k"] <= 6)].dropna(subset=["beta"])
            if post.empty:
                continue
            att = post["beta"].mean()
            se = np.sqrt((post["se"] ** 2).mean()) if not post["se"].isna().all() else np.nan
            n = panel[panel["dist_bin"] == dist]["pair_id"].nunique()
            rows.append({"Dimension": "Distance", "Category": dist, "ATT": att, "SE": se, "N pairs": n})
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(table_dir(), "table3_heterogeneity.csv"), index=False)
    log("  ✓ Saved")
    return df


def table_event_study_series():
    """
    Export complete event study series for transparency.
    Includes all coefficients shown in event study figure.
    """
    log("Table: Event study series (plotted coefficients)...")
    coefs = pd.read_csv(os.path.join(data_dir(), "event_study_coefs.csv"))
    keep_cols = [c for c in ["k", "beta", "se", "ci_lo", "ci_hi", "pvalue", "n_treated"] if c in coefs.columns]
    out = coefs[keep_cols].sort_values("k")
    out.to_csv(os.path.join(table_dir(), "table_event_study_coefs.csv"), index=False)
    log("  ✓ Saved")
    return out


def main():
    """
    Generate all paper tables:
    1. Summary statistics
    2. Main regression results and validity tests
    3. Heterogeneity analysis
    4. Event study coefficient series
    """
    ensure_dirs()
    
    log("=" * 60)
    log("STEP 4: GENERATE TABLES")
    log("=" * 60)
    
    t0 = time.time()
    
    # Check that previous steps have been run
    if not os.path.exists(os.path.join(data_dir(), "markets.csv")):
        log("ERROR: Run previous scripts first!")
        sys.exit(1)
    
    # Generate each table
    table1_summary_stats()
    table2_main_results()
    table3_heterogeneity()
    table_event_study_series()
    
    log(f"Done in {time.time() - t0:.1f}s")
    log(f"Output: {table_dir()}")


if __name__ == "__main__":
    main()

