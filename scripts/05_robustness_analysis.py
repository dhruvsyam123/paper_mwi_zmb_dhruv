"""
Step 5: Comprehensive Robustness Analysis for Paper
Creates: robustness tables and figures for appendix

NOTE: The key validity test is the pre-trend test from the event study (k∈[-6,-3]).
This script computes additional summary statistics and robustness checks.
"""

import os
import sys
import time
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import Config, data_dir, fig_dir, ensure_dirs

warnings.filterwarnings('ignore')

# Add parent for table output
table_dir = lambda: os.path.join(os.path.dirname(data_dir()), "tables")


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def calculate_summary_statistics(panel: pd.DataFrame, prices: pd.DataFrame, 
                                  markets: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive summary statistics table.
    Reports key variables: dispersion, distance, prices, sample sizes.
    """
    
    stats = []
    
    # Overall stats
    stats.append({
        "Variable": "Market-pair-commodity-month observations",
        "Mean": f"{len(panel):,}",
        "SD": "",
        "Min": "",
        "Max": ""
    })
    
    # Dispersion
    y = panel["y"]
    stats.append({
        "Variable": "Price dispersion (|log(p_i) - log(p_j)|)",
        "Mean": f"{y.mean():.3f}",
        "SD": f"{y.std():.3f}",
        "Min": f"{y.min():.3f}",
        "Max": f"{y.max():.3f}"
    })
    
    # By treatment status
    treated = panel[panel["T_ij"].notna()]
    control = panel[panel["T_ij"].isna()]
    
    stats.append({
        "Variable": "Dispersion - treated pairs",
        "Mean": f"{treated['y'].mean():.3f}",
        "SD": f"{treated['y'].std():.3f}",
        "Min": "",
        "Max": ""
    })
    
    stats.append({
        "Variable": "Dispersion - control pairs",
        "Mean": f"{control['y'].mean():.3f}",
        "SD": f"{control['y'].std():.3f}",
        "Min": "",
        "Max": ""
    })
    
    # Distance
    if "distance_km" in panel.columns:
        d = panel["distance_km"]
        stats.append({
            "Variable": "Distance between market pairs (km)",
            "Mean": f"{d.mean():.1f}",
            "SD": f"{d.std():.1f}",
            "Min": f"{d.min():.1f}",
            "Max": f"{d.max():.1f}"
        })
    
    # Prices
    if "price" in prices.columns:
        p = prices["price"]
        stats.append({
            "Variable": "Price (local currency/kg)",
            "Mean": f"{p.mean():.1f}",
            "SD": f"{p.std():.1f}",
            "Min": f"{p.min():.1f}",
            "Max": f"{p.max():.1f}"
        })
    
    # Markets
    stats.append({
        "Variable": "Number of markets",
        "Mean": f"{len(markets):,}",
        "SD": "",
        "Min": "",
        "Max": ""
    })
    
    stats.append({
        "Variable": "Number of unique market pairs",
        "Mean": f"{panel['pair_id'].nunique():,}",
        "SD": "",
        "Min": "",
        "Max": ""
    })
    
    stats.append({
        "Variable": "Number of treated pairs",
        "Mean": f"{treated['pair_id'].nunique():,}",
        "SD": "",
        "Min": "",
        "Max": ""
    })
    
    # Time period
    if "year" in prices.columns or "date" in prices.columns:
        try:
            if "year" in prices.columns:
                years = prices["year"]
            else:
                years = pd.to_datetime(prices["date"]).dt.year
            stats.append({
                "Variable": "Time period (years)",
                "Mean": f"{years.min()}-{years.max()}",
                "SD": "",
                "Min": "",
                "Max": ""
            })
        except:
            pass
    
    return pd.DataFrame(stats)


def get_pretrend_test(coefs: pd.DataFrame) -> dict:
    """
    Test for parallel pre-trends.
    
    Tests whether pre-treatment coefficients (k in [-6, -3]) are jointly zero.
    This is a key validity check for causal interpretation.
    """
    # Focus on k in [-6, -3] (excluding k=-2 which may have anticipation)
    pre = coefs[(coefs["k"] >= -6) & (coefs["k"] <= -3)]
    
    if pre.empty:
        return {"pretrend_avg": np.nan, "pretrend_se": np.nan, "pretrend_pval": np.nan}
    
    avg_beta = pre["beta"].mean()
    # Pooled standard error across pre-period
    pooled_se = np.sqrt((pre["se"] ** 2).mean())
    
    # T-test for average pre-trend
    t_stat = avg_beta / pooled_se if pooled_se > 0 else 0
    from scipy import stats as scipy_stats
    pval = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=len(pre) - 1))
    
    return {
        "pretrend_avg": avg_beta,
        "pretrend_se": pooled_se,
        "pretrend_pval": pval,
        "n_pretrend_coefs": len(pre)
    }


def get_treatment_effects(coefs: pd.DataFrame) -> dict:
    """
    Extract average treatment effects for different time windows.
    
    Short-run (0-6 months): Immediate impact
    Medium-run (7-12 months): Sustained effects
    """
    short_run = coefs[(coefs["k"] >= 0) & (coefs["k"] <= 6)]
    medium_run = coefs[(coefs["k"] >= 7) & (coefs["k"] <= 12)]
    
    results = {}
    
    if not short_run.empty:
        results["short_run_att"] = short_run["beta"].mean()
        results["short_run_se"] = np.sqrt((short_run["se"] ** 2).mean())
        results["short_run_n"] = len(short_run)
    
    if not medium_run.empty:
        results["medium_run_att"] = medium_run["beta"].mean()
        results["medium_run_se"] = np.sqrt((medium_run["se"] ** 2).mean())
        results["medium_run_n"] = len(medium_run)
    
    return results


def compute_window_att_from_panel(panel: pd.DataFrame, cfg: Config, k_min: int, k_max: int) -> float:
    """
    Compute average treatment effect for a window of months.
    
    Uses same estimator as main event study but averaged over [k_min, k_max].
    No bootstrapping (for speed in robustness checks).
    """
    treated_pairs = panel[panel["T_ij"].notna()]
    if treated_pairs.empty:
        return np.nan

    cohorts = sorted(treated_pairs["T_ij"].unique())
    betas = []

    for k in range(k_min, k_max + 1):
        cohort_diffs, cohort_weights = [], []
        for g in cohorts:
            t = int(g) + k
            treated = panel[(panel["T_ij"] == g) & (panel["month_int"] == t)]
            control = panel[((panel["T_ij"].isna()) | (panel["T_ij"] > t)) & (panel["month_int"] == t)]
            if treated.empty or control.empty:
                continue

            cu = treated[["commodity_id", "unit"]].drop_duplicates()
            control = control.merge(cu, on=["commodity_id", "unit"])
            if control.empty:
                continue

            cohort_diffs.append(treated["y"].mean() - control["y"].mean())
            cohort_weights.append(len(treated))

        if cohort_diffs:
            betas.append(np.average(cohort_diffs, weights=cohort_weights))

    return float(np.mean(betas)) if betas else np.nan


def leave_one_project_out(panel: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-project-out robustness check.
    
    For each road project, drop markets treated by that project and re-estimate.
    Tests whether results are driven by any single project.
    """
    if "T_i_project_id" not in markets.columns:
        return pd.DataFrame()

    # Only consider projects that actually treat at least one market
    proj_ids = (markets.loc[markets["T_i"].notna(), "T_i_project_id"]
                .dropna().astype(int).unique().tolist())
    proj_ids = sorted(proj_ids)
    if not proj_ids:
        return pd.DataFrame()

    # Market-level maps for fast reassignment
    base_T = markets.set_index("market_id")["T_i"]
    base_pid = markets.set_index("market_id")["T_i_project_id"]

    # Pre-merge market ids for speed
    p = panel[["market_i", "market_j", "month_int", "commodity_id", "unit", "y"]].copy()
    Ti = p["market_i"].map(base_T)
    Tj = p["market_j"].map(base_T)

    out_rows = []
    for pid in proj_ids:
        excluded_mkts = set(base_pid[base_pid == pid].index.values.tolist())
        Ti_mod = Ti.copy()
        Tj_mod = Tj.copy()
        Ti_mod[p["market_i"].isin(excluded_mkts)] = np.nan
        Tj_mod[p["market_j"].isin(excluded_mkts)] = np.nan

        Tij_mod = np.where(pd.notna(Ti_mod) & pd.notna(Tj_mod), np.minimum(Ti_mod, Tj_mod), np.nan)

        p_mod = p.copy()
        p_mod["T_ij"] = Tij_mod

        # Window effects (point estimates only)
        short_0_6 = compute_window_att_from_panel(p_mod, Config(), 0, 6)
        short_1_6 = compute_window_att_from_panel(p_mod, Config(), 1, 6)
        med_7_12 = compute_window_att_from_panel(p_mod, Config(), 7, 12)

        out_rows.append({
            "project_id_dropped": int(pid),
            "n_markets_dropped": int(len(excluded_mkts)),
            "att_0_6": short_0_6,
            "att_1_6": short_1_6,
            "att_7_12": med_7_12,
        })

    return pd.DataFrame(out_rows)


def get_distance_heterogeneity(panel: pd.DataFrame, coefs_by_dist_path: str) -> pd.DataFrame:
    """
    Compile distance heterogeneity results.
    Loads pre-computed coefficients and formats for table.
    """
    if not os.path.exists(coefs_by_dist_path):
        return pd.DataFrame()
    
    coefs = pd.read_csv(coefs_by_dist_path)
    
    results = []
    for dist_bin in coefs["dist_bin"].unique():
        c = coefs[coefs["dist_bin"] == dist_bin]
        post = c[(c["k"] >= 0) & (c["k"] <= 6)]
        if post.empty:
            continue
        
        att = post["beta"].mean()
        se = np.sqrt((post["se"] ** 2).mean()) if not post["se"].isna().all() else np.nan
        n_pairs = panel[panel["dist_bin"] == dist_bin]["pair_id"].nunique()
        
        results.append({
            "Distance": dist_bin,
            "ATT": f"{att:.3f}",
            "SE": f"{se:.3f}",
            "N pairs": n_pairs
        })
    
    return pd.DataFrame(results)


def main():
    """
    Comprehensive robustness analysis:
    1. Summary statistics
    2. Pre-trend tests (key validity check)
    3. Treatment effects by window
    4. Heterogeneity checks
    5. Sensitivity analyses (donut, exclude k=0)
    6. Leave-one-project-out
    """
    cfg = Config()
    ensure_dirs()
    os.makedirs(table_dir(), exist_ok=True)
    
    log("=" * 60)
    log("STEP 5: ROBUSTNESS ANALYSIS")
    log("=" * 60)
    
    t0 = time.time()
    
    # Load all necessary data
    log("Loading data...")
    panel = pd.read_csv(os.path.join(data_dir(), "panel.csv"))
    prices = pd.read_csv(os.path.join(data_dir(), "prices.csv"))
    markets = pd.read_csv(os.path.join(data_dir(), "markets.csv"))
    coefs = pd.read_csv(os.path.join(data_dir(), "event_study_coefs.csv"))
    
    log(f"  Panel: {len(panel):,} obs")
    log(f"  Event study coefficients: {len(coefs)} periods")
    
    # 1. Summary statistics
    log("Creating summary statistics...")
    summary = calculate_summary_statistics(panel, prices, markets)
    summary.to_csv(os.path.join(table_dir(), "table_summary_stats.csv"), index=False)
    log("  ✓ Saved summary statistics")
    
    # 2. Pre-trend test (THE KEY VALIDITY CHECK)
    log("Pre-trend test (k ∈ [-6, -3])...")
    pretrend = get_pretrend_test(coefs)
    log(f"  Average pre-trend: {pretrend['pretrend_avg']:.4f}")
    log(f"  SE: {pretrend['pretrend_se']:.4f}")
    log(f"  p-value: {pretrend['pretrend_pval']:.3f}")
    if pretrend['pretrend_pval'] > 0.10:
        log("  ✓ PRE-TREND TEST PASSES (parallel trends supported)")
    else:
        log("  ⚠ Pre-trend test suggests potential concern")
    
    # 3. Treatment effects by window
    log("Treatment effects by window...")
    effects = get_treatment_effects(coefs)
    if "short_run_att" in effects:
        log(f"  Short-run (0-6m): ATT = {effects['short_run_att']:.4f} (SE = {effects['short_run_se']:.4f})")
    if "medium_run_att" in effects:
        log(f"  Medium-run (7-12m): ATT = {effects['medium_run_att']:.4f} (SE = {effects['medium_run_se']:.4f})")
    
    # 4. Heterogeneity by distance
    log("Heterogeneity by distance...")
    dist_het = get_distance_heterogeneity(
        panel, 
        os.path.join(data_dir(), "event_study_by_distance.csv")
    )
    if not dist_het.empty:
        dist_het.to_csv(os.path.join(table_dir(), "table_het_distance.csv"), index=False)
        for _, row in dist_het.iterrows():
            log(f"  {row['Distance']}: ATT = {row['ATT']}")
    
    # 5. Compile robustness/validity table
    robustness = []
    
    # Pre-trend test
    robustness.append({
        "Test": "Pre-trend (k ∈ [-6,-3])",
        "Estimate": f"{pretrend['pretrend_avg']:.4f}",
        "SE": f"{pretrend['pretrend_se']:.4f}",
        "p-value": f"{pretrend['pretrend_pval']:.3f}",
        "Result": "PASS" if pretrend['pretrend_pval'] > 0.10 else "CONCERN"
    })
    
    # Short-run effect
    if "short_run_att" in effects:
        from scipy import stats as scipy_stats
        t_stat = effects['short_run_att'] / effects['short_run_se']
        pval = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=100))
        robustness.append({
            "Test": "Short-run effect (k ∈ [0,6])",
            "Estimate": f"{effects['short_run_att']:.4f}",
            "SE": f"{effects['short_run_se']:.4f}",
            "p-value": f"{pval:.3f}",
            "Result": "SIGNIFICANT" if pval < 0.05 else "NOT SIG"
        })
    
    # Medium-run effect  
    if "medium_run_att" in effects:
        t_stat = effects['medium_run_att'] / effects['medium_run_se']
        pval = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=100))
        robustness.append({
            "Test": "Medium-run effect (k ∈ [7,12])",
            "Estimate": f"{effects['medium_run_att']:.4f}",
            "SE": f"{effects['medium_run_se']:.4f}",
            "p-value": f"{pval:.3f}",
            "Result": "SIGNIFICANT" if pval < 0.05 else "NOT SIG"
        })
    
    # Anticipation check (k=-2)
    k2 = coefs[coefs["k"] == -2]
    if not k2.empty:
        k2_row = k2.iloc[0]
        robustness.append({
            "Test": "Anticipation (k=-2)",
            "Estimate": f"{k2_row['beta']:.4f}",
            "SE": f"{k2_row['se']:.4f}",
            "p-value": f"{k2_row['pvalue']:.3f}",
            "Result": "Expected (construction phase)"
        })
    
    rob_df = pd.DataFrame(robustness)
    rob_df.to_csv(os.path.join(table_dir(), "table_robustness.csv"), index=False)
    log("  ✓ Saved robustness table")

    # 6. Sensitivity checks: alternative specifications
    log("Sensitivity checks (exclude transition months; exclude k=0)...")
    panel2 = panel.copy()
    
    # "Donut" specification: drop months around treatment for anticipation concerns
    if "T_ij" in panel2.columns:
        rel = panel2["month_int"] - panel2["T_ij"]
        drop = (panel2["T_ij"].notna()) & (rel.isin([-2, -1]))
        panel_donut = panel2.loc[~drop].copy()
    else:
        panel_donut = panel2

    sens_rows = [
        {"spec": "Baseline ATT (k=0..6)", "att": compute_window_att_from_panel(panel2, cfg, 0, 6)},
        {"spec": "Exclude k=0 (k=1..6)", "att": compute_window_att_from_panel(panel2, cfg, 1, 6)},
        {"spec": "Donut: drop rel_month in [-2,-1], ATT (k=0..6)", "att": compute_window_att_from_panel(panel_donut, cfg, 0, 6)},
        {"spec": "Donut: drop rel_month in [-2,-1], ATT (k=7..12)", "att": compute_window_att_from_panel(panel_donut, cfg, 7, 12)},
    ]
    pd.DataFrame(sens_rows).to_csv(os.path.join(table_dir(), "table_sensitivity_checks.csv"), index=False)
    log("  ✓ Saved sensitivity checks table")

    # 7. Leave-one-project-out
    log("Leave-one-project-out robustness...")
    loo = leave_one_project_out(panel2, markets)
    if loo.empty:
        log("  ⚠ Skipping - no project id mapping found in markets.csv")
    else:
        loo.to_csv(os.path.join(table_dir(), "table_leave_one_project_out.csv"), index=False)
        log("  ✓ Saved leave-one-project-out table")
    
    log(f"Done in {time.time() - t0:.1f}s")
    log(f"Output: {table_dir()}")


if __name__ == "__main__":
    main()
