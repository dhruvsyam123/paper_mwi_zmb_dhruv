"""
Step 2: Estimate event study coefficients.
Creates: event_study_coefs.csv, event_study_by_commodity.csv, event_study_by_distance.csv
"""

import os
import sys
import time
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from config import Config, data_dir, ensure_dirs

warnings.filterwarnings('ignore')


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def estimate_event_study(panel: pd.DataFrame, cfg: Config, 
                         subset_col: str = None, subset_val=None,
                         bootstrap_reps: int = None) -> pd.DataFrame:
    """
    Estimate event study coefficients using stacked cohort design.
    
    For each relative month k:
    - Compare treated pairs at k months after their road completion
    - Against control pairs (never treated or not yet treated)
    - Average across cohorts weighted by sample size
    - Bootstrap standard errors clustered by pair
    """
    
    # Optional: subset analysis (by commodity, distance, etc.)
    if subset_col and subset_val is not None:
        panel = panel[panel[subset_col] == subset_val].copy()
    
    if bootstrap_reps is None:
        bootstrap_reps = cfg.bootstrap_reps
    
    # Identify treatment cohorts
    treated_pairs = panel[panel["T_ij"].notna()]
    if treated_pairs.empty:
        return pd.DataFrame()
    
    cohorts = sorted(treated_pairs["T_ij"].unique())
    results = []
    
    # Estimate coefficient for each event time k (relative to treatment)
    for k in range(-cfg.pre_months, cfg.post_months + 1):
        # Omit k=-1 (reference period)
        if k == -1:
            continue
        
        cohort_diffs, cohort_weights = [], []
        
        # For each cohort, compute treated-control difference at calendar time t = g + k
        for g in cohorts:
            t = int(g) + k  # Calendar time
            treated = panel[(panel["T_ij"] == g) & (panel["month_int"] == t)]
            # Control = never treated OR treated later
            control = panel[((panel["T_ij"].isna()) | (panel["T_ij"] > t)) & (panel["month_int"] == t)]
            
            if treated.empty or control.empty:
                continue
            
            # Match on commodities traded in treated pairs
            cu = treated[["commodity_id", "unit"]].drop_duplicates()
            control = control.merge(cu, on=["commodity_id", "unit"])
            if control.empty:
                continue
            
            # Compute difference in mean dispersion
            cohort_diffs.append(treated["y"].mean() - control["y"].mean())
            cohort_weights.append(len(treated))
        
        if not cohort_diffs:
            continue
        
        # Weighted average across cohorts
        beta_k = np.average(cohort_diffs, weights=cohort_weights)
        
        # Bootstrap standard errors (cluster by pair)
        rng = np.random.default_rng(cfg.seed + k)
        pairs = panel["pair_id"].unique()
        n_pairs = len(pairs)
        
        boots = []
        for _ in range(bootstrap_reps):
            # Resample pairs with replacement
            sampled = rng.choice(pairs, size=n_pairs, replace=True)
            sub = panel[panel["pair_id"].isin(set(sampled))]
            
            # Re-estimate beta_k on bootstrap sample
            c_diffs, c_wts = [], []
            for g in cohorts:
                t_cal = int(g) + k
                tr = sub[(sub["T_ij"] == g) & (sub["month_int"] == t_cal)]
                ct = sub[((sub["T_ij"].isna()) | (sub["T_ij"] > t_cal)) & (sub["month_int"] == t_cal)]
                if tr.empty or ct.empty:
                    continue
                cu_tr = tr[["commodity_id", "unit"]].drop_duplicates()
                ct = ct.merge(cu_tr, on=["commodity_id", "unit"])
                if ct.empty:
                    continue
                c_diffs.append(tr["y"].mean() - ct["y"].mean())
                c_wts.append(len(tr))
            
            if c_diffs:
                boots.append(np.average(c_diffs, weights=c_wts))
        
        # Compute SE, CI, and p-value from bootstrap distribution
        boots = np.array([b for b in boots if np.isfinite(b)])
        if len(boots) > 20:
            se = np.std(boots, ddof=1)
            ci = np.quantile(boots, [0.025, 0.975])
            pval = min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
        else:
            se, ci, pval = np.nan, [np.nan, np.nan], np.nan
        
        results.append({
            "k": k, "beta": beta_k, "se": se, 
            "ci_lo": ci[0], "ci_hi": ci[1], "pvalue": pval,
            "n_treated": sum(cohort_weights)
        })
    
    return pd.DataFrame(results)


def _is_staple(commodity: str) -> bool:
    """Classify commodity as staple food based on name keywords."""
    s = str(commodity).lower()
    return any(k in s for k in ["maize", "rice", "bean", "sorghum", "millet", "cassava", "wheat", "flour"])


def _is_perishable(commodity: str) -> bool:
    """Classify commodity as perishable based on name keywords."""
    s = str(commodity).lower()
    return any(k in s for k in ["tomato", "onion", "potato", "cabbage", "banana", "fish", "meat", "milk", "egg"])


def _build_panel_for_combos(pairs: pd.DataFrame, prices: pd.DataFrame, combos: pd.DataFrame) -> pd.DataFrame:
    """
    Build a pair-commodity-month panel for a specified set of (commodity_id, unit) combos.
    Uses the same merge logic as Step 1, but restricted to the provided combos.
    """
    if combos.empty:
        return pd.DataFrame()
    combos = combos[["commodity_id", "unit"]].drop_duplicates()
    pm = prices.merge(combos, on=["commodity_id", "unit"], how="inner").copy()
    if pm.empty:
        return pd.DataFrame()

    panel = pairs.merge(
        pm[["market_id", "month_int", "commodity_id", "commodity", "unit", "log_price"]].rename(
            columns={"market_id": "market_i", "log_price": "lp_i"}), on="market_i"
    )
    panel = panel.merge(
        pm[["market_id", "month_int", "commodity_id", "unit", "log_price"]].rename(
            columns={"market_id": "market_j", "log_price": "lp_j"}),
        on=["market_j", "month_int", "commodity_id", "unit"]
    )
    if panel.empty:
        return pd.DataFrame()

    panel["y"] = (panel["lp_i"] - panel["lp_j"]).abs()
    panel["pair_id"] = panel["market_i"].astype(str) + "_" + panel["market_j"].astype(str)
    return panel


def estimate_food_class_heterogeneity(cfg: Config, top_n_per_group: int = 8) -> pd.DataFrame:
    """
    Estimate separate event studies for different food types.
    
    Groups analyzed:
      - Staples vs non-staples (core consumption goods vs other foods)
    
    For efficiency, we select top-N commodities per group rather than
    re-estimating on the full panel.
    
    Note: Perishable analysis may be unreliable for this sample due to
    limited post-treatment coverage.
    """
    prices = pd.read_csv(os.path.join(data_dir(), "prices.csv"))
    pairs = pd.read_csv(os.path.join(data_dir(), "pairs.csv"))

    # Counts by commodity-unit combo
    counts = (prices.groupby(["commodity_id", "commodity", "unit"])
              .size().reset_index(name="n")
              .sort_values("n", ascending=False))
    counts["is_staple"] = counts["commodity"].apply(_is_staple)
    counts["is_perishable"] = counts["commodity"].apply(_is_perishable)

    def _top(group_df: pd.DataFrame) -> pd.DataFrame:
        return group_df.head(top_n_per_group)[["commodity_id", "unit"]]

    groups = [
        ("staple", _top(counts[counts["is_staple"] == True])),
        ("non_staple", _top(counts[counts["is_staple"] == False])),
    ]

    out = []
    for label, combos in groups:
        log(f"Estimating food class event study: {label} (top {len(combos)} combos)...")
        panel_g = _build_panel_for_combos(pairs, prices, combos)
        if panel_g.empty:
            log(f"  ⚠ No panel rows for {label}; skipping")
            continue
        coefs_g = estimate_event_study(panel_g, cfg, bootstrap_reps=50)
        if coefs_g.empty:
            continue
        coefs_g["group"] = label
        coefs_g["top_n_combos"] = len(combos)
        out.append(coefs_g)

    if not out:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


def main():
    """
    Estimate event study coefficients:
    1. Main specification (all pairs)
    2. Heterogeneity by commodity type
    3. Heterogeneity by distance
    4. Heterogeneity by food class (staples, etc.)
    """
    cfg = Config()
    ensure_dirs()
    
    log("=" * 60)
    log("STEP 2: ESTIMATE EVENT STUDY")
    log("=" * 60)
    
    t0 = time.time()
    
    # Load analysis panel
    log("Loading panel data...")
    panel_path = os.path.join(data_dir(), "panel.csv")
    if not os.path.exists(panel_path):
        log("ERROR: Run 01_prepare_data.py first!")
        sys.exit(1)
    
    panel = pd.read_csv(panel_path)
    log(f"  Panel: {len(panel):,} observations")
    
    # Main event study specification
    log("Estimating main event study...")
    coefs = estimate_event_study(panel, cfg)
    coefs.to_csv(os.path.join(data_dir(), "event_study_coefs.csv"), index=False)
    
    # Report short-run average treatment effect
    att = coefs[(coefs["k"] >= 0) & (coefs["k"] <= 6)]["beta"].mean()
    log(f"  Main ATT (0-6 mo): {att:.4f} ({att*100:.1f}%)")
    
    # Heterogeneity: estimate separately for each commodity
    log("Estimating by commodity type...")
    commodity_results = []
    
    # Get all commodities ordered by frequency
    top_commodities = (panel.groupby(["commodity_id", "commodity"])
                       .size().reset_index(name="n")
                       .sort_values("n", ascending=False))
    
    for _, row in top_commodities.iterrows():
        comm_id = row["commodity_id"]
        comm_name = row["commodity"]
        log(f"  {comm_name}...")
        # Use fewer bootstrap reps for speed
        c = estimate_event_study(panel, cfg, subset_col="commodity_id", subset_val=comm_id, 
                                 bootstrap_reps=50)
        if not c.empty:
            c["commodity_id"] = comm_id
            c["commodity"] = comm_name
            commodity_results.append(c)
    
    if commodity_results:
        pd.concat(commodity_results).to_csv(
            os.path.join(data_dir(), "event_study_by_commodity.csv"), index=False)
    
    # Heterogeneity: estimate separately for each distance bin
    log("Estimating by distance...")
    dist_results = []
    for dist_bin in ["0-50km", "50-100km", "100-200km", "200-300km"]:
        log(f"  {dist_bin}...")
        c = estimate_event_study(panel, cfg, subset_col="dist_bin", subset_val=dist_bin,
                                 bootstrap_reps=50)
        if not c.empty:
            c["dist_bin"] = dist_bin
            dist_results.append(c)
    
    if dist_results:
        pd.concat(dist_results).to_csv(
            os.path.join(data_dir(), "event_study_by_distance.csv"), index=False)

    # Heterogeneity: estimate for food classes (staples vs non-staples)
    log("Estimating food class heterogeneity (staple/non-staple; perishable/non-perishable)...")
    class_coefs = estimate_food_class_heterogeneity(cfg, top_n_per_group=8)
    if not class_coefs.empty:
        class_coefs.to_csv(os.path.join(data_dir(), "event_study_by_food_class.csv"), index=False)
        log("  ✓ Saved event_study_by_food_class.csv")
    else:
        log("  ⚠ No food class heterogeneity results produced")
    
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
