"""
Step 1: Load and prepare data for analysis.
Creates: markets.csv, pairs.csv, panel.csv
"""

import os
import sys
import math
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

from config import Config, project_root, data_dir, ensure_dirs

warnings.filterwarnings('ignore')


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def month_int(dt) -> int:
    """Convert date to integer month (year*12 + month)."""
    return int(dt.year) * 12 + int(dt.month)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Calculate distance between two lat/lon points in kilometers."""
    r = 6371.0088  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_prices(cfg: Config) -> pd.DataFrame:
    """Load and clean food price data."""
    log("Loading prices...")
    path = os.path.join(data_dir(), "all_food_prices_clean.csv")
    
    usecols = ["countryiso3", "date", "market_id", "market", "latitude", "longitude",
               "commodity_id", "commodity", "unit", "usdprice", "pricetype"]
    
    df = pd.read_csv(path, usecols=usecols, parse_dates=["date"], low_memory=False)
    log(f"  Raw rows: {len(df):,}")
    
    # Filter: selected countries, retail prices only, valid prices
    df = df[df["countryiso3"].isin(cfg.countries) & (df["pricetype"] == "Retail")]
    df = df.dropna(subset=["market_id", "commodity_id", "usdprice", "latitude", "longitude"])
    df = df[df["usdprice"] > 0].copy()
    log(f"  After filtering: {len(df):,}")
    
    # Convert to integer IDs and month integers
    df["market_id"] = df["market_id"].astype(int)
    df["commodity_id"] = df["commodity_id"].astype(int)
    df["month_int"] = df["date"].dt.to_period("M").dt.to_timestamp().map(month_int)
    
    # Aggregate to monthly level (median price per market-commodity-month)
    log("  Aggregating to monthly...")
    monthly = (df.groupby(["countryiso3", "month_int", "commodity_id", "commodity", 
                           "unit", "market_id", "market", "latitude", "longitude"], 
                          as_index=False)["usdprice"].median()
               .rename(columns={"usdprice": "price"}))
    monthly["log_price"] = np.log(monthly["price"])
    
    log(f"  Monthly observations: {len(monthly):,}")
    return monthly


def load_roads(cfg: Config) -> gpd.GeoDataFrame:
    """Load Chinese-funded road project data."""
    log("Loading roads...")
    path = os.path.join(data_dir(), "all_roads_filtered.gpkg")
    roads = gpd.read_file(path)
    roads = roads[roads["Recipient.ISO-3"].isin(cfg.countries)].copy()
    
    # Parse completion date and convert to month integer
    date_col = "Actual.Completion.Date.(MM/DD/YYYY)"
    roads[date_col] = pd.to_datetime(roads[date_col], errors="coerce")
    roads = roads.dropna(subset=[date_col])
    roads["T_road"] = roads[date_col].dt.to_period("M").dt.to_timestamp().map(month_int)
    
    log(f"  Roads: {len(roads)}")
    for iso in cfg.countries:
        n = len(roads[roads["Recipient.ISO-3"] == iso])
        log(f"    {iso}: {n} projects")
    
    return roads


def assign_treatment(prices: pd.DataFrame, roads: gpd.GeoDataFrame, cfg: Config) -> pd.DataFrame:
    """
    Assign treatment status to markets based on proximity to roads.
    Treatment = road completed within buffer_km of market.
    """
    log("Assigning treatment...")
    markets = prices[["countryiso3", "market_id", "market", "latitude", "longitude"]].drop_duplicates()
    log(f"  Markets: {len(markets)}")
    
    # Convert to GeoDataFrame and project to meters (for buffering)
    markets_gdf = gpd.GeoDataFrame(
        markets, geometry=gpd.points_from_xy(markets["longitude"], markets["latitude"]), crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    
    # Create buffer zones around road projects
    roads_m = roads.to_crs("EPSG:3857")
    buffer_m = cfg.buffer_km * 1000  # Convert km to meters
    roads_m["buffer"] = roads_m.geometry.buffer(buffer_m)
    
    # Preserve project metadata for leave-one-out robustness checks
    keep_cols = ["T_road", "buffer", "Recipient.ISO-3"]
    for c in ["id", "Title"]:
        if c in roads_m.columns:
            keep_cols.append(c)

    buf = roads_m.set_geometry("buffer")[keep_cols].rename(columns={"Recipient.ISO-3": "road_country"})
    
    # Find which markets fall within road buffers
    log("  Spatial join...")
    joined = gpd.sjoin(markets_gdf[["market_id", "countryiso3", "geometry"]], buf, 
                       predicate="intersects", how="left")
    
    # Only assign treatment if road is in same country
    treated = joined[joined["countryiso3"] == joined["road_country"]].copy()

    # For each market, use earliest road completion date (if multiple roads nearby)
    treated = treated.sort_values(["market_id", "T_road"])
    treated_first = treated.dropna(subset=["T_road"]).groupby("market_id", as_index=False).first()
    treated_first = treated_first.rename(columns={"T_road": "T_i"})

    # Rename metadata columns (if present)
    if "id" in treated_first.columns:
        treated_first = treated_first.rename(columns={"id": "T_i_project_id"})
    if "Title" in treated_first.columns:
        treated_first = treated_first.rename(columns={"Title": "T_i_project_title"})

    # Avoid column name collisions on merge (keep countryiso3 from markets)
    drop_collision_cols = [
        c for c in ["countryiso3", "road_country", "geometry", "index_right"] if c in treated_first.columns
    ]
    if drop_collision_cols:
        treated_first = treated_first.drop(columns=drop_collision_cols)
    
    result = markets.merge(treated_first, on="market_id", how="left")
    log(f"  Treated markets: {result['T_i'].notna().sum()}")
    
    return result


def build_pairs(markets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Create all valid market pairs (same country, within max distance).
    Pair treatment time = earliest treatment time of either market (if both treated).
    """
    log("Building market pairs...")
    mlist = markets.to_dict("records")
    n = len(mlist)
    
    rows = []
    for a in range(n):
        if a % 50 == 0:
            log(f"  Processing market {a}/{n}...")
        for b in range(a + 1, n):
            # Only pair markets within same country
            if mlist[a]["countryiso3"] != mlist[b]["countryiso3"]:
                continue
            
            # Calculate distance between markets
            d = haversine_km(mlist[a]["latitude"], mlist[a]["longitude"],
                             mlist[b]["latitude"], mlist[b]["longitude"])
            
            # Only keep pairs within distance threshold
            if d <= cfg.max_pair_km:
                T_i, T_j = mlist[a].get("T_i"), mlist[b].get("T_i")
                # Pair is treated only if BOTH markets are treated (use earliest date)
                T_ij = min(T_i, T_j) if (pd.notna(T_i) and pd.notna(T_j)) else np.nan
                rows.append({
                    "market_i": mlist[a]["market_id"],
                    "market_j": mlist[b]["market_id"],
                    "distance_km": d,
                    "T_ij": T_ij,
                    "countryiso3": mlist[a]["countryiso3"],
                })
    
    pairs = pd.DataFrame(rows)
    log(f"  Total pairs: {len(pairs):,}")
    log(f"  Treated pairs: {pairs['T_ij'].notna().sum()}")
    
    return pairs


def build_panel(prices: pd.DataFrame, pairs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Build analysis panel: pair-commodity-month observations.
    Outcome y = absolute log price difference between markets.
    """
    log("Building outcome panel...")
    
    # Focus on most-traded commodities to keep panel manageable
    counts = prices.groupby(["commodity_id", "unit"]).size().reset_index(name="n")
    top_cu = counts.nlargest(cfg.top_commodities, "n")[["commodity_id", "unit"]]
    pm = prices.merge(top_cu, on=["commodity_id", "unit"])
    log(f"  Top {len(top_cu)} commodities selected")
    
    # Merge prices for both markets in each pair
    log("  Merging prices for market_i...")
    panel = pairs.merge(
        pm[["market_id", "month_int", "commodity_id", "commodity", "unit", "log_price"]].rename(
            columns={"market_id": "market_i", "log_price": "lp_i"}), on="market_i")
    
    log("  Merging prices for market_j...")
    panel = panel.merge(
        pm[["market_id", "month_int", "commodity_id", "unit", "log_price"]].rename(
            columns={"market_id": "market_j", "log_price": "lp_j"}),
        on=["market_j", "month_int", "commodity_id", "unit"])
    
    # Outcome: absolute log price difference (price dispersion)
    panel["y"] = (panel["lp_i"] - panel["lp_j"]).abs()
    panel["pair_id"] = panel["market_i"].astype(str) + "_" + panel["market_j"].astype(str)
    
    # Create distance bins for heterogeneity analysis
    panel["dist_bin"] = pd.cut(panel["distance_km"], bins=[0, 50, 100, 200, 300], 
                               labels=["0-50km", "50-100km", "100-200km", "200-300km"])
    
    log(f"  Panel observations: {len(panel):,}")
    return panel


def main():
    """
    Main data preparation pipeline:
    1. Load raw price and road data
    2. Assign treatment based on spatial proximity
    3. Create market pairs
    4. Build analysis panel with price dispersion outcomes
    """
    cfg = Config()
    ensure_dirs()
    
    log("=" * 60)
    log("STEP 1: PREPARE DATA")
    log("=" * 60)
    
    t0 = time.time()
    
    # Load and process data
    prices = load_prices(cfg)
    roads = load_roads(cfg)
    markets = assign_treatment(prices, roads, cfg)
    pairs = build_pairs(markets, cfg)
    panel = build_panel(prices, pairs, cfg)
    
    # Save processed data for analysis scripts
    log("Saving data files...")
    markets.to_csv(os.path.join(data_dir(), "markets.csv"), index=False)
    pairs.to_csv(os.path.join(data_dir(), "pairs.csv"), index=False)
    panel.to_csv(os.path.join(data_dir(), "panel.csv"), index=False)
    prices[prices["countryiso3"].isin(cfg.countries)].to_csv(
        os.path.join(data_dir(), "prices.csv"), index=False)
    
    log(f"Done in {time.time() - t0:.1f}s")
    log(f"Output: {data_dir()}")


if __name__ == "__main__":
    main()

