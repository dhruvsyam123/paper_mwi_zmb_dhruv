"""
One-off: Generate an extended event study (-24 to +24 months) for inspection.

This script does NOT modify the main paper pipeline outputs.
It creates:
  - data/event_study_coefs_24m.csv
  - figures/fig_event_study_24m.png
"""

import os
import time
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from config import Config, data_dir, fig_dir, ensure_dirs
import importlib.util
from pathlib import Path

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    """
    Extended event study analysis: -24 to +24 months.
    
    This is a supplementary analysis to examine longer-run effects.
    Main analysis uses 12-month window; this extends to 24 months for robustness.
    """
    ensure_dirs()
    cfg = Config()
    # Extend window to 24 months pre/post
    cfg.pre_months = 24
    cfg.post_months = 24
    cfg.bootstrap_reps = 100

    log("Loading panel...")
    panel = pd.read_csv(os.path.join(data_dir(), "panel.csv"))
    log(f"Panel: {len(panel):,} rows")

    log("Estimating extended event study (-24 to +24)...")
    # Import estimation function from main event study script
    this_dir = Path(__file__).resolve().parent
    est_path = this_dir / "02_estimate_event_study.py"
    spec = importlib.util.spec_from_file_location("estimate_event_study_mod", est_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    estimate_event_study = getattr(mod, "estimate_event_study")

    # Run event study with extended window
    coefs = estimate_event_study(panel, cfg, bootstrap_reps=cfg.bootstrap_reps)
    out_csv = os.path.join(data_dir(), "event_study_coefs_24m.csv")
    coefs.to_csv(out_csv, index=False)
    log(f"✓ Saved {out_csv}")

    # Create extended plot
    log("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(coefs["k"], coefs["ci_lo"], coefs["ci_hi"], alpha=0.15, color="#2E86AB")
    ax.plot(coefs["k"], coefs["beta"], "o-", color="#2E86AB", markersize=6, linewidth=2.2, zorder=3)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="#d73027", linestyle="--", linewidth=2, label="Road completion")
    ax.set_xlabel("Months relative to road completion", fontsize=13)
    ax.set_ylabel("Effect on price dispersion (log points)", fontsize=13)
    ax.set_title("Event Study (Extended Window): -24 to +24 Months", fontsize=15, fontweight="bold")
    ax.set_xlim(-25, 25)
    ax.set_xticks(range(-24, 25, 4))
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)

    out_png = os.path.join(fig_dir(), "fig_event_study_24m.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    log(f"✓ Saved {out_png}")


if __name__ == "__main__":
    main()


