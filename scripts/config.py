"""Shared configuration for paper scripts."""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Config:
    # Geographic scope
    countries: Tuple[str, ...] = ("MWI", "ZMB")
    country_names: Dict[str, str] = field(default_factory=lambda: {"MWI": "Malawi", "ZMB": "Zambia"})
    
    # Treatment assignment: markets within 50km of road are "treated"
    buffer_km: float = 50.0
    
    # Market pairs: only consider markets within 300km of each other
    max_pair_km: float = 300.0
    
    # Data filters
    min_market_months: int = 18  # Minimum observations per market
    top_commodities: int = 10    # Number of most-traded commodities to analyze
    
    # Event study window
    pre_months: int = 12   # Months before road completion
    post_months: int = 12  # Months after road completion
    
    # Statistical inference
    bootstrap_reps: int = 100  # Bootstrap replications for standard errors
    seed: int = 42             # Random seed for reproducibility


def project_root() -> str:
    """Get path to project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def paper_dir() -> str:
    """Get path to paper directory (paper_mwi_zmb/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir() -> str:
    """Get path to data directory."""
    return os.path.join(paper_dir(), "data")


def fig_dir() -> str:
    """Get path to figures directory."""
    return os.path.join(paper_dir(), "figures")


def table_dir() -> str:
    """Get path to tables directory."""
    return os.path.join(paper_dir(), "tables")


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [data_dir(), fig_dir(), table_dir()]:
        os.makedirs(d, exist_ok=True)

