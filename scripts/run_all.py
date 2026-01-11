#!/usr/bin/env python
"""
Run all scripts to generate paper outputs.

Usage: python run_all.py
"""

import os
import sys
import time
import subprocess


def log(msg: str):
    """Print formatted section header."""
    print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)


def run_script(name: str) -> bool:
    """
    Execute a Python script and report success/failure.
    Returns True if script completed successfully.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, name)
    
    print(f"\n>>> Running {name}...\n", flush=True)
    t0 = time.time()
    
    # Run script as subprocess
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=script_dir,
        capture_output=False
    )
    
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n✓ {name} completed in {elapsed:.1f}s", flush=True)
        return True
    else:
        print(f"\n✗ {name} FAILED (exit code {result.returncode})", flush=True)
        return False


def main():
    """
    Run complete paper generation pipeline:
    1. Prepare data (load, clean, assign treatment)
    2. Estimate event study coefficients
    3. Generate all figures
    4. Generate all tables
    """
    log("PAPER GENERATION PIPELINE")
    print("Infrastructure and Market Integration: MWI + ZMB")
    
    t0 = time.time()
    
    # Core pipeline scripts (run in sequence)
    scripts = [
        "01_prepare_data.py",
        "02_estimate_event_study.py",
        "03_generate_figures.py",
        "04_generate_tables.py",
    ]
    
    # Execute each script; stop if any fail
    for script in scripts:
        if not run_script(script):
            log("PIPELINE FAILED")
            sys.exit(1)
    
    elapsed = time.time() - t0
    
    # Report success and output locations
    log("PIPELINE COMPLETE")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\nOutputs:")
    
    paper_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"  Figures: {os.path.join(paper_dir, 'figures')}")
    print(f"  Tables:  {os.path.join(paper_dir, 'tables')}")
    print(f"  Data:    {os.path.join(paper_dir, 'data')}")
    print(f"  Paper:   {os.path.join(paper_dir, 'paper.md')}")


if __name__ == "__main__":
    main()

