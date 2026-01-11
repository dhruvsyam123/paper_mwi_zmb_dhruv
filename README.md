# Price Dispersion Analysis: Malawi & Zambia

This project analyses the impact of road infrastructure improvements on food price dispersion in Malawi and Zambia using event study methodology.

## Project Structure

```
paper_mwi_zmb/
├── data/                           # Raw and processed data files
│   ├── all_food_prices_clean.csv   # Food price data
│   ├── all_roads_filtered.gpkg     # Road infrastructure data
│   ├── markets.csv                 # Market locations and metadata
│   ├── pairs.csv                   # Market pairs with distances
│   └── panel.csv                   # Panel data for analysis
│
├── scripts/                        # Analysis scripts (run in order)
│   ├── config.py                   # Configuration settings
│   ├── 01_prepare_data.py          # Data preparation
│   ├── 02_estimate_event_study.py  # Main event study estimation
│   ├── 03_generate_figures.py      # Generate figures
│   ├── 04_generate_tables.py       # Generate tables
│   ├── 05_robustness_analysis.py   # Robustness checks
│   ├── 06_generate_robustness_figure.py
│   ├── 07_event_study_extended_24m.py
│   └── run_all.py                  # Run all scripts sequentially
│
├── figures/                        # Output figures (PNG)
├── tables/                         # Output tables (CSV)
├── app/                           # Streamlit interactive app
└── marking/                       # Assignment guidelines
```

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - geopandas
  - matplotlib
  - seaborn
  - statsmodels
  - scipy

Install dependencies:
```bash
pip install pandas numpy geopandas matplotlib seaborn statsmodels scipy
```

## How to Run

### Option 1: Run All Scripts (Recommended)

From the `scripts/` directory:
```bash
cd scripts
python run_all.py
```

This will execute all analysis steps in order:
1. Prepare data (markets, pairs, panel)
2. Estimate event study
3. Generate figures
4. Generate tables
5. Run robustness checks
6. Generate robustness figures
7. Extended 24-month event study

### Option 2: Run Scripts Individually

```bash
cd scripts
python 01_prepare_data.py
python 02_estimate_event_study.py
python 03_generate_figures.py
python 04_generate_tables.py
python 05_robustness_analysis.py
python 06_generate_robustness_figure.py
python 07_event_study_extended_24m.py
```

## Configuration

Edit `scripts/config.py` to modify:
- Countries to analyze (default: MWI, ZMB)
- Treatment buffer distance (default: 50km)
- Maximum pair distance (default: 300km)
- Number of top commodities (default: 20)

## Output

After running the scripts, you'll find:

- **Figures**: `figures/` directory
  - Main event study plot
  - Heterogeneity by distance
  - Staple vs non-staple analysis
  - Maps and robustness checks

- **Tables**: `tables/` directory
  - Summary statistics
  - Event study coefficients
  - Heterogeneity analysis
  - Robustness checks

## Interactive App

To run the Streamlit interactive visualization:
```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notes

- The data preparation step (01) may take several minutes
- All output files are automatically saved to `figures/` and `tables/`
- The analysis uses a 50km buffer around road projects by default

