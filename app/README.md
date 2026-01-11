# 🛣️ Road Infrastructure Impact Simulator

Interactive visualization of how Chinese-funded road projects affect food price dispersion in Malawi and Zambia.

## Quick Start

```bash
# From the project root
pip install -r paper_mwi_zmb/app/requirements.txt
streamlit run paper_mwi_zmb/app/streamlit_app.py
```

The app will open at http://localhost:8501

## Features

The simulator provides two interactive views:

### ✏️ Draw Your Road
Simulate a hypothetical road investment:
1. Click the polyline tool in the map toolbar
2. Click points on the map to trace your road route
3. Double-click to finish
4. See which markets would be affected and the predicted dispersion reduction based on our estimated treatment effect of **-3.2%**

### 🎬 Time-Lapse Animation
Watch how markets respond to road completion over time (2012–2023). The key insight: **market dots turn greener as roads appear nearby**, indicating reduced price dispersion and improved market integration.

- **Green dots** = Low price dispersion (well-integrated markets)
- **Yellow dots** = Medium dispersion
- **Red dots** = High price dispersion (fragmented markets)
- **Larger dots with black border** = Treated markets (within 50km of a road)
- **Red shapes** = Completed road projects

## Prerequisites

Before running the app, generate the data by running:

```bash
cd paper_mwi_zmb/scripts
python run_all.py
```

To regenerate the animation:

```bash
python paper_mwi_zmb/app/generate_animation.py
```

## Data Sources

- **Food Prices**: World Food Programme (WFP) Food Prices dataset
- **Road Projects**: AidData Global Chinese Development Finance Dataset
- **Country Boundaries**: Natural Earth

## Key Finding

Road completion reduces pairwise food price dispersion by **3.2%** within 6 months (p=0.019), indicating improved market integration through reduced trade costs.
