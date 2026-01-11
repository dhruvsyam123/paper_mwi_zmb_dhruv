"""
Road Infrastructure Impact Simulator v4
========================================
Interactive visualization of how Chinese-funded roads affect 
food price dispersion in Malawi and Zambia.

Run with: streamlit run streamlit_app.py
"""

import os
import math
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Road Impact Simulator",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a polished, modern look
st.markdown("""
<style>
    /* Import distinctive fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    /* Global styling */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a5f2a 0%, #2d8a4e 50%, #1a5f2a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.15rem;
        color: #555;
        font-weight: 300;
        margin-bottom: 2rem;
        border-bottom: 2px solid #e8f5e9;
        padding-bottom: 1rem;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        color: #1a5f2a;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Key insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #2d8a4e;
        padding: 1.25rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .insight-box h4 {
        color: #1a5f2a;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .insight-box p {
        color: #333;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Watch for box - key narrative */
    .watch-for-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border: 2px solid #ff9800;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .watch-for-box h4 {
        color: #e65100;
        margin: 0 0 0.75rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .watch-for-box ul {
        margin: 0;
        padding-left: 1.25rem;
    }
    
    .watch-for-box li {
        color: #333;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    /* Instructions box */
    .instruction-box {
        background: #f8f9fa;
        border: 2px dashed #90a4ae;
        padding: 1.25rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .instruction-box h4 {
        color: #37474f;
        margin: 0 0 0.75rem 0;
        font-weight: 600;
    }
    
    .instruction-box ol {
        margin: 0;
        padding-left: 1.25rem;
    }
    
    .instruction-box li {
        color: #455a64;
        margin-bottom: 0.4rem;
    }
    
    /* Legend styling */
    .legend-item {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0.4rem 0;
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.95rem;
    }
    
    .legend-dot {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        border: 2px solid #333;
        flex-shrink: 0;
    }
    
    .legend-road {
        width: 30px;
        height: 8px;
        background: #c62828;
        border-radius: 2px;
        flex-shrink: 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        color: #1a5f2a;
        font-weight: 700;
    }
    
    .metric-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f5f5f5;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #2d8a4e !important;
        color: white !important;
    }
    
    /* Video container */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.85rem;
        color: #888;
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    project_root = os.path.dirname(base_dir)
    
    markets = pd.read_csv(os.path.join(data_dir, "markets.csv"))
    panel = pd.read_csv(os.path.join(data_dir, "panel.csv"))
    
    roads = gpd.read_file(os.path.join(project_root, "all_roads_filtered.gpkg"))
    roads = roads[roads["Recipient.ISO-3"].isin(["MWI", "ZMB"])].to_crs("EPSG:4326")
    
    # Parse road completion dates
    date_col = "Actual.Completion.Date.(MM/DD/YYYY)"
    roads["completion_month"] = pd.to_datetime(roads[date_col], errors='coerce')
    roads["month_int"] = roads["completion_month"].apply(
        lambda x: x.year * 12 + x.month if pd.notna(x) else None
    )
    
    coefs = pd.read_csv(os.path.join(data_dir, "event_study_coefs.csv"))
    
    try:
        ne_path = os.path.join(data_dir, "ne_countries", "ne_110m_admin_0_countries.shp")
        countries = gpd.read_file(ne_path)
        countries = countries[countries['ISO_A3'].isin(['MWI', 'ZMB', 'TZA', 'MOZ', 'ZWE', 'COD'])].to_crs("EPSG:4326")
    except:
        countries = None
    
    return markets, panel, roads, coefs, countries


def month_int_to_str(m: int) -> str:
    year = m // 12
    month = m % 12
    if month == 0:
        month = 12
        year -= 1
    return f"{year}-{month:02d}"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def point_to_polyline_distance(px, py, coords):
    """Calculate minimum distance from point to any segment of a polyline."""
    min_dist = float('inf')
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i][0], coords[i][1]
        lon2, lat2 = coords[i + 1][0], coords[i + 1][1]
        
        dx, dy = lon2 - lon1, lat2 - lat1
        if dx == 0 and dy == 0:
            dist = haversine_km(py, px, lat1, lon1)
        else:
            t = max(0, min(1, ((px - lon1) * dx + (py - lat1) * dy) / (dx * dx + dy * dy)))
            proj_lon = lon1 + t * dx
            proj_lat = lat1 + t * dy
            dist = haversine_km(py, px, proj_lat, proj_lon)
        min_dist = min(min_dist, dist)
    return min_dist


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING MAP - POLYLINE ONLY
# ─────────────────────────────────────────────────────────────────────────────

def create_draw_map(markets, countries, roads):
    """Create map with ONLY polyline drawing enabled."""
    
    center_lat = markets["latitude"].mean()
    center_lon = markets["longitude"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="cartodbpositron",
        width='100%',
        height='100%'
    )
    
    # Countries
    if countries is not None:
        for _, row in countries.iterrows():
            is_study = row['ISO_A3'] in ['MWI', 'ZMB']
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, study=is_study: {
                    "fillColor": "#e8f5e9" if study else "#f5f5f5",
                    "color": "#333" if study else "#bbb",
                    "weight": 2 if study else 1,
                    "fillOpacity": 0.4
                }
            ).add_to(m)
    
    # Existing roads (show in red)
    for _, road in roads.iterrows():
        try:
            folium.GeoJson(
                road.geometry.__geo_interface__,
                style_function=lambda x: {
                    "fillColor": "#c62828",
                    "color": "#7b1a1a", 
                    "weight": 3,
                    "fillOpacity": 0.6
                }
            ).add_to(m)
        except:
            pass
    
    # Markets
    for _, market in markets.iterrows():
        is_treated = pd.notna(market.get("T_i"))
        folium.CircleMarker(
            location=[market["latitude"], market["longitude"]],
            radius=8 if is_treated else 5,
            color="#1a5f2a" if is_treated else "#666",
            weight=2,
            fill=True,
            fillColor="#2d8a4e" if is_treated else "#aaa",
            fillOpacity=0.8,
            popup=f"<b>{market.get('market', 'Market')}</b><br>{market['countryiso3']}"
        ).add_to(m)
    
    # Drawing tools - POLYLINE ONLY (no polygon, no shapes)
    draw = Draw(
        draw_options={
            'polyline': {
                'shapeOptions': {
                    'color': '#00aa00',
                    'weight': 6,
                    'opacity': 0.9
                },
                'metric': True,
                'feet': False
            },
            'polygon': False,
            'rectangle': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': True, 'remove': True}
    )
    draw.add_to(m)
    
    return m


def analyze_drawing(drawn_data, markets, buffer_km=50, att=-0.032):
    """Analyze drawn polyline."""
    
    if not drawn_data:
        return None, []
    
    all_drawings = drawn_data.get("all_drawings", [])
    last_active = drawn_data.get("last_active_drawing")
    
    drawing = last_active if last_active else (all_drawings[-1] if all_drawings else None)
    
    if not drawing:
        return None, []
    
    geom_type = drawing.get("geometry", {}).get("type", "")
    coords = drawing.get("geometry", {}).get("coordinates", [])
    
    if not coords or geom_type != "LineString":
        return None, []
    
    if len(coords) < 2:
        return None, []
    
    # Calculate total length
    total_length = 0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        total_length += haversine_km(lat1, lon1, lat2, lon2)
    
    # Find affected markets
    affected = []
    for _, market in markets.iterrows():
        mlat, mlon = market["latitude"], market["longitude"]
        min_dist = point_to_polyline_distance(mlon, mlat, coords)
        
        if min_dist <= buffer_km:
            effect_mult = (1 - (min_dist / buffer_km) ** 0.7)
            predicted_effect = att * effect_mult
            
            affected.append({
                "market": market.get("market", f"ID {market['market_id']}"),
                "country": market["countryiso3"],
                "distance_km": round(min_dist, 1),
                "predicted_effect": predicted_effect,
                "is_treated": pd.notna(market.get("T_i"))
            })
    
    road_info = {
        "length_km": round(total_length, 1),
        "n_segments": len(coords) - 1
    }
    
    return road_info, affected


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<h1 class="main-title">🛣️ Road Infrastructure Impact Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Visualizing how Chinese-funded road projects affect food price dispersion in Malawi & Zambia</p>', unsafe_allow_html=True)
    
    # Load data
    try:
        markets, panel, roads, coefs, countries = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you've run the data preparation scripts first.")
        return
    
    # Tabs
    tab1, tab2 = st.tabs(["✏️  Draw Your Road", "🎬  Time-Lapse Animation"])
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: DRAW ROAD
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<h2 class="section-header">Simulate a Hypothetical Road</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>🎯 Purpose</h4>
            <p>Draw a road route to see which markets would be affected and the predicted impact on price dispersion based on our estimates.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_instr, col_buffer = st.columns([3, 1])
        
        with col_instr:
            st.markdown("""
            <div class="instruction-box">
                <h4>📐 How to Draw a Road</h4>
                <ol>
                    <li><strong>Click the polyline tool</strong> (diagonal line icon) in the map toolbar on the left</li>
                    <li><strong>Click on the map</strong> to place each point of your road path</li>
                    <li><strong>Keep clicking</strong> to add more segments (like connecting the dots)</li>
                    <li><strong>Double-click</strong> or press Enter to finish your road</li>
                </ol>
                <p style="margin-top: 0.75rem; color: #666; font-style: italic;">
                    Tip: Draw near the market dots to see the predicted impact!
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_buffer:
            buffer = st.slider(
                "Treatment radius (km)", 
                min_value=25, 
                max_value=100, 
                value=50, 
                step=5,
                help="Markets within this distance of the road are considered 'treated'"
            )
        
        st.markdown("---")
        
        col_map, col_results = st.columns([3, 2], gap="large")
        
        with col_map:
            m = create_draw_map(markets, countries, roads)
            output = st_folium(m, width=None, height=600, key="draw_map")
        
        with col_results:
            st.markdown("### 📊 Predicted Impact")
            
            road_info, affected = analyze_drawing(output, markets, buffer)
            
            if road_info:
                st.success(f"**Road detected!**  \n📏 Length: **{road_info['length_km']} km**  \n📍 Segments: **{road_info['n_segments']}**")
                
                if affected:
                    df = pd.DataFrame(affected)
                    
                    n_total = len(df)
                    n_new = len(df[~df["is_treated"]])
                    avg_effect = df["predicted_effect"].mean()
                    
                    # Metrics
                    c1, c2 = st.columns(2)
                    c1.metric("Markets Affected", n_total)
                    c2.metric("New Markets", n_new, help="Markets not already treated by existing roads")
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_effect:.1%}</div>
                        <div class="metric-label">Predicted Dispersion Reduction</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Effect chart
                    fig = go.Figure()
                    sorted_df = df.sort_values("distance_km")
                    fig.add_trace(go.Bar(
                        x=sorted_df["market"],
                        y=sorted_df["predicted_effect"] * 100,
                        marker_color=sorted_df["distance_km"],
                        marker_colorscale="RdYlGn_r",
                        marker_colorbar=dict(title="Distance<br>(km)"),
                        text=[f"{x:.1f}%" for x in sorted_df["predicted_effect"] * 100],
                        textposition="outside",
                        textfont=dict(size=10)
                    ))
                    fig.update_layout(
                        title=dict(text="Effect by Market", font=dict(size=14)),
                        xaxis_title="Market",
                        yaxis_title="Predicted Effect (%)",
                        template="plotly_white",
                        height=320,
                        xaxis_tickangle=-45,
                        margin=dict(b=80)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    with st.expander("📋 View Details"):
                        show_df = df[["market", "country", "distance_km", "predicted_effect"]].copy()
                        show_df["predicted_effect"] = show_df["predicted_effect"].apply(lambda x: f"{x:.1%}")
                        show_df.columns = ["Market", "Country", "Distance (km)", "Effect"]
                        st.dataframe(show_df.sort_values("Distance (km)"), hide_index=True, use_container_width=True)
                else:
                    st.warning("No markets within the treatment radius. Try drawing closer to the dots!")
            else:
                st.info("""
                **👆 Draw a road on the map to see predictions**
                
                Use the polyline tool from the toolbar, then click points on the map to trace your road route.
                """)
                
                st.markdown("""
                <div class="insight-box">
                    <h4>How It Works</h4>
                    <p>We use our estimated treatment effect of <strong>-3.2%</strong> (reduction in price dispersion) and scale it by distance: 
                    markets closer to the road see larger effects.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: ANIMATION
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<h2 class="section-header">Watch Markets Respond to Road Completion</h2>', unsafe_allow_html=True)
        
        col_video, col_info = st.columns([3, 2], gap="large")
        
        with col_video:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            mp4_path = os.path.join(app_dir, "animation.mp4")
            gif_path = os.path.join(app_dir, "animation.gif")
            
            if os.path.exists(mp4_path):
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.video(mp4_path)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("*Time-lapse animation: 2012–2023 • Each frame = 2 months*")
            elif os.path.exists(gif_path):
                st.image(gif_path, use_container_width=True)
                st.caption("*Time-lapse animation: 2012–2023*")
            else:
                st.warning("""
                **Animation not found!**  
                Generate it by running:
                ```bash
                python paper_mwi_zmb/app/generate_animation.py
                ```
                """)
        
        with col_info:
            # Key narrative
            st.markdown("""
            <div class="watch-for-box">
                <h4>👀 What to Watch For</h4>
                <p style="margin-bottom: 0.75rem; font-weight: 500;">
                    The key pattern: <strong>markets turn greener as roads appear nearby</strong>
                </p>
                <ul>
                    <li><strong>Red roads</strong> fade in when construction completes</li>
                    <li><strong>Market dots shift color</strong> from red/yellow → green</li>
                    <li><strong>Treated markets</strong> (larger dots with black border) show the effect most clearly</li>
                </ul>
                <p style="margin-top: 0.75rem; font-style: italic; color: #666;">
                    Greener = lower price dispersion = better market integration
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Legend
            st.markdown("#### Map Legend")
            st.markdown("""
            <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #ddd;">
                <div class="legend-item">
                    <div class="legend-dot" style="background: #2d8a4e;"></div>
                    <span><strong>Green dot</strong> — Low dispersion (well-integrated)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #ffc107;"></div>
                    <span><strong>Yellow dot</strong> — Medium dispersion</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #c62828;"></div>
                    <span><strong>Red dot</strong> — High dispersion (fragmented)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #888; border-width: 3px;"></div>
                    <span><strong>Thick border</strong> — Treated market (near road)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-road"></div>
                    <span><strong>Red shape</strong> — Completed road project</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Stats
            att = coefs[(coefs["k"] >= 0) & (coefs["k"] <= 6)]["beta"].mean()
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Road Projects", len(roads))
                st.metric("Treated Markets", int(markets["T_i"].notna().sum()))
            with col_b:
                st.metric("Countries", "2 (MWI, ZMB)")
                st.metric("Effect Size", f"{att:.1%}")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <strong>Data Sources:</strong> World Food Programme (food prices) • AidData (road projects) • Natural Earth (boundaries)<br>
        <strong>Methods:</strong> Staggered Difference-in-Differences with pair-level treatment assignment
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
