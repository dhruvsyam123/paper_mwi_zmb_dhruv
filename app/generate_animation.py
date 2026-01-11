"""
Generate smooth animated video/GIF of road impact over time.
Creates: animation.gif and animation.mp4

This animation shows the key insight: markets turn greener (lower price dispersion)
as road projects are completed nearby.
"""

import os
import sys
import math
from io import BytesIO

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """Load all data."""
    markets = pd.read_csv(os.path.join(DATA_DIR, "markets.csv"))
    panel = pd.read_csv(os.path.join(DATA_DIR, "panel.csv"))
    
    roads = gpd.read_file(os.path.join(PROJECT_ROOT, "all_roads_filtered.gpkg"))
    roads = roads[roads["Recipient.ISO-3"].isin(["MWI", "ZMB"])].to_crs("EPSG:4326")
    
    date_col = "Actual.Completion.Date.(MM/DD/YYYY)"
    roads["completion"] = pd.to_datetime(roads[date_col], errors='coerce')
    roads["month_int"] = roads["completion"].apply(
        lambda x: x.year * 12 + x.month if pd.notna(x) else None
    )
    
    try:
        ne_path = os.path.join(DATA_DIR, "ne_countries", "ne_110m_admin_0_countries.shp")
        countries = gpd.read_file(ne_path)
        countries = countries[countries['ISO_A3'].isin(['MWI', 'ZMB'])].to_crs("EPSG:4326")
    except:
        countries = None
    
    return markets, panel, roads, countries


def month_to_str(m):
    year = m // 12
    month = m % 12
    if month == 0:
        month = 12
        year -= 1
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return f"{months[month-1]} {year}"


def get_dispersion_color(value, vmin=0.16, vmax=0.30):
    """Return color from green (low) to red (high)."""
    norm = np.clip((value - vmin) / (vmax - vmin), 0, 1)
    # Use a green-yellow-red colormap (reversed so green=low)
    cmap = plt.cm.RdYlGn_r
    return cmap(norm)


def create_animation_frames(markets, panel, roads, countries, 
                            start_month, end_month, step=2):
    """Generate animation frames."""
    
    print(f"Generating frames from {month_to_str(start_month)} to {month_to_str(end_month)}...")
    
    # Figure setup - larger and more polished
    fig, ax = plt.subplots(figsize=(14, 11), dpi=120)
    
    # Get bounds with padding
    min_lon = markets["longitude"].min() - 2
    max_lon = markets["longitude"].max() + 2
    min_lat = markets["latitude"].min() - 2
    max_lat = markets["latitude"].max() + 2
    
    frames = []
    months = list(range(start_month, end_month + 1, step))
    
    for i, current_month in enumerate(months):
        ax.clear()
        
        # Background - subtle gradient effect
        ax.set_facecolor('#f5f8fa')
        
        # Draw countries
        if countries is not None:
            for _, row in countries.iterrows():
                if row.geometry.geom_type == 'Polygon':
                    x, y = row.geometry.exterior.xy
                    ax.fill(x, y, facecolor='#e8f5e9', edgecolor='#2d5a3d', 
                           linewidth=2, alpha=0.9)
                elif row.geometry.geom_type == 'MultiPolygon':
                    for poly in row.geometry.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, facecolor='#e8f5e9', edgecolor='#2d5a3d', 
                               linewidth=2, alpha=0.9)
        
        # Count completed roads
        completed_roads = 0
        
        # Draw roads (only if completed)
        for _, road in roads.iterrows():
            road_month = road.get("month_int")
            if pd.isna(road_month):
                continue
            
            if road_month <= current_month:
                completed_roads += 1
                # Fade in effect
                months_since = current_month - road_month
                alpha = min(1.0, 0.4 + months_since / 18)
                
                try:
                    if road.geometry.geom_type == 'MultiPolygon':
                        for poly in road.geometry.geoms:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, facecolor='#c62828', edgecolor='#7b1a1a', 
                                   linewidth=2.5, alpha=alpha, zorder=3)
                    elif road.geometry.geom_type == 'Polygon':
                        x, y = road.geometry.exterior.xy
                        ax.fill(x, y, facecolor='#c62828', edgecolor='#7b1a1a', 
                               linewidth=2.5, alpha=alpha, zorder=3)
                except:
                    pass
        
        # Get dispersion for this month
        month_data = panel[panel["month_int"] == current_month]
        market_disp = month_data.groupby("market_i")["y"].mean().to_dict()
        
        # Determine treated markets
        treated_now = set()
        for _, m in markets.iterrows():
            t_i = m.get("T_i")
            if pd.notna(t_i) and t_i <= current_month:
                treated_now.add(m["market_id"])
        
        # Calculate average dispersion
        disp_values = list(market_disp.values())
        avg_disp = np.mean(disp_values) if disp_values else 0.25
        
        # Draw markets
        for _, m in markets.iterrows():
            mid = m["market_id"]
            disp = market_disp.get(mid, 0.25)
            is_treated = mid in treated_now
            
            color = get_dispersion_color(disp)
            size = 220 if is_treated else 100
            edge = 'black' if is_treated else '#555'
            edge_width = 2.5 if is_treated else 1
            
            ax.scatter(m["longitude"], m["latitude"], 
                      s=size, c=[color], edgecolors=edge, 
                      linewidths=edge_width, zorder=5, alpha=0.9)
        
        # Title - clean and modern
        ax.text(0.5, 1.02, "Road Infrastructure Impact on Food Price Dispersion", 
               transform=ax.transAxes, fontsize=18, fontweight='bold', 
               ha='center', va='bottom', color='#1a3d2c',
               fontfamily='sans-serif')
        
        # Date display - prominent
        ax.text(0.5, 0.97, month_to_str(current_month), 
               transform=ax.transAxes, fontsize=24, fontweight='bold', 
               ha='center', va='top', color='#2d5a3d',
               fontfamily='sans-serif',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='#2d5a3d', linewidth=2, alpha=0.95))
        
        # Legend - positioned in corner
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2d8a4e', 
                   markersize=12, markeredgecolor='#333', markeredgewidth=1,
                   label='Low dispersion (integrated)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffc107', 
                   markersize=12, markeredgecolor='#333', markeredgewidth=1,
                   label='Medium dispersion'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#c62828', 
                   markersize=12, markeredgecolor='#333', markeredgewidth=1,
                   label='High dispersion (fragmented)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=14, markeredgecolor='black', markeredgewidth=2.5,
                   label='Treated market'),
            Patch(facecolor='#c62828', edgecolor='#7b1a1a', linewidth=2,
                  label='Completed road project'),
        ]
        legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=11, 
                          framealpha=0.95, fancybox=True, edgecolor='#ccc',
                          title='Legend', title_fontsize=12)
        legend.get_frame().set_linewidth(1.5)
        
        # Stats box - modern design
        n_treated = len(treated_now)
        stats_text = (f"Roads completed: {completed_roads}/{len(roads)}   •   "
                     f"Treated markets: {n_treated}   •   "
                     f"Avg. dispersion: {avg_disp:.3f}")
        
        ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=13, ha='center', va='bottom', fontweight='500',
               fontfamily='sans-serif',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='#2d5a3d', linewidth=1.5, alpha=0.95))
        
        # Key insight box (top right)
        insight_text = "Watch: Markets turn\ngreener as roads appear"
        ax.text(0.98, 0.98, insight_text, transform=ax.transAxes,
               fontsize=11, ha='right', va='top', style='italic',
               color='#1a5f2a', fontweight='500',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0', 
                        edgecolor='#ff9800', linewidth=1.5, alpha=0.95))
        
        # Axes
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_xlabel("Longitude", fontsize=12, fontweight='500')
        ax.set_ylabel("Latitude", fontsize=12, fontweight='500')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=10)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', color='#999')
        
        # Convert to image
        fig.canvas.draw()
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', 
                   pad_inches=0.15, facecolor='white')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        frames.append(img.copy())
        buf.close()
        
        # Progress
        if (i + 1) % 10 == 0 or i == len(months) - 1:
            print(f"  Frame {i + 1}/{len(months)} ({month_to_str(current_month)})")
    
    plt.close(fig)
    return frames


def save_gif(frames, output_path, duration=150):
    """Save frames as animated GIF."""
    print(f"Saving GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    print(f"  ✓ GIF saved ({os.path.getsize(output_path) / 1e6:.1f} MB)")


def save_mp4(frames, output_path, fps=8):
    """Save frames as MP4 video."""
    try:
        import imageio
        print(f"Saving MP4 to {output_path}...")
        with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
            for frame in frames:
                writer.append_data(np.array(frame))
        print(f"  ✓ MP4 saved ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    except ImportError:
        print("  ⚠ imageio not installed, skipping MP4. Run: pip install imageio[ffmpeg]")


def main():
    print("=" * 65)
    print("  GENERATING ROAD IMPACT ANIMATION")
    print("=" * 65)
    
    # Load data
    print("\nLoading data...")
    markets, panel, roads, countries = load_data()
    print(f"  Markets: {len(markets)}")
    print(f"  Roads: {len(roads)}")
    print(f"  Panel observations: {len(panel):,}")
    
    # Time range - focus on when roads were built
    road_months = roads["month_int"].dropna()
    if len(road_months) > 0:
        start = int(road_months.min()) - 12
        end = int(road_months.max()) + 24
    else:
        start = int(panel["month_int"].min())
        end = int(panel["month_int"].max())
    
    # Ensure within panel range
    start = max(start, int(panel["month_int"].min()))
    end = min(end, int(panel["month_int"].max()))
    
    print(f"\nTime range: {month_to_str(start)} to {month_to_str(end)}")
    
    # Generate frames (every 2 months for smooth animation)
    frames = create_animation_frames(markets, panel, roads, countries, 
                                     start, end, step=2)
    
    # Save outputs
    print("\n" + "-" * 40)
    print("Saving outputs...")
    gif_path = os.path.join(OUTPUT_DIR, "animation.gif")
    mp4_path = os.path.join(OUTPUT_DIR, "animation.mp4")
    
    save_gif(frames, gif_path, duration=150)  # 150ms per frame
    save_mp4(frames, mp4_path, fps=8)
    
    print("\n" + "=" * 65)
    print("  COMPLETE!")
    print("=" * 65)
    print(f"\nOutput files:")
    print(f"  📊 GIF: {gif_path}")
    print(f"  🎬 MP4: {mp4_path}")
    print("\nThe animation shows markets turning greener as roads are completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
