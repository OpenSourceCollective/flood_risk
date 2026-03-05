#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Flood Viewer — on‑map legends (no colormap selectors)

LLM Integration (Optional):
  Set environment variable to enable AI-generated risk explanations:
    export OPENAI_API_KEY="sk-..."        # For OpenAI GPT-4
    export ANTHROPIC_API_KEY="sk-ant-..."  # For Anthropic Claude
  
  If no API key is set, falls back to rule-based dynamic explanations.
"""
import json, os, base64
from typing import Optional, Tuple
import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_folium import st_folium
import folium
from flood_risk_compute import run as recompute_flood_risk
from branca.element import Element
import pandas as pd
import requests
import hashlib
import time
import sys
from pathlib import Path
import re

# Import fetch_physical functions directly (instead of subprocess)
try:
    from fetch_physical import main as fetch_physical_main
    DIRECT_FETCH_AVAILABLE = True
except ImportError:
    DIRECT_FETCH_AVAILABLE = False

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

# Contact email used for Nominatim per their usage policy
NOMINATIM_CONTACT_EMAIL = "axumaicollective@gmail.com"

st.set_page_config(page_title="Flood Risk Viewer", layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def read_summary(summary_path: str) -> dict:
    """Cached summary JSON reading to avoid repeated file I/O."""
    with open(summary_path, "r") as f:
        return json.load(f)

def raster_bounds_latlon(path: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
    return (bottom, left), (top, right)

def read_raster_array_and_stats(path: str, nodata=None, max_dim: int = 2000):
    with rasterio.open(path) as src:
        height, width = src.height, src.width
        if max(height, width) > max_dim:
            scale = max_dim / float(max(height, width))
            out_h = max(1, int(round(height * scale)))
            out_w = max(1, int(round(width * scale)))
            arr = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype("float32")
        else:
            arr = src.read(1).astype("float32")
        if nodata is None and src.nodata is not None:
            nodata = src.nodata
    mask = np.zeros_like(arr, dtype=bool)
    if nodata is not None:
        mask |= np.isclose(arr, nodata)
    mask |= ~np.isfinite(arr)
    valid = arr[~mask]
    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(valid, 2.0))
        vmax = float(np.percentile(valid, 98.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)
            if vmax <= vmin:
                vmin, vmax = 0.0, 1.0
    return arr, mask, vmin, vmax

def raster_to_rgba_image(path: str, cmap_name: str,
                         vmin: Optional[float] = None, vmax: Optional[float] = None,
                         nodata=None, max_dim: int = 2000, overlay_style: str = "transparent"):
    arr, mask, v_auto_min, v_auto_max = read_raster_array_and_stats(path, nodata=nodata, max_dim=max_dim)
    vmin = vmin if vmin is not None else v_auto_min
    vmax = vmax if vmax is not None else v_auto_max
    normed = (arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(arr, dtype="float32")
    normed = np.clip(normed, 0, 1)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(normed)
    
    # Apply different overlay styles
    if overlay_style == "transparent":
        # Make low values more transparent to preserve map labels
        rgba[:, :, 3] = rgba[:, :, 3] * (0.3 + 0.7 * normed)  # Scale alpha based on data intensity
    elif overlay_style == "contour":
        # Create contour-like effect with transparency
        rgba[:, :, 3] = np.where(normed > 0.3, rgba[:, :, 3] * 0.6, 0.0)
    elif overlay_style == "multiply":
        # Multiply blend mode simulation
        rgba[:, :, 3] = 0.4  # Fixed lower opacity for multiply effect
    
    rgba[mask, 3] = 0.0
    return (rgba * 255).astype("uint8"), (vmin, vmax)

@st.cache_data(ttl=3600, show_spinner=False, max_entries=50)
def raster_to_rgba_image_cached(path: str, cmap_name: str,
                                vmin: Optional[float] = None, vmax: Optional[float] = None,
                                nodata=None, max_dim: int = 2000, overlay_style: str = "transparent"):
    """Cached version of raster_to_rgba_image. Cache key includes all parameters.
    Saves 1-3 seconds per layer by avoiding repeated file I/O and array processing."""
    return raster_to_rgba_image(path, cmap_name, vmin, vmax, nodata, max_dim, overlay_style)

def add_image_overlay(m, img_rgba: np.ndarray, bounds, name: str, opacity: float = 0.7, overlay_style: str = "transparent"):
    # Adjust opacity based on overlay style
    if overlay_style == "transparent":
        adjusted_opacity = opacity * 0.6  # Reduce base opacity for transparent style
    elif overlay_style == "contour":
        adjusted_opacity = opacity * 0.8  # Higher opacity for contours
    elif overlay_style == "multiply":
        adjusted_opacity = opacity * 0.4  # Lower opacity for multiply blend
    else:
        adjusted_opacity = opacity
    
    overlay = folium.raster_layers.ImageOverlay(
        image=img_rgba,
        bounds=[[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]],
        opacity=adjusted_opacity,
        name=name,
        interactive=True,
        cross_origin=False,
        zindex=1,
        alt=name
    )
    overlay.add_to(m)

def make_continuous_legend_png(cmap_name: str, vmin: float, vmax: float, title: str, width_px=260) -> bytes:
    fig, ax = plt.subplots(figsize=(3.2, 1.0), dpi=200)
    fig.subplots_adjust(bottom=0.35, top=0.85, left=0.08, right=0.98)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    ax.set_title(title, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    bio.seek(0); img = Image.open(bio).convert("RGBA")
    w, h = img.size; new_h = int(h * (width_px / float(w)))
    img = img.resize((width_px, new_h), Image.LANCZOS)
    bio2 = BytesIO(); img.save(bio2, format="PNG")
    return bio2.getvalue()

@st.cache_data(ttl=3600, show_spinner=False, max_entries=50)
def make_continuous_legend_png_cached(cmap_name: str, vmin: float, vmax: float, title: str, width_px=260) -> bytes:
    """Cached legend generation. Matplotlib figure creation is expensive (~0.5-1.5s per legend).
    Cache key: colormap, vmin, vmax, title, width."""
    return make_continuous_legend_png(cmap_name, vmin, vmax, title, width_px)

LULC_CLASSES = [(1,"Water"),(2,"Urban / Impervious"),(3,"Vegetation"),(4,"Agriculture"),(5,"Wetland"),(6,"Other")]

def make_lulc_legend_png(width_px=240) -> bytes:
    palette = matplotlib.colormaps.get_cmap("tab10")
    row_h = 24
    img_h = row_h * len(LULC_CLASSES) + 16
    img = Image.new("RGBA", (width_px, img_h), (255, 255, 255, 220))
    draw = ImageDraw.Draw(img)
    x0, y = 10, 8; sw = 18
    for code, label in LULC_CLASSES:
        color = tuple(int(c * 255) for c in palette((code - 1) % 10)[:3]) + (255,)
        draw.rectangle([x0, y + 3, x0 + sw, y + 3 + sw], fill=color, outline=(40,40,40,255))
        draw.text((x0 + sw + 10, y + 2), f"{code} — {label}", fill=(10,10,10,255))
        y += row_h
    bio = BytesIO(); img.save(bio, format="PNG"); return bio.getvalue()

@st.cache_data(ttl=3600, show_spinner=False)
def make_lulc_legend_png_cached(width_px=240) -> bytes:
    """Cached LULC legend generation."""
    return make_lulc_legend_png(width_px)

def add_onmap_legend(map_obj, img_bytes: bytes, position: str = "bottomright", zindex: int = 1000, width_px: int = 200):
    b64 = base64.b64encode(img_bytes).decode("ascii")
    style_by_pos = {
        "bottomright": "position:absolute; bottom:10px; right:10px;",
        "bottomleft":  "position:absolute; bottom:10px; left:10px;",
        "topright":    "position:absolute; top:10px; right:10px;",
        "topleft":     "position:absolute; top:10px; left:10px;",
    }
    style = style_by_pos.get(position, style_by_pos["bottomright"])
    html = f'<div style="{style} z-index:{zindex}; background: rgba(255,255,255,0.8); padding:6px; border-radius:6px; box-shadow: 0 1px 4px rgba(0,0,0,0.25);"><img src="data:image/png;base64,{b64}" style="width:{width_px}px; height:auto;" /></div>'
    map_obj.get_root().html.add_child(Element(html))

# ---- Location search helpers (lat/lon parse, geocode, nearest-grid, raster sampler)

_GRID_CACHE = {}
_KD = None
_GRID_COORDS = None

def parse_latlon(text: str):
    if not isinstance(text, str):
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except Exception:
        return None
    return None

@st.cache_resource(show_spinner=False)
def load_grid_cells(path: str = "data/grid_cells.csv"):
    """Cached grid cells loading with KDTree construction.
    Uses @st.cache_resource since KDTree objects are not serializable.
    Saves 0.2-1 second on location searches by building KDTree once per session."""
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    coords = np.vstack([df['lat'].values, df['lon'].values]).T
    if KDTree is not None:
        try:
            kd = KDTree(coords)
        except Exception:
            kd = None
    else:
        kd = None
    return (df, kd, coords)

def nearest_grid_point(lat: float, lon: float, grid_path: str = "data/grid_cells.csv"):
    loaded = load_grid_cells(grid_path)
    if not loaded:
        return None
    df, kd, coords = loaded
    if kd is not None:
        dist, idx = kd.query([lat, lon], k=1)
        row = df.iloc[int(idx)].to_dict()
        row['__nn_dist_deg'] = float(dist)
        return row
    # fallback brute-force (euclidean degrees)
    d = np.sqrt((coords[:,0] - lat)**2 + (coords[:,1] - lon)**2)
    idx = int(np.argmin(d))
    row = df.iloc[idx].to_dict()
    row['__nn_dist_deg'] = float(d[idx])
    return row

def geocode_nominatim(query: str, limit: int = 6, user_agent_email: str = None):
    q = query.strip()
    if q == "":
        return []
    if user_agent_email is None:
        user_agent_email = NOMINATIM_CONTACT_EMAIL
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(q.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"geocode_{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            try:
                cache_file.unlink()
            except Exception:
                pass
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        "User-Agent": f"flood-risk-viewer ({user_agent_email})",
        "Accept": "application/json",
        "Referer": "http://localhost/",
    }
    params = {"q": q, "format": "json", "limit": limit}
    try:
        time.sleep(0.5)
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        # If blocked or forbidden, attempt a gentler fallback UA to see if it's a UA-block
        if resp.status_code == 403 or (isinstance(resp.text, str) and 'Access blocked' in resp.text):
            # fallback to a permissive curl UA (not ideal long-term — set a real contact email to avoid blocks)
            headers_fb = {**headers, 'User-Agent': 'curl/8.7.1'}
            try:
                resp = requests.get(url, headers=headers_fb, params=params, timeout=10)
            except Exception:
                pass
        resp.raise_for_status()
        data = resp.json()
        # normalize lat/lon to floats and small dict keys
        out = []
        for r in data:
            try:
                lat = float(r.get('lat'))
                lon = float(r.get('lon'))
                display = r.get('display_name') or r.get('name') or q
                out.append({'name': display, 'lat': lat, 'lon': lon, 'raw': r})
            except Exception:
                continue
        try:
            cache_file.write_text(json.dumps(out))
        except Exception:
            pass
        return out
    except Exception:
        return []

def sample_raster_at_point(path: str, lat: float, lon: float):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with rasterio.open(str(p)) as src:
            try:
                row, col = src.index(lon, lat)
                arr = src.read(1)
                if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                    val = float(arr[row, col])
                    if src.nodata is not None and np.isclose(val, src.nodata):
                        return None
                    if not np.isfinite(val):
                        return None
                    return val
            except Exception:
                # try sampling as fallback
                for val in rasterio.sample.sample_gen(src, [(lon, lat)]):
                    v = val[0]
                    if np.isfinite(v):
                        return float(v)
    except Exception:
        return None
    return None

def extract_high_risk_regions(risk_path: str, boundaries_paths: list, risk_threshold: float = 0.7) -> list:
    """
    Extract subregion names with high flood risk by intersecting risk raster with boundaries.
    Tries multiple boundary files (primary to fallback).
    Returns list of region names with mean risk > risk_threshold.
    """
    try:
        import geopandas as gpd
        
        # Try each boundary file in order
        gdf = None
        for bpath in boundaries_paths:
            if not os.path.exists(bpath):
                continue
            try:
                temp_gdf = gpd.read_file(bpath)
                if not temp_gdf.empty:
                    # Check for name column variants
                    name_cols = [c for c in temp_gdf.columns if c.lower() in ('name', 'admin_name', 'district', 'region')]
                    if name_cols:
                        gdf = temp_gdf
                        name_col = name_cols[0]
                        break
            except Exception:
                continue
        
        if gdf is None or gdf.empty:
            return []
        
        # Read risk raster
        with rasterio.open(risk_path) as src:
            risk_arr = src.read(1).astype("float32")
            transform = src.transform
            bounds = src.bounds  # (left, bottom, right, top)
        
        high_risk_regions = []
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            
            # Get region name (try multiple columns)
            region_name = None
            for col in ['name', 'admin_name', 'district', 'region']:
                if col in row and pd.notna(row[col]):
                    region_name = str(row[col]).strip()
                    if region_name:
                        break
            
            if not region_name:
                region_name = f"Region {idx}"
            
            # Simple check: does region geometry overlap with raster bounds?
            geom_bounds = geom.bounds  # (minx, miny, maxx, maxy)
            
            # Check if geometry overlaps with raster
            if (geom_bounds[2] < bounds[0] or geom_bounds[0] > bounds[2] or
                geom_bounds[3] < bounds[1] or geom_bounds[1] > bounds[3]):
                continue
            
            # Estimate mean risk in region (rasterize geometry and sample)
            try:
                from rasterio.features import rasterize as rio_rasterize
                geom_mask = rio_rasterize([geom], out_shape=risk_arr.shape, transform=transform, fill=0)
                region_risk_vals = risk_arr[geom_mask > 0]
                
                if region_risk_vals.size > 0:
                    region_mean_risk = float(np.nanmean(region_risk_vals[~np.isnan(region_risk_vals)]))
                    if region_mean_risk > risk_threshold:
                        high_risk_regions.append((region_name, region_mean_risk))
            except Exception:
                pass
        
        # Deduplicate by name, keeping highest risk value for each unique name
        region_dict = {}
        for name, risk in high_risk_regions:
            if name not in region_dict or risk > region_dict[name]:
                region_dict[name] = risk
        
        # Sort by risk (descending)
        sorted_regions = sorted(region_dict.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_regions]
    
    except Exception as e:
        return []

def calculate_risk_statistics(risk_path: str) -> dict:
    """
    Calculate comprehensive statistics from a flood risk raster.
    
    Returns:
        Dictionary with risk statistics including mean, percentiles, distribution, and spatial patterns.
    """
    with rasterio.open(risk_path) as src:
        risk_arr = src.read(1).astype("float32")
    
    # Remove nodata/NaN
    mask = np.isnan(risk_arr) | np.isinf(risk_arr)
    valid_risk = risk_arr[~mask]
    
    if valid_risk.size == 0:
        return None
    
    # Compute statistics
    stats = {
        'mean_risk': float(np.mean(valid_risk)),
        'median_risk': float(np.median(valid_risk)),
        'min_risk': float(np.min(valid_risk)),
        'max_risk': float(np.max(valid_risk)),
        'p25_risk': float(np.percentile(valid_risk, 25)),
        'p75_risk': float(np.percentile(valid_risk, 75)),
        'std_risk': float(np.std(valid_risk)),
        
        # Risk class percentages
        'high_risk_pct': 100.0 * np.sum(valid_risk > 0.6) / valid_risk.size,
        'moderate_risk_pct': 100.0 * np.sum((valid_risk > 0.3) & (valid_risk <= 0.6)) / valid_risk.size,
        'low_risk_pct': 100.0 * np.sum(valid_risk <= 0.3) / valid_risk.size,
        'very_high_risk_pct': 100.0 * np.sum(valid_risk > 0.8) / valid_risk.size,
    }
    
    # Spatial gradient (E-W slope via mean columns)
    h, w = risk_arr.shape
    west_mean = np.nanmean(risk_arr[:, :w//4])
    east_mean = np.nanmean(risk_arr[:, 3*w//4:])
    stats['gradient_direction'] = "west to east" if east_mean > west_mean else "east to west"
    stats['gradient_strength'] = "stronger" if abs(east_mean - west_mean) > 0.15 else "subtle"
    
    # Clustering (std as proxy for patchiness)
    stats['clustering_desc'] = "well-defined clusters" if stats['std_risk'] > 0.25 else "gradual transitions"
    
    return stats


def generate_llm_explanation(stats: dict, high_risk_regions: list, aoi_name: str) -> str:
    """
    Generate flood risk explanation using LLM API.
    Supports OpenAI, Anthropic, or local models.
    
    Returns explanation text, or None if LLM unavailable.
    """
    # Check for API keys
    openai_key = os.environ.get('OPENAI_API_KEY')
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not openai_key and not anthropic_key:
        # Silently return None - will use fallback
        return None
    
    # Build data-rich prompt
    regions_str = ', '.join(high_risk_regions[:5]) if high_risk_regions else 'None identified'
    
    prompt = f"""You are a GIS analyst writing a technical flood risk assessment for {aoi_name}.

KEY STATISTICS:
- Overall mean risk: {stats['mean_risk']:.2f} (0-1 scale)
- Risk distribution: {stats['high_risk_pct']:.1f}% high-risk (>0.8), {stats['moderate_risk_pct']:.1f}% moderate (0.3-0.6), {stats['low_risk_pct']:.1f}% low (≤0.3)
- Spatial pattern: {stats['clustering_desc']}, {stats['gradient_strength']} gradient {stats['gradient_direction']}
- High-risk neighborhoods: {regions_str}

Write a concise 2-3 paragraph assessment (max 250 words) for urban planners and emergency managers. Include:
1. Overall risk characterization and distribution
2. Spatial patterns and what they suggest about underlying drivers
3. Priority areas for intervention

Be specific, technical, and actionable. Do NOT include headers or bullet points."""
    
    try:
        # Try OpenAI first
        if openai_key:
            try:
                import openai
            except ImportError:
                st.warning("⚠️ OpenAI API key set but 'openai' package not installed. Install with: pip install openai")
                return None
            
            # Use OpenAI client API (v1.0+)
            client = openai.OpenAI(api_key=openai_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Faster, cheaper than gpt-4
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.7
            )
            result = response.choices[0].message.content.strip()
            st.sidebar.success("✓ Using OpenAI GPT-4o-mini for explanation")
            return result
        
        # Try Anthropic
        elif anthropic_key:
            try:
                import anthropic
            except ImportError:
                st.warning("⚠️ Anthropic API key set but 'anthropic' package not installed. Install with: pip install anthropic")
                return None
            
            client = anthropic.Anthropic(api_key=anthropic_key)
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Fast and cost-effective
                max_tokens=350,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip()
            st.sidebar.success("✓ Using Anthropic Claude Haiku for explanation")
            return result
    
    except Exception as e:
        # LLM failed, will fall back to rule-based
        st.sidebar.warning(f"⚠️ LLM generation failed: {str(e)}")
        return None
    
    return None


def generate_dynamic_explanation(stats: dict, high_risk_regions: list, aoi_name: str) -> str:
    """
    Generate varied flood risk explanation using rule-based templates.
    Fallback when LLM is unavailable.
    """
    # Dynamic vocabulary based on severity
    if stats['high_risk_pct'] > 30:
        severity = 'high'
        descriptor = 'critical'
        concern = 'urgent attention required'
        action = 'immediate mitigation measures recommended'
    elif stats['high_risk_pct'] > 10:
        severity = 'moderate'
        descriptor = 'moderate'
        concern = 'warrants monitoring'
        action = 'preventive planning advised'
    else:
        severity = 'low'
        descriptor = 'manageable'
        concern = 'generally stable conditions'
        action = 'routine maintenance sufficient'
    
    # Varied opening sentences
    import random
    openings = [
        f"Analysis of {aoi_name}'s flood susceptibility reveals {descriptor} risk conditions",
        f"Flood risk assessment for {aoi_name} indicates {descriptor} vulnerability levels",
        f"The flood risk surface across {aoi_name} shows {descriptor} hazard characteristics"
    ]
    
    opening = random.choice(openings)
    
    # Build regional mention
    regional_mention = ""
    if high_risk_regions:
        regions_str = ", ".join(high_risk_regions[:5])  # Top 5 regions
        if len(high_risk_regions) <= 3:
            regional_mention = f" particularly in {regions_str},"
        else:
            regional_mention = f" especially in {regions_str},"
    
    # Dynamic spatial pattern descriptions
    if stats['clustering_desc'] == 'well-defined clusters':
        spatial_insight = f"Risk is concentrated in discrete hotspots{regional_mention if not regional_mention else ''} suggesting localized drainage or topographic constraints."
    else:
        spatial_insight = f"Risk transitions gradually across the landscape, reflecting distributed hydrological factors."
    
    # Gradient interpretation
    if 'west to east' in stats['gradient_direction']:
        gradient_interpretation = "coastal accumulation patterns and urbanized low-lying areas"
    else:
        gradient_interpretation = "inland topographic influences and varied drainage characteristics"
    
    # Build varied narrative
    explanation = f"""{opening}. The flood risk surface exhibits values ranging from {stats['min_risk']:.2f} to {stats['max_risk']:.2f}, with a median risk of {stats['median_risk']:.2f}.

Risk distribution analysis shows {stats['high_risk_pct']:.1f}% of the mapped area exceeds the 0.8 high-risk threshold{regional_mention}. {stats['moderate_risk_pct']:.1f}% falls within the moderate class (0.3–0.6), and {stats['low_risk_pct']:.1f}% remains in the low-risk category. {spatial_insight} The {stats['gradient_strength']} {stats['gradient_direction']} gradient suggests {gradient_interpretation}.

{action.capitalize()}. The interquartile range ({stats['p25_risk']:.2f}–{stats['p75_risk']:.2f}) indicates {'substantial variability' if stats['p75_risk'] - stats['p25_risk'] > 0.3 else 'relatively consistent'} risk levels across the study area, with {stats['clustering_desc']} consistent with underlying hydrological and land-cover drivers."""
    
    return explanation


def generate_flood_risk_explanation(risk_path: str, aoi_name: str, boundaries_paths: list = None, use_ai: bool = False) -> str:
    """
    Generate a narrative GIS-aware explanation of the flood risk surface.
    
    Args:
        risk_path: Path to flood risk raster
        aoi_name: Area of interest name (e.g., "Accra")
        boundaries_paths: List of GeoJSON/Shapefile paths to try (primary to fallback)
        use_ai: If True, try LLM-based generation first; if False, use rule-based generation
    """
    try:
        # Calculate statistics
        stats = calculate_risk_statistics(risk_path)
        
        if stats is None:
            return "Unable to generate explanation: no valid data."
        
        # Extract high-risk subregions if boundaries available
        high_risk_regions = []
        if boundaries_paths:
            high_risk_regions = extract_high_risk_regions(risk_path, boundaries_paths, risk_threshold=0.65)
        
        # If AI is enabled, try LLM-based generation first
        if use_ai:
            llm_explanation = generate_llm_explanation(stats, high_risk_regions, aoi_name)
            if llm_explanation:
                return llm_explanation
        
        # Use rule-based dynamic generation (default or fallback)
        return generate_dynamic_explanation(stats, high_risk_regions, aoi_name)
    
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# ---------------- Sidebar ----------------
st.sidebar.markdown("---")
summary_path = "data/rasters/prepared_layers_summary.json"
if not os.path.exists(summary_path):
    st.error("Summary JSON not found. Run the fetch step first.")
    st.stop()

meta = read_summary(summary_path)
paths = meta["outputs"]

# Fixed cmaps
CMAPS = {
    "flood_risk_0to1": "viridis",
    "dist_to_river_m": "magma",
    "drainage_density_km_per_km2": "plasma",
    "soil_sand_pct": "cividis",
    "lulc_worldcover_proxy": "tab10",
}

# Controls: toggles + WEIGHTS (replaces per-layer opacity sliders)
st.sidebar.subheader("Flood risk index setup")
# This is the PLACE NAME used to fetch rasters (OSM + landcover + soil). It is separate from the point "Location search" above.
_default_aoi = meta.get("aoi_place", "Lagos, Nigeria") if isinstance(meta, dict) else "Lagos, Nigeria"
aoi_place = st.sidebar.text_input("City to compute flood risk for (e.g. Lagos, Nigeria)", value=_default_aoi, key="aoi_place_input")
fetch_first = st.sidebar.checkbox("Fetch physical layers before recompute", value=True, key="fetch_first")

st.sidebar.subheader("Flood risk index weights")
normalize_weights = st.sidebar.checkbox("Normalize weights to sum to 1", value=True, key="normalize_weights")

# defaults: all weights equal at 0.25
w_defaults = {"dist": 0.25, "drainage": 0.25, "soil": 0.25, "lulc": 0.25}

w_dist = st.sidebar.slider("Weight: distance to waterways", 0.0, 1.0, float(w_defaults["dist"]), 0.01, key="w_dist")
w_dd   = st.sidebar.slider("Weight: drainage density",      0.0, 1.0, float(w_defaults["drainage"]), 0.01, key="w_dd")
w_soil = st.sidebar.slider("Weight: soil infiltration",     0.0, 1.0, float(w_defaults["soil"]), 0.01, key="w_soil")
w_lulc = st.sidebar.slider("Weight: land cover (LULC)",      0.0, 1.0, float(w_defaults["lulc"]), 0.01, key="w_lulc")

# ===== Session state for smart caching =====
# Track the AOI location & weights from previous runs to detect changes
if "cached_aoi_place" not in st.session_state:
    st.session_state.cached_aoi_place = None
if "cached_summary_path" not in st.session_state:
    st.session_state.cached_summary_path = None
if "cached_paths" not in st.session_state:
    st.session_state.cached_paths = None
if "cached_meta" not in st.session_state:
    st.session_state.cached_meta = None

def _aoi_changed() -> bool:
    """Check if AOI location has changed since last fetch."""
    current_aoi = (aoi_place.strip() or _default_aoi).lower()
    cached_aoi = (st.session_state.cached_aoi_place or "").lower()
    return current_aoi != cached_aoi

def _summary_path_changed() -> bool:
    """Check if summary path has changed since last load."""
    return summary_path != st.session_state.cached_summary_path

def _run_fetch_physical_direct(place: str) -> str:
    """Run fetch_physical by calling main() directly (no subprocess overhead)."""
    place = (place or "").strip()
    if not place:
        place = "Lagos, Nigeria"
    
    # Save original sys.argv and replace with fetch_physical arguments
    orig_argv = sys.argv
    try:
        sys.argv = ["fetch_physical.py", "--place", place]
        # Call main() directly
        fetch_physical_main()
        return f"✓ Fetched layers for: {place}"
    except Exception as e:
        raise RuntimeError(f"fetch_physical failed: {str(e)}")
    finally:
        sys.argv = orig_argv

if st.sidebar.button("Compute flood risk", key="btn_recompute"):
    try:
        # Determine if we need to fetch (AOI changed) or just recompute (weights changed)
        need_fetch = fetch_first and _aoi_changed()
        need_recompute = True  # Always recompute after fetch or when weights change
        
        # Step 1: Fetch physical layers if AOI location changed
        if need_fetch:
            current_aoi = aoi_place.strip() or _default_aoi
            with st.sidebar.status(f"Fetching physical layers for '{current_aoi}'…", expanded=False):
                try:
                    if not DIRECT_FETCH_AVAILABLE:
                        st.sidebar.warning("Direct fetch unavailable, skipping layer fetch.")
                    else:
                        msg = _run_fetch_physical_direct(current_aoi)
                        st.sidebar.success(msg)
                        # Update cached AOI after successful fetch
                        st.session_state.cached_aoi_place = current_aoi
                        # Clear cache before reloading metadata (files have changed)
                        read_summary.clear()
                        # Reload metadata after fetch
                        meta = read_summary(summary_path)
                        paths = meta["outputs"]
                        st.session_state.cached_meta = meta
                        st.session_state.cached_paths = paths
                except Exception as e:
                    st.sidebar.error(f"Failed to fetch layers: {str(e)}")
                    st.sidebar.exception(e)
                    st.stop()
        else:
            if _aoi_changed():
                st.sidebar.info(f"ℹ AOI changed but 'Fetch physical layers' is disabled. Using existing data.")
            else:
                st.sidebar.info("✓ AOI unchanged. Skipping fetch, only recomputing weights.")
        
        # Step 2: Recompute flood risk (always, since weights may have changed)
        if need_recompute:
            with st.sidebar.status("Recomputing flood risk…", expanded=False):
                out_path, w_final = recompute_flood_risk(
                    summary_path=summary_path,
                    w_dist=w_dist, w_drainage=w_dd, w_soil=w_soil, w_lulc=w_lulc,
                    out_path_override=None,
                    normalize_weights=normalize_weights,
                )
                st.sidebar.success(f"Recomputed: {out_path}")
                st.sidebar.json({"weights_used": w_final})
        
        # Step 3: Reload metadata and rerun to update map
        # Clear ALL caches since files have been updated
        read_summary.clear()
        raster_to_rgba_image_cached.clear()
        
        meta = read_summary(summary_path)
        paths = meta["outputs"]
        st.session_state.cached_meta = meta
        st.session_state.cached_paths = paths
        st.session_state.cached_summary_path = summary_path
        st.rerun()

    except Exception as e:
        st.sidebar.error("Failed to recompute")
        st.sidebar.exception(e)

st.sidebar.markdown("---")
st.sidebar.subheader("Display settings")
overlay_opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.50, 0.05, key="overlay_opacity")
overlay_style = "solid"  # Fixed overlay style
use_ai_summary = st.sidebar.checkbox("Use AI-generated summary", value=False, key="use_ai_summary")

st.sidebar.markdown("---")
st.sidebar.subheader("Layers to display")
show_risk = st.sidebar.checkbox("Flood risk (0–1)", value=True, key="toggle_risk")
show_dist = st.sidebar.checkbox("Distance to river (m)", value=False, key="toggle_dist")
show_dd = st.sidebar.checkbox("Drainage density (km/km²)", value=False, key="toggle_dd")
show_soil = st.sidebar.checkbox("Soil sand fraction (%)", value=False, key="toggle_soil")
show_lulc = st.sidebar.checkbox("LULC (worldcover proxy)", value=False, key="toggle_lulc")



# --------------- Rainfall summary (above map) ----------------
from datetime import datetime, timedelta

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_nasa_power_rainfall(lat: float, lon: float, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch rainfall data from NASA POWER API for the given location and date range.
    
    Args:
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with columns: date, rainfall (mm/day)
    """
    try:
        # Convert dates to YYYYMMDD format for NASA POWER API
        start_s = datetime.fromisoformat(start_date).strftime("%Y%m%d")
        end_s = datetime.fromisoformat(end_date).strftime("%Y%m%d")
        
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters=PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}"
            f"&start={start_s}&end={end_s}&format=JSON"
        )
        
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract precipitation data
        prectot = data.get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {})
        
        if not prectot:
            return None
        
        # Convert to DataFrame
        rows = []
        for date_str, rainfall_val in sorted(prectot.items()):
            try:
                date_iso = datetime.strptime(date_str, "%Y%m%d").date().isoformat()
                rainfall = float(rainfall_val)
                if rainfall >= 0:  # Filter out missing data (usually -999)
                    rows.append([date_iso, rainfall])
            except (ValueError, TypeError):
                continue
        
        if not rows:
            return None
        
        df = pd.DataFrame(rows, columns=["date", "rainfall"])
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    
    except Exception as e:
        st.warning(f"Failed to fetch NASA POWER data: {e}")
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_rainfall_forecast(lat: float, lon: float, days: int = 7) -> Optional[pd.DataFrame]:
    """Fetch rainfall forecast from Open-Meteo API (free, no API key needed).
    
    Args:
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
        days: Number of forecast days (default 7)
    
    Returns:
        DataFrame with columns: date, rainfall_mm, rainfall_prob_pct
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum,precipitation_probability_max"
            f"&forecast_days={days}&timezone=auto"
        )
        
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        precip = daily.get("precipitation_sum", [])
        precip_prob = daily.get("precipitation_probability_max", [])
        
        if not dates or not precip:
            return None
        
        rows = []
        for i, date_str in enumerate(dates):
            rainfall = float(precip[i]) if i < len(precip) else 0.0
            prob = float(precip_prob[i]) if i < len(precip_prob) else 0.0
            rows.append([date_str, rainfall, prob])
        
        df = pd.DataFrame(rows, columns=["date", "rainfall_mm", "rainfall_prob_pct"])
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    except Exception as e:
        return None

def _assess_flood_warnings(forecast_df: pd.DataFrame, risk_path: str) -> dict:
    """Assess flood warnings based on forecast rainfall and terrain risk.
    
    Returns:
        Dict with warning levels and affected days
    """
    if forecast_df is None or len(forecast_df) == 0:
        return None
    
    try:
        # Read mean risk from raster
        with rasterio.open(risk_path) as src:
            risk_arr = src.read(1).astype("float32")
            mask = np.isnan(risk_arr) | np.isinf(risk_arr)
            valid_risk = risk_arr[~mask]
            mean_risk = float(np.mean(valid_risk)) if valid_risk.size > 0 else 0.5
            high_risk_pct = 100.0 * np.sum(valid_risk > 0.7) / valid_risk.size if valid_risk.size > 0 else 0.0
        
        # Define rainfall thresholds based on mean risk
        if mean_risk > 0.8:
            critical_threshold = 15.0
            high_threshold = 10.0
        elif mean_risk > 0.6:
            critical_threshold = 30.0
            high_threshold = 20.0
        elif mean_risk > 0.3:
            critical_threshold = 50.0
            high_threshold = 35.0
        else:
            critical_threshold = 75.0
            high_threshold = 50.0
        
        warnings = {
            "critical_days": [],
            "high_days": [],
            "moderate_days": [],
            "mean_risk": mean_risk,
            "high_risk_pct": high_risk_pct,
            "thresholds": {"critical": critical_threshold, "high": high_threshold}
        }
        
        for _, row in forecast_df.iterrows():
            rainfall = row["rainfall_mm"]
            date = row["date"]
            prob = row.get("rainfall_prob_pct", 0)
            
            if rainfall >= critical_threshold:
                warnings["critical_days"].append((date, rainfall, prob))
            elif rainfall >= high_threshold:
                warnings["high_days"].append((date, rainfall, prob))
            elif rainfall >= 10.0:
                warnings["moderate_days"].append((date, rainfall, prob))
        
        return warnings
    
    except Exception:
        return None
    
    except Exception as e:
        st.warning(f"Failed to fetch NASA POWER data: {e}")
        return None

@st.cache_data(ttl=600, show_spinner=False)
def _quarterly_rainfall_average(df: pd.DataFrame) -> pd.DataFrame:
    """Cached quarterly rainfall aggregation - returns data for stacked bar chart by year."""
    d = df.copy()
    # Extract year and quarter
    d["year"] = d["date"].dt.year
    d["quarter_num"] = d["date"].dt.quarter
    d["quarter"] = d["quarter_num"].map({1: "Q1 (Jan-Mar)", 2: "Q2 (Apr-Jun)", 
                                          3: "Q3 (Jul-Sep)", 4: "Q4 (Oct-Dec)"})
    
    # Group by quarter and year, then pivot to get years as columns
    q = d.groupby(["quarter", "year"], as_index=False)["rainfall"].mean()
    # Pivot to get years as columns for stacking
    q_pivot = q.pivot(index="quarter", columns="year", values="rainfall")
    # Ensure quarters are in correct order
    quarter_order = ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"]
    q_pivot = q_pivot.reindex(quarter_order)
    
    return q_pivot

# Sidebar control for meteorology data source
st.sidebar.markdown("---")
st.sidebar.subheader("Rainfall data (NASA POWER API)")

met_df = None

# Fetch from NASA POWER using AOI centroid
bbox = meta.get("bbox", [0, 0, 0, 0])
west, south, east, north = bbox
center_lat = (south + north) / 2.0
center_lon = (west + east) / 2.0

# Date range controls
col1, col2 = st.sidebar.columns(2)
with col1:
    # Default: last 3 years
    default_start = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
    start_date = st.date_input("Start date", value=datetime.fromisoformat(default_start), key="rain_start")
with col2:
    default_end = datetime.now().strftime("%Y-%m-%d")
    end_date = st.date_input("End date", value=datetime.fromisoformat(default_end), key="rain_end")

if st.sidebar.button("Fetch rainfall data", key="fetch_rainfall"):
    with st.spinner("Fetching NASA POWER data..."):
        met_df = _fetch_nasa_power_rainfall(
            center_lat, center_lon,
            start_date.isoformat(), end_date.isoformat()
        )
        if met_df is not None:
            st.sidebar.success(f"✓ Fetched {len(met_df)} days of data")
        else:
            st.sidebar.error("Failed to fetch data")
else:
    # Auto-fetch on first load (use cached data) - 3 years
    met_df = _fetch_nasa_power_rainfall(
        center_lat, center_lon,
        (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d"),
        datetime.now().strftime("%Y-%m-%d")
    )


# --------------- Map Build ----------------

# Flood risk explanation
with st.expander("ℹ️ Understanding the Flood Risk Index", expanded=False):
    st.markdown("""
**Flood Risk Index (0-1):**  
Composite score combining distance to water, drainage density, 
soil permeability, and land cover.

**Risk Categories & Rainfall Thresholds:**

| Range | Level | Terrain Characteristics | Critical Rainfall (24h) |
|---|---|---|---|
| 0.0–0.3 | **Low Risk** | Good drainage, distance from water, permeable surfaces | 75–150+ mm |
| 0.3–0.6 | **Moderate Risk** | Mixed conditions; some vulnerability to flooding | 50–75 mm |
| 0.6–0.8 | **High Risk** | Close to water, poor drainage, or impervious surfaces | 30–50 mm |
| 0.8–1.0 | **Very High Risk** | Critical areas highly susceptible to flooding | 15–30 mm |

**Important Caveats:**
- **Rainfall intensity matters**: 50mm over 1 hour is far more dangerous than 50mm over 24 hours
- **Antecedent conditions**: Previous rainfall saturates soil, significantly lowering flood thresholds
- **Local infrastructure**: Storm drains, retention basins, and maintenance affect actual flood risk
- **Seasonal variation**: Dry season vs wet season soil moisture changes vulnerability
- **Flash flooding**: High-intensity bursts (>20mm/hour) can cause flooding even in moderate-risk areas
""")

bbox = meta["bbox"]
west, south, east, north = bbox
center_lat = (south + north) / 2.0
center_lon = (west + east) / 2.0
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

# Legend positions cycle
legend_positions = ["bottomright", "bottomleft", "topright", "topleft"]
legend_idx = 0
def next_pos():
    nonlocal_vars = globals().setdefault("_legend_state", {"i":0})
    pos = legend_positions[nonlocal_vars["i"] % len(legend_positions)]
    nonlocal_vars["i"] += 1
    return pos

# Overlays + legends
if show_risk and "flood_risk_0to1" in paths and os.path.exists(paths["flood_risk_0to1"]):
    p = paths["flood_risk_0to1"]
    img, (rvmin, rvmax) = raster_to_rgba_image_cached(p, cmap_name=CMAPS["flood_risk_0to1"], overlay_style=overlay_style)
    add_image_overlay(m, img, raster_bounds_latlon(p), "Flood risk (0–1)", opacity=overlay_opacity, overlay_style=overlay_style)
    add_onmap_legend(m, make_continuous_legend_png_cached(CMAPS["flood_risk_0to1"], rvmin, rvmax, "Flood risk (0–1)"), position=next_pos())

if show_dist and "dist_to_river_m" in paths and os.path.exists(paths["dist_to_river_m"]):
    p = paths["dist_to_river_m"]
    img, (dvmin, dvmax) = raster_to_rgba_image_cached(p, cmap_name=CMAPS["dist_to_river_m"], overlay_style=overlay_style)
    add_image_overlay(m, img, raster_bounds_latlon(p), "Distance to river (m)", opacity=overlay_opacity, overlay_style=overlay_style)
    add_onmap_legend(m, make_continuous_legend_png_cached(CMAPS["dist_to_river_m"], dvmin, dvmax, "Distance to river (m)"), position=next_pos())

if show_dd and "drainage_density_km_per_km2" in paths and os.path.exists(paths["drainage_density_km_per_km2"]):
    p = paths["drainage_density_km_per_km2"]
    img, (ddvmin, ddvmax) = raster_to_rgba_image_cached(p, cmap_name=CMAPS["drainage_density_km_per_km2"], overlay_style=overlay_style)
    add_image_overlay(m, img, raster_bounds_latlon(p), "Drainage density (km/km²)", opacity=overlay_opacity, overlay_style=overlay_style)
    add_onmap_legend(m, make_continuous_legend_png_cached(CMAPS["drainage_density_km_per_km2"], ddvmin, ddvmax, "Drainage density (km/km²)"), position=next_pos())

if show_soil and "soil_sand_pct" in paths and os.path.exists(paths["soil_sand_pct"]):
    p = paths["soil_sand_pct"]
    img, (svmin, svmax) = raster_to_rgba_image_cached(p, cmap_name=CMAPS["soil_sand_pct"], overlay_style=overlay_style)
    add_image_overlay(m, img, raster_bounds_latlon(p), "Soil sand fraction (%)", opacity=overlay_opacity, overlay_style=overlay_style)
    add_onmap_legend(m, make_continuous_legend_png_cached(CMAPS["soil_sand_pct"], svmin, svmax, "Soil sand fraction (%)"), position=next_pos())

if show_lulc and "lulc_worldcover_proxy" in paths and os.path.exists(paths["lulc_worldcover_proxy"]):
    p = paths["lulc_worldcover_proxy"]
    img, _ = raster_to_rgba_image_cached(p, cmap_name=CMAPS["lulc_worldcover_proxy"], overlay_style=overlay_style)
    add_image_overlay(m, img, raster_bounds_latlon(p), "LULC (worldcover proxy)", opacity=overlay_opacity, overlay_style=overlay_style)
    add_onmap_legend(m, make_lulc_legend_png_cached(), position=next_pos())



# Add a marker for any user-selected location and sample nearby values
sel = st.session_state.get('selected_location') if 'selected_location' in st.session_state else None
if sel:
    try:
        lat = float(sel['lat'])
        lon = float(sel['lon'])
        name = sel.get('name', f"{lat:.6f}, {lon:.6f}")
        popup_html = f"<b>{name}</b><br/>{lat:.6f}, {lon:.6f}"
        # sample flood risk raster if available
        if 'flood_risk_0to1' in paths and os.path.exists(paths['flood_risk_0to1']):
            v = sample_raster_at_point(paths['flood_risk_0to1'], lat, lon)
            if v is not None:
                popup_html += f"<br/>Flood risk: {v:.3f}"
        # nearest grid info
        nn = nearest_grid_point(lat, lon)
        if nn is not None:
            nn_info = []
            if 'row' in nn and 'col' in nn:
                nn_info.append(f"r{nn.get('row')},c{nn.get('col')}")
            if 'LULC' in nn:
                nn_info.append(f"LULC:{nn.get('LULC')}")
            if 'DistRiver_m' in nn:
                nn_info.append(f"DistRiver_m:{nn.get('DistRiver_m')}")
            if nn_info:
                popup_html += "<br/>Nearest grid: " + ", ".join(nn_info)
        folium.Marker([lat, lon], popup=folium.Popup(popup_html, max_width=300), icon=folium.Icon(color='red', icon='map-marker')).add_to(m)
        # recenter map
        try:
            m.location = [lat, lon]
            m.zoom_start = 14
        except Exception:
            pass
    except Exception:
        pass

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width='stretch', returned_objects=[])

# --------------- Flood Risk Explanation ----------------
if show_risk and "flood_risk_0to1" in paths and os.path.exists(paths["flood_risk_0to1"]):
    st.markdown("---")
    st.subheader("Spatial Risk Analysis & Interpretation")
    aoi_name = (aoi_place or _default_aoi).strip() or "the mapped area"
    
    # Try multiple boundary sources (primary to fallback)
    boundaries_paths = [
        "data/admin_boundaries.geojson",  # Districts/regions
        "data/city_boundary.geojson",      # City/metro boundaries
        "data/neighborhoods.geojson",      # Neighborhoods/wards
    ]
    
    explanation = generate_flood_risk_explanation(
        paths["flood_risk_0to1"], 
        aoi_name,
        boundaries_paths=boundaries_paths,
        use_ai=use_ai_summary
    )
    st.markdown(explanation)
    
    # Rainfall forecast and flood warnings
    st.markdown("---")
    st.subheader("📅 Rainfall Forecast & Flood Warnings")
    
    bbox = meta.get("bbox", [0, 0, 0, 0])
    west, south, east, north = bbox
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    
    forecast_df = _fetch_rainfall_forecast(center_lat, center_lon, days=7)
    
    if forecast_df is not None and len(forecast_df) > 0:
        warnings = _assess_flood_warnings(forecast_df, paths["flood_risk_0to1"])
        
        # Display warnings if any
        if warnings:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if warnings["critical_days"]:
                    st.error(f"🚨 **{len(warnings['critical_days'])} Critical Warning(s)**")
                    for date, rainfall, prob in warnings["critical_days"]:
                        st.markdown(f"**{date.strftime('%a, %b %d')}**: {rainfall:.1f}mm ({prob:.0f}% prob)")
                elif warnings["high_days"]:
                    st.warning(f"⚠️ **{len(warnings['high_days'])} High Warning(s)**")
                    for date, rainfall, prob in warnings["high_days"]:
                        st.markdown(f"**{date.strftime('%a, %b %d')}**: {rainfall:.1f}mm ({prob:.0f}% prob)")
                else:
                    st.success("✓ No critical warnings")
            
            with col2:
                st.metric(
                    "Area Mean Risk", 
                    f"{warnings['mean_risk']:.2f}",
                    help="Average flood susceptibility across mapped area"
                )
                st.metric(
                    "High-Risk Area", 
                    f"{warnings['high_risk_pct']:.1f}%",
                    help="Percentage of area with risk > 0.7"
                )
            
            with col3:
                st.metric(
                    "Critical Threshold", 
                    f"{warnings['thresholds']['critical']:.0f}mm/24h",
                    help="Rainfall amount likely to cause flooding in high-risk areas"
                )
                st.metric(
                    "High Threshold", 
                    f"{warnings['thresholds']['high']:.0f}mm/24h",
                    help="Rainfall amount that may cause localized flooding"
                )
        
        # Show enhanced forecast table
        st.markdown("**7-Day Rainfall Forecast:**")
        
        # Build enhanced forecast display
        forecast_display = forecast_df.copy()
        
        # Store original rainfall values before formatting (for styling)
        rainfall_values = forecast_display["rainfall_mm"].values
        
        forecast_display["Day"] = forecast_display["date"].dt.strftime("%a")
        forecast_display["Date"] = forecast_display["date"].dt.strftime("%b %d")
        
        # Add risk assessment column
        def get_risk_level(rainfall, thresholds):
            if rainfall >= thresholds['critical']:
                return "🚨 Critical"
            elif rainfall >= thresholds['high']:
                return "⚠️ High"
            elif rainfall >= 10.0:
                return "⚪ Moderate"
            else:
                return "✓ Low"
        
        forecast_display["Flood Risk"] = forecast_display["rainfall_mm"].apply(
            lambda x: get_risk_level(x, warnings['thresholds']) if warnings else "—"
        )
        
        # Add weather icon based on rainfall amount
        def get_weather_icon(rainfall):
            if rainfall >= 50:
                return "⛈️"
            elif rainfall >= 20:
                return "🌧️"
            elif rainfall >= 5:
                return "🌦️"
            elif rainfall >= 0.5:
                return "🌤️"
            else:
                return "☀️"
        
        forecast_display[""] = forecast_display["rainfall_mm"].apply(get_weather_icon)
        
        # Format columns
        forecast_display["Rainfall"] = forecast_display["rainfall_mm"].apply(lambda x: f"{x:.1f} mm")
        forecast_display["Probability"] = forecast_display["rainfall_prob_pct"].apply(lambda x: f"{x:.0f}%")
        
        # Select and reorder columns for display
        display_cols = ["", "Day", "Date", "Rainfall", "Probability", "Flood Risk"]
        forecast_table = forecast_display[display_cols].copy()
        
        # Enhanced styling function
        def style_forecast_table(row):
            # Get the rainfall value using the row index
            idx = row.name
            rainfall = rainfall_values[idx] if idx < len(rainfall_values) else 0.0
            
            styles = [''] * len(row)
            
            if warnings:
                if rainfall >= warnings['thresholds']['critical']:
                    # Critical: black background
                    styles = ['background-color: #000000; font-weight: bold'] * len(row)
                elif rainfall >= warnings['thresholds']['high']:
                    # High: black background
                    styles = ['background-color: #000000; font-weight: 500'] * len(row)
                elif rainfall >= 10.0:
                    # Moderate: black background
                    styles = ['background-color: #000000'] * len(row)
           
            return styles
        
        # Display styled table
        styled_table = forecast_table.style.apply(style_forecast_table, axis=1)
        
        # Add bar chart visualization for rainfall (using numeric values)
        # We need to temporarily add the numeric column for the bar chart
        forecast_table_numeric = forecast_table.copy()
        forecast_table_numeric['Rainfall_numeric'] = rainfall_values
        
        st.dataframe(
            styled_table,
            hide_index=True,
            use_container_width=True,
            height=280
        )
        
        st.caption(f"📡 Forecast data from Open-Meteo API for {center_lat:.4f}°N, {center_lon:.4f}°E • Updates every 30 minutes")
    else:
        st.info("📡 Unable to fetch rainfall forecast. Check internet connection.")
    
    # Optional: Show layer contribution summary
    st.markdown("---")
    weights = meta.get("flood_risk_weights_used", {})
    # if weights:
    #     st.markdown("**Weights used in computation:**")
    #     w_col1, w_col2 = st.columns(2)
    #     with w_col1:
    #         st.metric("Distance to water", f"{weights.get('dist', 0):.2f}")
    #         st.metric("Drainage density", f"{weights.get('drainage', 0):.2f}")
    #     with w_col2:
    #         st.metric("Soil infiltration", f"{weights.get('soil', 0):.2f}")
    #         st.metric("Land cover (LULC)", f"{weights.get('lulc', 0):.2f}")

# --------------- Rainfall Chart (below map) ----------------
if met_df is not None and len(met_df) > 0:
    qdf = _quarterly_rainfall_average(met_df)
    st.subheader("Quarterly average rainfall by year")
    bbox = meta.get("bbox", [0, 0, 0, 0])
    west, south, east, north = bbox
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    # Calculate date range for caption
    date_range = f"{met_df['date'].min().strftime('%Y-%m-%d')} to {met_df['date'].max().strftime('%Y-%m-%d')}"
    st.caption(f"NASA POWER data at AOI centroid ({center_lat:.4f}°N, {center_lon:.4f}°E) — {date_range}")
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Define consistent color scheme for years (shades of blue)
    years = qdf.columns.tolist()
    # Use Blues colormap with adjusted range to avoid too light colors
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(years)))
    
    # Plot stacked bars
    qdf.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7)
    
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel("Average rainfall (mm/day)", fontsize=11)
    ax.set_xticklabels(qdf.index, rotation=0)
    ax.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
else:
    st.info("📊 Rainfall chart: Click 'Fetch rainfall data' in the sidebar to load data.")