#!/usr/bin/env python3
"""
Download admin boundaries from external sources for better coverage.
For Accra/Ghana: Uses GADM (Global Administrative Areas) database.
"""
import json
import os
import requests
import zipfile
from pathlib import Path

def download_gadm_boundaries(country_code: str, admin_level: int, output_dir: str = "data"):
    """
    Download GADM boundaries for a country.
    
    Args:
        country_code: ISO 3-letter country code (e.g., 'GHA' for Ghana)
        admin_level: 0=country, 1=regions, 2=districts, 3=sub-districts
        output_dir: Output directory
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # GADM provides simplified GeoJSON (lower resolution, smaller files)
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country_code}_{admin_level}.json"
    
    output_file = os.path.join(output_dir, f"gadm_{country_code}_level{admin_level}.geojson")
    
    print(f"Downloading GADM level {admin_level} for {country_code}...")
    print(f"  URL: {url}")
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ Downloaded: {output_file} ({size_mb:.2f} MB)")
        return output_file
    
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def process_gadm_for_city(gadm_file: str, city_name: str, bbox: tuple, output_file: str):
    """
    Extract boundaries within a city's bounding box from GADM file.
    
    Args:
        gadm_file: Path to GADM GeoJSON file
        city_name: City name for filtering (optional)
        bbox: (west, south, east, north)
        output_file: Output GeoJSON path
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box
        
        print(f"Processing {gadm_file} for {city_name}...")
        
        gdf = gpd.read_file(gadm_file)
        
        # Create bounding box polygon
        west, south, east, north = bbox
        bbox_poly = box(west, south, east, north)
        
        # Filter to features intersecting the bbox
        gdf_filtered = gdf[gdf.intersects(bbox_poly)].copy()
        
        if gdf_filtered.empty:
            print(f"  ⚠ No features found within bbox")
            return None
        
        # Simplify to just geometry + name
        name_cols = [c for c in gdf_filtered.columns if 'NAME' in c.upper() or c == 'name']
        if name_cols:
            gdf_filtered['name'] = gdf_filtered[name_cols[0]]
        else:
            gdf_filtered['name'] = gdf_filtered.index.astype(str)
        
        gdf_filtered = gdf_filtered[['geometry', 'name']].copy()
        
        # Save
        gdf_filtered.to_file(output_file, driver="GeoJSON")
        print(f"  ✓ Saved {len(gdf_filtered)} features to {output_file}")
        return output_file
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def setup_city_boundaries(city_name: str, country_code: str, bbox: tuple, output_dir: str = "data"):
    """
    Setup boundaries for any city.
    
    Args:
        city_name: City name (e.g., "Nairobi", "Lagos", "Accra")
        country_code: ISO 3-letter country code (e.g., "KEN", "NGA", "GHA")
        bbox: Bounding box as (west, south, east, north) in decimal degrees
        output_dir: Output directory for GeoJSON files
    
    Country codes:
        - Kenya: KEN
        - Nigeria: NGA
        - Ghana: GHA
        - South Africa: ZAF
        - Tanzania: TZA
        - Uganda: UGA
        - Ethiopia: ETH
    
    To find bbox: Use https://boundingbox.klokantech.com/ (select CSV format)
    """
    print(f"\n=== Setting up boundaries for {city_name}, {country_code} ===")
    print(f"Bounding box: {bbox}\n")
    
    # Download admin boundaries (level 2 = districts)
    gadm_file = download_gadm_boundaries(country_code, 2, output_dir)
    
    if gadm_file and os.path.exists(gadm_file):
        # Extract city districts
        admin_file = process_gadm_for_city(
            gadm_file, 
            city_name, 
            bbox,
            os.path.join(output_dir, "admin_boundaries.geojson")
        )
        
        # Also try level 3 for neighborhoods
        gadm_file_3 = download_gadm_boundaries(country_code, 3, output_dir)
        if gadm_file_3 and os.path.exists(gadm_file_3):
            neigh_file = process_gadm_for_city(
                gadm_file_3,
                city_name,
                bbox,
                os.path.join(output_dir, "neighborhoods.geojson")
            )
    
    print("\n✓ Setup complete!")
    print("\nGenerated files:")
    for fname in ["admin_boundaries.geojson", "neighborhoods.geojson"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  • {fpath} ({size_kb:.1f} KB)")


# Predefined city configurations
CITY_CONFIGS = {
    "accra": {
        "country_code": "GHA",
        "bbox": (-0.35, 5.45, 0.05, 5.75),
        "name": "Accra"
    },
    "nairobi": {
        "country_code": "KEN",
        "bbox": (36.65, -1.45, 37.10, -1.15),
        "name": "Nairobi"
    },
    "lagos": {
        "country_code": "NGA",
        "bbox": (3.20, 6.40, 3.70, 6.70),
        "name": "Lagos"
    },
    "dar es salaam": {
        "country_code": "TZA",
        "bbox": (39.10, -7.00, 39.50, -6.65),
        "name": "Dar es Salaam"
    },
    "kampala": {
        "country_code": "UGA",
        "bbox": (32.50, 0.25, 32.70, 0.42),
        "name": "Kampala"
    },
    "addis ababa": {
        "country_code": "ETH",
        "bbox": (38.65, 8.90, 38.90, 9.10),
        "name": "Addis Ababa"
    },
    "johannesburg": {
        "country_code": "ZAF",
        "bbox": (27.90, -26.35, 28.20, -26.05),
        "name": "Johannesburg"
    },
}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python3 download_gadm_boundaries.py <city_name>")
        print("  python3 download_gadm_boundaries.py <city_name> <country_code> <west> <south> <east> <north>")
        print("\nPredefined cities:")
        for city, config in CITY_CONFIGS.items():
            print(f"  - {city.title()}")
        print("\nExamples:")
        print("  python3 download_gadm_boundaries.py nairobi")
        print("  python3 download_gadm_boundaries.py lagos")
        print("  python3 download_gadm_boundaries.py \"custom city\" KEN 36.0 -1.5 37.0 -1.0")
        print("\nCountry codes: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3")
        print("Find bbox: https://boundingbox.klokantech.com/ (select CSV format)")
        sys.exit(1)
    
    city_input = sys.argv[1].lower()
    
    # Check if it's a predefined city
    if city_input in CITY_CONFIGS:
        config = CITY_CONFIGS[city_input]
        setup_city_boundaries(
            city_name=config["name"],
            country_code=config["country_code"],
            bbox=config["bbox"]
        )
    elif len(sys.argv) >= 7:
        # Custom city with manual parameters
        city_name = sys.argv[1]
        country_code = sys.argv[2].upper()
        west = float(sys.argv[3])
        south = float(sys.argv[4])
        east = float(sys.argv[5])
        north = float(sys.argv[6])
        bbox = (west, south, east, north)
        
        setup_city_boundaries(city_name, country_code, bbox)
    else:
        print(f"Error: '{sys.argv[1]}' is not a predefined city.")
        print("Use --help to see usage.")
        sys.exit(1)
