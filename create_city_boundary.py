#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_city_boundary.py

Derive a city boundary polygon from grid_cells.geojson using convex hull.
This boundary will be used to mask rasters to the city limits.

Output:
  - data/city_boundary.geojson
"""
import json
import geopandas as gpd
from shapely.geometry import box

def create_city_boundary_from_grid(grid_geojson_path, output_path):
    """
    Load grid cells GeoJSON and create a city boundary as the convex hull.
    """
    # Load grid cells
    gdf = gpd.read_file(grid_geojson_path)
    print(f"Loaded {len(gdf)} grid cells from {grid_geojson_path}")
    
    # Create convex hull of all grid cells
    boundary = gdf.geometry.unary_union.convex_hull
    print(f"Created convex hull boundary")
    
    # Create GeoDataFrame with boundary
    boundary_gdf = gpd.GeoDataFrame(
        {"name": ["City Boundary"]},
        geometry=[boundary],
        crs=gdf.crs
    )
    
    # Save to GeoJSON
    boundary_gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved city boundary to {output_path}")
    print(f"Boundary bounds: {boundary.bounds}")
    
    return boundary_gdf

if __name__ == "__main__":
    grid_path = "data/grid_cells.geojson"
    boundary_path = "data/city_boundary.geojson"
    create_city_boundary_from_grid(grid_path, boundary_path)
