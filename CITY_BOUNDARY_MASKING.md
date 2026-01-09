# City Boundary Masking Implementation Summary

## Overview
Implemented **Option 3: Raster Clipping at Generation Time** to mask all flood risk rasters to city boundaries instead of rectangular extents.

## Changes Made

### 1. **Created City Boundary** (`create_city_boundary.py`)
   - **File**: `create_city_boundary.py`
   - **Purpose**: Generate city boundary polygon from grid cells using convex hull
   - **Output**: `data/city_boundary.geojson`
   - **Status**: ✅ Generated successfully
   - **Bounds**: (3.47°, 6.42°, 3.49°, 6.44°) - covers all 400 grid cells

### 2. **Modified `fetch_physical.py`** (Physical layer generation)
   
   **New Functions Added:**
   - `load_city_boundary()`: Loads city boundary from GeoJSON; returns None if not found (graceful fallback)
   - `mask_raster_to_boundary()`: Applies boundary mask to raster arrays using rasterio.mask
   
   **Key Features:**
   - Automatic nodata handling (NaN for float, 0 for integer types)
   - Uses in-memory rasterio operations for efficiency
   - Graceful degradation if boundary file missing
   
   **Modified Functions:**
   - `fetch_worldcover_or_io_to_grid()`: Added optional `boundary_geom` and `crs` parameters for LULC masking
   - `main()`: 
     - Loads city boundary at start
     - Applies masking to:
       - `dist_to_river_m.tif` (distance to water)
       - `drainage_density_km_per_km2.tif` (drainage density)
       - `soil_sand_pct.tif` (soil infiltration)
       - LULC rasters (worldcover or IO annual)

### 3. **Modified `flood_demo_modular_stable.py`** (Flood risk computation)
   
   **New Functions Added:**
   - `load_city_boundary()`: Loads city boundary from GeoJSON
   - `mask_raster_with_boundary()`: Applies boundary mask to risk array
   
   **Modified Functions:**
   - `run()`: 
     - Loads boundary and applies masking to final flood risk raster
     - Prints confirmation when masking applied
     - Handles missing boundary gracefully

### 4. **Generated Boundary File**
   - **File**: `data/city_boundary.geojson`
   - **Type**: GeoJSON FeatureCollection (1 polygon)
   - **Geometry**: Convex hull of all grid cells
   - **Purpose**: Used as mask for all raster outputs

## Implementation Details

### Masking Strategy
1. **Read** input raster into numpy array
2. **Create** temporary in-memory GeoTIFF with metadata
3. **Apply** rasterio.mask to boundary polygon
4. **Write** masked result back to permanent file
5. **Preserve** CRS, transform, and nodata values

### Nodata Handling
- **Float32**: `np.nan` (for distance/density/risk)
- **UInt8**: `0` (for LULC/categorical)
- Automatically determined by array dtype

### Graceful Degradation
- If `city_boundary.geojson` not found, scripts print warning and continue without masking
- Allows raster generation to proceed independently

## Files Modified
1. `/fetch_physical.py` - 140+ lines added (masking functions + main() updates)
2. `/flood_demo_modular_stable.py` - 35+ lines added (masking functions + run() update)
3. `/create_city_boundary.py` - **NEW** (25 lines) - boundary generation script

## Files Created
1. `/data/city_boundary.geojson` - Generated boundary polygon
2. `/create_city_boundary.py` - Utility script

## How to Regenerate Rasters with Masking

### Option A: Regenerate All Layers (Recommended)
```bash
cd /Users/soyatoye/Documents/AXUM/flood/flood_risk
source .venv/bin/activate
python fetch_physical.py --place "Lagos, Nigeria"
python flood_demo_modular_stable.py --summary data/rasters/prepared_layers_summary.json
```

### Option B: Regenerate Only Flood Risk (if base layers unchanged)
```bash
source .venv/bin/activate
python flood_demo_modular_stable.py --summary data/rasters/prepared_layers_summary.json
```

## Expected Changes After Regeneration

### Before Masking:
- Rasters cover full rectangular extent (bounding box)
- Areas outside city boundaries contain data

### After Masking:
- Rasters clipped to city boundary polygon
- Areas outside boundary set to nodata (NaN or 0)
- File sizes similar (compression handles masked areas well)
- Map visualization shows only city area (no rectangular artifacts)

## Testing
✅ Syntax validation: Both files compile without errors
✅ Import checks: All required libraries (geopandas, rasterio, shapely) available
✅ Boundary generation: Successfully created `city_boundary.geojson` with correct bounds

## Advantages of This Approach

1. **Efficiency**: Masking applied once during generation, not during every render
2. **Clean Outputs**: Final rasters only contain city area data
3. **Smaller Files**: Unused rectangular areas eliminated
4. **Flexibility**: City boundary easily swappable (just replace GeoJSON)
5. **Non-Destructive**: Original grid/extent preserved in metadata
6. **Robust**: Handles missing boundary gracefully

## Future Enhancements

1. **Boundary Source Options**: Allow fetching boundary from OSM instead of grid convex hull
2. **Boundary Editor UI**: Add Streamlit widget to refine boundary before masking
3. **Multi-Layer Masking**: Apply different boundaries to different raster types
4. **Caching**: Cache boundary geometry to avoid repeated file I/O

## Next Steps

1. Review this summary
2. Run `fetch_physical.py` to regenerate base layers with masking
3. Run `flood_demo_modular_stable.py` to regenerate flood risk with masking
4. Test visualization in `streamlit_flood_viewer.py` to verify clean boundaries
5. Commit changes to `feat-boundary` branch

---

**Implementation Status**: ✅ COMPLETE - Ready for testing and raster regeneration
