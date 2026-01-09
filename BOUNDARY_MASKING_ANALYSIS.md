# City Boundary Masking - Problem Analysis & Fix Plan

## Root Cause Analysis

### The Problem
The overlays appear as tiny rectangular boxes because there's a **scale mismatch** between two coordinate systems:

1. **OSM Geocoding Extent** (LARGE): 2.686°E - 4.371°E, 6.347°N - 6.717°N
   - Used by `fetch_physical.py` to generate all rasters
   - Results in ~1685 × 371 pixel rasters covering ~1.7° × 0.37° area
   - Covers entire Lagos region

2. **Grid Cells Extent** (TINY): 3.47°E - 3.49°E, 6.42°N - 6.44°N
   - Only 0.02° × 0.02° area (0.04 km² at equator)
   - Contains 400 grid cells (20×20 grid)
   - Represents a small neighborhood in Lagos

### What Happened
- Current implementation masked rasters to the **grid cells boundary** (tiny 0.02° × 0.02° area)
- This masked away 99.9% of the raster data
- Left only a barely-visible 2×2 pixel rectangle in a 1685×371 image
- Users saw what looks like a rectangular artifact, not a city mask

### Why It's Wrong
The masking solution was conceptually correct (Option 3) but applied to the **wrong boundary**:
- ❌ Masked to: Grid cells (0.02° × 0.02°) — this is the analysis grid, not the city
- ✅ Should mask to: Lagos city administrative boundary (~0.5° × 0.3°) — the actual city limits

## Solution: Three Options

### Option A: Use Grid Cells As Entire Analysis Area (RECOMMENDED)
**Approach**: Regenerate rasters using ONLY the grid cell extent, then mask to grid boundary.

**Steps**:
1. Modify `fetch_physical.py` to use grid_cells bounds as the OSM bbox (instead of geocoding entire city)
2. This makes rasters ~20×20 pixels covering just the grid area
3. Mask to grid boundary becomes meaningful (masks to full raster area)
4. Overlays cover entire analysis area

**Pros**:
- ✅ Simple: no need for separate city boundary file
- ✅ Efficient: smaller rasters (~100 KB instead of ~1.7 MB)
- ✅ Clean: overlays exactly match analysis area
- ✅ Flexible: works with any grid size

**Cons**:
- ⚠️ Requires raster regeneration (time-consuming)
- ⚠️ Loses context of surrounding city areas

### Option B: Fetch Actual Lagos Administrative Boundary
**Approach**: Download Lagos city administrative boundary from OSM, use for masking.

**Steps**:
1. Query OSM for admin_level=6 (city boundary) for Lagos
2. Save to `data/lagos_admin_boundary.geojson`
3. Mask all rasters to this boundary
4. Overlays cover appropriate city area

**Pros**:
- ✅ Contextually correct (shows actual city limits)
- ✅ No raster regeneration needed
- ✅ Works with existing rasters

**Cons**:
- ⚠️ May not perfectly align with grid (gaps at edges)
- ⚠️ Requires OSM boundary fetch (additional dependency)
- ⚠️ Still ~1.7 MB rasters (most pixels masked away)

### Option C: Hybrid - Clip Rasters to Grid + Expand Extent
**Approach**: Regenerate rasters with minimal padding around grid cells, mask to grid boundary.

**Steps**:
1. Calculate grid cell bounds: (3.47, 6.42, 3.49, 6.44)
2. Add small buffer (e.g., 0.01°) for context: (3.46, 6.41, 3.50, 6.45)
3. Regenerate rasters using this smaller extent
4. Mask to grid boundary or buffer boundary

**Pros**:
- ✅ Shows analysis area + context
- ✅ Much smaller rasters than current (~300 KB)
- ✅ No missing data at grid edges

**Cons**:
- ⚠️ Requires raster regeneration
- ⚠️ Still need to decide buffer size

---

## Recommended Fix: Option A (Use Grid Cell Extent)

### Why This Is Best
1. **Correct semantics**: Grid defines the analysis area; mask should respect that
2. **Most efficient**: Smaller rasters, faster processing
3. **Cleanest**: No wasted pixels, overlays cover entire analysis area
4. **Aligned**: Raster resolution matches grid perfectly
5. **Future-proof**: Works with any grid configuration

### Implementation Plan

#### Phase 1: Add Grid-Based Extent Option to fetch_physical.py
- Add `--use-grid-cells` flag to accept grid bounds from file
- Load grid bounds from `data/grid_cells.geojson`
- Use grid bounds instead of geocoding-based bbox
- Regenerate all rasters

#### Phase 2: Update flood_demo_modular_stable.py
- No changes needed (masking already implemented correctly)
- Just re-run with regenerated base layers

#### Phase 3: Update City Boundary
- Grid boundary already correct in `data/city_boundary.geojson`
- No changes needed

### Code Changes Needed

**In fetch_physical.py**:
```python
# New function
def load_grid_bounds_from_geojson(grid_path="data/grid_cells.geojson"):
    """Load grid cell bounds from GeoJSON file."""
    gdf = gpd.read_file(grid_path)
    bounds = gdf.geometry.total_bounds  # [west, south, east, north]
    return tuple(bounds)

# Modified main()
if args.use_grid_cells:
    bbox = load_grid_bounds_from_geojson()
    print(f"Using grid cell extent: {bbox}")
else:
    aoi = geocode_aoi(CFG.place_name)
    bbox = bbox_from_gdf(aoi, buffer_deg=CFG.buffer_deg)
```

**Add CLI argument**:
```python
ap.add_argument("--use-grid-cells", action="store_true", 
                help="Use grid_cells.geojson bounds instead of geocoding")
```

### Expected Results

**Before (Current - Wrong)**:
- Raster extent: 1.7° × 0.37° (entire Lagos region)
- Grid cells: 0.02° × 0.02° (tiny invisible rectangle)
- Coverage: 0.1% of raster
- Overlay: barely visible pixel block

**After (Fixed)**:
- Raster extent: 0.02° × 0.02° (exactly grid cells)
- Grid cells: 0.02° × 0.02° (entire raster)
- Coverage: 100% of raster
- Overlay: full coverage of analysis area

### Regeneration Commands

```bash
# Regenerate all physical layers using grid cell extent
python fetch_physical.py --place "Lagos, Nigeria" --use-grid-cells

# Regenerate flood risk (uses regenerated base layers)
python flood_demo_modular_stable.py --summary data/rasters/prepared_layers_summary.json
```

### Time Estimate
- Code changes: 30 minutes
- Raster regeneration: 10-20 minutes (depends on OSM/cloud API response times)
- Testing: 10 minutes
- **Total**: ~1 hour

---

## Alternative: Why Not Option B?

While Option B (fetch Lagos admin boundary) seems more intuitive, it has a critical issue:

**The grid cells are only a small neighborhood in Lagos** (3.47-3.49°E, 6.42-6.44°N). Masking to the full Lagos city boundary would:
- Still result in ~99% masked pixels
- Show overlays covering only a tiny portion of the city boundary
- Create visual confusion ("why is the boundary so large?")
- Not actually solve the fundamental problem

**Option B is only viable if** the grid cells actually covered most of Lagos (which they don't).

---

## Recommendation

**Implement Option A**: Modify `fetch_physical.py` to accept grid cell extent via `--use-grid-cells` flag. This:
1. ✅ Fixes the rectangular overlay problem (100% coverage)
2. ✅ Fixes the area coverage problem (covers intended analysis area)
3. ✅ Maintains all existing masking logic (no regression)
4. ✅ Adds flexibility for future grids
5. ✅ Is quick to implement (~1 hour including regeneration)

Shall I proceed with implementing Option A?
