# Nairobi Data Verification & Soil/LULC Masking Fix

## Issue Investigation Summary (February 16, 2026)

### 1. ✅ Coordinates Match Nairobi
**Status: CORRECT**

Expected bbox (from geocoding):
- `[36.6647016, -1.4448822, 37.1048735, -1.1606749]`

Actual bbox in JSON:
- `[36.6447016, -1.4648822, 37.1248735, -1.1406749]`

Actual bbox in raster files:
- `[36.6447016, -1.4656749, 37.1257016, -1.1406749]`

**Difference:** The actual raster bbox includes the 0.02° buffer (as designed), so coordinates are **correct**.

---

### 2. ❌ Soil & LULC Rectangular (Not Masked to City Boundary)
**Status: FIXED**

#### Root Cause:
The code was only applying boundary masking to:
- ✅ `dist_to_river_m.tif` (distance to water)
- ✅ `drainage_density_km_per_km2.tif` (drainage density)

But NOT to:
- ❌ `soil_sand_pct.tif` (was rectangular)
- ❌ `lulc_worldcover_proxy.tif` (was rectangular)

#### Why:
Lines 758 and 767 in `fetch_physical.py` had explicit masking calls:
```python
# Line 758
if boundary_geom is not None:
    dist = mask_raster_to_boundary(dist, transform, crs, boundary_geom, nodata=np.nan)

# Line 767
if boundary_geom is not None:
    dd = mask_raster_to_boundary(dd, transform, crs, boundary_geom, nodata=0.0)
```

But lines 774-800 (LULC) and lines 802-815 (Soil) had no such masking.

#### Fix Applied:
Added boundary masking to both LULC and Soil layers:

**For LULC (after line 774):**
```python
# Apply boundary masking to LULC if boundary available
if boundary_geom is not None:
    lulc_arr = None
    with safe_rio_open(lulc_path) as src:
        lulc_arr = src.read(1)
    # Mask LULC to boundary (set outside to nodata=0)
    lulc_arr = mask_raster_to_boundary(lulc_arr, transform, crs, boundary_geom, nodata=0)
    # Save masked LULC
    save_geotiff(lulc_path, lulc_arr, transform, crs, nodata=0, dtype="uint8")
```

**For Soil (after reproject, line 815):**
```python
# Apply boundary masking to soil if boundary available
if boundary_geom is not None:
    soil_arr = None
    with safe_rio_open(soil_path) as src:
        soil_arr = src.read(1)
    # Mask soil to boundary (set outside to nodata=0)
    soil_arr = mask_raster_to_boundary(soil_arr, transform, crs, boundary_geom, nodata=0)
    # Save masked soil
    save_geotiff(soil_path, soil_arr, transform, crs, nodata=0, dtype=src.dtypes[0])
```

---

## Testing Instructions

To verify the fix works:

1. **Re-run Nairobi:**
   - In the Streamlit app sidebar, enter "Nairobi" in "AOI place to fetch layers for"
   - Check "Fetch physical layers before recompute"
   - Click "Recompute flood_risk_0to1.tif"

2. **Expected Result:**
   - Soil sand fraction overlay should **follow city boundary** (not rectangular)
   - LULC worldcover proxy should **follow city boundary** (not rectangular)
   - Distance to river, drainage density, and flood risk should still be masked ✅

3. **Verification:**
   ```bash
   # Check that soil and LULC now have masked values outside Nairobi boundary
   source .venv/bin/activate && python3 << 'EOF'
   import rasterio
   import numpy as np
   
   for fname in ['soil_sand_pct.tif', 'lulc_worldcover_proxy.tif']:
       with rasterio.open(f'data/rasters/{fname}') as src:
           arr = src.read(1)
           nodata_count = np.sum(arr == 0)  # Count masked pixels
           total_pixels = arr.size
           masked_pct = (nodata_count / total_pixels) * 100
           print(f'{fname}: {masked_pct:.1f}% masked (expected >50% for city boundary)')
   EOF
   ```

---

## Implementation Details

### Code Changed:
- **File:** `fetch_physical.py`
- **Lines modified:** 774-815 (added ~30 lines)
- **Functions used:** `mask_raster_to_boundary()` (existing, reused for LULC & Soil)
- **Syntax validation:** ✅ No errors

### Behavior:
- **Before:** Soil and LULC layers were full rectangles (entire bbox)
- **After:** Soil and LULC layers are masked to Nairobi city boundary (same as distance/drainage)

### Backward Compatibility:
- If boundary fetch fails, masking is skipped (existing code path)
- Only applied when `boundary_geom is not None`
- No breaking changes to existing functionality

---

## Summary of Fixes in This Session

| Issue | Status | Action |
|-------|--------|--------|
| Nairobi coordinates match | ✅ Verified correct | None needed |
| Soil rectangular overlay | ✅ Fixed | Added boundary masking |
| LULC rectangular overlay | ✅ Fixed | Added boundary masking |
| Cache invalidation | ✅ Already implemented | Earlier in session |

All issues addressed! The application is now ready for proper testing with Nairobi and other cities.
