# Quick Start: Optimized Workflow Testing

## What Changed

### Before (Subprocess Model)
```
User clicks "Recompute"
  → Spawn new Python process
  → Re-import all modules
  → Run fetch_physical.py
  → Run recompute
  → Total: 35-50 seconds
```

### After (Direct Calls + Smart Caching)
```
User clicks "Recompute"
  ├─ If AOI changed: Fetch new layers (25-35s)
  └─ Always: Recompute with weights (1-2s)
  
User adjusts weights again and clicks "Recompute"
  └─ AOI unchanged → Skip fetch, instant recompute (1-2s)
```

## Testing the Optimization

### Test 1: Instant Weight-Only Update (Fast Path)
**Expected: ~1-2 seconds**

1. Start app: `streamlit run streamlit_flood_viewer_modified_location_recompute.py`
2. Keep "AOI place to fetch layers for" set to current value (e.g., "Lagos, Nigeria")
3. Adjust a weight slider (e.g., "Weight: distance to waterways")
4. Click "Recompute flood_risk_0to1.tif"
5. **Expected output:** 
   - ℹ "AOI unchanged. Skipping fetch, only recomputing weights."
   - Recompute completes in ~1-2 seconds
   - Map updates with new flood risk raster

### Test 2: Full Fetch + Recompute (Slow Path)
**Expected: ~25-35 seconds**

1. In sidebar, change "AOI place to fetch layers for" to new city (e.g., "Accra, Ghana")
2. Ensure "Fetch physical layers before recompute" is ✓ checked
3. Click "Recompute flood_risk_0to1.tif"
4. **Expected output:**
   - Status: "Fetching physical layers for 'Accra, Ghana'…"
   - Fetches OSM boundary, SoilGrids, LULC, drainage
   - Then: "Recomputing flood risk…"
   - Total time: ~25-35 seconds
   - Map shows new AOI with flood risk overlay

### Test 3: Skip Fetch (Disable Mode)
**Expected: ~1-2 seconds (instant recompute)**

1. Change AOI to new city (e.g., "Nairobi, Kenya")
2. **Uncheck** "Fetch physical layers before recompute"
3. Click "Recompute flood_risk_0to1.tif"
4. **Expected output:**
   - ℹ "AOI changed but 'Fetch physical layers' is disabled. Using existing data."
   - Recomputes with old data + new weights
   - No API calls made
   - Instant result (~1-2s)

### Test 4: Repeated AOI (Cached)
**Expected: ~1-2 seconds second time**

1. Set AOI to "Kinshasa, Congo"
2. Click "Recompute" → Full fetch (25-35s) ✓ cached_aoi_place="kinshasa, congo"
3. Change only weights (slider)
4. Click "Recompute" → Instant (1-2s) ✓ AOI unchanged, skip fetch
5. Change AOI back to "Kinshasa, Congo" (exact same)
6. Click "Recompute" → Instant (1-2s) ✓ Match cached location, skip fetch

### Test 5: Error Handling
**Expected: Clear error messages**

1. Set AOI to "XyzNonExistentPlace12345"
2. Ensure "Fetch physical layers before recompute" is ✓ checked
3. Click "Recompute flood_risk_0to1.tif"
4. **Expected output:**
   - Red error box with helpful message
   - Exception shown in sidebar
   - No partial/corrupted state

## Performance Metrics to Observe

### Console/Sidebar Messages
- "✓ AOI unchanged. Skipping fetch, only recomputing weights." → Fast path (1-2s)
- "ℹ AOI changed but 'Fetch physical layers' is disabled." → Skip fetch mode
- "Fetching physical layers for 'City, Country'…" → Full fetch (25-35s)

### In Application Metrics
- Status containers show elapsed time
- Success messages show output file path
- JSON shows weights used for that run

## Expected Session State Changes

After successful operations:
```json
{
  "cached_aoi_place": "accra, ghana",
  "cached_summary_path": "data/rasters/prepared_layers_summary.json",
  "cached_meta": { ... },
  "cached_paths": {
    "dist_to_river_m": "data/rasters/dist_to_river_m.tif",
    ...
  }
}
```

## Comparison: Old vs New

| Workflow | Old (Subprocess) | New (Direct) | Speedup |
|----------|------------------|--------------|---------|
| Adjust weights only | 35-45s | 1-2s | **95% faster** |
| Change AOI | 40-50s | 25-35s + 1-2s = 26-37s | **27% faster** |
| Repeat same AOI | 40-50s | 1-2s | **95% faster** |

## Troubleshooting

### Issue: "Direct fetch unavailable, skipping layer fetch"
- **Cause:** `fetch_physical` module couldn't be imported
- **Fix:** Verify `fetch_physical.py` exists in same directory
- **Check:** `python -c "from fetch_physical import main; print('OK')"`

### Issue: Recompute slower than expected
- **Check:** Is fetch_first checkbox checked when AOI should be cached?
- **Check:** Are you observing the session state messages?
- **Tip:** First run fetches, second run should be instant

### Issue: Map doesn't update
- **Check:** Click "Recompute", watch sidebar for "Recomputed: data/rasters/flood_risk_0to1.tif"
- **Check:** Look for successful weights_used JSON output
- **Note:** App calls `st.rerun()` to refresh map after recompute
