# Performance Optimization Plan - February 16, 2026

## Executive Summary

The application has regressed significantly in performance despite earlier optimizations. Multiple new features were added after the initial optimization (direct function calls + smart caching), which have introduced substantial overhead. This document outlines critical bottlenecks and a comprehensive optimization strategy.

## Current Performance Issues

### 🔴 Critical Issues (Causing Major Slowdown)

#### 1. **NO CACHING for Raster Processing** (Most Critical)
**Impact:** Every page reload/rerun re-reads and re-processes ALL raster files from disk
- `raster_to_rgba_image()` called 5+ times per render (once per layer)
- Each call:
  - Reads multi-megabyte raster files from disk with `rasterio.open()`
  - Resamples large arrays (2000x2000 pixels)
  - Computes percentile statistics (`np.percentile`)
  - Applies colormap transformations
  - Converts to RGBA (multiply by 255, type conversions)
- **Estimated time per layer:** 1-3 seconds
- **Total time for 5 layers:** 5-15 seconds per page render

**Evidence:** No `@st.cache_data` or `@st.cache_resource` decorators found in code

#### 2. **Legend PNG Generation on Every Render** (High Impact)
**Impact:** Matplotlib figure creation is extremely slow
- `make_continuous_legend_png()` called 4+ times per render
- Each call:
  - Creates matplotlib figure/subplot (slow initialization)
  - Renders colorbar with PIL operations
  - Saves to BytesIO, reopens with PIL, resizes
- **Estimated time per legend:** 0.5-1.5 seconds
- **Total time:** 2-6 seconds per render

**Evidence:** Lines 129-144 - no caching decorator

#### 3. **Meteorology CSV Processing on Every Render** (Moderate Impact)
**Impact:** File I/O and DataFrame operations repeated unnecessarily
- `_load_meteorology_timeseries()` reads CSV, parses dates, cleans data
- `_quarterly_rainfall_average()` computes aggregations
- Matplotlib bar chart generation
- **Estimated time:** 0.5-2 seconds
- Called on every widget interaction (slider, checkbox, text input change)

**Evidence:** Lines 525-635 - no caching

#### 4. **Grid Cells KDTree Rebuilt or Checked Repeatedly**
**Impact:** Spatial indexing overhead on every location search
- Global cache `_GRID_CACHE` exists but may rebuild unnecessarily
- CSV read + KDTree construction can be slow for large grids
- **Estimated time:** 0.2-1 second (if not cached properly)

#### 5. **Session State Not Used for Display Metadata**
**Impact:** Summary JSON and paths re-parsed constantly
- `read_summary()` called at top level (line 372)
- Runs on every Streamlit script rerun (widget changes trigger full reruns)
- **Estimated time:** 0.1-0.5 seconds

### ⚠️ Moderate Issues (Contributing to Slowdown)

#### 6. **Geocoding Without Aggressive Caching**
- Nominatim API calls with 0.5s sleep
- Cache exists but is file-based (disk I/O overhead)
- Should use in-memory cache + file backup

#### 7. **Inefficient Raster Sampling for Location Marker**
- `sample_raster_at_point()` opens rasterio source every time
- Called for selected location marker (line 693)
- Could reuse already-loaded arrays from display rendering

#### 8. **Map Rebuilding from Scratch**
- Folium map object recreated on every render
- All overlays re-added even if layers haven't changed
- No detection of "display-only" changes (opacity slider, style dropdown)

### 📊 Performance Breakdown Estimate

| Operation | Current Time | After Caching | Savings |
|-----------|--------------|---------------|---------|
| Raster loading + RGBA conversion (5 layers) | 5-15s | 0.1-0.5s | ~95% |
| Legend PNG generation (4 legends) | 2-6s | 0.1s | ~95% |
| Meteorology CSV + chart | 0.5-2s | 0.05s | ~95% |
| Summary JSON parsing | 0.1-0.5s | 0.01s | ~90% |
| Grid KDTree loading | 0.2-1s | 0.01s | ~95% |
| Location geocoding (cached) | 0.5-1s | 0.01s | ~98% |
| **TOTAL ESTIMATED** | **8.3-25.5s** | **0.28-1.13s** | **~90-95%** |

---

## Optimization Strategy

### Phase 1: Critical Caching (Immediate - High ROI) ⚡⚡⚡

#### 1.1 Cache Raster RGBA Conversion
**Implementation:**
```python
@st.cache_data(ttl=3600, show_spinner=False)
def raster_to_rgba_image_cached(path: str, cmap_name: str,
                                vmin: Optional[float] = None, 
                                vmax: Optional[float] = None,
                                nodata=None, max_dim: int = 2000, 
                                overlay_style: str = "transparent"):
    """Cached version - use file path + params as cache key."""
    return raster_to_rgba_image(path, cmap_name, vmin, vmax, nodata, max_dim, overlay_style)
```

**Benefits:**
- Cache keyed by: file path, colormap, style, dimensions
- Invalidates automatically if file changes (Streamlit tracks file mtime)
- Saves 5-15 seconds per page load
- **ROI: 🔥 HIGHEST - ~40-60% total speedup**

#### 1.2 Cache Legend PNG Generation
**Implementation:**
```python
@st.cache_data(ttl=3600, show_spinner=False)
def make_continuous_legend_png_cached(cmap_name: str, vmin: float, vmax: float, 
                                     title: str, width_px=260) -> bytes:
    """Cached legend generation - matplotlib figures are expensive."""
    return make_continuous_legend_png(cmap_name, vmin, vmax, title, width_px)

@st.cache_data(ttl=3600, show_spinner=False)
def make_lulc_legend_png_cached(width_px=240) -> bytes:
    """Cached LULC legend."""
    return make_lulc_legend_png(width_px)
```

**Benefits:**
- Cache keyed by: colormap, vmin, vmax, title, width
- Legends rarely change during session
- Saves 2-6 seconds per render
- **ROI: 🔥 HIGH - ~20-30% speedup**

#### 1.3 Cache Meteorology Data Processing
**Implementation:**
```python
@st.cache_data(ttl=3600, show_spinner=False)
def _load_meteorology_timeseries_cached(csv_path: str) -> Optional[pd.DataFrame]:
    """Cached met data loading."""
    return _load_meteorology_timeseries(csv_path)

@st.cache_data(ttl=600, show_spinner=False)
def _quarterly_rainfall_average_cached(df: pd.DataFrame) -> pd.DataFrame:
    """Cached quarterly aggregation."""
    return _quarterly_rainfall_average(df)
```

**Benefits:**
- File I/O and DataFrame ops run once per file
- Chart generation still needed but data prep is instant
- Saves 0.5-2 seconds
- **ROI: 🔥 MEDIUM - ~5-10% speedup**

#### 1.4 Cache Summary JSON + Paths
**Implementation:**
```python
@st.cache_data(ttl=3600, show_spinner=False)
def read_summary_cached(summary_path: str) -> dict:
    """Cached summary JSON reading."""
    return read_summary(summary_path)
```

**Benefits:**
- Avoids repeated JSON parsing
- Saves 0.1-0.5 seconds
- **ROI: 🔥 LOW-MEDIUM - ~2-5% speedup**

#### 1.5 Cache Grid Cells + KDTree
**Implementation:**
```python
@st.cache_resource(show_spinner=False)
def load_grid_cells_cached(path: str = "data/grid_cells.csv"):
    """Cached grid loading with KDTree. Use cache_resource since KDTree is not serializable."""
    # Existing logic here
    df = pd.read_csv(path)
    coords = np.vstack([df['lat'].values, df['lon'].values]).T
    if KDTree is not None:
        kd = KDTree(coords)
    else:
        kd = None
    return (df, kd, coords)
```

**Benefits:**
- KDTree built once per session
- Saves 0.2-1 second on location searches
- **ROI: 🔥 MEDIUM - ~3-8% speedup for location searches**

### Phase 2: Smart Rendering (Medium Priority) ⚡⚡

#### 2.1 Detect Display-Only Changes
**Problem:** Opacity slider or overlay style change triggers full raster reloading
**Solution:** Add session state flags to detect if data layers changed vs. display-only

```python
# Track which data has changed
if "last_render_paths" not in st.session_state:
    st.session_state.last_render_paths = {}

# Before rendering map
current_paths_hash = hashlib.md5(json.dumps(paths, sort_keys=True).encode()).hexdigest()
data_changed = st.session_state.get("last_render_paths") != current_paths_hash

if data_changed:
    # Re-cache everything
    st.session_state.last_render_paths = current_paths_hash
```

**Benefits:**
- Skip expensive operations when only opacity/style changed
- Saves 3-10 seconds on display-only adjustments
- **ROI: 🔥 MEDIUM - ~15-25% speedup for display tweaks**

#### 2.2 Lazy Raster Loading
**Problem:** All layers loaded even if not displayed
**Solution:** Only load RGBA images for checked layers

```python
# BEFORE (current):
# Loads all 5 layers unconditionally

# AFTER:
if show_risk:
    img_risk = raster_to_rgba_image_cached(...)  # Only if needed
```

**Current Code Already Does This** (lines 656-681) - ✅ No action needed

#### 2.3 Optimize Raster Sampling for Marker
**Problem:** Opens raster file again for point sampling
**Solution:** Cache raster arrays, reuse for both display and sampling

```python
@st.cache_data(ttl=3600, show_spinner=False)
def read_raster_array_cached(path: str, max_dim: int = 2000):
    """Cache just the array reading, separate from RGBA conversion."""
    with rasterio.open(path) as src:
        # ... existing read logic
    return arr, mask, transform, crs
```

**Benefits:**
- Avoid duplicate file reads
- Saves 0.2-0.5 seconds per location marker
- **ROI: 🔥 LOW - ~2-5% speedup for marker operations**

### Phase 3: Geocoding Optimization (Lower Priority) ⚡

#### 3.1 Two-Tier Geocoding Cache
**Implementation:**
```python
# In-memory cache (fast)
_GEOCODE_MEMORY_CACHE = {}

@st.cache_data(ttl=86400, show_spinner=False)  # 24hr cache
def geocode_nominatim_cached(query: str, limit: int = 6):
    """Dual-layer cache: memory + disk."""
    return geocode_nominatim(query, limit)
```

**Benefits:**
- Instant lookup for repeat queries
- Saves 0.5-1 second on cached queries
- **ROI: 🔥 LOW - ~5-10% speedup for location searches**

### Phase 4: Advanced Optimizations (Future) ⚡

#### 4.1 Incremental Map Updates
- Use Streamlit fragments (@st.experimental_fragment) for map area
- Only refresh map when layer data changes, not on every widget interaction

#### 4.2 Web Workers / Async Processing
- Move raster processing to background threads (if Streamlit supports)
- Show loading spinners while computing in background

#### 4.3 Pre-computed Pyramid/Tiles
- Generate multiple resolution levels (like map tiles)
- Serve appropriate resolution based on zoom level

---

## Implementation Priority

### Immediate (This Session) - Expected 70-85% speedup
1. ✅ Add `@st.cache_data` to `raster_to_rgba_image` wrapper
2. ✅ Add `@st.cache_data` to legend generation functions
3. ✅ Add `@st.cache_data` to meteorology loading
4. ✅ Add `@st.cache_data` to `read_summary`
5. ✅ Add `@st.cache_resource` to `load_grid_cells`

### Short-term (Next Session) - Additional 5-15% speedup
6. Implement display-only change detection
7. Optimize raster sampling with array caching
8. Add geocoding `@st.cache_data` decorator

### Medium-term (Future Enhancement) - Additional 5-10% speedup
9. Experiment with Streamlit fragments for map isolation
10. Investigate async/background processing options

---

## Testing Plan

### Before Implementation Baseline
- [ ] Record time for: Initial page load
- [ ] Record time for: Weight slider change + recompute
- [ ] Record time for: Layer toggle (checkbox)
- [ ] Record time for: Opacity slider change
- [ ] Record time for: Location search + marker placement

### After Phase 1 (Caching)
- [ ] Verify: Initial load ~same (cache miss)
- [ ] Verify: Second render <1 second (cache hit)
- [ ] Verify: Weight change still triggers recompute correctly
- [ ] Verify: Layer toggle <1 second
- [ ] Verify: Display-only changes <0.5 second

### Regression Testing
- [ ] Verify: Recompute button still fetches when AOI changes
- [ ] Verify: Recompute button updates flood risk correctly
- [ ] Verify: Location search still works (geocoding + marker)
- [ ] Verify: All layers display correctly with all overlay styles
- [ ] Verify: Legends update when changing layer selections

---

## Expected Performance After Optimization

| Scenario | Before (Current) | After Phase 1 | Speedup |
|----------|------------------|---------------|---------|
| **Initial page load** | 8-25s | 7-20s | ~10-20% |
| **Second render (cached)** | 8-25s | 0.5-2s | **~90-95%** ⚡⚡⚡ |
| **Weight slider change** | 8-25s | 1-3s | **~85-90%** ⚡⚡⚡ |
| **Layer toggle** | 6-15s | 0.5-1.5s | **~90-95%** ⚡⚡⚡ |
| **Opacity/style change** | 6-15s | 0.3-1s | **~93-95%** ⚡⚡⚡ |
| **Location search** | 2-5s | 0.2-1s | **~80-90%** ⚡⚡ |

**Overall User Experience:**
- First load: Slightly slower (acceptable - data loading)
- All subsequent interactions: **Near-instant (<1-2 seconds)** ⚡⚡⚡
- 10-20x improvement for typical usage patterns

---

## Risk Assessment

### Low Risk
- ✅ `@st.cache_data` is stable API (not experimental)
- ✅ Caching is transparent to user (no behavior change)
- ✅ File-based invalidation works automatically (mtime tracking)
- ✅ Can easily disable caching by removing decorators

### Medium Risk
- ⚠️ Cache may grow large if many different parameters used
  - **Mitigation:** Set `max_entries=50` parameter
  - **Mitigation:** Set `ttl=3600` (1 hour expiry)
- ⚠️ Cached data may become stale if files updated externally
  - **Mitigation:** Streamlit tracks file mtime automatically
  - **Mitigation:** Manual cache clear button if needed

### High Risk
- ❌ None identified

---

## Next Steps

1. **Review this plan** - Confirm priorities and approach
2. **Implement Phase 1 caching** - 5 critical cache decorators
3. **Test thoroughly** - Verify correctness and performance gains
4. **Measure results** - Document actual speedup achieved
5. **Plan Phase 2** - If needed, implement display-change detection

---

## Notes

- **Why regression happened:** Each new feature (rainfall chart, location search, overlay styles, grid sampling) added uncached I/O or computation
- **Root cause:** No caching strategy applied to new features
- **Prevention:** Always wrap expensive operations in `@st.cache_data` or `@st.cache_resource`
- **Streamlit caching docs:** https://docs.streamlit.io/library/advanced-features/caching
