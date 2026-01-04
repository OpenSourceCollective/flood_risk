#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_physical_improved.py

Improved version of fetch_physical.py addressing:
1) dist_to_river_m.tif ignoring inland waterbodies:
   - Distance is now to the union of OSM waterway lines AND OSM inland water polygons.
   - Output filename remains dist_to_river_m.tif for backward compatibility (now effectively "distance to water").

2) drainage_density_km_per_km2.tif appearing patchy:
   - Fishnet cell size default reduced (0.005°).
   - Optional Gaussian smoothing applied to reduce blockiness (set --smooth-sigma 0 to disable).

Outputs (unchanged filenames):
- data/rasters/dist_to_river_m.tif
- data/rasters/drainage_density_km_per_km2.tif
- data/rasters/soil_sand_pct.tif
- data/rasters/lulc_worldcover_proxy.tif OR data/rasters/lulc_io_annual_proxy.tif
- data/rasters/prepared_layers_summary.json

Run:
  python fetch_physical_improved.py --place "Lagos, Nigeria"
"""
import os, sys, math, json, warnings, argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.merge import merge
from pyproj import CRS
import requests

from rasterio.errors import RasterioIOError

def safe_rio_open(*args, **kwargs):
    """
    Wrapper around rasterio.open with clearer error messages.

    Works for BOTH reading and writing:
      - safe_rio_open("file.tif")
      - safe_rio_open("file.tif", "w", **profile)
    """
    try:
        return rasterio.open(*args, **kwargs)
    except RasterioIOError as e:
        target = args[0] if args else "<unknown>"
        raise RasterioIOError(f"Rasterio failed to open: {target}\nOriginal error: {e}") from e

def download_url_to_file(url: str, out_path: str, timeout: int = 240):
    """Download a URL to a local file (helps when GDAL https/vsi is flaky)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    if os.path.getsize(out_path) < 1024:
        raise RuntimeError(f"Downloaded file too small (<1KB): {out_path} from {url}")
    # If it's supposed to be a TIFF, validate header
    if out_path.lower().endswith((".tif", ".tiff")):
        validate_geotiff_or_dump(out_path, source_url=url)
    return out_path

def looks_like_tiff_bytes(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        return head in (b"II*\x00", b"MM\x00*")
    except Exception:
        return False

def validate_geotiff_or_dump(path: str, source_url: str = ""):
    """
    Ensure file is a real GeoTIFF. If not, write a .debug file next to it for inspection.
    """
    if not looks_like_tiff_bytes(path):
        dbg = path + ".debug"
        try:
            with open(path, "rb") as f:
                content = f.read(4096)
            with open(dbg, "wb") as f:
                f.write(content)
        except Exception:
            pass
        raise RuntimeError(f"Downloaded file is not a TIFF: {path} (from {source_url}). "
                           f"First bytes dumped to: {dbg}")

def validate_raster_read(path: str):
    """
    Open and read a small window to ensure GDAL can actually read it.
    """
    with safe_rio_open(path) as src:
        # Read a small 64x64 window from the top-left (or smaller if tiny)
        h = min(64, src.height)
        w = min(64, src.width)
        arr = src.read(1, window=((0, h), (0, w)))
        if arr is None or arr.size == 0:
            raise RuntimeError(f"Raster read returned empty array: {path}")

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import osmnx as ox
except Exception:
    ox = None

def features_from_bbox_compat(north, south, east, west, tags):
    """
    Call osmnx.features.features_from_bbox across multiple OSMnx API variants.

    Different OSMnx versions have different signatures:
      - features_from_bbox(north, south, east, west, tags)
      - features_from_bbox(north, south, east, west, tags=...)
      - features_from_bbox(bbox, tags=...) with bbox=(west, south, east, north) OR (north, south, east, west)

    We try common patterns and raise the last TypeError if all fail.
    """
    if ox is None:
        raise RuntimeError("osmnx is required (pip install osmnx).")
    from osmnx import features as ox_features

    last_err = None
    # 1) positional (north, south, east, west, tags)
    try:
        return ox_features.features_from_bbox(north, south, east, west, tags)
    except TypeError as e:
        last_err = e
    # 2) positional + tags kw
    try:
        return ox_features.features_from_bbox(north, south, east, west, tags=tags)
    except TypeError as e:
        last_err = e
    # 3) bbox as single positional (try both common orders)
    try:
        return ox_features.features_from_bbox((west, south, east, north), tags=tags)
    except TypeError as e:
        last_err = e
    try:
        return ox_features.features_from_bbox((north, south, east, west), tags=tags)
    except TypeError as e:
        last_err = e
    # 4) bbox keyword (try both)
    try:
        return ox_features.features_from_bbox(bbox=(west, south, east, north), tags=tags)
    except TypeError as e:
        last_err = e
    try:
        return ox_features.features_from_bbox(bbox=(north, south, east, west), tags=tags)
    except TypeError as e:
        last_err = e

    raise last_err

try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter
except Exception:
    distance_transform_edt = None
    gaussian_filter = None

@dataclass
class Config:
    place_name: str = "Lagos, Nigeria"
    grid_res_deg: float = 0.001
    fishnet_cell_deg: float = 0.005
    buffer_deg: float = 0.02
    soil_res_m: int = 250
    out_dir: str = "data/rasters"
    tmp_dir: str = "data/tmp"
    crs_epsg: int = 4326
    worldcover_year: int = 2024
    smooth_drainage_sigma: float = 1.25  # pixels; 0 disables

CFG = Config()

def ensure_dirs():
    os.makedirs(CFG.out_dir, exist_ok=True)
    os.makedirs(CFG.tmp_dir, exist_ok=True)

def geocode_aoi(place: str) -> gpd.GeoDataFrame:
    if ox is None:
        raise RuntimeError("osmnx is required (pip install osmnx).")
    gdf = ox.geocode_to_gdf(place)
    return gdf.to_crs(epsg=CFG.crs_epsg)

def bbox_from_gdf(gdf: gpd.GeoDataFrame, buffer_deg: float = 0.02):
    minx, miny, maxx, maxy = gdf.total_bounds
    return (float(minx - buffer_deg), float(miny - buffer_deg),
            float(maxx + buffer_deg), float(maxy + buffer_deg))

def make_raster_grid_from_bbox(bbox: Tuple[float, float, float, float], res_deg: float):
    west, south, east, north = bbox
    width = int(math.ceil((east - west) / res_deg))
    height = int(math.ceil((north - south) / res_deg))
    transform = from_origin(west, north, res_deg, res_deg)
    return transform, (height, width)

def save_geotiff(path, array, transform, crs, nodata=None, dtype=None, compress="deflate"):
    if dtype is None:
        dtype = array.dtype
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": compress,
        "nodata": nodata,
    }
    with safe_rio_open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)

def rasterize_geoms(geoms, out_shape, transform, burn_value=1, all_touched=False, dtype="uint8"):
    if len(geoms) == 0:
        return np.zeros(out_shape, dtype=dtype)
    shapes = [(geom, burn_value) for geom in geoms]
    return rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype=dtype,
    )

def fetch_osm_waterways_and_waterpolys(bbox):
    if ox is None:
        raise RuntimeError("osmnx is required (pip install osmnx).")
    west, south, east, north = bbox

    tags_lines = {"waterway": ["river", "stream", "canal", "drain"]}
    tags_polys = {
        "natural": ["water", "wetland"],
        "water": True,
        "landuse": ["reservoir"],
    }

    def _features(tags):
        return features_from_bbox_compat(north, south, east, west, tags)

    g_lines = _features(tags_lines)
    g_polys = _features(tags_polys)

    if g_lines is None or g_lines.empty:
        raise RuntimeError("No waterways returned from OSM for the given bbox.")

    g_lines = g_lines[g_lines.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy().to_crs(epsg=4326)

    if g_polys is None or g_polys.empty:
        g_polys = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    else:
        g_polys = g_polys[g_polys.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy().to_crs(epsg=4326)

    return g_lines, g_polys

def compute_distance_to_water_raster(water_lines, water_polys, transform, out_shape, grid_res_deg: float):
    if distance_transform_edt is None:
        raise RuntimeError("scipy is required for distance transform (pip install scipy).")
    lines_r = rasterize_geoms(water_lines.geometry.values, out_shape, transform, burn_value=1, all_touched=True, dtype="uint8")
    polys_r = rasterize_geoms(water_polys.geometry.values, out_shape, transform, burn_value=1, all_touched=True, dtype="uint8")
    water_mask = (lines_r > 0) | (polys_r > 0)
    dist_px = distance_transform_edt(~water_mask)
    meters_per_deg = 111_320.0
    dist_m = dist_px * grid_res_deg * meters_per_deg
    dist_m[water_mask] = 0.0
    return dist_m.astype("float32")

def fishnet_grid(bbox, cell_deg):
    from shapely.geometry import box
    west, south, east, north = bbox
    xs = np.arange(west, east, cell_deg)
    ys = np.arange(south, north, cell_deg)
    polys = [box(x, y, x + cell_deg, y + cell_deg) for x in xs for y in ys]
    return gpd.GeoDataFrame({"geometry": polys}, crs="EPSG:4326")

def compute_drainage_density_grid(water_lines, bbox, cell_deg):
    grid = fishnet_grid(bbox, cell_deg)
    metric = 3857
    ww_m = water_lines.to_crs(epsg=metric)
    grid_m = grid.to_crs(epsg=metric)
    sindex = ww_m.sindex
    lengths_km = []
    areas_km2 = grid_m.geometry.area / 1e6
    for poly in grid_m.geometry:
        possible = list(sindex.intersection(poly.bounds))
        if not possible:
            lengths_km.append(0.0)
            continue
        sub = ww_m.iloc[possible]
        inter = sub.intersection(poly)
        total_len_m = 0.0
        for g in inter:
            if g is None:
                continue
            total_len_m += g.length
        lengths_km.append(total_len_m / 1000.0)
    grid["dd_km_per_km2"] = np.divide(lengths_km, areas_km2, out=np.zeros_like(areas_km2), where=areas_km2 > 0)
    return grid

def rasterize_grid_values(grid, value_col, transform, out_shape):
    shapes = list(zip(grid.geometry.values, grid[value_col].values))
    return rasterize(shapes=shapes, out_shape=out_shape, transform=transform, fill=0.0,
                     all_touched=True, dtype="float32")

def smooth_raster(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or gaussian_filter is None:
        return arr.astype("float32")
    sm = gaussian_filter(arr.astype("float32"), sigma=float(sigma))
    return np.maximum(sm, 0.0).astype("float32")

SOILGRID_WCS = "https://maps.isric.org/mapserv?map=/map/sand.map"

def build_wcs_url(bbox, coverage, res_m, format_str="GEOTIFF_INT16"):
    west, south, east, north = bbox
    subset_parts = "&".join([f"SUBSET=long({west},{east})", f"SUBSET=lat({south},{north})"])
    url = (
        f"{SOILGRID_WCS}&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={coverage}&FORMAT={format_str}&{subset_parts}"
        f"&RESX={res_m}m&RESY={res_m}m&CRS=EPSG:4326"
        f"&SUBSETTINGCRS=EPSG:4326&OUTPUTCRS=EPSG:4326"
    )
    return url

def _looks_like_geotiff(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        return head in (b"II*\x00", b"MM\x00*")
    except Exception:
        return False

def download_soilgrids_bbox(bbox, out_tif, coverage="sand_0-5cm_Q0.5", res_m=250):
    url = build_wcs_url(bbox, coverage, res_m, format_str="GEOTIFF_INT16")
    print(f"[SoilGrids] Requesting WCS: {url}")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with open(out_tif, "wb") as f:
        f.write(r.content)
    if not _looks_like_geotiff(out_tif):
        dbg = out_tif + ".txt"
        with open(dbg, "wb") as f:
            f.write(r.content)
        raise RuntimeError(f"SoilGrids did not return a GeoTIFF. See debug: {dbg}")
    print(f"[SoilGrids] Saved: {out_tif}")

def _pc_client():
    from pystac_client import Client
    import planetary_computer as pc
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def fetch_worldcover_items(bbox, year=2024):
    client = _pc_client()
    dt = f"{int(year)}-01-01/{int(year)}-12-31"
    search = client.search(collections=["esa-worldcover"], bbox=list(map(float, bbox)), datetime=dt)
    items = list(search.get_items())
    if not items:
        search = client.search(collections=["esa-worldcover"], bbox=list(map(float, bbox)))
        items = list(search.get_items())
    srcs = []
    for it in items:
        assets = it.assets
        key = next((k for k in assets.keys() if "map" in k.lower()), None)
        if key is None:
            continue
        href = assets[key].href
        local = os.path.join(CFG.tmp_dir, "pc_tiles", f"worldcover_{it.id}.tif")
        try:
            download_url_to_file(href, local, timeout=240)
            validate_raster_read(local)
            srcs.append(safe_rio_open(local))
        except Exception as e:
            print(f"[WARN] Failed to fetch/open WorldCover tile {it.id}: {e}")
    return srcs

def fetch_io_lulc_items(bbox, year=2024):
    client = _pc_client()
    dt = f"{int(year)}-01-01/{int(year)}-12-31"
    collections = ["io-lulc-annual-v02", "io-lulc-annual"]
    for coll in collections:
        try:
            search = client.search(collections=[coll], bbox=list(map(float, bbox)), datetime=dt)
            items = list(search.get_items())
            if items:
                break
        except Exception:
            items = []
    srcs = []
    for it in items:
        assets = it.assets
        key = "data" if "data" in assets else next(iter(assets.keys()))
        href = assets[key].href
        local = os.path.join(CFG.tmp_dir, "pc_tiles", f"worldcover_{it.id}.tif")
        try:
            download_url_to_file(href, local, timeout=240)
            validate_raster_read(local)
            srcs.append(safe_rio_open(local))
        except Exception as e:
            print(f"[WARN] Failed to fetch/open WorldCover tile {it.id}: {e}")
    return srcs

def reclassify_worldcover_to_proxy(wc_arr):
    out = np.full_like(wc_arr, 6, dtype=np.uint8)
    out[(wc_arr == 80)] = 1
    out[(wc_arr == 50)] = 2
    out[(wc_arr == 90) | (wc_arr == 95)] = 5
    out[(wc_arr == 10) | (wc_arr == 20) | (wc_arr == 30)] = 3
    out[(wc_arr == 40)] = 4
    return out

def reclassify_io_to_proxy(io_arr):
    out = np.full_like(io_arr, 6, dtype=np.uint8)
    out[(io_arr == 0)] = 1
    out[(io_arr == 6)] = 2
    out[(io_arr == 1) | (io_arr == 2) | (io_arr == 5)] = 3
    out[(io_arr == 4)] = 4
    out[(io_arr == 3)] = 5
    out[(io_arr == 7) | (io_arr == 8)] = 6
    return out

def fetch_worldcover_or_io_to_grid(bbox, target_transform, target_shape, out_path_wc, out_path_io, year=2024):
    srcs = fetch_worldcover_items(bbox, year=year)
    label = None
    if not srcs:
        srcs = fetch_io_lulc_items(bbox, year=year)
        label = "io"
    if not srcs:
        raise RuntimeError("No LULC tiles found from Planetary Computer for the given bbox/year.")
    print(f"[DEBUG] Merging {len(srcs)} LULC tiles...")
    mosaic, mosaic_transform = merge(srcs)
    for s in srcs:
        s.close()
    arr = mosaic[0]

    import rasterio.warp
    tmp_reproj = os.path.join(CFG.tmp_dir, "lulc_reproj.tif")
    profile = {
        "driver": "GTiff",
        "height": target_shape[0],
        "width": target_shape[1],
        "count": 1,
        "dtype": arr.dtype.name,
        "crs": "EPSG:4326",
        "transform": target_transform,
        "compress": "deflate",
        "nodata": 0,
    }
    with safe_rio_open(tmp_reproj, "w", **profile) as dst:
        rasterio.warp.reproject(
            source=arr,
            destination=rasterio.band(dst, 1),
            src_transform=mosaic_transform,
            src_crs="EPSG:4326",
            dst_transform=target_transform,
            dst_crs="EPSG:4326",
            resampling=rasterio.warp.Resampling.nearest,
        )
    print(f"[DEBUG] Opening reprojected LULC: {tmp_reproj}")
    with safe_rio_open(tmp_reproj) as src:
        data = src.read(1)

    if label == "io":
        proxy = reclassify_io_to_proxy(data)
        out_path = out_path_io
        provider = "IO Annual v02"
    else:
        proxy = reclassify_worldcover_to_proxy(data)
        out_path = out_path_wc
        provider = "ESA WorldCover"

    save_geotiff(out_path, proxy, target_transform, CRS.from_epsg(4326), nodata=0, dtype="uint8")
    return out_path, provider

def parse_args():
    ap = argparse.ArgumentParser(description="Fetch & prepare physical flood layers (improved).")
    ap.add_argument("--place", default=CFG.place_name)
    ap.add_argument("--grid-res-deg", type=float, default=CFG.grid_res_deg)
    ap.add_argument("--fishnet-deg", type=float, default=CFG.fishnet_cell_deg)
    ap.add_argument("--buffer-deg", type=float, default=CFG.buffer_deg)
    ap.add_argument("--soil-res-m", type=int, default=CFG.soil_res_m)
    ap.add_argument("--smooth-sigma", type=float, default=CFG.smooth_drainage_sigma)
    ap.add_argument("--worldcover-year", type=int, default=CFG.worldcover_year)
    return ap.parse_args()

def main():
    args = parse_args()
    CFG.place_name = args.place
    CFG.grid_res_deg = float(args.grid_res_deg)
    CFG.fishnet_cell_deg = float(args.fishnet_deg)
    CFG.buffer_deg = float(args.buffer_deg)
    CFG.soil_res_m = int(args.soil_res_m)
    CFG.smooth_drainage_sigma = float(args.smooth_sigma)
    CFG.worldcover_year = int(args.worldcover_year)

    ensure_dirs()
    crs = CRS.from_epsg(CFG.crs_epsg)

    print("=== Fetch & Prepare Layers (improved) ===")
    aoi = geocode_aoi(CFG.place_name)
    bbox = bbox_from_gdf(aoi, buffer_deg=CFG.buffer_deg)
    transform, out_shape = make_raster_grid_from_bbox(bbox, CFG.grid_res_deg)

    # Water features
    water_lines, water_polys = fetch_osm_waterways_and_waterpolys(bbox)

    dist = compute_distance_to_water_raster(water_lines, water_polys, transform, out_shape, grid_res_deg=CFG.grid_res_deg)
    dist_path = os.path.join(CFG.out_dir, "dist_to_river_m.tif")
    save_geotiff(dist_path, dist, transform, crs, nodata=np.nan, dtype="float32")

    dd_grid = compute_drainage_density_grid(water_lines, bbox, CFG.fishnet_cell_deg)
    dd = rasterize_grid_values(dd_grid, "dd_km_per_km2", transform, out_shape)
    dd = smooth_raster(dd, CFG.smooth_drainage_sigma)
    dd_path = os.path.join(CFG.out_dir, "drainage_density_km_per_km2.tif")
    save_geotiff(dd_path, dd, transform, crs, nodata=0.0, dtype="float32")

    # LULC
    wc_out = os.path.join(CFG.out_dir, "lulc_worldcover_proxy.tif")
    io_out = os.path.join(CFG.out_dir, "lulc_io_annual_proxy.tif")
    lulc_path, provider = fetch_worldcover_or_io_to_grid(bbox, transform, out_shape, wc_out, io_out, year=CFG.worldcover_year)

    # Soil
    soil_raw = os.path.join(CFG.tmp_dir, "soil_sand_raw.tif")
    download_soilgrids_bbox(bbox, soil_raw, coverage="sand_0-5cm_Q0.5", res_m=CFG.soil_res_m)

    soil_path = os.path.join(CFG.out_dir, "soil_sand_pct.tif")
    import rasterio.warp
    print(f"[DEBUG] Opening SoilGrids raster: {soil_raw}")
    with safe_rio_open(soil_raw) as src:
        profile = {
            "driver": "GTiff",
            "height": out_shape[0],
            "width": out_shape[1],
            "count": 1,
            "dtype": src.dtypes[0],
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "deflate",
            "nodata": None,
        }
        with safe_rio_open(soil_path, "w", **profile) as dst:
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=rasterio.warp.Resampling.bilinear,
            )

    outputs = {
        "dist_to_river_m": dist_path,
        "drainage_density_km_per_km2": dd_path,
        "soil_sand_pct": soil_path,
        "lulc": lulc_path,
        "lulc_proxy": lulc_path,
    }
    if provider == "ESA WorldCover":
        outputs["lulc_worldcover_proxy"] = lulc_path
    else:
        outputs["lulc_io_annual_proxy"] = lulc_path

    summary = {
        "aoi_place": CFG.place_name,
        "bbox": bbox,
        "grid_res_deg": float(CFG.grid_res_deg),
        "fishnet_cell_deg": float(CFG.fishnet_cell_deg),
        "smooth_drainage_sigma": float(CFG.smooth_drainage_sigma),
        "soil_res_m": int(CFG.soil_res_m),
        "worldcover_year": int(CFG.worldcover_year),
        "lulc_provider": provider,
        "outputs": outputs,
        "notes": {
            "dist_to_river_m": "Improved: distance to union(OSM waterways + inland water polygons).",
            "drainage_density_km_per_km2": "Improved: smaller fishnet + optional Gaussian smoothing."
        }
    }

    summary_path = os.path.join(CFG.out_dir, "prepared_layers_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {summary_path}")
    print("=== Done ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
