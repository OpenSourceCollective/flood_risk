#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flood_demo_modular_stable.py

Compute a composite flood susceptibility / flood risk raster (0–1) from prepared layers.

Inputs expected in prepared_layers_summary.json -> outputs:
  - dist_to_river_m
  - drainage_density_km_per_km2
  - soil_sand_pct
  - lulc_*_proxy  (any of: lulc_worldcover_proxy, lulc_io_annual_proxy, lulc_proxy, lulc)

Output:
  - flood_risk_0to1.tif
  - prepared_layers_summary.json updated with flood_risk_0to1 path + weights used

Run (CLI):
  python flood_demo_modular_stable.py --summary data/rasters/prepared_layers_summary.json \
         --w-dist 0.35 --w-drainage 0.25 --w-soil 0.20 --w-lulc 0.20
"""
import os, sys, json, argparse
import numpy as np
import rasterio

def robust_minmax(x: np.ndarray, lower_q=2.0, upper_q=98.0) -> np.ndarray:
    """Percentile-based min-max scaling to [0,1] with outlier resistance."""
    x = x.astype("float32")
    lo = np.nanpercentile(x, lower_q); hi = np.nanpercentile(x, upper_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(x); hi = np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype="float32")
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)

def lulc_class_to_risk_weight(lulc_arr: np.ndarray) -> np.ndarray:
    """
    Map your proxy LULC classes to a relative flood-risk contribution.
      1 water, 2 built, 3 veg, 4 ag, 5 wetland, 6 other
    """
    mapping = {1:1.00, 2:0.85, 5:0.75, 4:0.55, 6:0.45, 3:0.35}
    out = np.zeros_like(lulc_arr, dtype="float32")
    for k, v in mapping.items():
        out[lulc_arr == k] = v
    return out

def compute_risk(dist_to_river_m, drainage_density, soil_sand_pct, lulc_arr, weights):
    """
    Compute a weighted composite flood risk index:
      - distance-to-river is inverted (closer => higher risk)
      - soil sand is inverted (more sand => more infiltration => lower risk)
    """
    dist_norm = robust_minmax(dist_to_river_m); dist_risk = 1.0 - dist_norm
    drainage_norm = robust_minmax(drainage_density)
    soil_norm = robust_minmax(soil_sand_pct); soil_risk = 1.0 - soil_norm
    lulc_risk = lulc_class_to_risk_weight(lulc_arr)

    risk = (weights["dist"]*dist_risk +
            weights["drainage"]*drainage_norm +
            weights["soil"]*soil_risk +
            weights["lulc"]*lulc_risk)
    return robust_minmax(risk)

def _pick_lulc_key(outputs: dict) -> str:
    for k in ("lulc_worldcover_proxy", "lulc_io_annual_proxy", "lulc_proxy", "lulc"):
        if k in outputs:
            return k
    raise KeyError("No LULC proxy raster key found in meta['outputs']. Expected one of "
                   "lulc_worldcover_proxy, lulc_io_annual_proxy, lulc_proxy, lulc.")

def run(summary_path, w_dist=0.35, w_drainage=0.25, w_soil=0.20, w_lulc=0.20,
        out_path_override=None, normalize_weights=False):
    """
    Compute flood_risk_0to1.tif from prepared layers.

    If normalize_weights=True, weights are scaled to sum to 1 (useful when sliders are changed).
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    p = meta.get("outputs", {})

    lulc_key = _pick_lulc_key(p)

    with rasterio.open(p["dist_to_river_m"]) as src:
        dist = src.read(1); transform = src.transform; crs = src.crs
    with rasterio.open(p["drainage_density_km_per_km2"]) as src:
        dd = src.read(1)
    with rasterio.open(p["soil_sand_pct"]) as src:
        soil = src.read(1)
    with rasterio.open(p[lulc_key]) as src:
        lulc = src.read(1)

    weights = {"dist": float(w_dist), "drainage": float(w_drainage),
               "soil": float(w_soil), "lulc": float(w_lulc)}

    if normalize_weights:
        s = sum(weights.values())
        if s > 0:
            weights = {k: v/s for k, v in weights.items()}

    risk01 = compute_risk(dist, dd, soil, lulc, weights)

    out_path = out_path_override or os.path.join(os.path.dirname(summary_path), "flood_risk_0to1.tif")
    profile = {
        "driver": "GTiff",
        "height": risk01.shape[0],
        "width": risk01.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": np.nan,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(risk01.astype("float32"), 1)

    meta.setdefault("outputs", {})
    meta["outputs"]["flood_risk_0to1"] = out_path
    meta["flood_risk_weights_used"] = weights
    meta["lulc_key_used"] = lulc_key

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved flood risk: {out_path}")
    print(f"Updated summary: {summary_path}")
    return out_path, weights

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/rasters/prepared_layers_summary.json")
    ap.add_argument("--w-dist", type=float, default=0.35)
    ap.add_argument("--w-drainage", type=float, default=0.25)
    ap.add_argument("--w-soil", type=float, default=0.20)
    ap.add_argument("--w-lulc", type=float, default=0.20)
    ap.add_argument("--normalize-weights", action="store_true",
                    help="If set, normalize weights to sum to 1.")
    ap.add_argument("--out", default=None, help="Optional output path for flood_risk_0to1.tif")
    args = ap.parse_args()

    run(
        summary_path=args.summary,
        w_dist=args.w_dist,
        w_drainage=args.w_drainage,
        w_soil=args.w_soil,
        w_lulc=args.w_lulc,
        out_path_override=args.out,
        normalize_weights=args.normalize_weights,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
