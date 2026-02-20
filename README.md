# FloodRisk

Welcome to the Flood Risk repo. This project builds a lightweight, open flood-risk index from physical layers (distance to water, drainage density, soil texture, and land cover) and serves an interactive Streamlit viewer for exploration and recomputation.

## Why Flood Risk?

Flood exposure is growing in many cities, yet high‑quality risk maps can be slow to produce. This repo provides a reproducible, transparent pipeline that generates a 0–1 flood‑risk index map from open data so individuals/teams can rapidly obtain relative flood risk evaluation of different locations.

## Key Features

- End‑to‑end pipeline to fetch physical layers and build raster products.
- Composite flood‑risk index with configurable weights and optional normalization.
- Interactive Streamlit user interface.
- Meteorology downloader and a simple rainfall forecasting evaluator.
- Cached downloads and preprocessing to reduce repeated work.

## How it works

1. **Fetch physical layers** using [fetch_physical.py](fetch_physical.py):
	- Waterways + inland water bodies from OpenStreetMap (OSMnx)
	- Soil sand percentage from SoilGrids WCS
	- Land‑cover from ESA WorldCover or IO LULC (Planetary Computer)
	- Drainage density from OSM waterway length per grid cell
2. **Compute flood risk** with [flood_risk_compute.py](flood_risk_compute.py):
	- Normalizes each layer (robust min‑max)
	- Inverts distance‑to‑water and soil sand (less infiltration = higher risk)
	- Combines layers into a 0–1 raster using weights
3. **Explore in UI** with [ui.py](ui.py):
	- Live recompute with weight sliders
	- Overlay legends for each layer
	- Location search + nearest grid lookup

Outputs are stored in data/rasters and summarized in data/rasters/prepared_layers_summary.json.

# Future enhancements

- Add river discharge, elevation/DEM‑derived slope and other pertinent layers.
- Expand climate drivers and add forecast‑conditioned risk.
- Add per‑cell explainability and uncertainty layers.

# Acknowledgements

- OpenStreetMap contributors
- NASA POWER, SoilGrids, ESA WorldCover, and the Microsoft Planetary Computer

# Who we are

We are [Axum AI](https://axumai.org/) leveraging AI and Tech to empower Africa and Africans with custom, open-source solutions with a focus on social impact and development

# Dev Setup

- Clone repo and install venv environment

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

- Run the app

```bash
streamlit run ui.py
```

- Optional

    - Fetch meteorology data with 'python3 fetch_metrology.py --start 2023-01-01 --end 2024-12-31'


    - Fetch physical layers and build rasters with  'python3 fetch_physical.py --place "Lagos, Nigeria" '


    - Compute flood risk with already fetched layers 'python3 flood_risk_compute.py --summary data/rasters/prepared_layers_summary.json'
