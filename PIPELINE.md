# Pipeline

All scripts take a working directory `W` as the first argument and read or write files inside it. The steps below are the order used for the paper. Steps 1–3 download inputs (see DATA_SOURCES.md for the sources).

> The scripts were run step by step for the paper. A single `run_all.sh` that runs the whole chain will be added and tested before publication.

## 1. Inputs
| Path in W | Content |
|---|---|
| `geo/traffic_census_2021_converted.parquet`, `geo/traffic_census_2015_converted.parquet` | Road Traffic Census sections (shiwaku/mlit-road-traffic-census-converter, release data-v1) |
| `estat_all/shp/r2ka*.shp` | 2020 census small-area boundaries with population (e-Stat) |
| `nox_ef_nilim671.csv` | NOx emission factor coefficients (NILIM Technical Note 671, Table 6.4), included in this repository under `inputs/` |
| `air/x/TD*.txt`, `air/x/TM*.txt` | Annual air quality monitoring data and station metadata: `python src/tenbou_dl.py W/air 2015,...,2023` |
| `tropomi_asia_2019.nc` | TROPOMI-inferred surface NO2, Asia, 2019 (Cooper et al. 2022, Zenodo 5484305) |
| noise model | `src/noise_lib.py` of japan-night-road-noise (for step 7) |

## 2. Stations
```
python src/stations_no2only.py W        # W/stations.csv: annual NO2 per station and fiscal year (>= 6,000 h)
cp W/stations.csv W/stations_all.csv
```

## 3. Receptors and features (2021 census network)
```
python src/receptors.py W               # W/rec/r_XX.parquet, sa_XX.parquet (12,298,351 locations)
python src/station_pop.py W
python src/station_features.py W        # also writes W/sections_nox.parquet
python src/grid_features.py W           # W/rec_grid.parquet, W/station_grid.csv
python src/road_receptors.py W          # road kernel at receptors
python src/sat_receptors.py W W/tropomi_asia_2019.nc   # W/rec_sat.parquet, W/station_sat*.csv
```

## 4. Main model, fiscal years 2019–2022 (2021 census)
```
python src/cv_main.py W cv.csv 2019 2020 2021 2022
python src/fit_final_sat.py W 2022
python src/predict_national_sat.py W 2022
```

## 5. Fiscal years 2015–2023 (2015 census for 2015–2019, 2021 census for 2020–2023)
```
python src/features_multiyear.py W 2015 2015 2020
python src/features_multiyear.py W 2021 2020 2025
python src/road_multiyear.py W 2015 2015,2020 01 02 ... 47
python src/road_multiyear.py W 2021 2020,2025 01 02 ... 47
python src/station_popgrid.py W
python src/release_build.py W OUT        # cross-validation, station predictions, receptor and area files for all years
```

## 6. Comparisons and sensitivity
```
python src/compare_external.py W EXT OUT     # Anenberg et al. 2022 and Cooper et al. 2022 at stations
python src/sat_background.py W EXT OUT
python src/sens_kriging.py W OUT
python src/sens_emission.py W OUT
```

## 7. Night noise and joint exposure
```
python src/noise_at_receptors.py W path/to/japan-night-road-noise/src W/nz 01 02 ... 47   # NOISE_NET=2021 (default), 2015T or 2021T
python src/receptor_noise_no2.py W OUT
python src/joint_noise.py W A38-14_GML/A38-14.dbf    # secondary medical areas (National Land Numerical Information A38)
```

## 8. Trends and urban form
```
python src/measured_trends.py TN_DIR W/stations_all.csv OUT   # TN_DIR: noise monitoring 2015-2023 (tn2015...tn2023)
python src/urban_form.py W OUT
python src/measured_trends_robust.py TN_DIR W/stations_all.csv OUT   # same estimate with subsets of sites (Extended Data Table 2)
```

## 9. Numbers in the paper, sensitivity of exposure, figures and web map
```
python src/unified_stats.py W RELEASE_PARQUET A38_DBF OUT        # key numbers, Table 1, correlations by spatial unit
python src/ed_thresholds.py W RELEASE_PARQUET OUT                # Extended Data Table 1 (other thresholds)
# Placement on building land (Extended Data Table 3); LU_DIR: National Land Numerical Information L03-b (2021)
python src/popalloc_building.py W W2 LU_DIR 01 02 ... 47
NOISE_CUT=200 python src/noise_at_receptors.py W2 path/to/japan-night-road-noise/src W2/nz 01 02 ... 47
python src/popalloc_compare.py W W2 RELEASE_PARQUET OUT 01 02 ... 47
# Cutoff distance of 500 m for noise sources (Extended Data Table 3)
NOISE_CUT=500 python src/noise_at_receptors.py W path/to/japan-night-road-noise/src W/nz 01 02 ... 47
python src/cutoff_sensitivity.py W RELEASE_PARQUET NEAREST_ROAD_PARQUET OUT
python src/release_results.py WORK OUT/data                # arithmetic-mean Lnight column and summary tables in data/results/
python src/make_figures.py WORK BOUNDARIES.gpkg figures/   # Figures 1-4
python src/make_webmap_data.py RELEASE_PARQUET path/to/japan-night-road-noise/docs/muni.geojson docs/muni.geojson
```
