# NO2 and nighttime road traffic noise at residential locations in Japan, FY2015–2023

Estimates for the paper "Diverging trends in nitrogen dioxide and nighttime road traffic noise in Japanese cities" (Daisuke Matsushita, Osaka Metropolitan University, ORCID 0000-0002-6765-9366). Code: https://github.com/m138/japan-traffic-no2 ([code DOI]).

Annual mean NO2 (ppb) at 12,298,351 residential locations covering the 2020 population of Japan, for fiscal years 2015–2023, with the part explained by nearby census roads and the nighttime road traffic noise (Lnight, 2021 traffic) of the companion model japan-night-road-noise. Leave-one-prefecture-out R² against monitoring stations: 0.78–0.81 (all), 0.46–0.60 (roadside).

Files: see the data dictionary below. License: CC BY 4.0. Please cite the paper and this dataset, and credit the data sources (MLIT Road Traffic Census; e-Stat 2020 census boundaries; NIES air quality and noise monitoring data of the Ministry of the Environment; Cooper et al. 2022 TROPOMI surface NO2).

# Data dictionary

Units: NO2 in ppb (multiply by 1.88 for µg/m³); noise in dB (A-weighted). Fiscal year (FY) YYYY runs from April YYYY to March YYYY+1. All means are population-weighted with the 2020 census population.

## small_area_no2_noise_2015_2023.csv
| Column | Meaning |
|---|---|
| KEY_CODE | 2020 census small-area code (e-Stat `KEY_CODE`; first 2 digits = prefecture, first 5 = municipality) |
| pop2020 | 2020 census population of the small area |
| NO2_YYYY | Mean annual NO2 at the residential locations of the area, FY YYYY (2015–2023) |
| road_YYYY | Part of NO2_YYYY explained by Road Traffic Census roads within 1 km |
| Lnight_2021_mean | Population-weighted arithmetic mean (dB) of nighttime (22:00–06:00) road traffic noise Lnight, 2021 traffic (japan-night-road-noise v4; 40 dB floor). Used for all area-level analyses in the paper |
| Lnight_2021_energy_mean | Population-weighted energy mean of the same Lnight (for reference) |
| share_Lnight55 | Share of residents with Lnight ≥ 55 dB |
| share_double_2022 | Share of residents with Lnight ≥ 55 dB and NO2 (FY2022) ≥ 20 µg/m³ |

## no2_2015_2023_by_municipality.csv, no2_2015_2023_by_prefecture.csv
| Column | Meaning |
|---|---|
| muni / pref | 5-digit municipality code (2020 boundaries; wards of designated cities and of Tokyo are separate rows, 1,895 rows) / 2-digit prefecture code |
| w | 2020 population |
| NO2_YYYY, road_YYYY | As above, aggregated |

## receptors_no2_noise_2015_2023.parquet
| Column | Meaning |
|---|---|
| rid | Location id (0 … 12,298,350) |
| KEY_CODE | Small area that contains the location |
| lon, lat | Coordinates (JGD2011, degrees). Locations are random points inside the small area, about one per 10 residents; they are not addresses |
| pop_w | Population represented by the location (small-area population divided equally) |
| Lnight_2021 | Nighttime road traffic noise (dB), 2021 traffic. Floor of 40 dB: 40 means no census road within 200 m or a lower level |
| NO2_YYYY, road_YYYY | NO2 and its road part, FY 2015–2023 |

## stations_lopo_2015_2023.csv
| Column | Meaning |
|---|---|
| fy, code | Fiscal year and NIES station code |
| kind | gen = general (ambient) station, road = roadside station |
| pref, lat, lon | Prefecture code and coordinates |
| NO2_obs | Measured annual mean NO2 (≥ 6,000 h of valid data) |
| NO2_lopo | Prediction when the station's prefecture was left out of the fit |

## section_nox_2015_2021.csv
| Column | Meaning |
|---|---|
| census | Census year (2015 or 2021) |
| section_id | Census section id (`census` field of the converted census files) |
| muni, road_type, motorway | Municipality code, census road type code, motorway flag (1 = motorway) |
| v24_small, v24_large | 24-hour traffic counts, both directions (vehicles/day) |
| speed_small, speed_large, speed_source | Travel speed (km/h) used for emission factors and its source (12h = 12-hour daytime mean, offpeak, peak, limit = posted limit) |
| length_km | Geodesic length of the section |
| nox_g_km_day, nox_t_year | NOx emissions with emission factors of the census year |

## results/
Summary tables behind the paper (English column names): key_numbers.csv, table1_by_municipality_type.csv, correlation_by_level.csv, measured_trends_2015_2023.csv and measured_trends_robustness.csv (site fixed-effect trends of measured NO2 and noise), sites_by_year.csv, sensitivity_cutoff_500m.csv, sensitivity_building_land_placement.csv, noise_validation_cutoffs.csv, sensitivity_kriging.csv, sensitivity_emissions.csv, comparison_external_models.csv.

## Other files
cv_lopo_2015_2023.csv (cross-validation by year and station type), coef_2015_2023.csv (model coefficients and kriging kernels), national_2015_2023.csv (national means and shares above WHO levels).
