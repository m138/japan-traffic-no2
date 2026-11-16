"""Make docs/muni.geojson for the web map: municipal NO2 (FY2015-2023), nighttime noise and high co-exposure.

Usage:
  python src/make_webmap_data.py receptors_no2_noise_2015_2023.parquet BOUNDARIES.geojson docs/muni.geojson
BOUNDARIES.geojson: simplified municipal polygons with properties c (5-digit code), n (name) and pop,
for example docs/muni.geojson of https://github.com/m138/japan-night-road-noise (same 2020 census municipalities).
All values are population-weighted over the residential points of each municipality.
"""
import sys, json
import numpy as np, pandas as pd

rec, bnd, out = sys.argv[1:4]
YRS = range(2015, 2024)
cols = ['KEY_CODE', 'pop_w', 'Lnight_2021'] + [f'NO2_{y}' for y in YRS] + ['road_2022']
R = pd.read_parquet(rec, columns=cols)
R['muni'] = R.KEY_CODE.str[:5]
w = R.pop_w
ug = 1.88
D = pd.DataFrame({'muni': R.muni, 'w': w})
for y in YRS:
    D[f'n{y}'] = w * R[f'NO2_{y}'] * ug
D['road'] = w * R.road_2022 * ug
D['p20'] = w * (R.NO2_2022 * ug >= 20)
D['p55'] = w * (R.Lnight_2021 >= 55)
D['dbl'] = w * ((R.NO2_2022 * ug >= 20) & (R.Lnight_2021 >= 55))
S = D.groupby('muni').sum()
for c in S.columns:
    if c != 'w':
        S[c] = S[c] / S['w']
NAT = D.drop(columns='muni').sum()
NAT = {c: float(NAT[c] / NAT['w']) for c in NAT.index if c != 'w'}

g = json.load(open(bnd))
keep = []
for f in g['features']:
    p = f['properties']; c = p['c']
    if c not in S.index:
        continue
    s = S.loc[c]
    f['properties'] = {'c': c, 'n': p['n'], 'pop': p['pop'],
                       'no2': [round(float(s[f'n{y}']), 2) for y in YRS],
                       'chg': round(float(100 * (s['n2023'] / s['n2015'] - 1)), 1),
                       'road': round(float(s['road']), 2),
                       'p20': round(float(100 * s['p20']), 2), 'p55': round(float(100 * s['p55']), 2),
                       'dbl': round(float(100 * s['dbl']), 2)}
    keep.append(f)
g['features'] = keep
g['national'] = {'no2': [round(NAT[f'n{y}'], 2) for y in YRS], 'chg': round(100 * (NAT['n2023'] / NAT['n2015'] - 1), 1),
                 'road': round(NAT['road'], 2), 'p20': round(100 * NAT['p20'], 2), 'p55': round(100 * NAT['p55'], 2),
                 'dbl': round(100 * NAT['dbl'], 2)}
open(out, 'w').write(json.dumps(g, ensure_ascii=False, separators=(',', ':')))
print(len(keep), 'municipalities;', g['national'])
