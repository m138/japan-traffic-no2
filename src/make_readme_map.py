"""Map for the README: share of residents with high co-exposure (Lnight >= 55 dB and NO2 >= 20 ug/m3, FY2022) by municipality.

Usage: python src/make_readme_map.py RELEASE_PARQUET BOUNDARIES.gpkg OUT.png
RELEASE_PARQUET: receptors_no2_noise_2015_2023.parquet (Zenodo).
BOUNDARIES.gpkg: municipal polygons with a 5-digit code column 'muni' (2020 census municipalities).
"""
import sys, numpy as np, pandas as pd, geopandas as gpd, shapely
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RP, BND, OUT = sys.argv[1:4]
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic'})
R = pd.read_parquet(RP, columns=['KEY_CODE', 'pop_w', 'Lnight_2021', 'NO2_2022'])
R['dbl'] = (R.Lnight_2021 >= 55) & (R.NO2_2022 * 1.88 >= 20)
R['muni'] = R.KEY_CODE.str[:5]
R['wd'] = R.pop_w * R.dbl
M = R.groupby('muni')[['pop_w', 'wd']].sum()
M['share'] = 100 * M.wd / M.pop_w
G = gpd.read_file(BND, columns=['muni', 'geometry']).to_crs(6677).merge(M[['share']], left_on='muni', right_index=True, how='left')

bins = [0, 1, 5, 10, 20, 30, 100]
cols = ['#f1efe8', '#fbd9b8', '#f5ab79', '#e6743f', '#c2471d', '#7f2704']
G['cls'] = pd.cut(G.share.fillna(0), bins, include_lowest=True, right=False, labels=False)
fig, ax = plt.subplots(figsize=(6.5, 7))
isl = G.muni.str.startswith('47') | G.muni.isin([f'13{c}' for c in ['361', '362', '363', '364', '381', '382', '401', '402', '421']])


def draw(a, D):
    for i, c in enumerate(cols):
        D[D.cls == i].plot(ax=a, color=c, lw=0)
    a.set_axis_off(); a.set_aspect('equal')


draw(ax, G[~isl])
xmin, ymin, xmax, ymax = G[~isl].total_bounds
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ia = ax.inset_axes([0.04, 0.66, 0.2, 0.26]); draw(ia, G[G.muni.str.startswith('47')])
bb = gpd.GeoSeries([shapely.box(127.6, 26.05, 128.35, 26.9)], crs=4326).to_crs(6677).total_bounds
ia.set_xlim(bb[0], bb[2]); ia.set_ylim(bb[1], bb[3]); ia.set_axis_on(); ia.set_xticks([]); ia.set_yticks([]); ia.set_xlabel(''); ia.set_ylabel('')
for sp in ia.spines.values():
    sp.set_color('#c3c2b7'); sp.set_visible(True)
ia.set_title('Okinawa Island', fontsize=7, loc='left')
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
ax.legend(h, ['<1%', '1–5%', '5–10%', '10–20%', '20–30%', '≥30%'],
          title='Share of residents with high co-exposure\n($L_\\mathrm{night}$ ≥ 55 dB and NO$_2$ ≥ 20 µg m$^{-3}$,\nFY2022, municipality)',
          loc='lower right', frameon=False, fontsize=7, title_fontsize=8)
fig.tight_layout(); fig.savefig(OUT, dpi=300); plt.close()
print('national share %.2f%%' % (100 * R.wd.sum() / R.pop_w.sum()))
