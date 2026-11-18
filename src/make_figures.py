"""Figures 1-4 of the paper.

Usage: python src/make_figures.py WORK BOUNDARIES.gpkg OUTDIR
WORK is the work folder of the pipeline (see PIPELINE.md) and must contain:
  release/data/receptors_no2_noise_2015_2023.parquet  (residential points)
  release/data/stations_lopo_2015_2023.csv           (cross-validated predictions)
  comparison_external/compare_external_station_preds.csv
  rq34/measured_trends_2015_2023.csv
  results_unified/{key_numbers,table1_by_municipality_type,correlation_by_level}.csv
  sensitivity/noise_cutoff/cutoff_500m_comparison.csv
BOUNDARIES.gpkg: municipal polygons with a 5-digit code column 'muni' (2020 census municipalities).
Outputs: OUTDIR/fig{1..4}.{png,pdf}
"""
import os, sys, numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap, BoundaryNorm
import shapely.geometry as sg

C, BND, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
fam = 'Liberation Sans' if any('Liberation Sans' in f.name for f in fm.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': fam, 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
                     'axes.edgecolor': '#52514e', 'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
INK, INK2, GRID = '#1f1f1d', '#6b6a66', '#e5e4e0'
BLUE, LBLUE, ORANGE, LORANGE = '#2a78d6', '#9cc3ef', '#d9622b', '#f2b48f'
MM = 1 / 25.4


def panel(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight='bold', fontsize=8, va='bottom', ha='left')


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=300)
    fig.savefig(f'{OUT}/{name}.pdf')
    plt.close(fig)


def r2(o, p):
    return 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()


# ---------- Figure 1 ----------
a = pd.read_csv(f'{C}/comparison_external/compare_external_station_preds.csv', dtype={'code': str})
b = pd.read_csv(f'{C}/release/data/stations_lopo_2015_2023.csv', dtype={'code': str})
a = a[a.fy == 2019].dropna(subset=['Anenberg2022_2019', 'Cooper2022_2019_recal'])
m = a.merge(b[b.fy == 2019][['code', 'NO2_lopo']], on='code')
cols = [('ours_bg', 'Background\nonly'), ('ours', '+ Near-road\ncontribution'), ('NO2_lopo', '+ Satellite\n(main model)'),
        ('Anenberg2022_2019', 'Anenberg\net al.'), ('Cooper2022_2019_recal', 'Cooper et al.\n(calibrated)')]
R_all = [r2(m.NO2, m[c]) for c, _ in cols]
rd = m[m.kind == 'road']
R_road = [r2(rd.NO2, rd[c]) for c, _ in cols]
fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM), gridspec_kw={'width_ratios': [1.45, 1], 'wspace': .32})
ax = axs[0]
x = np.arange(len(cols)); w = .36
b1 = ax.bar(x - w / 2, R_all, w, color=BLUE)
b2 = ax.bar(x + w / 2, R_road, w, color=LBLUE)
for i, (u, v) in enumerate(zip(R_all, R_road)):
    ax.text(i - w / 2, max(u, 0) + .015, f'{u:.2f}'.replace('-', '−'), ha='center', fontsize=6)
    ax.text(i + w / 2, max(v, 0) + .015, f'{v:.2f}'.replace('-', '−'), ha='center', fontsize=6)
ax.axhline(0, color=INK2, lw=.6)
ax.axvline(2.5, color=GRID, lw=.8, ls='--')
ax.text(1, .98, 'This study', ha='center', fontsize=6.5, color=INK2, transform=ax.get_xaxis_transform())
ax.text(3.5, .98, 'Existing global models', ha='center', fontsize=6.5, color=INK2, transform=ax.get_xaxis_transform())
ax.set_xticks(x); ax.set_xticklabels([l for _, l in cols], fontsize=6.3)
ax.set_ylim(-0.35, 1); ax.set_ylabel('R² (leave-one-prefecture-out, FY2019)')
ax.legend([b1, b2], [f'All stations (n = {len(m):,})', f'Roadside stations (n = {len(rd):,})'], loc='lower left', frameon=False)
panel(ax, 'a', -0.13)
ax = axs[1]
s = b[b.fy == 2022]
for k, col, lab in [('gen', BLUE, 'Ambient'), ('road', ORANGE, 'Roadside')]:
    t = s[s.kind == k]
    ax.scatter(t.NO2_obs, t.NO2_lopo, s=3, color=col, alpha=.55, lw=0, label=f'{lab} (n = {len(t):,}; R² {r2(t.NO2_obs, t.NO2_lopo):.2f})')
t = s[~s.kind.isin(['gen', 'road'])]
ax.scatter(t.NO2_obs, t.NO2_lopo, s=9, color='0.25', lw=0, zorder=5, label=f'Remote background (n = {len(t)})')
lim = (0, 32)
ax.plot(lim, lim, color=INK2, lw=.6, ls='--'); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
ax.set_xlabel('Measured NO$_2$ (ppb)'); ax.set_ylabel('Cross-validated prediction (ppb)')
ax.text(.03, .97, f'All stations: R² = {r2(s.NO2_obs, s.NO2_lopo):.2f}, RMSE = {np.sqrt(((s.NO2_obs - s.NO2_lopo) ** 2).mean()):.2f} ppb',
        transform=ax.transAxes, va='top', fontsize=6.3)
ax.legend(loc='lower right', frameon=False, markerscale=2.5, handletextpad=.2, fontsize=6)
panel(ax, 'b', -0.2)
save(fig, 'fig1')
print('fig1', [round(v, 5) for v in R_all], [round(v, 3) for v in R_road])

# ---------- Figure 2 ----------
t = pd.read_csv(f'{C}/rq34/measured_trends_2015_2023.csv')
fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM), gridspec_kw={'wspace': .3})
fig.subplots_adjust(bottom=.18, top=.92)
for ax in axs:
    ax.axvspan(2019.5, 2021.5, color='#efeeea', lw=0)
    ax.axhline(0, color=INK2, lw=.6)
    ax.set_xticks(range(2015, 2024)); ax.set_xlim(2014.7, 2023.3); ax.set_xlabel('Fiscal year')
for ser, col, lab in [('NO2 自排局（%変化）', ORANGE, 'Roadside stations (n = 418)'), ('NO2 一般局（%変化）', BLUE, 'Ambient stations (n = 1,299)')]:
    d = t[t.series == ser]
    axs[0].fill_between(d.fy, d.lo, d.hi, color=col, alpha=.25, lw=0)
    axs[0].plot(d.fy, d.est, color=col, marker='o', ms=2.5, lw=1.2, label=lab)
axs[0].set_ylabel('Change in NO$_2$ from FY2015 (%)'); axs[0].set_ylim(-42, 4)
axs[0].legend(loc='lower left', frameon=False)
d = t[t.series == '夜間騒音（実測、dB差）']
axs[1].fill_between(d.fy, d.lo, d.hi, color='#5b4bb7', alpha=.22, lw=0)
axs[1].plot(d.fy, d.est, color='#5b4bb7', marker='o', ms=2.5, lw=1.2, label='Nighttime noise (n = 7,302 sites)')
axs[1].set_ylabel('Change in $L_\\mathrm{night}$ from FY2015 (dB)'); axs[1].set_ylim(-1.6, 0.5)
axs[1].legend(loc='lower left', frameon=False)
for ax in axs:
    ax.text(2020.5, .97, 'COVID-19', ha='center', va='top', fontsize=6.3, color=INK2, transform=ax.get_xaxis_transform())
panel(axs[0], 'a', -0.16); panel(axs[1], 'b', -0.16)
save(fig, 'fig2')

# ---------- shared point data ----------
R = pd.read_parquet(f'{C}/release/data/receptors_no2_noise_2015_2023.parquet',
                    columns=['KEY_CODE', 'lon', 'lat', 'pop_w', 'Lnight_2021', 'NO2_2022'])
R['ug'] = R.NO2_2022 * 1.88
R['dbl'] = (R.Lnight_2021 >= 55) & (R.ug >= 20)
R['muni'] = R.KEY_CODE.str[:5]
NAT = 100 * (R.pop_w * R.dbl).sum() / R.pop_w.sum()

# ---------- Figure 3 ----------
T1 = pd.read_csv(f'{C}/results_unified/table1_by_municipality_type.csv')
K = pd.read_csv(f'{C}/results_unified/key_numbers.csv', index_col=0).iloc[:, 0]
names = {'東京23区': "Tokyo's 23 wards", '政令指定都市': 'Designated cities', '人口30万以上の市': 'Cities ≥ 300,000',
         '10〜30万の市': 'Cities 100,000–300,000', '5〜10万の市': 'Cities 50,000–100,000', '5万未満の市': 'Cities < 50,000', '町村': 'Towns and villages'}
T1 = T1[T1.ctype != '全国']
dist = [('< 20 m', K['距離 <20 m の住民のうち二重暴露 %']), ('20–50 m', K['距離 20〜50 m の住民のうち二重暴露 %']),
        ('50–100 m', K['距離 50〜100 m の住民のうち二重暴露 %']), ('100–200 m', K['距離 100〜200 m の住民のうち二重暴露 %']),
        ('≥ 200 m', K['距離 ≥200 m の住民のうち二重暴露 %'])]
M = R.groupby('muni').apply(lambda g: pd.Series({'pop': g.pop_w.sum(), 'dbl': 100 * (g.pop_w * g.dbl).sum() / g.pop_w.sum()}))
G = gpd.read_file(BND).to_crs(6677)
G = G.merge(M, left_on='muni', right_index=True, how='left')
miss = G.dbl.isna().sum()
bins = [0, 1, 5, 10, 20, 30, 100]
cmap = ListedColormap(['#f1efe8', '#fbd9b8', '#f5ab79', '#e6743f', '#c2471d', '#7f2704'])
norm = BoundaryNorm(bins, cmap.N)

fig = plt.figure(figsize=(180 * MM, 120 * MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.7], height_ratios=[1, 1], wspace=.12, hspace=.45, left=.2, right=.99, top=.95, bottom=.08)
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(T1))[::-1]
ax.barh(y, T1['二重暴露%'], color=ORANGE, height=.65)
for yy, v in zip(y, T1['二重暴露%']):
    ax.text(v + .6, yy, f'{v:.1f}%' if v < 10 else f'{v:.0f}%', va='center', fontsize=6)
ax.axvline(NAT, color=INK2, lw=.7, ls='--')
ax.set_yticks(y); ax.set_yticklabels([names[c] for c in T1.ctype])
ax.set_xlabel('Residents with high co-exposure (%)'); ax.set_xlim(0, 38)
panel(ax, 'a', -0.62)
ax = fig.add_subplot(gs[1, 0])
y = np.arange(len(dist))[::-1]
ax.barh(y, [v for _, v in dist], color=ORANGE, height=.65)
for yy, (_, v) in zip(y, dist):
    ax.text(v + .6, yy, f'{v:.0f}%' if v >= 1 else '0%', va='center', fontsize=6)
ax.axvline(NAT, color=INK2, lw=.7, ls='--')
ax.set_yticks(y); ax.set_yticklabels([l for l, _ in dist])
ax.set_ylabel('Distance to nearest main road'); ax.set_xlabel('Residents with high co-exposure (%)'); ax.set_xlim(0, 48)
panel(ax, 'b', -0.62)


def draw(a, D):
    D.plot(ax=a, column='dbl', cmap=cmap, norm=norm, linewidth=.05, edgecolor='#ffffff', missing_kwds={'color': '#dddddd'})
    a.set_axis_off()


ax = fig.add_subplot(gs[:, 1])
isl = G.muni.str.startswith('47') | G.muni.isin(['13361', '13362', '13363', '13364', '13381', '13382', '13401', '13402', '13421'])
draw(ax, G[~isl])
xmin, ymin, xmax, ymax = G[~isl & ~G.muni.str.startswith('01')].total_bounds
hx = G[G.muni.str.startswith('01')].total_bounds
ax.set_xlim(xmin, max(xmax, hx[2])); ax.set_ylim(ymin, hx[3])
H = 18000  # half-width of each enlarged map (m): both insets are 36 km squares at the same scale
for (lab, ctr, pos) in [('Osaka', (135.50, 34.67), [0.0, 0.66, 0.28, 0.30]),
                        ('Tokyo', (139.75, 35.67), [0.30, 0.66, 0.28, 0.30])]:
    c = gpd.GeoSeries([sg.Point(*ctr)], crs=4326).to_crs(6677).iloc[0]
    bb = (c.x - H, c.y - H, c.x + H, c.y + H)
    ia = ax.inset_axes(pos); draw(ia, G.cx[bb[0]:bb[2], bb[1]:bb[3]])
    ia.set_xlim(bb[0], bb[2]); ia.set_ylim(bb[1], bb[3]); ia.set_axis_on(); ia.set_xticks([]); ia.set_yticks([]); ia.set_xlabel(''); ia.set_ylabel(''); ia.set_anchor('N')
    for s in ia.spines.values():
        s.set_visible(True); s.set_color(INK2); s.set_linewidth(.5)
    ia.set_title(lab, fontsize=6.5, loc='left', pad=2)
    ax.add_patch(plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec=INK, lw=.5))
h = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(cmap.N)]
ax.legend(h, ['< 1%', '1–5%', '5–10%', '10–20%', '20–30%', '≥ 30%'], title='High co-exposure\n(share of residents)',
          loc='lower right', frameon=False, fontsize=6, title_fontsize=6.5)
panel(ax, 'c', -0.02, 0.98)
save(fig, 'fig3')
print('fig3 national', round(NAT, 2), 'munis without data', miss)

# ---------- Figure 4 ----------
CL = pd.read_csv(f'{C}/results_unified/correlation_by_level.csv')
CO = pd.read_csv(f'{C}/sensitivity/noise_cutoff/cutoff_500m_comparison.csv', index_col=0)
fig = plt.figure(figsize=(180 * MM, 118 * MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], wspace=.35, hspace=.5)
ax = fig.add_subplot(gs[0, 0])
lv = CL.iloc[:5]
labs = ['Points', 'Small\nareas', 'Munici-\npalities', 'Medical\nareas', 'Prefec-\ntures']
ax.plot(range(5), lv.r, color=BLUE, marker='o', ms=3.5, lw=1.2)
for i, v in enumerate(lv.r):
    ax.text(i, v + .035, f'{v:.2f}', ha='center', fontsize=6)
wi = CL.iloc[5].r
ax.axhline(wi, color=INK2, lw=.7, ls='--'); ax.text(4.35, wi + .015, f'Within\nmunicipalities ({wi:.2f})', ha='right', va='bottom', fontsize=5.5, color=INK2)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=6); ax.set_ylim(0, 1); ax.set_xlim(-.4, 4.4)
ax.set_ylabel('Correlation of nighttime noise and NO$_2$')
panel(ax, 'a', -0.35, 1.06)
ax = fig.add_subplot(gs[0, 1:])
rows = [('NO$_2$', 91.93), ('Nighttime noise,\ncutoff 200 m (main)', float(CO.loc['騒音分散の市区町村間 %', '200 m（主解析）'])),
        ('Nighttime noise,\ncutoff 500 m', float(CO.loc['騒音分散の市区町村間 %', '500 m']))]
y = np.arange(len(rows))[::-1]
for yy, (lab, bt) in zip(y, rows):
    ax.barh(yy, bt, color=BLUE, height=.55); ax.barh(yy, 100 - bt, left=bt, color=LBLUE, height=.55)
    ax.text(bt / 2, yy, f'{bt:.0f}%', ha='center', va='center', color='white', fontsize=6.3) if bt > 8 else None
    ax.text(bt + (100 - bt) / 2, yy, f'{100 - bt:.0f}%', ha='center', va='center', color=INK, fontsize=6.3) if 100 - bt > 8 else None
ax.set_yticks(y); ax.set_yticklabels([l for l, _ in rows]); ax.set_xlim(0, 100)
ax.set_xlabel('Share of population-weighted variance (%)')
h = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=LBLUE)]
ax.legend(h, ['Between municipalities', 'Within municipalities'], loc='upper center', bbox_to_anchor=(.5, 1.25), ncol=2, frameon=False)
panel(ax, 'b', -0.26, 1.06)
box = (139.68, 35.64, 139.80, 35.72)
Z = R[(R.lon > box[0]) & (R.lon < box[2]) & (R.lat > box[1]) & (R.lat < box[3])]
for j, (col, lab, cm, vmin, vmax) in enumerate([('Lnight_2021', 'Nighttime noise, $L_\\mathrm{night}$ (dB)', 'magma_r', 40, 70),
                                                 ('ug', 'NO$_2$ (µg m$^{-3}$)', 'viridis_r', 15, 35)]):
    ax = fig.add_subplot(gs[1, j])
    sc = ax.scatter(Z.lon, Z.lat, c=Z[col], s=.4, lw=0, cmap=cm, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3]); ax.set_aspect(1 / np.cos(np.radians(35.68)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(INK2); s.set_linewidth(.5)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=.06, pad=.04); cb.set_label(lab, fontsize=6.3); cb.ax.tick_params(labelsize=6)
    if j == 0:
        panel(ax, 'c', -0.05, 1.03)
        km = 1 / (111.32 * np.cos(np.radians(35.68)))
        ax.plot([box[0] + .005, box[0] + .005 + 2 * km], [box[1] + .004] * 2, color=INK, lw=1.2)
        ax.text(box[0] + .005 + km, box[1] + .007, '2 km', ha='center', fontsize=6, bbox=dict(fc='white', ec='none', pad=.5))
ax = fig.add_subplot(gs[1, 2])
zz = Z.sample(min(len(Z), 20000), random_state=1)
ax.scatter(zz.ug, zz.Lnight_2021, s=.6, lw=0, alpha=.35, color=INK2, rasterized=True)
ax.set_xlabel('NO$_2$ (µg m$^{-3}$)'); ax.set_ylabel('$L_\\mathrm{night}$ (dB)'); ax.set_xlim(15, 35); ax.set_ylim(38, 72)
rr = np.corrcoef(Z.ug, Z.Lnight_2021)[0, 1]
ax.text(.03, .97, f'Points in the map area\n(n = {len(Z):,}; r = {rr:.2f})', transform=ax.transAxes, va='top', fontsize=6)
save(fig, 'fig4')
print('fig4 points', len(Z), round(rr, 3))
