"""Make the data files of the grid map (docs/grid.html): NO2 of each fiscal year, the change in NO2 and high co-exposure
as population-weighted values of the residential points in grid cells of about 0.4 km, with two coarser levels for small scales.

Usage: python src/make_grid_data.py receptors_no2_noise_2015_2023.parquet OUT_DIR
Writes OUT_DIR/grid_no2_YYYY.js (fiscal years 2015-2023), grid_change.js and grid_coexposure.js. Each file holds, for every
level, the non-empty blocks of 256 x 256 cells as 8-bit PNG images in base64. Codes: 0 no residents, 1 few residents,
3-255 the value on a linear scale:
  no2_YYYY     annual mean NO2 in ug/m3 (the parquet file gives ppb; 1 ppb = 1.88 ug/m3), 0 to 50.4 in steps of 0.2
  change       NO2 of fiscal year 2023 minus NO2 of fiscal year 2015 in ug/m3, -28 to +12.32 in steps of 0.16
  coexposure   percent of residents with Lnight >= 55 dB and NO2 >= 20 ug/m3 (fiscal year 2022), 0 to 100.8 in steps of 0.4
The change is given as a difference, not in percent: where the 2015 value is at the lower bound of the model
(0.5 ppb = 0.94 ug/m3), a ratio has no meaning.
Needs Pillow in addition to requirements.txt (it is installed with matplotlib).
"""
import sys, io, json, base64
import numpy as np, pyarrow.parquet as pq
from PIL import Image

rec, out = sys.argv[1:3]
YRS = range(2015, 2024)
BOX = (122.9, 146.0, 24.0, 45.6)                     # lon0, lon1, lat0, lat1
LEVELS = [(0.004, 3), (0.016, 10), (0.064, 20)]      # cell size in degrees, residents below which a cell is shown in gray
BLOCK = 256
UG = 1.88                                            # ug/m3 per ppb at 25 degrees C
SCALE = {'no2': (0.0, 50.4), 'change': (-28.0, 12.32), 'coexposure': (0.0, 100.8)}

R = pq.read_table(rec, columns=['lon', 'lat', 'pop_w', 'Lnight_2021'] + [f'NO2_{y}' for y in YRS]).to_pandas()
w = R.pop_w.values.astype('float64'); x0, x1, y0, y1 = BOX
dbl = (R.Lnight_2021.values >= 55) & (R.NO2_2022.values * UG >= 20)


def code(v, lo, hi):
    return (np.clip(np.round((v - lo) / (hi - lo) * 252), 0, 252) + 3).astype(np.uint8)


keys = [f'no2_{y}' for y in YRS] + ['change', 'coexposure']
data = {k: {'box': BOX, 'block': BLOCK, 'scale': SCALE[k.split('_')[0]], 'levels': []} for k in keys}
for res, minpop in LEVELS:
    nx = int(np.ceil((x1 - x0) / res)); ny = int(np.ceil((y1 - y0) / res))
    ix = np.floor((R.lon.values - x0) / res).astype(np.int64); iy = np.floor((R.lat.values - y0) / res).astype(np.int64)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cell, inv = np.unique(iy[m] * nx + ix[m], return_inverse=True); n = len(cell)
    W = np.bincount(inv, w[m], n); ok = W >= minpop; Ws = np.maximum(W, 1e-9)
    mean = {y: np.bincount(inv, (w * R[f'NO2_{y}'].values)[m], n) / Ws * UG for y in YRS}
    val = {f'no2_{y}': np.where(ok, code(mean[y], *SCALE['no2']), 1).astype(np.uint8) for y in YRS}
    val['change'] = np.where(ok, code(mean[2023] - mean[2015], *SCALE['change']), 1).astype(np.uint8)
    val['coexposure'] = np.where(ok, code(np.bincount(inv, (w * dbl)[m], n) / Ws * 100, *SCALE['coexposure']), 1).astype(np.uint8)
    cx, cy = (cell % nx) // BLOCK, (cell // nx) // BLOCK; ncx = int(np.ceil(nx / BLOCK)); key = cy * ncx + cx
    order = np.argsort(key, kind='stable'); ks, start = np.unique(key[order], return_index=True); end = np.append(start[1:], len(order))
    for k in keys:
        blocks = {}
        for kk, a, b in zip(ks, start, end):
            j = order[a:b]; img = np.zeros((BLOCK, BLOCK), np.uint8)
            img[(cell[j] // nx) % BLOCK, (cell[j] % nx) % BLOCK] = val[k][j]            # row 0 is the southern edge of the block
            buf = io.BytesIO(); Image.fromarray(img, mode='L').save(buf, 'PNG', optimize=True)
            blocks[int(kk)] = base64.b64encode(buf.getvalue()).decode()
        data[k]['levels'].append({'res': res, 'nx': nx, 'ny': ny, 'ncx': ncx, 'minpop': minpop, 'blocks': blocks})
    print(f'res {res}: {nx} x {ny}, cells with residents {n:,}, shown in color {int(ok.sum()):,}, blocks {len(ks)}')
    if res == LEVELS[0][0]:
        c = (mean[2023] - mean[2015])[ok]
        print('  cells: NO2 2015 max', round(float(mean[2015][ok].max()), 1), '| change in ug/m3, min, median, max:', round(float(c.min()), 2), round(float(np.median(c)), 2), round(float(c.max()), 2),
              '| share of cells with co-exposure above 0:', round(float((val['coexposure'][ok] > 3).mean() * 100), 1), '%')
tot = 0
for k in keys:
    s = 'window.GRID_DATA=window.GRID_DATA||{};window.GRID_DATA.%s=%s;\n' % (k, json.dumps(data[k], separators=(',', ':')))
    open(f'{out}/grid_{k}.js', 'w').write(s); tot += len(s); print(f'grid_{k}.js {len(s) / 1e6:.2f} MB')
print(f'total {tot / 1e6:.1f} MB')
