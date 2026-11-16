"""Extended Data: share of residents doubly exposed under alternative thresholds (night noise x NO2, FY2022 NO2, 2021 noise).
Also the share of the doubly exposed living in Tokyo 23 wards + designated cities, and within 50/100 m of a census road.
Usage: ed_thresholds.py W RELEASE_PARQUET OUT"""
import pandas as pd, numpy as np, shapely, glob, os, sys
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W,RP,OUT=sys.argv[1:4]; os.makedirs(OUT,exist_ok=True)
files=sorted(glob.glob(f'{W}/rec/r_*.parquet'))
G=nl.load_sections(f'{W}/geo/traffic_census_2021_converted.parquet',f'{W}/nox_ef_nilim671.csv'); G=G[G.ok]
dist=[]
for f in files:
    R=pd.read_parquet(f); z=int(R.epsg.iloc[0]); Gz=G.to_crs(z); Gz=Gz[Gz.intersects(shapely.box(R.x.min(),R.y.min(),R.x.max(),R.y.max()).buffer(5000))]
    pts=shapely.points(R.x.values,R.y.values); tr=shapely.STRtree(Gz.geometry.values); idx=tr.nearest(pts)
    dist.append(shapely.distance(pts,Gz.geometry.values[idx]))
D=pd.read_parquet(RP,columns=['KEY_CODE','pop_w','Lnight_2021','NO2_2022'])
D['dist']=np.concatenate(dist); assert len(D)==sum(len(x) for x in dist)
D['muni']=D.KEY_CODE.astype(str).str[:5]; pref=D.muni.str[:2]; city=D.muni.str[2:].astype(int)
D['big']=((city>=101)&(city<=199))  # Tokyo 23 wards (13101-13123) and wards of designated cities
w=D.pop_w.values; L=D.Lnight_2021.values; G2=D.NO2_2022.values*1.88
rows=[]
for ln in (45,50,55,60):
    for ng in (10,20,30,40):
        b=(L>=ln)&(G2>=ng); wb=w[b].sum()
        rows.append({'夜間騒音 Lnight (dB)':ln,'NO2 (µg/m³)':ng,'二重暴露の住民 %':100*wb/w.sum(),'人数（万人）':wb/1e4,
          'うち東京23区・政令市 %':100*w[b&D.big.values].sum()/wb if wb>0 else np.nan,
          'うち道路から50 m以内 %':100*w[b&(D.dist.values<50)].sum()/wb if wb>0 else np.nan,
          'うち道路から100 m以内 %':100*w[b&(D.dist.values<100)].sum()/wb if wb>0 else np.nan})
T=pd.DataFrame(rows)
ref={'全住民 %':100,'東京23区・政令市':100*w[D.big.values].sum()/w.sum(),'50m':100*w[D.dist.values<50].sum()/w.sum(),'100m':100*w[D.dist.values<100].sum()/w.sum(),
     '騒音のみ55':100*w[L>=55].sum()/w.sum(),'NO2のみ20':100*w[G2>=20].sum()/w.sum()}
T.round(2).to_csv(f'{OUT}/ed_double_exposure_thresholds.csv',index=False,encoding='utf-8-sig')
pd.Series(ref).round(2).to_csv(f'{OUT}/ed_reference_shares.csv',encoding='utf-8-sig')
print(T.round(1).to_string()); print(pd.Series(ref).round(1))
