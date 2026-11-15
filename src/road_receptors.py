"""Road NOx dispersion kernel (cut 1000 m, h=20 m) at all receptors, per prefecture. Usage: road_receptors.py W [pref ...]"""
import pandas as pd, numpy as np, geopandas as gpd, shapely, glob, sys, os, time
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W=sys.argv[1]; PREFS=sys.argv[2:] or [f'{i:02d}' for i in range(1,48)]
G=nl.load_geom(f'{W}/geo/traffic_census_2021_converted.parquet'); G['id']=G.census.astype(str)
S=pd.read_parquet(f'{W}/sections_nox.parquet'); S=S[S.ok]
K=gpd.GeoDataFrame(S.merge(G[['id','geometry']],on='id'),geometry='geometry',crs=G.crs)
os.makedirs(f'{W}/road',exist_ok=True)
for pc in PREFS:
    o=f'{W}/road/c_{pc}.parquet'
    if os.path.exists(o): continue
    t0=time.time(); R=pd.read_parquet(f'{W}/rec/r_{pc}.parquet'); z=int(R.epsg.iloc[0])
    Kz=K.to_crs(z); Kz=Kz[Kz.intersects(shapely.box(R.x.min(),R.y.min(),R.x.max(),R.y.max()).buffer(1100))]
    f=nl.road_field(R.x.values,R.y.values,Kz,1000,[20],chunk=3000)
    pd.DataFrame({'C1000_h20':f[20].astype(np.float32)}).to_parquet(o)
    print(pc,len(R),len(Kz),round(time.time()-t0),'s',flush=True)
