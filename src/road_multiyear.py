"""Receptor road kernels for one network and several EF years. Usage: road_multiyear.py W NET 'Y1,Y2' pref [pref ...]"""
import pandas as pd, numpy as np, shapely, sys, os, time
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W,NET,YS=sys.argv[1:4]; YRS=[int(y) for y in YS.split(',')]; PREFS=sys.argv[4:]
K=nl.sections_multi(f'{W}/geo/traffic_census_{NET}_converted.parquet',f'{W}/nox_ef_nilim671.csv',YRS); K['id']=K.id.astype(str); qc=[f'q{y}' for y in YRS]
for pc in PREFS:
    o=f'{W}/my/{NET}/road/c_{pc}.parquet'
    if os.path.exists(o): continue
    t0=time.time(); R=pd.read_parquet(f'{W}/rec/r_{pc}.parquet'); z=int(R.epsg.iloc[0])
    Kz=K.to_crs(z); Kz=Kz[Kz.intersects(shapely.box(R.x.min(),R.y.min(),R.x.max(),R.y.max()).buffer(1100))]
    f=nl.road_field_multi(R.x.values,R.y.values,Kz,qc)
    pd.DataFrame({f'C{c[1:]}':f[:,i].astype(np.float32) for i,c in enumerate(qc)}).to_parquet(o)
    print(NET,pc,len(R),round(time.time()-t0),'s',flush=True)
