"""Road kernel (C1000_h20) and emission densities (3 and 10 km disk means of a 250 m grid, raw g/day/km2) for one census network and
several emission-factor years, at stations and at all receptors. Usage: features_multiyear.py W NET(2015|2021) EFYEAR [EFYEAR ...] [--stations-only]
Writes W/my/{NET}/station_feat.csv, rec_E.parquet, road/c_{pref}.parquet (columns C{year}, E3_{year}, E10_{year})."""
import pandas as pd, numpy as np, geopandas as gpd, shapely, sys, os, glob, time
from scipy.signal import oaconvolve
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
args=[a for a in sys.argv[1:] if not a.startswith('--')]; W,NET=args[0],args[1]; YRS=[int(y) for y in args[2:]]; ST_ONLY='--stations-only' in sys.argv
O=f'{W}/my/{NET}'; os.makedirs(f'{O}/road',exist_ok=True)
K=nl.sections_multi(f'{W}/geo/traffic_census_{NET}_converted.parquet',f'{W}/nox_ef_nilim671.csv',YRS); K['id']=K.id.astype(str)
qc=[f'q{y}' for y in YRS]; print(NET,len(K),{c:round(K[c].sum()) for c in qc},flush=True)
# emission grid (as grid_features.py)
RES=0.25; xy=lambda lon,lat: ((lon-137)*np.cos(np.radians(36))*111.32,(lat-36)*110.57)
X0=xy(122,0)[0]; Y0=xy(0,24)[1]; NX=int((xy(154,0)[0]-X0)/RES)+1; NY=int((xy(0,46)[1]-Y0)/RES)+1
cell=lambda gx,gy: (np.clip(((gy-Y0)/RES).astype(int),0,NY-1),np.clip(((gx-X0)/RES).astype(int),0,NX-1))
def disk(r):
    k=int(np.ceil(r/RES)); y,x=np.mgrid[-k:k+1,-k:k+1]; m=((x*RES)**2+(y*RES)**2<=r*r).astype(np.float32); return m/(m.sum()*RES*RES)
Em={c:np.zeros((NY,NX),np.float32) for c in qc}
for z,k in K.groupby(K.geometry.centroid.x.map(nl.utm_epsg)):
    kz=gpd.GeoSeries(k.geometry.values,crs=K.crs).to_crs(z); seg=nl.segments(kz.values,np.arange(len(k)))
    L=np.hypot(seg.bx-seg.ax,seg.by-seg.ay).values; n=np.maximum(1,np.ceil(L/25).astype(int))
    rep=np.repeat(np.arange(len(seg)),n); t=(np.concatenate([np.arange(m) for m in n])+0.5)/np.repeat(n,n)
    px=seg.ax.values[rep]+t*(seg.bx.values[rep]-seg.ax.values[rep]); py=seg.ay.values[rep]+t*(seg.by.values[rep]-seg.ay.values[rep])
    ll=gpd.GeoSeries(gpd.points_from_xy(px,py),crs=z).to_crs(4326); iy,ix=cell(*xy(ll.x.values,ll.y.values)); sid=seg.id.values[rep].astype(int)
    for c in qc: np.add.at(Em[c],(iy,ix),(k[c].values[sid]*L[rep]/n[rep]/1000).astype(np.float32))
print('emission t/yr',{c:round(float(Em[c].sum())*365/1e6) for c in qc},flush=True)
St=pd.read_csv(f'{W}/stations_all.csv',dtype={'code':str}).drop_duplicates('code').dropna(subset=['lat','lon'])
siy,six=cell(*xy(St.lon.values,St.lat.values)); outS={'code':St.code.values}
if not ST_ONLY: R=pd.read_parquet(f'{W}/rec_ll.parquet'); riy,rix=cell(*xy(R.lon.values,R.lat.values)); outR={}
for c in qc:
    for r in (3,10):
        g=oaconvolve(Em[c],disk(r),mode='same').clip(min=0); outS[f'E{r}_{c[1:]}']=g[siy,six]
        if not ST_ONLY: outR[f'E{r}_{c[1:]}']=g[riy,rix].astype(np.float32)
        del g
if not ST_ONLY: pd.DataFrame(outR).to_parquet(f'{O}/rec_E.parquet'); del outR
# station road kernel
St=St.reset_index(drop=True); Cst=np.full((len(St),len(qc)),np.nan)
for z,u in St.groupby(St.lon.map(nl.utm_epsg)):
    pts=gpd.GeoSeries(gpd.points_from_xy(u.lon,u.lat),crs=4326).to_crs(z); px,py=pts.x.values,pts.y.values
    Kz=K.to_crs(z); Kz=Kz[Kz.intersects(shapely.box(px.min(),py.min(),px.max(),py.max()).buffer(2000))]
    Cst[u.index.values]=nl.road_field_multi(px,py,Kz,qc)
for i,c in enumerate(qc): outS[f'C{c[1:]}']=Cst[:,i]
pd.DataFrame(outS).to_csv(f'{O}/station_feat.csv',index=False); print('stations',len(St),flush=True)
