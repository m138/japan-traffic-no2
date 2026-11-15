"""Emission and population densities on a 250 m national grid (equirectangular km around 137E/36N), disk-averaged at 1/3/10 km,
sampled at stations and at all receptors (one consistent method for both)."""
import pandas as pd, numpy as np, geopandas as gpd, shapely, glob, sys, os
from scipy.signal import oaconvolve
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W=sys.argv[1]; RES=0.25
xy=lambda lon,lat: ((lon-137)*np.cos(np.radians(36))*111.32,(lat-36)*110.57)
X0,X1=xy(122,0)[0],xy(154,0)[0]; Y0,Y1=xy(0,24)[1],xy(0,46)[1]
NX=int((X1-X0)/RES)+1; NY=int((Y1-Y0)/RES)+1; print('grid',NX,NY,flush=True)
cell=lambda gx,gy: (np.clip(((gy-Y0)/RES).astype(int),0,NY-1),np.clip(((gx-X0)/RES).astype(int),0,NX-1))
# emissions: densify census lines to 25 m points carrying q*len
G=nl.load_geom(f'{W}/geo/traffic_census_2021_converted.parquet'); G['id']=G.census.astype(str)
S=pd.read_parquet(f'{W}/sections_nox.parquet'); S=S[S.ok]; K=G.merge(S[['id','nox_gkmd']],on='id')
Em=np.zeros((NY,NX),np.float32)
for z,k in K.groupby(K.geometry.centroid.x.map(nl.utm_epsg)):
    kz=gpd.GeoSeries(k.geometry.values,crs=G.crs).to_crs(z)
    seg=nl.segments(kz.values,k.nox_gkmd.values)   # id column carries q
    L=np.hypot(seg.bx-seg.ax,seg.by-seg.ay).values; n=np.maximum(1,np.ceil(L/25).astype(int))
    rep=np.repeat(np.arange(len(seg)),n); t=(np.concatenate([np.arange(m) for m in n])+0.5)/np.repeat(n,n)
    px=seg.ax.values[rep]+t*(seg.bx.values[rep]-seg.ax.values[rep]); py=seg.ay.values[rep]+t*(seg.by.values[rep]-seg.ay.values[rep])
    ll=gpd.GeoSeries(gpd.points_from_xy(px,py),crs=z).to_crs(4326); gx,gy=xy(ll.x.values,ll.y.values)
    w=(seg.id.values[rep].astype(float)*L[rep]/n[rep]/1000)   # g/day
    iy,ix=cell(gx,gy); np.add.at(Em,(iy,ix),w.astype(np.float32))
R=pd.read_parquet(f'{W}/rec_ll.parquet'); rgx,rgy=xy(R.lon.values,R.lat.values)
Pp=np.zeros((NY,NX),np.float32); iy,ix=cell(rgx,rgy); np.add.at(Pp,(iy,ix),R.w.values.astype(np.float32))
print('emission total t/yr',round(Em.sum()*365/1e6),' pop',round(Pp.sum()/1e6,2),'M',flush=True)
def disk(rkm):
    r=int(np.ceil(rkm/RES)); y,x=np.mgrid[-r:r+1,-r:r+1]; k=((x*RES)**2+(y*RES)**2<=rkm**2).astype(np.float32); return k/(k.sum()*RES*RES)   # -> per km2
St=pd.read_csv(f'{W}/stations.csv',dtype={'code':str}).drop_duplicates('code'); sgx,sgy=xy(St.lon.values,St.lat.values)
siy,six=cell(sgx,sgy)
outS={'code':St.code.values}; outR={}
for name,grid,radii in (('E',Em,(3,10)),('pop',Pp,(1,3,10))):
    for r in radii:
        c=oaconvolve(grid,disk(r),mode='same').clip(min=0)
        outS[f'g{name}{r}']=np.log1p(c[siy,six]); outR[f'g{name}{r}']=np.log1p(c[iy,ix]).astype(np.float32); print(name,r,flush=True); del c
pd.DataFrame(outS).to_csv(f'{W}/station_grid.csv',index=False)
pd.DataFrame(outR).to_parquet(f'{W}/rec_grid.parquet')
print(pd.DataFrame(outS).describe().round(2).to_string())
