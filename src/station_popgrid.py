"""Population densities (1/3/10 km disks on the 250 m grid from receptors, as grid_features.py) at all stations in stations_all.csv."""
import pandas as pd, numpy as np, sys
from scipy.signal import oaconvolve
W=sys.argv[1]; RES=0.25; xy=lambda lon,lat: ((lon-137)*np.cos(np.radians(36))*111.32,(lat-36)*110.57)
X0=xy(122,0)[0]; Y0=xy(0,24)[1]; NX=int((xy(154,0)[0]-X0)/RES)+1; NY=int((xy(0,46)[1]-Y0)/RES)+1
cell=lambda gx,gy: (np.clip(((gy-Y0)/RES).astype(int),0,NY-1),np.clip(((gx-X0)/RES).astype(int),0,NX-1))
def disk(r):
    k=int(np.ceil(r/RES)); y,x=np.mgrid[-k:k+1,-k:k+1]; m=((x*RES)**2+(y*RES)**2<=r*r).astype(np.float32); return m/(m.sum()*RES*RES)
R=pd.read_parquet(f'{W}/rec_ll.parquet'); Pp=np.zeros((NY,NX),np.float32); np.add.at(Pp,cell(*xy(R.lon.values,R.lat.values)),R.w.values.astype(np.float32))
St=pd.read_csv(f'{W}/stations_all.csv',dtype={'code':str}).dropna(subset=['lat']).drop_duplicates('code'); iy,ix=cell(*xy(St.lon.values,St.lat.values))
out={'code':St.code.values}
for r in (1,3,10): out[f'gpop{r}']=np.log1p(oaconvolve(Pp,disk(r),mode='same').clip(min=0)[iy,ix])
pd.DataFrame(out).to_csv(f'{W}/station_popgrid_all.csv',index=False)
