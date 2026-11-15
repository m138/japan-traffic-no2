import pandas as pd, numpy as np, geopandas as gpd, glob, sys
from scipy.spatial import cKDTree
W=sys.argv[1]
R=[]
for f in sorted(glob.glob(f'{W}/rec/r_*.parquet')):
    d=pd.read_parquet(f); ll=gpd.GeoSeries(gpd.points_from_xy(d.x,d.y),crs=int(d.epsg.iloc[0])).to_crs(4326)
    R.append(pd.DataFrame({'lon':ll.x.values,'lat':ll.y.values,'w':d.w.values}))
R=pd.concat(R); R.to_parquet(f'{W}/rec_ll.parquet')
xy=lambda lon,lat: np.c_[(lon-137)*np.cos(np.radians(36))*111.32,(lat-36)*110.57]
T=cKDTree(xy(R.lon.values,R.lat.values)); w=R.w.values
P=pd.read_csv(f'{W}/stations.csv',dtype={'code':str}).drop_duplicates('code')
q=xy(P.lon.values,P.lat.values); out={'code':P.code.values}
for r in (1,3,10):
    idx=T.query_ball_point(q,r); out[f'lpop{r}']=np.log1p([w[i].sum()/(np.pi*r*r) for i in idx])
pd.DataFrame(out).to_csv(f'{W}/station_pop.csv',index=False); print(pd.DataFrame(out).describe().round(2))
