"""Sensitivity of receptor placement: re-place each small area's receptors on building land (MLIT land-use mesh L03-b 2021, 100 m,
class 0700 'building land') instead of uniformly over the small area. Seeded per prefecture. Same number of points and weights per small area; small areas
without building cells keep the original points; points on census carriageways are redrawn.
Usage: popalloc_building.py W W2 LU_DIR pref [pref ...]   (writes W2/rec/r_{pref}.parquet)"""
import pandas as pd, numpy as np, geopandas as gpd, shapely, glob, os, sys, zipfile, io
from dbfread import DBF
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W,W2,LU=sys.argv[1:4]; PREFS=sys.argv[4:]; os.makedirs(f'{W2}/rec',exist_ok=True)
def lu_cells():
    cache=f'{W2}/_building_cells.parquet'
    if os.path.exists(cache): return pd.read_parquet(cache)
    rows=[]
    for z in sorted(glob.glob(f'{LU}/L03b_*.zip')):
        with zipfile.ZipFile(z) as f:
            dbf=[n for n in f.namelist() if n.endswith('.dbf')][0]; tmp=f'{LU}/_t.dbf'; open(tmp,'wb').write(f.read(dbf))
        T=pd.DataFrame(iter(DBF(tmp))); T=T[T.L03b_002=='0700']; rows.append(T.L03b_001.astype(str))
    c=pd.concat(rows).unique(); s=pd.Series(c)
    lat=s.str[0:2].astype(int)/1.5+s.str[4].astype(int)*5/60+s.str[6].astype(int)*30/3600+s.str[8].astype(int)*3/3600+1.5/3600
    lon=s.str[2:4].astype(int)+100+s.str[5].astype(int)*7.5/60+s.str[7].astype(int)*45/3600+s.str[9].astype(int)*4.5/3600+2.25/3600
    D=pd.DataFrame({'lon':lon.values,'lat':lat.values}); D.to_parquet(cache); return D
C=lu_cells(); print('building-land cells',len(C),flush=True)
K=nl.load_geom(f'{W}/geo/traffic_census_2021_converted.parquet'); S=pd.read_parquet(f'{W}/sections_nox.parquet').set_index('id')
K=K[K.census.astype(str).isin(S.index)]; K['halfw']=S.halfw.loc[K.census.astype(str)].values
for pc in PREFS:
    R=pd.read_parquet(f'{W}/rec/r_{pc}.parquet'); z=int(R.epsg.iloc[0])
    A=gpd.read_file(glob.glob(f'{W}/estat_all/shp/r2ka{pc}*.shp')[0]); A=A[A.HCODE==8101][['KEY_CODE','geometry']].dissolve(by='KEY_CODE').reset_index().to_crs(z)
    rng=np.random.default_rng([20261115,int(pc)])
    b=A.to_crs(6668).total_bounds; Cb=C[(C.lon>=b[0]-.01)&(C.lon<=b[2]+.01)&(C.lat>=b[1]-.01)&(C.lat<=b[3]+.01)]
    Cz=gpd.GeoDataFrame({'lat':Cb.lat.values},geometry=gpd.points_from_xy(Cb.lon,Cb.lat),crs=6668).to_crs(z); Cz=gpd.sjoin(Cz,A,predicate='within')[['KEY_CODE','geometry','lat']]
    dy=3/3600*111320; Cz['dx']=4.5/3600*111320*np.cos(np.radians(Cz.lat))
    cells={k:g for k,g in Cz.groupby('KEY_CODE')}; geo=A.set_index('KEY_CODE').geometry
    Kz=K.to_crs(z); Kz=Kz[Kz.intersects(shapely.box(R.x.min(),R.y.min(),R.x.max(),R.y.max()).buffer(100))]; tr=shapely.STRtree(Kz.geometry.values)
    def on_road(x,y):
        pts=shapely.points(x,y); pi,ki=tr.query(pts,predicate='dwithin',distance=30); on=np.zeros(len(pts),bool)
        d=shapely.distance(pts[pi],Kz.geometry.values[ki]); on[pi[d<Kz.halfw.values[ki]]]=True; return on
    X=R.x.values.copy(); Y=R.y.values.copy(); moved=np.zeros(len(R),bool)
    for key,idx in R.groupby('KEY_CODE').indices.items():
        g=cells.get(key)
        if g is None: continue
        poly=geo[key]; cx=g.geometry.x.values; cy=g.geometry.y.values; dx=g.dx.values; k=len(idx); px=np.empty(k); py=np.empty(k); todo=np.arange(k)
        for _ in range(30):
            if len(todo)==0: break
            j=rng.integers(0,len(g),len(todo)); x=cx[j]+rng.uniform(-.5,.5,len(todo))*dx[j]; y=cy[j]+rng.uniform(-.5,.5,len(todo))*dy
            ok=shapely.contains_xy(poly,x,y); px[todo[ok]]=x[ok]; py[todo[ok]]=y[ok]; todo=todo[~ok]
        if len(todo): j=rng.integers(0,len(g),len(todo)); px[todo]=cx[j]; py[todo]=cy[j]
        X[idx]=px; Y[idx]=py; moved[idx]=True
    for _ in range(20):
        on=on_road(X,Y)&moved
        if not on.any(): break
        for i in np.where(on)[0]:
            g=cells[R.KEY_CODE.iat[i]]; j=rng.integers(0,len(g)); X[i]=g.geometry.x.values[j]+rng.uniform(-.5,.5)*g.dx.values[j]; Y[i]=g.geometry.y.values[j]+rng.uniform(-.5,.5)*dy
    O=R.copy(); O['x']=X; O['y']=Y; O['moved']=moved; O.to_parquet(f'{W2}/rec/r_{pc}.parquet')
    print(pc,'points',len(R),'moved',round(moved.mean()*100,1),'% of points',round(R.w[moved].sum()/R.w.sum()*100,1),'% of pop; still on road',int(on.sum()),flush=True)
