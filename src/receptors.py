"""Receptors as in the noise model v4: ~1 per 10 residents in 2020 census small areas (seed 20261106); points on census carriageways removed and their population reassigned."""
import pandas as pd, numpy as np, geopandas as gpd, shapely, sys, glob, os, time
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W=sys.argv[1]; PER=10; rng=np.random.default_rng(20261106)
K=nl.load_geom(f'{W}/geo/traffic_census_2021_converted.parquet')
S=pd.read_parquet(f'{W}/sections_nox.parquet').set_index('id')
K=K[K.census.astype(str).isin(S.index)]; K['halfw']=S.halfw.loc[K.census.astype(str)].values
os.makedirs(f'{W}/rec',exist_ok=True)
for f in sorted(glob.glob(f'{W}/estat_all/shp/r2ka*.shp')):
    pc=os.path.basename(f)[4:6]
    if os.path.exists(f'{W}/rec/r_{pc}.parquet'): continue
    t0=time.time()
    A=gpd.read_file(f); A=A[A.HCODE==8101][['KEY_CODE','PREF','CITY','JINKO','geometry']]
    A=A.dissolve(by='KEY_CODE',aggfunc={'PREF':'first','CITY':'first','JINKO':'sum'}).reset_index(); A=A[A.JINKO>0]
    z=nl.utm_epsg(A.to_crs(4326).union_all().centroid.x); A=A.to_crs(z)
    n=np.maximum(1,np.round(A.JINKO.values/PER).astype(int)); px=[];py=[];kc=[]
    for (key,geom),k in zip(A[['KEY_CODE','geometry']].itertuples(index=False),n):
        x0,y0,x1,y1=geom.bounds; got=np.empty((0,2)); tries=0
        while len(got)<k and tries<50:
            m=max(4*(k-len(got)),16); c=np.c_[rng.uniform(x0,x1,m),rng.uniform(y0,y1,m)]
            ins=shapely.contains_xy(geom,c[:,0],c[:,1]); got=np.vstack([got,c[ins]]); tries+=1
        if len(got)==0: rp=geom.representative_point(); got=np.array([[rp.x,rp.y]])
        got=got[:k]; px.append(got[:,0]); py.append(got[:,1]); kc+=[key]*len(got)
    px=np.concatenate(px); py=np.concatenate(py); kc=np.array(kc)
    # remove receptors on census carriageways (distance to centreline < half carriageway width)
    Kz=K.to_crs(z); Kz=Kz[Kz.intersects(shapely.box(*A.total_bounds).buffer(100))]
    pts=shapely.points(px,py); tr=shapely.STRtree(Kz.geometry.values)
    pi,ki=tr.query(pts,predicate='dwithin',distance=30)
    d=shapely.distance(pts[pi],Kz.geometry.values[ki]); on=np.zeros(len(pts),bool); on[pi[d<Kz.halfw.values[ki]]]=True
    D=pd.DataFrame({'KEY_CODE':kc,'x':px,'y':py,'on':on})
    keep=~D.on | D.groupby('KEY_CODE').on.transform('all')
    D=D[keep].copy(); pop=A.set_index('KEY_CODE').JINKO; D['w']=pop.loc[D.KEY_CODE].values/D.groupby('KEY_CODE').x.transform('size').values
    D['epsg']=z; D['PREF']=pc
    D.drop(columns='on').to_parquet(f'{W}/rec/r_{pc}.parquet')
    A[['KEY_CODE','PREF','CITY','JINKO']].to_parquet(f'{W}/rec/sa_{pc}.parquet')
    print(pc,len(A),'receptors',len(D),'removed',int((~keep).sum()),round(time.time()-t0),'s',flush=True)
