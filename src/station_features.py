import pandas as pd, numpy as np, geopandas as gpd, shapely, sys, os
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W=sys.argv[1]
P=pd.read_csv(f'{W}/stations.csv',dtype={'code':str,'muni':str})
U=P.drop_duplicates('code')[['code','lat','lon']].copy()
G=nl.load_geom(f'{W}/geo/traffic_census_2021_converted.parquet'); G['id']=G.census.astype(str)
S=pd.read_parquet(f'{W}/sections_nox.parquet'); S=S[S.ok]
K=gpd.GeoDataFrame(S.merge(G[['id','geometry']],on='id'),geometry='geometry',crs=G.crs)
HS=[2,5,10,20,50]; out=[]
for z,u in U.groupby(U.lon.map(nl.utm_epsg)):
    pts=gpd.GeoSeries(gpd.points_from_xy(u.lon,u.lat),crs=4326).to_crs(z); px,py=pts.x.values,pts.y.values
    Kz=K.to_crs(z); Kz=Kz[Kz.intersects(shapely.box(px.min(),py.min(),px.max(),py.max()).buffer(12000))]
    r={'code':u.code.values}
    for cut in (300,1000,3000):
        f=nl.road_field(px,py,Kz,cut,HS)
        for h in HS: r[f'C{cut}_h{h}']=f[h]
    # emission within radius (segment midpoints)
    SG=nl.segments(Kz.geometry.values,Kz.id.values); L=np.hypot(SG.bx-SG.ax,SG.by-SG.ay).values
    q=Kz.set_index('id').nox_gkmd.loc[SG.id.values].values*L/1000   # g/day per segment
    mx=(SG.ax.values+SG.bx.values)/2; my=(SG.ay.values+SG.by.values)/2; tr=shapely.STRtree(shapely.points(mx,my))
    for R in (500,1000,3000,10000):
        pi,si=tr.query(shapely.points(px,py),predicate='dwithin',distance=R)
        e=np.zeros(len(px)); np.add.at(e,pi,q[si]); r[f'E{R}']=e/(np.pi*(R/1000)**2)   # g/day/km2
    # nearest census road
    trk=shapely.STRtree(Kz.geometry.values); idx=trk.nearest(shapely.points(px,py))
    r['dnear']=shapely.distance(shapely.points(px,py),Kz.geometry.values[idx]); r['qnear']=Kz.nox_gkmd.values[idx]; r['grp_near']=Kz.grp.values[idx]
    r['x']=px; r['y']=py; r['epsg']=z
    out.append(pd.DataFrame(r)); print(z,len(u),flush=True)
F=pd.concat(out); F.to_csv(f'{W}/station_features.csv',index=False)
print(F.describe().T[['mean','50%','max']].round(3).to_string())
