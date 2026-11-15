"""Road Traffic Census sections with NOx emissions for the release: one row per section and census (2015, 2021).
NOx in g/km/day and t/year with emission factors of the census year (linear between NILIM 671 table years). Usage: release_sections.py W OUTCSV"""
import pandas as pd, numpy as np, sys, os
from pyproj import Geod
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W,OUT=sys.argv[1:3]; EF=f'{W}/nox_ef_nilim671.csv'; parts=[]
for net,(y0,y1,fy) in {2015:(2015,2020,2015),2021:(2020,2025,2021)}.items():
    K=nl.sections_multi(f'{W}/geo/traffic_census_{net}_converted.parquet',EF,[y0,y1]); a=(fy-y0)/5
    q=(1-a)*K[f'q{y0}']+a*K[f'q{y1}']
    geod=Geod(ellps='GRS80'); L=np.array([geod.geometry_length(g) for g in K.to_crs(4326).geometry.values])/1000   # geodesic km
    parts.append(pd.DataFrame({'census':net,'section_id':K.id.astype(str).values,'muni':K.muni.values,'road_type':K.rt.values,'motorway':K.motorway.values,
        'v24_small':K.v24s.values,'v24_large':K.v24l.values,'speed_small':K.Vs.round(1).values,'speed_large':K.Vl.round(1).values,'speed_source':K.speed_src.values,
        'length_km':L.round(3),'nox_g_km_day':q.round(1).values,'nox_t_year':(q.values*L*365/1e6).round(4)}))
    print(net,len(K),'NOx t/yr',round(parts[-1].nox_t_year.sum()),flush=True)
pd.concat(parts).to_csv(OUT,index=False)
