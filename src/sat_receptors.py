"""Satellite background at stations and receptors. Usage: sat_receptors.py W tropomi_asia_2019.nc"""
import pandas as pd, numpy as np, sys, os
sys.path.insert(0,os.path.dirname(__file__)); import sat_lib as sl
W,FN=sys.argv[1:3]; g=sl.load(FN)
St=pd.read_csv(f'{W}/stations_all.csv',dtype={'code':str}).drop_duplicates('code')
pd.DataFrame({'code':St.code.values,'sat3':sl.sample(g,St.lat.values,St.lon.values)}).to_csv(f'{W}/station_sat.csv',index=False)
R=pd.read_parquet(f'{W}/rec_ll.parquet'); s=sl.sample(g,R.lat.values,R.lon.values)
print('receptors',len(s),'missing',int(np.isnan(s).sum()))
# fill rare missing cells (coast/islands) with the nearest valid value along the receptor list is not spatial; use a wider window instead
if np.isnan(s).any():
    g11=sl.load(FN,11); m=np.isnan(s); s[m]=sl.sample(g11,R.lat.values[m],R.lon.values[m]); print('still missing',int(np.isnan(s).sum()))
pd.DataFrame({'sat3':s.astype(np.float32)}).to_parquet(f'{W}/rec_sat.parquet')
