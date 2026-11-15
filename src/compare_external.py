"""Compare our hybrid model (LOPO CV) with global 1-km products at Japanese monitoring stations.
Products: Anenberg et al. 2022 (LUR [Larkin 2017] + OMI, 1 km, 2019/2020), Cooper et al. 2022 (OMI/TROPOMI + GEOS-Chem, 1 km, 2019, ~13:30 overpass).
Our model is evaluated by leave-one-prefecture-out CV for each FY with the same features (census R3 traffic).
External products are evaluated (a) as published and (b) after LOPO linear recalibration (a+b*ext fit on other prefectures)."""
import pandas as pd, numpy as np, sys, statsmodels.api as sm, warnings, rasterio, netCDF4
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
warnings.filterwarnings('ignore')
W,EXT,OUT=sys.argv[1:4]
P=pd.read_csv(f'{W}/stations_all.csv',dtype={'code':str,'muni':str})
F=pd.read_csv(f'{W}/station_features.csv',dtype={'code':str}).merge(pd.read_csv(f'{W}/station_grid.csv',dtype={'code':str}),on='code',how='left')
def ext_anen(D,y):
    with rasterio.open(f'{EXT}/anenberg_{y}.tif') as r:
        v=np.array([x[0] for x in r.sample(zip(D.lon,D.lat))],float); v[v<=-998]=np.nan; return v
def ext_cooper(D):
    d=netCDF4.Dataset(f'{EXT}/cooper_asia_2019.nc'); lon=d['LON_CENTER'][:,0]; lat=d['LAT_CENTER'][0,:]
    ii=np.abs(lat[None,:]-D.lat.values[:,None]).argmin(1); jj=np.abs(lon[None,:]-D.lon.values[:,None]).argmin(1)
    Z=d['surface_no2_ppb']; v=np.array([Z[i,j] for i,j in zip(ii,jj)],float); v[~np.isfinite(v)|(v<0)]=np.nan; return v
def gp(k=None): return GPR(k if k is not None else Ck(4.0)*Matern(length_scale=30.0,length_scale_bounds=(2,500),nu=1.5)+WhiteKernel(2.0),optimizer=None if k is not None else 'fmin_l_bfgs_b')
def lopo(D,X,krige=True):
    X=sm.add_constant(X); y=D.NO2.values; pred=np.full(len(D),np.nan)
    if krige: K0=gp().fit(D[['gx','gy']].values,y-sm.OLS(y,X).fit().predict(X)).kernel_
    for p in D.pref.unique():
        te=(D.pref==p).values; tr=~te; m=sm.OLS(y[tr],X[tr]).fit(); pred[te]=m.predict(X[te])
        if krige: pred[te]+=gp(K0).fit(D.loc[tr,['gx','gy']].values,y[tr]-m.predict(X[tr])).predict(D.loc[te,['gx','gy']].values)
    return pred
def stats(D,col,lab,fy):
    rows=[]
    for k in ['gen','road','all']:
        s=(D.kind==k).values if k!='all' else np.ones(len(D),bool); y=D.NO2.values[s]; p=D[col].values[s]; e=p-y
        rows.append({'fy':fy,'method':lab,'kind':k,'n':int(s.sum()),'R2':1-np.sum(e**2)/np.sum((y-y.mean())**2),'RMSE':np.sqrt(np.mean(e**2)),'bias':e.mean(),'r':np.corrcoef(p,y)[0,1],'slope':sm.OLS(y,sm.add_constant(p)).fit().params[1]})
    return rows
rows=[]; keep=[]
for fy in [2019,2020,2021,2022]:
    D=P[P.fy==fy].merge(F,on='code').dropna(subset=['NO2','lat','lon']).reset_index(drop=True)
    D['pref']=D.muni.str[:2]; D['gx']=(D.lon-137)*np.cos(np.radians(36))*111.32; D['gy']=(D.lat-36)*110.57
    bg=pd.DataFrame({'lE10':D.gE10,'lE3':D.gE3,'lp1':D.gpop1,'lp3':D.gpop3,'lp10':D.gpop10})
    D['ours_bg']=lopo(D,bg); D['ours']=lopo(D,bg.assign(sCr=np.sqrt(D['C1000_h20']))); D['ours_nokrige']=lopo(D,bg.assign(sCr=np.sqrt(D['C1000_h20'])),False)
    exts={}
    if fy in (2019,2020): exts[f'Anenberg2022_{fy}']=ext_anen(D,fy)
    if fy==2019: exts['Cooper2022_2019']=ext_cooper(D)
    if fy in (2021,2022): exts['Anenberg2022_2020（年ずれ）']=ext_anen(D,2020)
    ok=np.ones(len(D),bool)
    for k,v in exts.items(): D[k]=v; ok&=np.isfinite(v)
    D=D[ok].reset_index(drop=True); bg=pd.DataFrame({'lE10':D.gE10,'lE3':D.gE3,'lp1':D.gpop1,'lp3':D.gpop3,'lp10':D.gpop10})
    D['ours_bg']=lopo(D,bg); D['ours']=lopo(D,bg.assign(sCr=np.sqrt(D['C1000_h20']))); D['ours_nokrige']=lopo(D,bg.assign(sCr=np.sqrt(D['C1000_h20'])),False)  # refit on the common station set
    for k in exts: D[k+'_recal']=lopo(D,pd.DataFrame({'x':D[k]}),False)
    for k in exts: D[k+'+ours']=lopo(D,bg.assign(sCr=np.sqrt(D['C1000_h20']),ext=D[k]))   # does the external product add information to ours?
    cols=['ours_bg','ours','ours_nokrige']+[c for k in exts for c in (k,k+'_recal',k+'+ours')]
    for c in cols: rows+=stats(D,c,c,fy)
    keep.append(D[['fy','code','kind','pref','lat','lon','NO2']+cols])
R=pd.DataFrame(rows); R.round(4).to_csv(f'{OUT}/compare_external_stations.csv',index=False,encoding='utf-8-sig')
pd.concat(keep).round(3).to_csv(f'{OUT}/compare_external_station_preds.csv',index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); print(R.pivot_table(index=['fy','method'],columns='kind',values=['R2','RMSE','bias','n'],sort=False).round(2).to_string())
