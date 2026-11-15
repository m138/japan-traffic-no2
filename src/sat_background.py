"""Add satellite-derived surface NO2 (Cooper et al. 2022; ~1 km, 13:30 overpass) as a background predictor to the main model.
Products: OMI-downscaled 2019 (Zenodo 5424752) and TROPOMI 2019 (Zenodo 5484305). Values at the station cell and window means (3x3, 5x5, 11x11 cells ~ 3, 5, 11 km).
Static 2019 products are used for all FYs (spatial pattern as predictor; regression rescales). LOPO CV with kriging as in fit_cv.py."""
import pandas as pd, numpy as np, sys, statsmodels.api as sm, warnings, netCDF4
from scipy.ndimage import uniform_filter
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
warnings.filterwarnings('ignore')
W,EXT,OUT=sys.argv[1:4]
P=pd.read_csv(f'{W}/stations_all.csv',dtype={'code':str,'muni':str})
F=pd.read_csv(f'{W}/station_features.csv',dtype={'code':str}).merge(pd.read_csv(f'{W}/station_grid.csv',dtype={'code':str}),on='code',how='left')
st=P.drop_duplicates('code').set_index('code')[['lat','lon']]
def sample(fn,latname,lonname):
    d=netCDF4.Dataset(fn); lat=np.ravel(d[latname][:]); lon=np.ravel(d[lonname][:])
    li=np.where((lat>23)&(lat<46.5))[0]; lj=np.where((lon>122)&(lon<149))[0]
    Z=np.array(d['surface_no2_ppb'][li.min():li.max()+1,lj.min():lj.max()+1],float); Z[~np.isfinite(Z)|(Z<0)]=np.nan
    lat=lat[li.min():li.max()+1]; lon=lon[lj.min():lj.max()+1]
    ii=np.abs(lat[None,:]-st.lat.values[:,None]).argmin(1); jj=np.abs(lon[None,:]-st.lon.values[:,None]).argmin(1)
    out={'w1':Z[ii,jj]}; Z0=np.nan_to_num(Z); M=np.isfinite(Z).astype(float)
    for k in (3,5,11):
        out[f'w{k}']=(uniform_filter(Z0,k,mode='nearest')/np.maximum(uniform_filter(M,k,mode='nearest'),1e-9))[ii,jj]
    return pd.DataFrame(out,index=st.index)
S_omi=sample(f'{EXT}/cooper_asia_2019.nc','LAT_CENTER','LON_CENTER').add_prefix('omi_')
S_tro=sample(f'{EXT}/tropomi_asia_2019.nc','LAT_CENTER','LON_CENTER').add_prefix('tro_')
SAT=S_omi.join(S_tro).reset_index(); SAT.to_csv(f'{OUT}/station_satellite_2019.csv',index=False)
print(SAT.describe().round(2).to_string())
def gp(k=None): return GPR(k if k is not None else Ck(4.0)*Matern(length_scale=30.0,length_scale_bounds=(2,500),nu=1.5)+WhiteKernel(2.0),optimizer=None if k is not None else 'fmin_l_bfgs_b')
rows=[]; coefs=[]
for fy in [2019,2020,2021,2022]:
    D=P[P.fy==fy].merge(F,on='code').merge(SAT,on='code').dropna(subset=['NO2','lat','lon']).reset_index(drop=True)
    D=D[D[[c for c in SAT.columns if c!='code']].notna().all(1)].reset_index(drop=True)
    D['pref']=D.muni.str[:2]; D['gx']=(D.lon-137)*np.cos(np.radians(36))*111.32; D['gy']=(D.lat-36)*110.57
    y=D.NO2.values; xy=D[['gx','gy']].values
    base=pd.DataFrame({'lE10':D.gE10,'lE3':D.gE3,'lp1':D.gpop1,'lp3':D.gpop3,'lp10':D.gpop10,'sCr':np.sqrt(D['C1000_h20'])})
    specs={'C 主モデル':base,'S0 衛星のみ（TROPOMI 5x5）':pd.DataFrame({'s':D.tro_w5})}
    for prod in ('omi','tro'):
        for w in ('w1','w3','w5','w11'): specs[f'C+{prod}_{w}']=base.assign(sat=D[f'{prod}_{w}'])
    specs['C+tro_w5+w11']=base.assign(s5=D.tro_w5,s11=D.tro_w11)
    specs['C（人口・排出なし）背景=TROPOMI w5,w11 + √道路']=pd.DataFrame({'s5':D.tro_w5,'s11':D.tro_w11,'sCr':np.sqrt(D['C1000_h20'])})
    for name,X in specs.items():
        X=sm.add_constant(X); pred=np.full(len(D),np.nan); full=sm.OLS(y,X).fit()
        K0=gp().fit(xy,y-full.predict(X)).kernel_
        for p in D.pref.unique():
            te=(D.pref==p).values; tr=~te; m=sm.OLS(y[tr],X[tr]).fit()
            pred[te]=m.predict(X[te])+gp(K0).fit(xy[tr],y[tr]-m.predict(X[tr])).predict(xy[te])
        for k in ['gen','road','all']:
            s=(D.kind==k).values if k!='all' else np.ones(len(D),bool); e=pred[s]-y[s]
            rows.append({'fy':fy,'spec':name,'kind':k,'n':int(s.sum()),'R2':1-np.sum(e**2)/np.sum((y[s]-y[s].mean())**2),'RMSE':np.sqrt(np.mean(e**2)),'bias':e.mean()})
        coefs.append({'fy':fy,'spec':name,**full.params.round(4).to_dict(),'R2_in':full.rsquared,'kernel':str(K0)})
    print(fy,'done',flush=True)
R=pd.DataFrame(rows); R.round(4).to_csv(f'{OUT}/sat_background_cv.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(coefs).to_csv(f'{OUT}/sat_background_coef.csv',index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); print(R.pivot_table(index=['fy','spec'],columns='kind',values=['R2','RMSE'],sort=False).round(3).to_string())
