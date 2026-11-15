"""FY2015-2023 main model (background + sqrt road kernel + satellite, GP residual), fitted separately for each FY.
Traffic: H27 census (2015 network) for FY2015-2019, R3 census (2021 network) for FY2020-2023. NOx emission factors interpolated
linearly in time between the NILIM 671 table years (2015/2020 or 2020/2025); because the kernel and the grid are linear in emissions,
features are interpolated between the two EF-year versions. Population (2020 census receptors) and satellite (TROPOMI 2019) are fixed.
Usage: release_build.py W OUT
Release version: same models as multiyear_fit_predict.py, and also writes station-level LOPO predictions,
receptor-level NO2/road part for every FY with night noise (paper A v4), and small-area/municipality/prefecture tables."""
import pandas as pd, numpy as np, sys, os, glob, warnings, statsmodels.api as sm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
warnings.filterwarnings('ignore'); W,OUT=sys.argv[1:3]; CVONLY='--cv-only' in sys.argv; os.makedirs(OUT,exist_ok=True)
FYS=range(2015,2024); NETY={2015:(2015,2020),2021:(2020,2025)}
net=lambda fy: 2015 if fy<=2019 else 2021
def interp(df,stem,fy,suffix=''):
    y0,y1=NETY[net(fy)]; a=(fy-y0)/5; return (1-a)*df[f'{stem}{suffix}{y0}'].values+a*df[f'{stem}{suffix}{y1}'].values
P=pd.read_csv(f'{W}/stations_all.csv',dtype={'code':str,'muni':str})
G=pd.read_csv(f'{W}/station_popgrid_all.csv',dtype={'code':str}); SAT=pd.read_csv(f'{W}/station_sat_all.csv',dtype={'code':str})
SF={n:pd.read_csv(f'{W}/my/{n}/station_feat.csv',dtype={'code':str}) for n in NETY}
def gp(k=None): return GPR(k if k is not None else Ck(4.0)*Matern(length_scale=30.0,length_scale_bounds=(2,500),nu=1.5)+WhiteKernel(2.0),optimizer=None if k is not None else 'fmin_l_bfgs_b')
def design(C,E3,E10,p1,p3,p10,sat):
    return pd.DataFrame({'const':1.0,'lE10':np.log1p(E10),'lE3':np.log1p(E3),'lp1':p1,'lp3':p3,'lp10':p10,'sCr':np.sqrt(C),'sat':sat})
rows=[]; coefs=[]; models={}; SP=[]
for fy in FYS:
    F=SF[net(fy)]; f=pd.DataFrame({'code':F.code,'C':interp(F,'C',fy),'E3':interp(F,'E3',fy,'_'),'E10':interp(F,'E10',fy,'_')})
    D=P[P.fy==fy].merge(f,on='code').merge(G,on='code').merge(SAT,on='code').dropna(subset=['NO2','lat','lon','sat3','C','gpop1']).reset_index(drop=True); D['pref']=D.muni.str[:2]
    X=design(D.C,D.E3,D.E10,D.gpop1,D.gpop3,D.gpop10,D.sat3); y=D.NO2.values; xy=np.c_[(D.lon-137)*np.cos(np.radians(36))*111.32,(D.lat-36)*110.57]
    full=sm.OLS(y,X).fit(); g=gp().fit(xy,y-full.predict(X)); K0=g.kernel_; pred=np.full(len(D),np.nan)
    for p in D.pref.unique():
        te=(D.pref==p).values; tr=~te; m=sm.OLS(y[tr],X[tr]).fit()
        pred[te]=m.predict(X[te])+gp(K0).fit(xy[tr],y[tr]-m.predict(X[tr])).predict(xy[te])
    SP.append(pd.DataFrame({'fy':fy,'code':D.code,'kind':D.kind,'pref':D.pref,'lat':D.lat,'lon':D.lon,'NO2_obs':y,'NO2_lopo':pred.round(3)}))
    for k in ['gen','road','all']:
        s=(D.kind==k).values if k!='all' else np.ones(len(D),bool); e=pred[s]-y[s]
        rows.append({'fy':fy,'census':'H27' if net(fy)==2015 else 'R3','kind':k,'n':int(s.sum()),'obs_mean':y[s].mean(),'R2':1-np.sum(e**2)/np.sum((y[s]-y[s].mean())**2),'RMSE':np.sqrt(np.mean(e**2)),'bias':e.mean()})
    coefs.append({'fy':fy,**full.params.round(4).to_dict(),'R2_in':round(full.rsquared,4),'kernel':str(K0)}); models[fy]=(full.params,g)
    print(fy,len(D),'R2 all',round(rows[-1]['R2'],3),'road',round(rows[-2]['R2'],3),flush=True)
pd.DataFrame(rows).round(4).to_csv(f'{OUT}/cv_lopo_2015_2023.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(coefs).to_csv(f'{OUT}/coef_2015_2023.csv',index=False,encoding='utf-8-sig')
pd.concat(SP).to_csv(f'{OUT}/stations_lopo_2015_2023.csv',index=False,encoding='utf-8')
if CVONLY: sys.exit()
# national prediction
files=sorted(glob.glob(f'{W}/rec/r_*.parquet')); RG=pd.read_parquet(f'{W}/rec_grid.parquet'); RS=pd.read_parquet(f'{W}/rec_sat.parquet').sat3.values; LL=pd.read_parquet(f'{W}/rec_ll.parquet')
RE={n:pd.read_parquet(f'{W}/my/{n}/rec_E.parquet') for n in NETY}
gxy=np.c_[(LL.lon.values-137)*np.cos(np.radians(36))*111.32,(LL.lat.values-36)*110.57]
KEY=[];PREF=[];Wt=[];RC={n:[] for n in NETY}
for f in files:
    R=pd.read_parquet(f); pc=os.path.basename(f)[2:4]; KEY.append(R.KEY_CODE.values); PREF.append(np.full(len(R),pc)); Wt.append(R.w.values)
    for n in NETY: RC[n].append(pd.read_parquet(f'{W}/my/{n}/road/c_{pc}.parquet'))
KEY=np.concatenate(KEY); PREF=np.concatenate(PREF); Wt=np.concatenate(Wt); RC={n:pd.concat(v,ignore_index=True) for n,v in RC.items()}
assert len(KEY)==len(RG)==len(RS)==len(LL)
LN=np.concatenate([pd.read_parquet(f'{W}/noise_rec/n_{os.path.basename(f)[2:4]}.parquet').Ln.values for f in files]).astype(np.float32); LN=np.clip(np.nan_to_num(LN,nan=40.0),40.0,None); assert len(LN)==len(KEY)   # 40 dB floor as in the noise model
REC={'KEY_CODE':KEY,'lon':LL.lon.values.round(5),'lat':LL.lat.values.round(5),'pop_w':Wt.astype(np.float32),'Lnight_2021':LN.round(1)}
nat=[]; SA=pd.DataFrame({'KEY_CODE':KEY,'w':Wt}).groupby('KEY_CODE',sort=True).w.sum().to_frame()
ug=1.88
for fy in FYS:
    b,g=models[fy]; n=net(fy); C=interp(RC[n],'C',fy); E3=interp(RE[n],'E3',fy,'_'); E10=interp(RE[n],'E10',fy,'_')
    road=b['sCr']*np.sqrt(C)
    lin=b['const']+b['lE10']*np.log1p(E10)+b['lE3']*np.log1p(E3)+b['lp1']*RG.gpop1.values+b['lp3']*RG.gpop3.values+b['lp10']*RG.gpop10.values+road+b['sat']*RS
    res=np.concatenate([g.predict(gxy[j:j+50000]) for j in range(0,len(gxy),50000)])
    no2=np.clip(lin+res,0.5,None); tot=Wt.sum(); REC[f'NO2_{fy}']=no2.astype(np.float32).round(2); REC[f'road_{fy}']=road.astype(np.float32).round(2)
    nat.append({'fy':fy,'census':'H27' if n==2015 else 'R3','NO2_ppb':np.sum(Wt*no2)/tot,'road_ppb':np.sum(Wt*road)/tot,'p_ge10ug':np.sum(Wt*(no2>=10/ug))/tot,'p_ge20ug':np.sum(Wt*(no2>=20/ug))/tot,'p_ge40ug':np.sum(Wt*(no2>=40/ug))/tot})
    T=pd.DataFrame({'KEY_CODE':KEY,'wN':Wt*no2,'wR':Wt*road}).groupby('KEY_CODE',sort=True).sum()
    SA[f'NO2_{fy}']=(T.wN/SA.w).astype(np.float32); SA[f'road_{fy}']=(T.wR/SA.w).astype(np.float32)
    print(nat[-1],flush=True)
pd.DataFrame(nat).round(4).to_csv(f'{OUT}/national_2015_2023.csv',index=False,encoding='utf-8-sig')
R=pd.DataFrame(REC); R.insert(0,'rid',np.arange(len(R),dtype=np.int32)); R.to_parquet(f'{OUT}/receptors_no2_noise_2015_2023.parquet',compression='zstd',index=False)
ugm=1.88; lw=10**(R.Lnight_2021/10)*R.pop_w; R['dbl']=((R.Lnight_2021>=55)&(R.NO2_2022>=20/ugm))*R.pop_w; R['n55']=(R.Lnight_2021>=55)*R.pop_w
T=pd.DataFrame({'KEY_CODE':R.KEY_CODE,'lw':lw,'dbl':R.dbl,'n55':R.n55}).groupby('KEY_CODE',sort=True).sum()
SA=SA.reset_index(); T=T.loc[SA.KEY_CODE].values; SA['Lnight_2021_energy_mean']=np.round(10*np.log10(T[:,0]/SA.w.values),1)
SA['share_Lnight55']=(T[:,2]/SA.w.values).round(4); SA['share_double_2022']=(T[:,1]/SA.w.values).round(4)
SA=SA.rename(columns={'w':'pop2020'}); SA.round(3).to_csv(f'{OUT}/small_area_no2_noise_2015_2023.csv',index=False)
SA['w']=SA.pop2020; SA['muni']=SA.KEY_CODE.astype(str).str[:5]; SA['pref']=SA.KEY_CODE.astype(str).str[:2]
for lev in ('muni','pref'):
    cols=[c for c in SA.columns if c.startswith(('NO2_','road_'))]
    A=SA.assign(**{c:SA[c]*SA.w for c in cols}).groupby(lev)[['w']+cols].sum(); A[cols]=A[cols].div(A.w,axis=0)
    A.round(3).to_csv(f'{OUT}/no2_2015_2023_by_{"municipality" if lev=="muni" else "prefecture"}.csv',encoding='utf-8-sig')
