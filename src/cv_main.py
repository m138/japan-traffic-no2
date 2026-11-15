"""LOPO CV (regression + kriging) for the background-only model, model C (v1 specification) and the main model (C + satellite), by FY.
Usage: cv_main.py W OUTCSV FY [FY ...]"""
import pandas as pd, numpy as np, sys, statsmodels.api as sm, warnings
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
warnings.filterwarnings('ignore'); W,OUT=sys.argv[1:3]; FYS=[int(x) for x in sys.argv[3:]]
P=pd.read_csv(f'{W}/stations_all.csv',dtype={'code':str,'muni':str})
F=pd.read_csv(f'{W}/station_features.csv',dtype={'code':str}).merge(pd.read_csv(f'{W}/station_grid.csv',dtype={'code':str}),on='code').merge(pd.read_csv(f'{W}/station_sat.csv',dtype={'code':str}),on='code')
def gp(k=None): return GPR(k if k is not None else Ck(4.0)*Matern(length_scale=30.0,length_scale_bounds=(2,500),nu=1.5)+WhiteKernel(2.0),optimizer=None if k is not None else 'fmin_l_bfgs_b')
rows=[]; preds=[]
for fy in FYS:
    D=P[P.fy==fy].merge(F,on='code').dropna(subset=['NO2','sat3','lat','lon']).reset_index(drop=True); D['pref']=D.muni.str[:2]
    y=D.NO2.values; xy=np.c_[(D.lon-137)*np.cos(np.radians(36))*111.32,(D.lat-36)*110.57]
    bg=pd.DataFrame({'lE10':D.gE10,'lE3':D.gE3,'lp1':D.gpop1,'lp3':D.gpop3,'lp10':D.gpop10})
    specs={'A 背景のみ':bg,'C 背景＋√道路寄与（v1の形）':bg.assign(sCr=np.sqrt(D.C1000_h20)),'主モデル C＋衛星':bg.assign(sCr=np.sqrt(D.C1000_h20),sat=D.sat3),'衛星＋√道路寄与だけ':pd.DataFrame({'sat':D.sat3,'sCr':np.sqrt(D.C1000_h20)})}
    for name,X in specs.items():
        X=sm.add_constant(X); full=sm.OLS(y,X).fit(); K0=gp().fit(xy,y-full.predict(X)).kernel_; pred=np.full(len(D),np.nan); lin=pred.copy()
        for p in D.pref.unique():
            te=(D.pref==p).values; tr=~te; m=sm.OLS(y[tr],X[tr]).fit(); lin[te]=m.predict(X[te])
            pred[te]=lin[te]+gp(K0).fit(xy[tr],y[tr]-m.predict(X[tr])).predict(xy[te])
        for lab,pp in (('回帰のみ',lin),('回帰+クリギング',pred)):
            for k in ['gen','road','all']:
                s=(D.kind==k).values if k!='all' else np.ones(len(D),bool); e=pp[s]-y[s]
                rows.append({'fy':fy,'model':name,'pred':lab,'kind':k,'n':int(s.sum()),'R2':1-np.sum(e**2)/np.sum((y[s]-y[s].mean())**2),'RMSE':np.sqrt(np.mean(e**2)),'bias':e.mean(),'r':np.corrcoef(pp[s],y[s])[0,1]})
        if name.startswith('主モデル'): preds.append(D[['fy','code','kind','pref','lat','lon','NO2']].assign(pred=pred))
        print(fy,name,'| coef',full.params.round(3).to_dict(),flush=True)
R=pd.DataFrame(rows); R.round(4).to_csv(OUT,index=False,encoding='utf-8-sig'); pd.concat(preds).round(3).to_csv(OUT.replace('.csv','_station_preds.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); print(R[R.pred=='回帰+クリギング'].pivot_table(index=['fy','model'],columns='kind',values=['R2','RMSE'],sort=False).round(3).to_string())
