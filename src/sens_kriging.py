"""Sensitivity of the main model (C: background + sqrt road kernel h=20 m, 1 km) to kriging choices, LOPO CV.
K1 main: GP on all-station residuals, hyperparameters fixed from full data.
K2 GP on general-station residuals only (roadside residuals not spread to neighbours).
K3 hyperparameters re-estimated inside each fold.
K4 Matern nu=0.5 / K5 nu=2.5.
K6 regression fitted on general stations only + kriging on general residuals (roadside evaluated out of sample by type too).
K7 add roadside indicator? no (not available at receptors). K7 = station-type weights: none. Instead K7 = no kriging."""
import pandas as pd, numpy as np, sys, statsmodels.api as sm, warnings
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
warnings.filterwarnings('ignore')
W,OUT=sys.argv[1:3]; FYS=[2021,2022]
P=pd.read_csv(f'{W}/stations.csv',dtype={'code':str,'muni':str})
F=pd.read_csv(f'{W}/station_features.csv',dtype={'code':str}).merge(pd.read_csv(f'{W}/station_grid.csv',dtype={'code':str}),on='code',how='left')
def gp(nu=1.5,k=None): return GPR(k if k is not None else Ck(4.0)*Matern(length_scale=30.0,length_scale_bounds=(2,500),nu=nu)+WhiteKernel(2.0),optimizer=None if k is not None else 'fmin_l_bfgs_b')
rows=[]; hyp=[]
for fy in FYS:
    D=P[P.fy==fy].merge(F,on='code').dropna(subset=['NO2']).reset_index(drop=True)
    D['pref']=D.muni.str[:2]; D['gx']=(D.lon-137)*np.cos(np.radians(36))*111.32; D['gy']=(D.lat-36)*110.57
    X=sm.add_constant(pd.DataFrame({'lE10':D.gE10,'lE3':D.gE3,'lp1':D.gpop1,'lp3':D.gpop3,'lp10':D.gpop10,'sCr':np.sqrt(D['C1000_h20'])}))
    y=D.NO2.values; xy=D[['gx','gy']].values; gen=(D.kind=='gen').values
    def run(name,nu=1.5,krige_on='all',refit=False,reg_on='all',krige=True):
        pred=np.full(len(D),np.nan)
        rs=np.ones(len(D),bool) if reg_on=='all' else gen
        m0=sm.OLS(y[rs],X[rs]).fit(); ks=np.ones(len(D),bool) if krige_on=='all' else gen
        K0=gp(nu).fit(xy[ks],(y-m0.predict(X))[ks]).kernel_ if krige else None
        if krige: hyp.append({'fy':fy,'spec':name,'kernel':str(K0)})
        for p in D.pref.unique():
            te=(D.pref==p).values; tr=~te
            m=sm.OLS(y[tr&rs],X[tr&rs]).fit(); pred[te]=m.predict(X[te])
            if krige:
                s=tr&ks; g=gp(nu).fit(xy[s],y[s]-m.predict(X[s])) if refit else gp(nu,K0).fit(xy[s],y[s]-m.predict(X[s]))
                pred[te]+=g.predict(xy[te])
        for k in ['gen','road','all']:
            s=(D.kind==k).values if k!='all' else np.ones(len(D),bool); e=pred[s]-y[s]
            rows.append({'fy':fy,'spec':name,'kind':k,'n':int(s.sum()),'R2':1-np.sum(e**2)/np.sum((y[s]-y[s].mean())**2),'RMSE':np.sqrt(np.mean(e**2)),'bias':e.mean(),'r':np.corrcoef(pred[s],y[s])[0,1]})
        print(fy,name,flush=True)
    run('K1 主解析（全局の残差、超パラメータ固定、ν=1.5）')
    run('K2 一般局の残差だけでクリギング',krige_on='gen')
    run('K3 超パラメータを各分割で推定',refit=True)
    run('K4 ν=0.5',nu=0.5)
    run('K5 ν=2.5',nu=2.5)
    run('K6 回帰も一般局だけ＋一般局の残差でクリギング',krige_on='gen',reg_on='gen')
    run('K7 クリギングなし',krige=False)
R=pd.DataFrame(rows); R.round(4).to_csv(f'{OUT}/sens_kriging.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(hyp).to_csv(f'{OUT}/sens_kriging_kernels.csv',index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); print(R.pivot_table(index=['fy','spec'],columns='kind',values=['R2','RMSE','bias'],sort=False).round(3).to_string())
