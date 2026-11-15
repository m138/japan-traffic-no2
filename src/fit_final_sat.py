"""Main model (2026-11-14 onward): NO2 = b0 + b·[gE10, gE3, gpop1, gpop3, gpop10, sqrt(C1000_h20), sat3] + GP residual (Matern 1.5).
sat3 = TROPOMI-inferred surface NO2 2019 (Cooper et al. 2022), 3x3-cell mean. Fit on all stations of FY and save."""
import pandas as pd, numpy as np, sys, pickle, statsmodels.api as sm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
W=sys.argv[1]; FY=int(sys.argv[2])
P=pd.read_csv(f'{W}/stations_all.csv',dtype={'code':str,'muni':str})
F=pd.read_csv(f'{W}/station_features.csv',dtype={'code':str}).merge(pd.read_csv(f'{W}/station_grid.csv',dtype={'code':str}),on='code').merge(pd.read_csv(f'{W}/station_sat.csv',dtype={'code':str}),on='code')
D=P[P.fy==FY].merge(F,on='code').dropna(subset=['NO2','sat3','lat','lon']).reset_index(drop=True)
X=pd.DataFrame({'lE10':D.gE10,'lE3':D.gE3,'lp1':D.gpop1,'lp3':D.gpop3,'lp10':D.gpop10,'sCr':np.sqrt(D.C1000_h20),'sat':D.sat3})
m=sm.OLS(D.NO2.values,sm.add_constant(X)).fit(); print(len(D),'stations'); print(m.summary().tables[1])
gx=np.c_[(D.lon-137)*np.cos(np.radians(36))*111.32,(D.lat-36)*110.57]
g=GPR(Ck(4.0)*Matern(length_scale=30.0,length_scale_bounds=(2,500),nu=1.5)+WhiteKernel(2.0)).fit(gx,D.NO2.values-m.predict(sm.add_constant(X)))
print('GP kernel',g.kernel_)
pickle.dump({'params':m.params,'gp':g,'fy':FY},open(f'{W}/final_model_sat_{FY}.pkl','wb'))
pd.DataFrame({'term':m.params.index,'coef':m.params.values,'se':m.bse.values}).to_csv(f'{W}/final_coef_sat_{FY}.csv',index=False)
