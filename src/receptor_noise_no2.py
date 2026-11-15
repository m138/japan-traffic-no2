"""Receptor-level joint structure of night noise (paper A model v4 at the NO2 receptors) and NO2 (FY2022 main model v3).
Population-weighted Pearson and Spearman, overall, within municipality, by urbanicity (10 km population density tertiles) and
for receptors with/without a census road within 200 m; co-exposure table; check against small-area noise v4. Usage: receptor_noise_no2.py W OUT"""
import pandas as pd, numpy as np, glob, os, sys
from scipy.stats import rankdata
W,OUT=sys.argv[1:3]; os.makedirs(OUT,exist_ok=True)
files=sorted(glob.glob(f'{W}/rec/r_*.parquet'))
R=pd.concat([pd.read_parquet(f,columns=['KEY_CODE','w']) for f in files],ignore_index=True)
N=pd.concat([pd.read_parquet(f'{W}/noise_rec/n_{os.path.basename(f)[2:4]}.parquet') for f in files],ignore_index=True)
P=pd.read_parquet(f'{W}/receptor_no2_sat_2022.parquet'); G=pd.read_parquet(f'{W}/rec_grid.parquet',columns=['gpop10'])
assert len(R)==len(N)==len(P)==len(G) and (R.KEY_CODE.values==P.KEY_CODE.values).all()
D=pd.DataFrame({'KEY':R.KEY_CODE.astype(str).values,'w':R.w.values,'Lraw':N.Ln.values,'no2':P.no2.values,'road':P.road.values,'lp10':G.gpop10.values})
D['near']=D.Lraw.notna(); D['L']=np.clip(D.Lraw.fillna(40),40,None); D['muni']=D.KEY.str[:5]; D['pref']=D.KEY.str[:2]
def wcorr(x,y,w):
    mx,my=np.average(x,weights=w),np.average(y,weights=w); return np.sum(w*(x-mx)*(y-my))/np.sqrt(np.sum(w*(x-mx)**2)*np.sum(w*(y-my)**2))
def wspear(x,y,w): return wcorr(rankdata(x),rankdata(y),w)
rows=[]
def add(lab,S):
    rows.append({'集団':lab,'受音点':len(S),'人口(百万)':S.w.sum()/1e6,'NO2とPearson':wcorr(S.L.values,S.no2.values,S.w.values),'NO2とSpearman':wspear(S.L.values,S.no2.values,S.w.values),
                 '道路寄与分とPearson':wcorr(S.L.values,S.road.values,S.w.values),'背景分(NO2−道路)とPearson':wcorr(S.L.values,(S.no2-S.road).values,S.w.values)})
add('全受音点',D); add('200 m以内にセンサス道路あり',D[D.near]); add('200 m以内に道路なし（騒音は40 dB）',D[~D.near])
for c in ['L','no2','road']: D[c+'_dm']=D[c]-D.groupby('muni')[c].transform('mean')
rows.append({'集団':'市区町村内の偏差','受音点':len(D),'人口(百万)':D.w.sum()/1e6,'NO2とPearson':wcorr(D.L_dm.values,D.no2_dm.values,D.w.values),'道路寄与分とPearson':wcorr(D.L_dm.values,D.road_dm.values,D.w.values)})
D['urb']=pd.qcut(D.lp10,3,labels=['人口密度 低','中','高'])
for u,S in D.groupby('urb',observed=True): add(f'都市度：{u}',S)
T=pd.DataFrame(rows); T.round(3).to_csv(f'{OUT}/receptor_noise_no2_correlation.csv',index=False,encoding='utf-8-sig'); print(T.round(3).to_string())
# co-exposure at receptor level
D['Ncat']=pd.cut(D.L,[0,45,50,55,60,200],right=False,labels=['<45','45-50','50-55','55-60','>=60'])
D['Gcat']=pd.cut(D.no2*1.88,[0,10,20,30,1000],right=False,labels=['<10','10-20','20-30','>=30 µg/m3'])
C=D.pivot_table(index='Ncat',columns='Gcat',values='w',aggfunc='sum',observed=False)/D.w.sum()
(C*100).round(2).to_csv(f'{OUT}/receptor_coexposure_table.csv',encoding='utf-8-sig'); print((C*100).round(1).to_string())
# among people with night noise >=55 dB, share with NO2 >= 20 ug; and vice versa
hiN=D.L>=55; hiG=D.no2*1.88>=20
print('P(NO2>=20 | L>=55)',round(D.w[hiN&hiG].sum()/D.w[hiN].sum(),3),' P(L>=55 | NO2>=20)',round(D.w[hiN&hiG].sum()/D.w[hiG].sum(),3))
# check: small-area Lmean from these receptors vs noise v4 small-area table
S=D.assign(wL=D.w*D.L).groupby('KEY')[['w','wL']].sum(); S['Lm']=S.wL/S.w
V=pd.read_csv(f'{W}/noise/small_area_noise_v4.csv',dtype={'KEY_CODE':str}); V['Lm4']=V.wL/V.w; M=S.join(V.set_index('KEY_CODE')[['Lm4']],how='inner')
print('check vs v4: small areas',len(M),' r',round(np.corrcoef(M.Lm,M.Lm4)[0,1],4),' mean diff',round((M.Lm-M.Lm4).mean(),3),' national mean',round(np.average(D.L,weights=D.w),2),'vs',round(np.average(V.Lm4,weights=V.w),2))
