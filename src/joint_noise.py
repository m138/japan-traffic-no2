"""Joint structure of modelled NO2 (FY2022) and night noise v4 at small-area, municipality, secondary-area and prefecture level."""
import pandas as pd, numpy as np, sys
from dbfread import DBF
from scipy import stats
W=sys.argv[1]; A38=sys.argv[2]
N=pd.read_csv(f'{W}/small_area_no2_2022.csv'); Z=pd.read_csv(f'{W}/noise/small_area_noise_v4.csv')
D=N.merge(Z[['KEY_CODE','w','w55','w60','wL','PREF','CITY']],on='KEY_CODE',suffixes=('','_n'))
print('small areas',len(N),'matched',len(D),' pop',round(D.w.sum()/1e6,2),'M')
D['muni']=D.PREF.astype(str).str.zfill(2)+D.CITY.astype(str).str.zfill(3)
g=pd.DataFrame(iter(DBF(A38,encoding='cp932'))).drop_duplicates('A38b_003')
mp={m:r.A38b_003 for _,r in g.iterrows() for m in r.A38b_001.split(',')}; nm14={r.A38b_003:r.A38b_004 for _,r in g.iterrows()}
for new,old in {'03216':'03305','04216':'04423','40231':'40305','22138':'22131','22139':'22131','22140':'22131'}.items(): mp[new]=mp[old]
merge={'0704':'0708','0705':'0708','1401':'1412','1402':'1412','1403':'1412','2301':'2313','2303':'2313','2802':'2811','2803':'2811','2806':'2812','2807':'2812','3701':'3706','3703':'3706','3704':'3707','3705':'3707','4301':'4312','4307':'4312'}
gun={'前橋':'1001','渋川':'1002','伊勢崎':'1003','高崎・安中':'1004','藤岡':'1005','富岡':'1006','吾妻':'1007','沼田':'1008','桐生':'1009','太田・館林':'1010'}
for k in list(mp): a=mp[k]; mp[k]=merge.get(a, gun[nm14[a]] if a.startswith('10') else a)
D['area']=D.muni.map(mp); D['pref']=D.muni.str[:2]
rows=[]
def agg(by):
    X=D.groupby(by)[['w','wN','wR','w10ug','w20ug','wL','w55','w60']].sum(); X=X[X.w>0]
    return pd.DataFrame({'pop':X.w,'NO2':X.wN/X.w,'road':X.wR/X.w,'p10ug':X.w10ug/X.w,'p20ug':X.w20ug/X.w,'Lmean':X.wL/X.w,'n55':X.w55/X.w,'n60':X.w60/X.w})
def wcorr(x,y,w):
    mx,my=np.average(x,weights=w),np.average(y,weights=w); return np.sum(w*(x-mx)*(y-my))/np.sqrt(np.sum(w*(x-mx)**2)*np.sum(w*(y-my)**2))
S=pd.DataFrame({'pop':D.w,'NO2':D.wN/D.w,'road':D.wR/D.w,'Lmean':D.wL/D.w,'n55':D.w55/D.w,'muni':D.muni}).query('pop>0')
levels={'小地域':S,'市区町村':agg('muni'),'二次医療圏':agg('area'),'都道府県':agg('pref')}
for lab,X in levels.items():
    for e in ['Lmean','n55']:
        rows.append({'単位':lab,'n':len(X),'騒音':e,'Pearson(人口重み)':wcorr(X[e].values,X.NO2.values,X['pop'].values),'Spearman':stats.spearmanr(X[e],X.NO2)[0],
                     '道路寄与分とのPearson(人口重み)':wcorr(X[e].values,X.road.values,X['pop'].values)})
# within-municipality (small areas demeaned by municipality, pop-weighted)
for c in ['NO2','Lmean']: S[c+'_dm']=S[c]-S.groupby('muni')[c].transform(lambda v: v.mean())
rows.append({'単位':'小地域（市区町村内の偏差）','n':len(S),'騒音':'Lmean','Pearson(人口重み)':wcorr(S.Lmean_dm.values,S.NO2_dm.values,S['pop'].values),'Spearman':stats.spearmanr(S.Lmean_dm,S.NO2_dm)[0]})
R=pd.DataFrame(rows); print(R.round(3).to_string()); R.round(4).to_csv(f'{W}/noise_no2_correlation.csv',index=False,encoding='utf-8-sig')
for lab,X in levels.items(): X.round(5).to_csv(f'{W}/no2_by_{ {"小地域":"smallarea","市区町村":"municipality","二次医療圏":"area","都道府県":"prefecture"}[lab]}.csv',encoding='utf-8-sig')
# co-exposure: population share by noise and NO2 category (small-area means)
S['Ncat']=pd.cut(S.Lmean,[0,45,50,55,100],right=False,labels=['<45','45-50','50-55','>=55'])
S['Gcat']=pd.cut(S.NO2*1.88,[0,10,20,30,1000],right=False,labels=['<10','10-20','20-30','>=30 ug/m3'])
T=S.pivot_table(index='Ncat',columns='Gcat',values='pop',aggfunc='sum',observed=False)/S['pop'].sum()
print((T*100).round(1).to_string()); T.round(5).to_csv(f'{W}/coexposure_table.csv',encoding='utf-8-sig')
