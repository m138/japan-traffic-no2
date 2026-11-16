"""All receptor-level statistics in the manuscript, recomputed from the released receptor file (Lnight 2021, NO2 FY2015-2023)
so that main text, Table 1 and Extended Data agree. Adds: correlation by spatial level, within/between-municipality covariance
decomposition, small-area-mean double exposure, Table 1 by municipality type (NO2 FY2022 cross-section).
Usage: unified_stats.py W RELEASE_PARQUET A38_DBF OUT"""
import pandas as pd, numpy as np, geopandas as gpd, shapely, glob, os, sys
from dbfread import DBF
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W,RP,A38,OUT=sys.argv[1:5]; os.makedirs(OUT,exist_ok=True); ug=1.88
D=pd.read_parquet(RP,columns=['KEY_CODE','pop_w','Lnight_2021','NO2_2015','NO2_2022','NO2_2023','road_2015','road_2022','road_2023'])
cache=f'{OUT}/_nearest_road.parquet'
if os.path.exists(cache): Q=pd.read_parquet(cache)
else:
    G=nl.load_sections(f'{W}/geo/traffic_census_2021_converted.parquet',f'{W}/nox_ef_nilim671.csv'); G=G[G.ok]
    G['rclass']=np.where((G.motorway=='1')|G.rt.str[0].isin(['1','2']),'高速道路',np.where(G.rt.str[0]=='3','一般国道',np.where(G.rt.str[0].isin(['4','5','6']),'都道府県道','市道ほか')))
    dist=[];rc=[]
    for f in sorted(glob.glob(f'{W}/rec/r_*.parquet')):
        R=pd.read_parquet(f); z=int(R.epsg.iloc[0]); Gz=G.to_crs(z); Gz=Gz[Gz.intersects(shapely.box(R.x.min(),R.y.min(),R.x.max(),R.y.max()).buffer(5000))]
        pts=shapely.points(R.x.values,R.y.values); idx=shapely.STRtree(Gz.geometry.values).nearest(pts)
        dist.append(shapely.distance(pts,Gz.geometry.values[idx])); rc.append(Gz.rclass.values[idx])
    Q=pd.DataFrame({'dist':np.concatenate(dist),'rclass':np.concatenate(rc)}); Q.to_parquet(cache)
D=pd.concat([D,Q],axis=1); w=D.pop_w.values; D['L']=D.Lnight_2021; D['G']=D.NO2_2022*ug
D['muni']=D.KEY_CODE.str[:5]; pref=D.muni.str[:2]; city=D.muni.str[2:].astype(int)
ward=(city>=101)&(city<=199)&(pref!='13'); tokyo=(pref=='13')&(city>=101)&(city<=123)
D['city']=np.where(ward,D.muni.str[:3]+'00',D.muni); pop=D.city.map(D.groupby('city').pop_w.sum()).values
D['ctype']=np.select([tokyo,ward,(city>=300)&~ward,pop>=300000,pop>=100000,pop>=50000],['東京23区','政令指定都市','町村','人口30万以上の市','10〜30万の市','5〜10万の市'],'5万未満の市')
D['dband']=pd.cut(D.dist,[-1,20,50,100,200,1e9],labels=['<20 m','20〜50 m','50〜100 m','100〜200 m','≥200 m'])
D['both']=(D.L>=55)&(D.G>=20)
def wavg(x,ww): return np.average(x,weights=ww)
def wcorr(x,y,ww):
    mx,my=wavg(x,ww),wavg(y,ww); return np.sum(ww*(x-mx)*(y-my))/np.sqrt(np.sum(ww*(x-mx)**2)*np.sum(ww*(y-my)**2))
out={}
b=D.both.values; out['二重暴露 %']=100*w[b].sum()/w.sum(); out['二重暴露 万人']=w[b].sum()/1e4
out['55dB以上 %']=100*w[D.L>=55].sum()/w.sum(); out['NO2 20µg以上 %']=100*w[D.G>=20].sum()/w.sum()
out['55dB以上のうちNO2 20µg以上 %']=100*w[b].sum()/w[D.L>=55].sum()
out['NO2 2022 ppb']=wavg(D.NO2_2022,w); out['WHO10µg超 2022 %']=100*w[D.G>=10].sum()/w.sum()
out['WHO10µg超 2015 %']=100*w[D.NO2_2015*ug>=10].sum()/w.sum(); out['WHO10µg超 2023 %']=100*w[D.NO2_2023*ug>=10].sum()/w.sum()
out['NO2 2015 ppb']=wavg(D.NO2_2015,w); out['NO2 2023 ppb']=wavg(D.NO2_2023,w); out['道路分変化 2015→2023 %']=(wavg(D.road_2023,w)/wavg(D.road_2015,w)-1)*100
for k,v in (('二重暴露のうち東京23区 %',D.ctype=='東京23区'),('二重暴露のうち政令市 %',D.ctype=='政令指定都市'),('二重暴露のうち道路50m以内 %',D.dist<50),('二重暴露のうち道路100m以内 %',D.dist<100)):
    out[k]=100*w[b&v.values].sum()/w[b].sum()
for r in ['都道府県道','一般国道','高速道路','市道ほか']: out[f'二重暴露の最寄り道路 {r} %']=100*w[b&(D.rclass==r).values].sum()/w[b].sum()
for band in D.dband.cat.categories: m=(D.dband==band).values; out[f'距離 {band} の住民のうち二重暴露 %']=100*w[m&b].sum()/w[m].sum()
# correlations by level
g=pd.DataFrame(iter(DBF(A38,encoding='cp932'))).drop_duplicates('A38b_003')
mp={m:r.A38b_003 for _,r in g.iterrows() for m in r.A38b_001.split(',')}; nm14={r.A38b_003:r.A38b_004 for _,r in g.iterrows()}
for new,old in {'03216':'03305','04216':'04423','40231':'40305','22138':'22131','22139':'22131','22140':'22131'}.items(): mp[new]=mp[old]
merge={'0704':'0708','0705':'0708','1401':'1412','1402':'1412','1403':'1412','2301':'2313','2303':'2313','2802':'2811','2803':'2811','2806':'2812','2807':'2812','3701':'3706','3703':'3706','3704':'3707','3705':'3707','4301':'4312','4307':'4312'}
gun={'前橋':'1001','渋川':'1002','伊勢崎':'1003','高崎・安中':'1004','藤岡':'1005','富岡':'1006','吾妻':'1007','沼田':'1008','桐生':'1009','太田・館林':'1010'}
for k in list(mp): a=mp[k]; mp[k]=merge.get(a, gun[nm14[a]] if a.startswith('10') else a)
D['area']=D.muni.map(mp); D['pref']=pref; print('unmapped area pop',D.pop_w[D.area.isna()].sum())
D['wL']=w*D.L; D['wG']=w*D.NO2_2022; D['wb']=w*b
rows=[{'単位':'住居地点','n':len(D),'r':wcorr(D.L.values,D.NO2_2022.values,w)}]
for lab,key in (('小地域','KEY_CODE'),('市区町村','muni'),('二次医療圏','area'),('都道府県','pref')):
    A=D.groupby(key)[['pop_w','wL','wG','wb']].sum(); A=A[A.pop_w>0]; rows.append({'単位':lab,'n':len(A),'r':wcorr((A.wL/A.pop_w).values,(A.wG/A.pop_w).values,A.pop_w.values)})
    if lab=='小地域':
        SL=D.KEY_CODE.map(A.wL/A.pop_w).values; SG=D.KEY_CODE.map(A.wG/A.pop_w).values*ug
        out['小地域平均で数えた二重暴露 %']=100*w[(SL>=55)&(SG>=20)].sum()/w.sum()
for c in ['L','NO2_2022','road_2022']: D[c+'_m']=D.groupby('muni')[c].transform(lambda v: v)  # placeholder
mL=D.groupby('muni').apply(lambda s: wavg(s.L,s.pop_w)); mG=D.groupby('muni').apply(lambda s: wavg(s.NO2_2022,s.pop_w))
Lm=D.muni.map(mL).values; Gm=D.muni.map(mG).values; L=D.L.values; Gv=D.NO2_2022.values; Lb=wavg(L,w); Gb=wavg(Gv,w)
covW=np.sum(w*(L-Lm)*(Gv-Gm)); covB=np.sum(w*(Lm-Lb)*(Gm-Gb)); vLW=np.sum(w*(L-Lm)**2); vLB=np.sum(w*(Lm-Lb)**2); vGW=np.sum(w*(Gv-Gm)**2); vGB=np.sum(w*(Gm-Gb)**2)
rows.append({'単位':'住居地点（市区町村内の偏差）','n':len(D),'r':covW/np.sqrt(vLW*vGW)})
out.update({'騒音分散のうち市区町村間 %':100*vLB/(vLB+vLW),'NO2分散のうち市区町村間 %':100*vGB/(vGB+vGW),'共分散のうち市区町村間 %':100*covB/(covB+covW),
            '市区町村内相関':covW/np.sqrt(vLW*vGW),'市区町村間相関（人口重み、地点で重み付け）':covB/np.sqrt(vLB*vGB),
            '騒音×道路分 r':wcorr(L,D.road_2022.values,w),'騒音×背景分 r':wcorr(L,(D.NO2_2022-D.road_2022).values,w)})
for c,lab in (('L','騒音'),('G','NO2')): pass
R=pd.DataFrame(rows); R.round(3).to_csv(f'{OUT}/correlation_by_level.csv',index=False,encoding='utf-8-sig')
# Table 1
def summ(S):
    ww=S.pop_w.values; bb=S.both.values
    return pd.Series({'人口(百万)':ww.sum()/1e6,'NO2 2022 ppb':wavg(S.NO2_2022,ww),'NO2変化% 2015→2023':(wavg(S.NO2_2023,ww)/wavg(S.NO2_2015,ww)-1)*100,
        '55dB以上%':100*ww[S.L>=55].sum()/ww.sum(),'NO2 20µg以上%':100*ww[S.G>=20].sum()/ww.sum(),'二重暴露%':100*ww[bb].sum()/ww.sum(),
        '二重暴露に占める割合%':100*ww[bb].sum()/w[b].sum(),'相関':wcorr(S.L.values,S.NO2_2022.values,ww)})
order=['東京23区','政令指定都市','人口30万以上の市','10〜30万の市','5〜10万の市','5万未満の市','町村']
T=D.groupby('ctype').apply(summ).loc[order]; T.loc['全国']=summ(D); T.round(3).to_csv(f'{OUT}/table1_by_municipality_type.csv',encoding='utf-8-sig')
pd.Series(out).round(3).to_csv(f'{OUT}/key_numbers.csv',encoding='utf-8-sig')
pd.set_option('display.width',250); print(pd.Series(out).round(2).to_string()); print(R.round(3).to_string()); print(T.round(2).to_string())
