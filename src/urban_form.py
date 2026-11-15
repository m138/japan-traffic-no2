"""RQ3/RQ4: noise and NO2 by urban form, at the receptor level. Classes: municipality type/size, local population density (1 km),
distance to the nearest census road and its type. Exposures: NO2 FY2022 (v3), FY2015 and FY2023 (multi-year model, small-area
means assigned to receptors), night noise from the 2021 (R3) and 2015 (H27) census with two-way totals (same model as v4).
Usage: urban_form.py W OUT"""
import pandas as pd, numpy as np, geopandas as gpd, shapely, glob, os, sys
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W,OUT=sys.argv[1:3]; os.makedirs(OUT,exist_ok=True)
files=sorted(glob.glob(f'{W}/rec/r_*.parquet')); pcs=[os.path.basename(f)[2:4] for f in files]
# nearest census road (R3) distance and type
G=nl.load_sections(f'{W}/geo/traffic_census_2021_converted.parquet',f'{W}/nox_ef_nilim671.csv'); G=G[G.ok]
G['rclass']=np.where((G.motorway=='1')|G.rt.str[0].isin(['1','2']),'高速道路',np.where(G.rt.str[0]=='3','一般国道',np.where(G.rt.str[0].isin(['4','5','6']),'都道府県道','市道ほか')))
parts=[]
for f,pc in zip(files,pcs):
    R=pd.read_parquet(f); z=int(R.epsg.iloc[0]); Gz=G.to_crs(z); Gz=Gz[Gz.intersects(shapely.box(R.x.min(),R.y.min(),R.x.max(),R.y.max()).buffer(5000))]
    pts=shapely.points(R.x.values,R.y.values); tr=shapely.STRtree(Gz.geometry.values); idx=tr.nearest(pts)
    parts.append(pd.DataFrame({'KEY':R.KEY_CODE.astype(str).values,'w':R.w.values,'dist':shapely.distance(pts,Gz.geometry.values[idx]),'rclass':Gz.rclass.values[idx]}))
D=pd.concat(parts,ignore_index=True)
N21=pd.concat([pd.read_parquet(f'{W}/noise_rec_2021T/n_{p}.parquet') for p in pcs],ignore_index=True).Ln.values
N15=pd.concat([pd.read_parquet(f'{W}/noise_rec_2015T/n_{p}.parquet') for p in pcs],ignore_index=True).Ln.values
P=pd.read_parquet(f'{W}/receptor_no2_sat_2022.parquet'); Gr=pd.read_parquet(f'{W}/rec_grid.parquet',columns=['gpop1'])
D['L21']=np.clip(np.nan_to_num(N21,nan=40),40,None); D['L15']=np.clip(np.nan_to_num(N15,nan=40),40,None); D['no2']=P.no2.values; D['road']=P.road.values
SA=pd.read_csv(glob.glob(f'{W}/my/out/small_area_no2_2015_2023.csv')[0],dtype={'KEY_CODE':str}).set_index('KEY_CODE')
for fy in (2015,2023): D[f'no2_{fy}']=D.KEY.map(SA[f'NO2_{fy}']).values; D[f'road_{fy}']=D.KEY.map(SA[f'road_{fy}']).values
# municipality class
D['muni']=D.KEY.str[:5]; pref=D.muni.str[:2]; city=D.muni.str[2:].astype(int)
ward=(city>=101)&(city<=199)&(pref!='13'); tokyo=(pref=='13')&(city>=101)&(city<=123)
D['city']=np.where(ward,D.muni.str[:3]+'00',D.muni)
cpop=D.groupby('city').w.sum(); pop=D.city.map(cpop).values
D['ctype']=np.select([tokyo,ward,(city>=300)&~ward,pop>=300000,pop>=100000,pop>=50000],['東京23区','政令指定都市','町村','人口30万以上の市','10〜30万の市','5〜10万の市'],'5万未満の市')
dens=np.expm1(Gr.gpop1.values); D['dens']=pd.cut(dens,[-1,1000,4000,10000,1e9],labels=['<1,000人/km²','1,000〜4,000','4,000〜10,000','≥10,000'])
D['dband']=pd.cut(D.dist,[-1,20,50,100,200,1e9],labels=['<20 m','20〜50 m','50〜100 m','100〜200 m','≥200 m'])
ug=1.88
def wcorr(x,y,w):
    mx,my=np.average(x,weights=w),np.average(y,weights=w); v=np.sqrt(np.sum(w*(x-mx)**2)*np.sum(w*(y-my)**2)); return np.sum(w*(x-mx)*(y-my))/v if v>0 else np.nan
def summ(S):
    w=S.w.values; a=lambda c: np.average(S[c].values,weights=w)
    hiN=S.L21.values>=55; hiG=S.no2.values*ug>=20
    return pd.Series({'人口(百万)':w.sum()/1e6,'NO2 2022 ppb':a('no2'),'うち道路 ppb':a('road'),'NO2 2015':a('no2_2015'),'NO2 2023':a('no2_2023'),'NO2変化% 2015→2023':(a('no2_2023')/a('no2_2015')-1)*100,
        '道路分 2015':a('road_2015'),'道路分 2023':a('road_2023'),'道路分変化%':(a('road_2023')/a('road_2015')-1)*100,
        '夜間騒音 平均dB 2021':a('L21'),'夜間騒音 平均dB 2015':a('L15'),'騒音エネルギー変化% 2015→2021':(np.average(10**(S.L21.values/10),weights=w)/np.average(10**(S.L15.values/10),weights=w)-1)*100,
        '55dB以上%':100*w[hiN].sum()/w.sum(),'NO2 20µg以上%':100*w[hiG].sum()/w.sum(),'両方%':100*w[hiN&hiG].sum()/w.sum(),'騒音とNO2の相関':wcorr(S.L21.values,S.no2.values,w)})
out=[]
for var,lab in (('ctype','市区町村の種類'),('dens','人口密度（1 km）'),('dband','最寄りのセンサス道路までの距離'),('rclass','最寄りのセンサス道路の種類')):
    T=D.groupby(var,observed=True).apply(summ); T.index=[f'{lab}：{i}' for i in T.index]; out.append(T)
out.append(summ(D).to_frame('全国').T)
T=pd.concat(out); T.round(3).to_csv(f'{OUT}/urban_form_summary.csv',encoding='utf-8-sig'); pd.set_option('display.width',300); print(T.round(2).to_string())
# share of the doubly exposed population by class
both=(D.L21>=55)&(D.no2*ug>=20)
for var in ('ctype','dens','dband','rclass'):
    s=(D.w*both).groupby(D[var],observed=True).sum(); print(var,(100*s/s.sum()).round(1).to_dict())
