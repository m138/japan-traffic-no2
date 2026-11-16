"""Compare exposure statistics between the original receptor placement (uniform within small areas) and building-land placement
for the test prefectures. NO2 at moved points = original nearest receptor's non-road part + road part rescaled by the new road
kernel (road part is linear in sqrt(kernel), r=0.99996). Night noise recomputed with the same model.
Usage: popalloc_compare.py W W2 RELEASE_PARQUET OUT pref [pref ...]"""
import pandas as pd, numpy as np, glob, os, sys, shapely
from scipy.spatial import cKDTree
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
W,W2,RP,OUT=sys.argv[1:5]; PREFS=sys.argv[5:]; os.makedirs(OUT,exist_ok=True); ug=1.88
files=sorted(glob.glob(f'{W}/rec/r_*.parquet')); pcs=[os.path.basename(f)[2:4] for f in files]
n=[len(pd.read_parquet(f,columns=['w'])) for f in files]; off=dict(zip(pcs,np.cumsum([0]+n[:-1])))
D=pd.read_parquet(RP,columns=['KEY_CODE','pop_w','Lnight_2021','NO2_2022','road_2022'])
G=nl.load_sections(f'{W}/geo/traffic_census_2021_converted.parquet',f'{W}/nox_ef_nilim671.csv'); G=G[G.ok]
coef=[]
rows=[]; allp=[]
for pc in PREFS:
    R0=pd.read_parquet(f'{W}/rec/r_{pc}.parquet'); R1=pd.read_parquet(f'{W2}/rec/r_{pc}.parquet'); z=int(R0.epsg.iloc[0])
    o=off[pc]; S=D.iloc[o:o+len(R0)].reset_index(drop=True); assert (S.KEY_CODE.values==R0.KEY_CODE.values).all()
    C0=pd.read_parquet(f'{W}/road/c_{pc}.parquet').C1000_h20.values; C1=pd.read_parquet(f'{W2}/road/c_{pc}.parquet').C1000_h20.values
    m=C0>0; k=np.median(S.road_2022.values[m]/np.sqrt(C0[m])); coef.append(k)
    bg=S.NO2_2022.values-S.road_2022.values
    j=cKDTree(np.c_[R0.x,R0.y]).query(np.c_[R1.x,R1.y])[1]          # nearest original receptor for the smooth (non-road) part
    no2_1=bg[j]+k*np.sqrt(C1); no2_0=S.NO2_2022.values
    L1=np.clip(np.nan_to_num(pd.read_parquet(f'{W2}/noise_rec/n_{pc}.parquet').Ln.values,nan=40),40,None); L0=S.Lnight_2021.values
    Gz=G.to_crs(z); Gz=Gz[Gz.intersects(shapely.box(R0.x.min(),R0.y.min(),R0.x.max(),R0.y.max()).buffer(5000))]; tr=shapely.STRtree(Gz.geometry.values)
    d0=shapely.distance(shapely.points(R0.x,R0.y),Gz.geometry.values[tr.nearest(shapely.points(R0.x,R0.y))])
    d1=shapely.distance(shapely.points(R1.x,R1.y),Gz.geometry.values[tr.nearest(shapely.points(R1.x,R1.y))])
    city=S.KEY_CODE.str[2:5].astype(int).values; big=(city>=101)&(city<=199)
    for lab,L,N,dd in (('元の配置（小地域内に一様）',L0,no2_0,d0),('建物用地に配分',L1,no2_1,d1)):
        w=S.pop_w.values; b=(L>=55)&(N*ug>=20)
        rows.append({'都府県':pc,'配置':lab,'人口(万)':w.sum()/1e4,'道路50m以内 %':100*w[dd<50].sum()/w.sum(),'道路100m以内 %':100*w[dd<100].sum()/w.sum(),
            '平均Lnight':np.average(L,weights=w),'55dB以上 %':100*w[L>=55].sum()/w.sum(),'NO2平均 ppb':np.average(N,weights=w),'NO2 20µg以上 %':100*w[N*ug>=20].sum()/w.sum(),
            '二重暴露 %':100*w[b].sum()/w.sum(),'二重暴露のうち50m以内 %':100*w[b&(dd<50)].sum()/max(w[b].sum(),1),'二重暴露のうち100m以内 %':100*w[b&(dd<100)].sum()/max(w[b].sum(),1),
            '二重暴露のうち区部（23区・政令市の区）%':100*w[b&big].sum()/max(w[b].sum(),1),
            '騒音とNO2の相関':np.corrcoef(L,N)[0,1] if False else (lambda x,y: np.sum(w*(x-np.average(x,weights=w))*(y-np.average(y,weights=w)))/np.sqrt(np.sum(w*(x-np.average(x,weights=w))**2)*np.sum(w*(y-np.average(y,weights=w))**2)))(L,N)})
    allp.append(pd.DataFrame({'pc':pc,'big':big,'w':S.pop_w.values,'L0':L0,'L1':L1,'N0':no2_0,'N1':no2_1,'d0':d0,'d1':d1}))
T=pd.DataFrame(rows); A=pd.concat(allp)
for lab,(L,N,dd) in (('元の配置（小地域内に一様）',('L0','N0','d0')),('建物用地に配分',('L1','N1','d1'))):
    w=A.w.values; b=(A[L]>=55)&(A[N]*ug>=20)
    wc=lambda x,y: np.sum(w*(x-np.average(x,weights=w))*(y-np.average(y,weights=w)))/np.sqrt(np.sum(w*(x-np.average(x,weights=w))**2)*np.sum(w*(y-np.average(y,weights=w))**2))
    T.loc[len(T)]={'都府県':'全国計' if len(PREFS)==47 else f'{len(PREFS)}都道府県計','配置':lab,'人口(万)':w.sum()/1e4,'道路50m以内 %':100*w[A[dd]<50].sum()/w.sum(),'道路100m以内 %':100*w[A[dd]<100].sum()/w.sum(),
       '平均Lnight':np.average(A[L],weights=w),'55dB以上 %':100*w[A[L]>=55].sum()/w.sum(),'NO2平均 ppb':np.average(A[N],weights=w),'NO2 20µg以上 %':100*w[A[N]*ug>=20].sum()/w.sum(),
       '二重暴露 %':100*w[b].sum()/w.sum(),'二重暴露のうち50m以内 %':100*w[b&(A[dd]<50)].sum()/w[b].sum(),'二重暴露のうち100m以内 %':100*w[b&(A[dd]<100)].sum()/w[b].sum(),
       '二重暴露のうち区部（23区・政令市の区）%':100*w[b&A.big.values].sum()/w[b].sum(),'騒音とNO2の相関':wc(A[L].values,A[N].values),'二重暴露 (万人)':w[b].sum()/1e4}
T.round(2).to_csv(f'{OUT}/popalloc_building_comparison.csv',index=False,encoding='utf-8-sig')
pd.set_option('display.width',300); print('road coef',np.round(coef,4)); print(T.round(2).to_string())
