"""Sensitivity of the v2 main model to emission assumptions (station level, LOPO CV with kriging).
For each case the section NOx emissions are recomputed, then the road kernel (C1000_h20) and the emission densities (gE3, gE10, 250 m grid,
disk means) at stations; population densities and the satellite term are unchanged. Usage: sens_emission.py W OUT"""
import pandas as pd, numpy as np, geopandas as gpd, sys, os, warnings, statsmodels.api as sm
from scipy.signal import oaconvolve
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
sys.path.insert(0,os.path.dirname(__file__)); import no2_lib as nl
warnings.filterwarnings('ignore'); W,OUT=sys.argv[1:3]; EFP=f'{W}/nox_ef_nilim671.csv'
G=gpd.read_parquet(f'{W}/geo/traffic_census_2021_converted.parquet').drop_duplicates('census')   # attributes only (same order as load_sections)
base=nl.load_sections(f'{W}/geo/traffic_census_2021_converted.parquet',EFP)
n=lambda c: pd.to_numeric(G[c],errors='coerce').values
vs=lambda a,b: pd.DataFrame({'a':n(a),'b':n(b)}).replace(0,np.nan).mean(axis=1).values
V_off_s=vs('昼間（非混雑時）／上り／旅行速度／小型車（ｋｍ／ｈ）','昼間（非混雑時）／下り／旅行速度／小型車（ｋｍ／ｈ）')
V_off_l=vs('昼間（非混雑時）／上り／旅行速度／大型車（ｋｍ／ｈ）','昼間（非混雑時）／下り／旅行速度／大型車（ｋｍ／ｈ）')
V_pk=vs('朝夕（混雑時）／上り／旅行速度／合計（ｋｍ／ｈ）','朝夕（混雑時）／下り／旅行速度／合計（ｋｍ／ｈ）')
st=(base.motorway=='1')|base.rt.isin(['1','2']); V_lim=np.where(st,base.vlim.where(base.vlim>0).fillna(80),base.vlim.where(base.vlim>0).fillna(40))
def emis(year=2020,Vs=None,Vl=None,fl=1.0,fs=1.0):
    Vs=base.Vs.values if Vs is None else np.where(np.isfinite(Vs),Vs,base.Vs.values)
    Vl=base.Vl.values if Vl is None else np.where(np.isfinite(Vl),Vl,base.Vl.values)
    return base.v24s.fillna(0).values*nl.ef(Vs,'small',year,EFP)*fs+base.v24l.fillna(0).values*nl.ef(Vl,'large',year,EFP)*fl
mean_ef=lambda veh: np.average(nl.ef(base.Vs.values if veh=='small' else base.Vl.values,veh,2020,EFP),weights=(base.v24s if veh=='small' else base.v24l).fillna(0))
CASES={'E0 主解析（2020年次、昼間12時間速度）':emis(),
 'E1 排出係数2015年次':emis(2015),'E2 排出係数2025年次':emis(2025),
 'E3 速度＝非混雑時':emis(Vs=V_off_s,Vl=np.where(np.isfinite(V_off_l),V_off_l,V_off_s)),
 'E4 速度＝混雑時':emis(Vs=V_pk,Vl=V_pk),'E5 速度＝指定最高速度':emis(Vs=V_lim,Vl=V_lim),
 'E6 大型車の係数×1.5':emis(fl=1.5),'E7 大型車の係数×0.5':emis(fl=0.5),
 'E8 速度・車種によらず一定（台数だけ）':base.v24s.fillna(0).values*mean_ef('small')+base.v24l.fillna(0).values*mean_ef('small')}
P=pd.read_csv(f'{W}/stations_all.csv',dtype={'code':str,'muni':str}); U=P.drop_duplicates('code')[['code','lat','lon']].reset_index(drop=True)
SG0=pd.read_csv(f'{W}/station_grid.csv',dtype={'code':str})[['code','gpop1','gpop3','gpop10']]; SAT=pd.read_csv(f'{W}/station_sat.csv',dtype={'code':str})
RES=0.25; xy=lambda lon,lat: ((lon-137)*np.cos(np.radians(36))*111.32,(lat-36)*110.57)
X0=xy(122,0)[0]; Y0=xy(0,24)[1]; NX=int((xy(154,0)[0]-X0)/RES)+1; NY=int((xy(0,46)[1]-Y0)/RES)+1
cell=lambda gx,gy: (np.clip(((gy-Y0)/RES).astype(int),0,NY-1),np.clip(((gx-X0)/RES).astype(int),0,NX-1))
def disk(r):
    k=int(np.ceil(r/RES)); y,x=np.mgrid[-k:k+1,-k:k+1]; m=((x*RES)**2+(y*RES)**2<=r*r).astype(np.float32); return m/(m.sum()*RES*RES)
# geometry prepared once: 25 m points per segment, in lon/lat grid cells, carrying section index and length share
ok=base.ok.values; idx=np.where(ok)[0]; geo=gpd.GeoSeries(base.geometry.values[idx],crs=base.crs)
pts_cache=[]
for z in np.unique(geo.centroid.x.map(nl.utm_epsg)):
    m=(geo.centroid.x.map(nl.utm_epsg)==z).values; kz=geo[m].to_crs(z)
    seg=nl.segments(kz.values,idx[m]); L=np.hypot(seg.bx-seg.ax,seg.by-seg.ay).values; nn=np.maximum(1,np.ceil(L/25).astype(int))
    rep=np.repeat(np.arange(len(seg)),nn); t=(np.concatenate([np.arange(k) for k in nn])+0.5)/np.repeat(nn,nn)
    px=seg.ax.values[rep]+t*(seg.bx.values[rep]-seg.ax.values[rep]); py=seg.ay.values[rep]+t*(seg.by.values[rep]-seg.ay.values[rep])
    ll=gpd.GeoSeries(gpd.points_from_xy(px,py),crs=z).to_crs(4326); iy,ix=cell(*xy(ll.x.values,ll.y.values))
    pts_cache.append((seg.id.values[rep].astype(int),(L[rep]/nn[rep]/1000),iy,ix))
sgx,sgy=xy(U.lon.values,U.lat.values); siy,six=cell(sgx,sgy)
def gp(k=None): return GPR(k if k is not None else Ck(4.0)*Matern(length_scale=30.0,length_scale_bounds=(2,500),nu=1.5)+WhiteKernel(2.0),optimizer=None if k is not None else 'fmin_l_bfgs_b')
rows=[]; coefs=[]; tot=[]
for name,q in CASES.items():
    K=base.copy(); K['nox_gkmd']=q; K['id']=K.id.astype(str)
    Kg=gpd.GeoDataFrame(K[K.ok],geometry='geometry',crs=base.crs)
    feats=[]
    for z,u in U.groupby(U.lon.map(nl.utm_epsg)):
        pts=gpd.GeoSeries(gpd.points_from_xy(u.lon,u.lat),crs=4326).to_crs(z); px,py=pts.x.values,pts.y.values
        Kz=Kg.to_crs(z); Kz=Kz[Kz.intersects(gpd.GeoSeries.from_xy([px.min(),px.max()],[py.min(),py.max()]).union_all().envelope.buffer(2000))]
        f=nl.road_field(px,py,Kz,1000,[20]); feats.append(pd.DataFrame({'code':u.code.values,'C1000_h20':f[20]}))
    F=pd.concat(feats)
    Em=np.zeros((NY,NX),np.float32)
    for sid,lw,iy,ix in pts_cache: np.add.at(Em,(iy,ix),(q[sid]*lw).astype(np.float32))
    F=F.merge(pd.DataFrame({'code':U.code.values,'gE3':np.log1p(oaconvolve(Em,disk(3),mode='same').clip(min=0)[siy,six]),'gE10':np.log1p(oaconvolve(Em,disk(10),mode='same').clip(min=0)[siy,six])}),on='code')
    F=F.merge(SG0,on='code').merge(SAT,on='code'); tot.append({'case':name,'NOx_t_per_yr':q[ok].sum()*0+np.nansum(q[ok]*0)+Em.sum()*365/1e6})
    for fy in (2021,2022):
        D=P[P.fy==fy].merge(F,on='code').dropna(subset=['NO2','sat3']).reset_index(drop=True); D['pref']=D.muni.str[:2]
        X=sm.add_constant(pd.DataFrame({'lE10':D.gE10,'lE3':D.gE3,'lp1':D.gpop1,'lp3':D.gpop3,'lp10':D.gpop10,'sCr':np.sqrt(D.C1000_h20),'sat':D.sat3}))
        y=D.NO2.values; xyk=np.c_[(D.lon-137)*np.cos(np.radians(36))*111.32,(D.lat-36)*110.57]; full=sm.OLS(y,X).fit()
        K0=gp().fit(xyk,y-full.predict(X)).kernel_; pred=np.full(len(D),np.nan)
        for p in D.pref.unique():
            te=(D.pref==p).values; tr=~te; m=sm.OLS(y[tr],X[tr]).fit()
            pred[te]=m.predict(X[te])+gp(K0).fit(xyk[tr],y[tr]-m.predict(X[tr])).predict(xyk[te])
        for k in ['gen','road','all']:
            s=(D.kind==k).values if k!='all' else np.ones(len(D),bool); e=pred[s]-y[s]
            rows.append({'case':name,'fy':fy,'kind':k,'n':int(s.sum()),'R2':1-np.sum(e**2)/np.sum((y[s]-y[s].mean())**2),'RMSE':np.sqrt(np.mean(e**2)),'bias':e.mean()})
        # road share at stations: mean of b_sCr*sqrt(C) (for scale comparison across cases)
        coefs.append({'case':name,'fy':fy,**full.params.round(4).to_dict(),'road_part_mean_gen':float(np.mean(full.params['sCr']*np.sqrt(D.C1000_h20[D.kind=='gen']))),'road_part_mean_road':float(np.mean(full.params['sCr']*np.sqrt(D.C1000_h20[D.kind=='road'])))})
    print(name,'done',flush=True)
R=pd.DataFrame(rows); R.round(4).to_csv(f'{OUT}/sens_emission_cv.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(coefs).round(4).to_csv(f'{OUT}/sens_emission_coef.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(tot).round(0).to_csv(f'{OUT}/sens_emission_totals.csv',index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); print(pd.DataFrame(tot).round(0).to_string())
print(R.pivot_table(index=['fy','case'],columns='kind',values=['R2','RMSE'],sort=False).round(3).to_string())
print(pd.DataFrame(coefs)[['case','fy','sCr','sat','road_part_mean_gen','road_part_mean_road']].round(3).to_string())
