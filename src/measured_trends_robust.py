"""Robustness of the measured trends: sites and observations by year, and site fixed-effect trends restricted to
(a) all sites measured >=2 years (main), (b) sites measured in both FY2015 and FY2023, (c) sites measured in >=5 years,
(d) NO2 stations reporting all 9 years. Usage: measured_trends_robust.py TN_DIR STATIONS_CSV OUT"""
import pandas as pd, numpy as np, glob, io, sys, os, statsmodels.api as sm
TN,ST,OUT=sys.argv[1:4]; os.makedirs(OUT,exist_ok=True)
def rd(f):
    b=open(f,'rb').read()
    for enc in ('utf-8-sig','cp932'):
        try: return pd.read_csv(io.StringIO(b.decode(enc)),dtype=str)
        except Exception: pass
M=pd.concat([rd(f) for f in sorted(glob.glob(f'{TN}/tn*/**/souon*.csv',recursive=True))]); M.columns=[c.strip() for c in M.columns]
M['fy']=M.report_year.astype(int); M['L']=pd.to_numeric(M.night_leq,errors='coerce')
M['site']=M.pref_code.str.zfill(2)+'_'+M.jis_code.str.zfill(5)+'_'+M.noise_measure_point_code.str.strip()
M=M.dropna(subset=['L']).drop_duplicates(['site','fy'])
P=pd.read_csv(ST,dtype={'code':str}); P=P[(P.fy>=2015)&(P.fy<=2023)&(P.NO2>0)].dropna(subset=['NO2'])
def fe(d,y,unit,log=False):
    d=d.copy(); n=d.groupby(unit).fy.transform('nunique'); d=d[n>=2].reset_index(drop=True); d['yv']=np.log(d[y]) if log else d[y]
    X=pd.get_dummies(d.fy,prefix='fy',dtype=float).drop(columns='fy_2015'); Xd=X-X.groupby(d[unit]).transform('mean'); yd=d.yv-d.groupby(unit).yv.transform('mean')
    m=sm.OLS(yd.values,Xd.values).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d[unit])[0]}); i=list(Xd.columns).index('fy_2023')
    b,s=m.params[i],m.bse[i]; f=(lambda v:(np.exp(v)-1)*100) if log else (lambda v:v)
    pre=list(Xd.columns).index('fy_2019'); return {'2019':f(m.params[pre]),'2023':f(b),'lo':f(b-1.96*s),'hi':f(b+1.96*s),'sites':d[unit].nunique(),'obs':len(d)}
yrs=M.groupby('site').fy.agg(set)
cnt=pd.DataFrame({'地点数（夜間騒音）':M.groupby('fy').site.nunique(),'うち前年度にも測定':[len(set(M.site[M.fy==y])&set(M.site[M.fy==y-1])) for y in sorted(M.fy.unique())]})
cnt['自排局']=P[P.kind=='road'].groupby('fy').code.nunique(); cnt['一般局']=P[P.kind=='gen'].groupby('fy').code.nunique()
cnt.to_csv(f'{OUT}/sites_by_year.csv',encoding='utf-8-sig'); print(cnt.to_string())
print('noise obs per site:',M.groupby('site').size().describe().round(2).to_dict())
rows=[]
both=yrs[yrs.apply(lambda s:{2015,2023}<=s)].index; five=yrs[yrs.apply(len)>=5].index
for lab,d in (('全地点（2回以上）',M),('2015と2023の両年度に測定',M[M.site.isin(both)]),('5年度以上測定',M[M.site.isin(five)])):
    rows.append({'系列':'夜間騒音（dB差）','標本':lab,**fe(d,'L','site')})
for k,lab in (('road','NO2 自排局（%）'),('gen','NO2 一般局（%）')):
    d=P[P.kind==k]; full=d.groupby('code').fy.nunique(); rows.append({'系列':lab,'標本':'全局（2回以上）',**fe(d,'NO2','code',True)})
    rows.append({'系列':lab,'標本':'9年度すべて測定',**fe(d[d.code.isin(full[full==9].index)],'NO2','code',True)})
T=pd.DataFrame(rows); T.round(3).to_csv(f'{OUT}/measured_trends_robustness.csv',index=False,encoding='utf-8-sig'); print(T.round(2).to_string())
