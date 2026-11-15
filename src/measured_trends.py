"""Measured trends FY2015-2023: roadside night noise (continuous road-traffic-noise monitoring, tenbou gisplus tn{year}.zip, night Leq)
vs NO2 (air monitoring, roadside and general stations). Within-site change via site fixed effects with year dummies (sites measured
in >=2 years). Usage: measured_trends.py TN_DIR STATIONS_CSV OUT"""
import pandas as pd, numpy as np, glob, io, sys, statsmodels.formula.api as smf
TN,ST,OUT=sys.argv[1:4]
def rd(f):
    b=open(f,'rb').read()
    for enc in ('utf-8-sig','cp932'):
        try: return pd.read_csv(io.StringIO(b.decode(enc)),dtype=str)
        except Exception: pass
M=pd.concat([rd(f) for f in sorted(glob.glob(f'{TN}/tn*/**/souon*.csv',recursive=True))]); M.columns=[c.strip() for c in M.columns]
M['fy']=M.report_year.astype(int); M['L']=pd.to_numeric(M.night_leq,errors='coerce'); M['Ld']=pd.to_numeric(M.day_leq,errors='coerce')
M['site']=M.pref_code.str.zfill(2)+'_'+M.jis_code.str.zfill(5)+'_'+M.noise_measure_point_code.str.strip()
M=M.dropna(subset=['L']).drop_duplicates(['site','fy'])
P=pd.read_csv(ST,dtype={'code':str}); P=P[(P.fy>=2015)&(P.fy<=2023)&(P.NO2>0)].dropna(subset=['NO2'])   # a few general stations report 0
def fe(df,y,unit,log=False):
    """Site fixed effects by within transformation (site-demeaned outcome on site-demeaned year dummies), SE clustered by site."""
    import statsmodels.api as sm
    d=df.copy(); n=d.groupby(unit).fy.transform('nunique'); d=d[n>=2].reset_index(drop=True); d['yv']=np.log(d[y]) if log else d[y]
    X=pd.get_dummies(d.fy,prefix='fy',dtype=float).drop(columns='fy_2015'); Xd=X-X.groupby(d[unit]).transform('mean'); yd=d.yv-d.groupby(unit).yv.transform('mean')
    m=sm.OLS(yd.values,Xd.values).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d[unit])[0]}); m.params=pd.Series(m.params,index=Xd.columns); m.bse=pd.Series(m.bse,index=Xd.columns)
    out=[{'fy':2015,'est':0.0,'lo':0.0,'hi':0.0}]
    for fy in range(2016,2024):
        k=f'fy_{fy}'; b=m.params[k]; s=m.bse[k]; out.append({'fy':fy,'est':b,'lo':b-1.96*s,'hi':b+1.96*s})
    o=pd.DataFrame(out)
    if log: o[['est','lo','hi']]=(np.exp(o[['est','lo','hi']])-1)*100
    o['n_sites']=d[unit].nunique(); o['n_obs']=len(d); return o
R=[]
for lab,df,y,u,lg in (('夜間騒音（実測、dB差）',M,'L','site',False),('昼間騒音（実測、dB差）',M.dropna(subset=['Ld']),'Ld','site',False),
                       ('NO2 自排局（%変化）',P[P.kind=='road'],'NO2','code',True),('NO2 一般局（%変化）',P[P.kind=='gen'],'NO2','code',True)):
    o=fe(df,y,u,lg); o.insert(0,'series',lab); R.append(o)
R=pd.concat(R); R.round(3).to_csv(f'{OUT}/measured_trends_2015_2023.csv',index=False,encoding='utf-8-sig')
pd.set_option('display.width',200); print(R.pivot_table(index='fy',columns='series',values='est',sort=False).round(2).to_string())
print(R.groupby('series')[['n_sites','n_obs']].first())
# night noise as acoustic energy change for comparison with NO2 (%)
L=R[R.series.str.startswith('夜間')].set_index('fy'); print('night noise 2023 vs 2015: %.2f dB (%.2f to %.2f) = energy %+.1f%%'%(L.est[2023],L.lo[2023],L.hi[2023],(10**(L.est[2023]/10)-1)*100))
