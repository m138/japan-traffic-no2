"""Build station table: annual mean NO2 / NOx (ppb) per station and FY with coordinates and station type."""
import pandas as pd, numpy as np, glob, io, sys
W=sys.argv[1]
def rd(f):
    return pd.read_csv(io.StringIO(open(f,'rb').read().decode('cp932',errors='replace')),dtype=str)
TM=pd.concat([rd(f) for f in glob.glob(f'{W}/air/x/TM*.txt')]); TM.columns=[c.strip() for c in TM.columns]
dms=lambda d,m,s: pd.to_numeric(TM[d])+pd.to_numeric(TM[m])/60+pd.to_numeric(TM[s])/3600
TM['lat']=dms('緯度_度','緯度_分','緯度_秒'); TM['lon']=dms('経度_度','経度_分','経度_秒')
TM=TM.rename(columns={'年度':'fy','国環研局番':'code'})[['fy','code','lat','lon','標高(m)']].drop_duplicates(['fy','code'])
rows=[]
for f in glob.glob(f'{W}/air/x/TD*.txt'):
    d=rd(f); d.columns=[c.strip() for c in d.columns]
    d=d.rename(columns={'測定年度':'fy','測定局コード':'code','項目コード_英数字':'item','測定局区分コード':'kind','用途地域名':'zone','市区町村コード':'muni','測定局名':'name'})
    d['hrs']=pd.to_numeric(d['測定時間(時間)'],errors='coerce'); d['val']=pd.to_numeric(d['年平均値(ppm)'],errors='coerce')*1000
    rows.append(d[['fy','code','name','muni','kind','zone','item','hrs','val']])
D=pd.concat(rows); D['item']=D.item.str.strip()
D=D[(D.hrs>=6000)&D.val.notna()]
P=D.pivot_table(index=['fy','code','name','muni','kind','zone'],columns='item',values='val').reset_index()
P=P.merge(TM,on=['fy','code'],how='left')
P['kind']=P.kind.map({'1':'gen','2':'road'}); P['muni']=P.muni.str.zfill(5)
print(P.groupby(['fy','kind']).agg(n=('code','size'),no2=('NO2','mean'),nolat=('lat',lambda x:x.isna().sum())))
P.to_csv(f'{W}/stations.csv',index=False)
