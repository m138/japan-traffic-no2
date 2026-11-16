"""Night-noise search distance 200 m (main) vs 500 m: double high exposure, concentration, variance decomposition between/within
municipalities, correlation by level and small-area-mean classification. 500 m noise from noise_at_receptors.py with NOISE_CUT=500.
Usage: cutoff_sensitivity.py W RELEASE_PARQUET NEAREST_ROAD_PARQUET OUT"""
import pandas as pd, numpy as np, glob, sys, os
W,RP,NR,OUT=sys.argv[1:5]; os.makedirs(OUT,exist_ok=True); ug=1.88
pcs=[f[-10:-8] for f in sorted(glob.glob(f'{W}/rec/r_*.parquet'))]
L5=np.clip(np.nan_to_num(np.concatenate([pd.read_parquet(f'{W}/noise_rec_c500/n_{p}.parquet').Ln.values for p in pcs]),nan=40),40,None)
D=pd.read_parquet(RP,columns=['KEY_CODE','pop_w','Lnight_2021','NO2_2022']); d=pd.read_parquet(NR).dist.values
w=D.pop_w.values; W_=w.sum(); G=D.NO2_2022.values; muni=D.KEY_CODE.str[:5].values
city=D.KEY_CODE.str[2:5].astype(int).values; big=(city>=101)&(city<=199)
def wc(x,y,ww): mx,my=np.average(x,weights=ww),np.average(y,weights=ww); return np.sum(ww*(x-mx)*(y-my))/np.sqrt(np.sum(ww*(x-mx)**2)*np.sum(ww*(y-my)**2))
def gm(key,v):
    s=pd.DataFrame({'k':key,'w':w,'wv':w*v}).groupby('k')[['w','wv']].sum(); return pd.Series(key).map(s.wv/s.w).values, s
rows=[]
for lab,L in (('200 m（主解析）',D.Lnight_2021.values),('500 m',L5)):
    b=(L>=55)&(G*ug>=20)
    Lm,_=gm(muni,L); Gm,_=gm(muni,G); Lb,Gb=np.average(L,weights=w),np.average(G,weights=w)
    vLW,vLB=np.sum(w*(L-Lm)**2),np.sum(w*(Lm-Lb)**2); vGW,vGB=np.sum(w*(G-Gm)**2),np.sum(w*(Gm-Gb)**2); cW,cB=np.sum(w*(L-Lm)*(G-Gm)),np.sum(w*(Lm-Lb)*(Gm-Gb))
    SL,sa=gm(D.KEY_CODE.values,L); SG,_=gm(D.KEY_CODE.values,G); a=(SL>=55)&(SG*ug>=20)
    mL=pd.DataFrame({'k':muni,'w':w,'a':w*L,'g':w*G}).groupby('k').sum()
    rows.append({'探索範囲':lab,'55dB以上 %':100*w[L>=55].sum()/W_,'二重高暴露 %':100*w[b].sum()/W_,'二重高暴露 万人':w[b].sum()/1e4,
      'うち23区・政令市 %':100*w[b&big].sum()/w[b].sum(),'うち道路50m以内 %':100*w[b&(d<50)].sum()/w[b].sum(),'うち道路100m以内 %':100*w[b&(d<100)].sum()/w[b].sum(),
      '地点相関':wc(L,G,w),'市区町村平均の相関':wc((mL.a/mL.w).values,(mL.g/mL.w).values,mL.w.values),'市区町村内相関':cW/np.sqrt(vLW*vGW),
      '騒音分散の市区町村間 %':100*vLB/(vLB+vLW),'NO2分散の市区町村間 %':100*vGB/(vGB+vGW),'共分散の市区町村間 %':100*cB/(cB+cW),
      '小地域平均で判定 %':100*w[a].sum()/W_,'地点該当のうち小地域平均で非該当 %':100*w[b&~a].sum()/w[b].sum(),'地点非該当で小地域平均のみ該当（全人口比）%':100*w[~b&a].sum()/W_})
T=pd.DataFrame(rows).set_index('探索範囲').T; T.round(2).to_csv(f'{OUT}/cutoff_500m_comparison.csv',encoding='utf-8-sig'); print(T.round(2).to_string())
