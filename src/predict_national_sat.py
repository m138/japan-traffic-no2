"""Predict annual NO2 (FY model with satellite background, main model from 2026-11-14) at all receptors; aggregate to small areas, municipalities, prefectures, secondary medical areas."""
import pandas as pd, numpy as np, glob, sys, os, pickle
W=sys.argv[1]; FY=int(sys.argv[2]) if len(sys.argv)>2 else 2022
M=pickle.load(open(f'{W}/final_model_sat_{FY}.pkl','rb')); b=M['params']; gp=M['gp']
G=pd.read_parquet(f'{W}/rec_grid.parquet'); SAT=pd.read_parquet(f'{W}/rec_sat.parquet').sat3.values; LL=pd.read_parquet(f'{W}/rec_ll.parquet')
files=sorted(glob.glob(f'{W}/rec/r_*.parquet')); i0=0; parts=[]
for f in files:
    pc=os.path.basename(f)[2:4]; R=pd.read_parquet(f); n=len(R); g=G.iloc[i0:i0+n].reset_index(drop=True); ll=LL.iloc[i0:i0+n]; i0+=n
    C=pd.read_parquet(f'{W}/road/c_{pc}.parquet').C1000_h20.values.astype(float)
    road=b['sCr']*np.sqrt(C)
    lin=b['const']+b['lE10']*g.gE10+b['lE3']*g.gE3+b['lp1']*g.gpop1+b['lp3']*g.gpop3+b['lp10']*g.gpop10+road+b['sat']*SAT[i0-n:i0]
    gx=np.c_[(ll.lon.values-137)*np.cos(np.radians(36))*111.32,(ll.lat.values-36)*110.57]
    res=np.concatenate([gp.predict(gx[j:j+20000]) for j in range(0,n,20000)])
    no2=np.clip(lin.values+res,0.5,None)
    parts.append(pd.DataFrame({'KEY_CODE':R.KEY_CODE.values,'PREF':pc,'w':R.w.values,'no2':no2.astype(np.float32),'road':road.astype(np.float32)}))
    print(pc,n,round(np.average(no2,weights=R.w),2),flush=True)
P=pd.concat(parts); P.to_parquet(f'{W}/receptor_no2_sat_{FY}.parquet')
ug=1.88
for lab,t in (('w10ug',10/ug),('w20ug',20/ug),('w40ug',40/ug)): P[lab]=P.w*(P.no2>=t)
P['wN']=P.w*P.no2; P['wR']=P.w*P.road
S=P.groupby('KEY_CODE')[['w','wN','wR','w10ug','w20ug','w40ug']].sum().reset_index()
S.to_csv(f'{W}/small_area_no2_sat_{FY}.csv',index=False)
tot=P.w.sum(); print('national pop-weighted NO2 ppb',round(P.wN.sum()/tot,2),' road part',round(P.wR.sum()/tot,2),' >=10ug',round(P.w10ug.sum()/tot,3),' >=20ug',round(P.w20ug.sum()/tot,3),' >=40ug',round(P.w40ug.sum()/tot,4))
