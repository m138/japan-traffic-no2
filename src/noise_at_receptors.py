"""Night road noise (paper A model v4: ASJ RTN-Model 2018 simplified, census roads, 200 m, region/road-type bias correction, 40 dB floor)
computed at the NO2 receptors, so that both exposures are at the same points. Setup copied from japan-night-road-noise/src/noise_national.py.
Usage: noise_at_receptors.py W NOISE_LIB_DIR NZ_DIR pref [pref ...]
Environment NOISE_NET=2015|2021 (default 2021, directional volumes as in v4) or NOISE_NET=2015T|2021T (two-way totals, needed for H27)   (NZ_DIR has census27.csv, national_validation_points.csv, geo/)"""
import pandas as pd, numpy as np, shapely, sys, os, time
sys.path.insert(0,sys.argv[2]); import noise_lib as nl
W,_,S=sys.argv[1:4]; PREFS=sys.argv[4:]; H=4.0; CUT=200.0
REG={**{p:'北海道・東北' for p in ['01','02','03','04','05','06','07']},**{p:'関東' for p in ['08','09','10','11','12','13','14']},**{p:'中部' for p in ['15','16','17','18','19','20','21','22','23']},
     **{p:'近畿' for p in ['24','25','26','27','28','29','30']},**{p:'中国・四国' for p in ['31','32','33','34','35','36','37','38','39']},**{p:'九州・沖縄' for p in ['40','41','42','43','44','45','46','47']}}
V=pd.read_csv(f'{S}/national_validation_points.csv',dtype={'pref':str})
bias_gen=V[V.grp=='gen'].groupby(V.pref.map(REG)).err.mean().to_dict(); V['rtc']=V.rt.astype(str).str[0].where(V.rt.astype(str).str[0].isin(['1','2']),'other')
bias_mw=V[V.grp=='mw'].groupby('rtc').err.mean().to_dict()
prof=nl.night_profile(f'{S}/census27.csv')
NET=os.environ.get('NOISE_NET','2021'); TAG='' if NET=='2021' else '_'+NET
if NET.endswith('T'):
    import geopandas as gpd
    G=gpd.read_parquet(f'{S}/geo/traffic_census_{NET[:4]}_converted.parquet'); n=lambda c: pd.to_numeric(G[c],errors='coerce')
    K=pd.DataFrame({'id':G.census.astype(str),'rt':G['道路種別'].astype(str).str.strip(),'muni':G['市区町村コード'].astype(str).str.zfill(5),
      'motorway':G['自動車専用道路の別'].astype(str).str.strip(),'oneway':G['一方通行フラグ'].astype(str).str.strip(),
      'u12s':n('昼間１２時間自動車類交通量（上下合計）／小型車（台）'),'u12l':n('昼間１２時間自動車類交通量（上下合計）／大型車（台）'),
      'u24s':n('２４時間自動車類交通量（上下合計）／小型車（台）'),'u24l':n('２４時間自動車類交通量（上下合計）／大型車（台）'),
      'd12s':np.nan,'d12l':np.nan,'d24s':np.nan,'d24l':np.nan,
      'roadw':n('幅員構成／道路部幅員（ｍ）'),'carw':n('幅員構成／車道部幅員（ｍ）'),'medw':n('幅員構成／中央帯幅員（ｍ）'),'vlim':n('指定最高速度（ｋｍ／ｈ）')})
    K=gpd.GeoDataFrame(K,geometry=G.geometry.values,crs=G.crs)   # night energy is linear in volume, so two-way totals give the same S
else: K=nl.load_sections(f'{S}/geo/traffic_census_2021_converted.parquet')
SRC=nl.source_table(K.drop_duplicates('id'),prof).set_index('id')
K=K[K.id.isin(SRC.index[SRC.ok])].dissolve(by='id').reset_index(); K['pref']=K.muni.str[:2]
rtc=K.rt.astype(str).str[0].where(K.rt.astype(str).str[0].isin(['1','2']),'other')
b=np.where(SRC.loc[K.id,'grp'].values=='mw',rtc.map(bias_mw).values,K.pref.map(REG).map(bias_gen).fillna(np.mean(list(bias_gen.values()))).values)
K['S']=SRC.loc[K.id,'S'].values*10**(-b/10); K['off']=SRC.loc[K.id,'off'].values; K['geometry']=K.geometry.simplify(0.00002)
os.makedirs(f'{W}/noise_rec{TAG}',exist_ok=True)
for pc in PREFS:
    o=f'{W}/noise_rec{TAG}/n_{pc}.parquet'
    if os.path.exists(o): continue
    t0=time.time(); R=pd.read_parquet(f'{W}/rec/r_{pc}.parquet'); z=int(R.epsg.iloc[0]); px,py=R.x.values,R.y.values
    Kz=K.to_crs(z); Kz=Kz[Kz.intersects(shapely.box(px.min(),py.min(),px.max(),py.max()).buffer(CUT+100))]
    pts=shapely.points(px,py); rtree=shapely.STRtree(pts); SG=nl.segments(Kz.geometry.values,Kz.id.values); src=Kz.set_index('id'); E=np.zeros(len(pts))
    for i0 in range(0,len(SG),20000):
        sg=SG.iloc[i0:i0+20000]; lines=shapely.linestrings(np.stack([np.c_[sg.ax,sg.ay],np.c_[sg.bx,sg.by]],axis=1))
        si,ri=rtree.query(lines,predicate='dwithin',distance=CUT)
        if len(si)==0: continue
        np.add.at(E,ri,nl.seg_energy(sg.ax.values[si],sg.ay.values[si],sg.bx.values[si],sg.by.values[si],px[ri],py[ri],src.S.loc[sg.id.values[si]].values,src.off.loc[sg.id.values[si]].values,H))
    Ln=np.where(E>0,10*np.log10(np.maximum(E,1e-30)),np.nan)
    pd.DataFrame({'Ln':Ln.astype(np.float32)}).to_parquet(o); print(pc,len(R),round(time.time()-t0),'s',flush=True)
