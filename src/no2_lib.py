"""Road NOx emissions from the 2021 road traffic census and a line-source dispersion kernel (shared with the noise model geometry)."""
import pandas as pd, numpy as np, geopandas as gpd, shapely
def ef(V,veh,year=2020,path=None):
    E=pd.read_csv(path); r=E[(E.year==year)&(E.veh==veh)].iloc[0]
    V=np.clip(V,20,110 if veh=='small' else 90)
    return r.A/V+r.B*V+r.C*V**2+r.D          # g/km/vehicle
def load_geom(gp):
    """Section geometry with all pieces of a census section merged (the converted census file stores a section as several rows).
    Before v3 (2026-11-14) only the first row was used, which dropped about 29% of the R3 road length."""
    import os
    cache=gp.replace('.parquet','_dissolved.parquet')
    if os.path.exists(cache): return gpd.read_parquet(cache)
    G=gpd.read_parquet(gp)[['census','geometry']]; G['census']=G.census.astype(str)
    D=G.dissolve(by='census').reset_index()[['census','geometry']]; D.to_parquet(cache); return D
def load_sections(gp,efpath,year=2020):
    G=gpd.read_parquet(gp); G=G.drop_duplicates('census'); G['census']=G.census.astype(str)
    G=G.drop(columns='geometry').merge(load_geom(gp),on='census',how='left'); G=gpd.GeoDataFrame(G,geometry='geometry',crs=4326)
    H27={'昼間（非混雑時）／{d}／旅行速度／小型車（ｋｍ／ｈ）':'昼間非混雑時／{d}／旅行速度（ｋｍ／ｈ）','昼間（非混雑時）／{d}／旅行速度／大型車（ｋｍ／ｈ）':'昼間非混雑時／{d}／旅行速度（ｋｍ／ｈ）',
         '朝夕（混雑時）／{d}／旅行速度／合計（ｋｍ／ｈ）':'混雑時／{d}／旅行速度（ｋｍ／ｈ）'}   # H27 (2015 census) names; no vehicle split
    def col(c):
        if c in G.columns: return G[c]
        for k,v in H27.items():
            for d in ('上り','下り'):
                if c==k.format(d=d): return G[v.format(d=d)]
        raise KeyError(c)
    n=lambda c: pd.to_numeric(col(c),errors='coerce')
    vs=lambda a,b: pd.concat([n(a),n(b)],axis=1).replace(0,np.nan).mean(axis=1)
    K=pd.DataFrame({'id':G.census.astype(str).values,'rt':G['道路種別'].astype(str).str.strip().values,'muni':G['市区町村コード'].astype(str).str.zfill(5).values,
       'motorway':G['自動車専用道路の別'].astype(str).str.strip().values,'oneway':G['一方通行フラグ'].astype(str).str.strip().values,
       'v24s':n('２４時間自動車類交通量（上下合計）／小型車（台）').values,'v24l':n('２４時間自動車類交通量（上下合計）／大型車（台）').values,
       'carw':n('幅員構成／車道部幅員（ｍ）').values,'medw':n('幅員構成／中央帯幅員（ｍ）').values,'vlim':n('指定最高速度（ｋｍ／ｈ）').values})
    vs_off_s=vs('昼間（非混雑時）／上り／旅行速度／小型車（ｋｍ／ｈ）','昼間（非混雑時）／下り／旅行速度／小型車（ｋｍ／ｈ）')
    vs_off_l=vs('昼間（非混雑時）／上り／旅行速度／大型車（ｋｍ／ｈ）','昼間（非混雑時）／下り／旅行速度／大型車（ｋｍ／ｈ）')
    vs_pk=vs('朝夕（混雑時）／上り／旅行速度／合計（ｋｍ／ｈ）','朝夕（混雑時）／下り／旅行速度／合計（ｋｍ／ｈ）')
    vs_12=vs('昼間１２時間平均旅行速度（時間帯別交通量加重）／上り（ｋｍ／ｈ）','昼間１２時間平均旅行速度（時間帯別交通量加重）／下り（ｋｍ／ｈ）')
    st=(K.motorway=='1')|K.rt.isin(['1','2'])
    vdef=np.where(st,K.vlim.where(K.vlim>0).fillna(80),K.vlim.where(K.vlim>0).fillna(40))
    K['Vs']=vs_12.values; K['Vs']=K.Vs.fillna(pd.Series(vs_off_s.values)).fillna(pd.Series(vs_pk.values)).fillna(pd.Series(vdef))
    K['Vl']=pd.Series(vs_off_l.values).fillna(K.Vs).values
    K['speed_src']=np.where(vs_12.notna().values,'12h',np.where(vs_off_s.notna().values,'offpeak',np.where(vs_pk.notna().values,'peak','limit')))
    K['grp']=np.where(st,'mw','gen')
    K['nox_gkmd']=K.v24s.fillna(0)*ef(K.Vs.values,'small',year,efpath)+K.v24l.fillna(0)*ef(K.Vl.values,'large',year,efpath)   # g/km/day
    med=K.medw.fillna(0).values; carw=K.carw.fillna(7).values
    K['off']=np.where(K.oneway.values=='1',0.0,med/2+np.maximum((carw-med)/2,1.5)/2)
    K['halfw']=carw/2
    K['ok']=K.v24s.notna()|K.v24l.notna()
    K['geometry']=G.geometry.values
    return gpd.GeoDataFrame(K,geometry='geometry',crs=G.crs)
def segments(geoms,ids):
    rows=[]
    for cid,geom in zip(ids,geoms):
        for ln in getattr(geom,'geoms',[geom]):
            c=shapely.get_coordinates(ln)
            if len(c)<2: continue
            a,b=c[:-1],c[1:]; m=np.hypot(*(b-a).T)>0.01
            for (ax,ay),(bx,by) in zip(a[m],b[m]): rows.append((cid,ax,ay,bx,by))
    return pd.DataFrame(rows,columns=['id','ax','ay','bx','by'])
def seg_kernel(ax,ay,bx,by,px,py,q,off,h):
    """Direction-averaged ground-level concentration kernel ~ q * integral dt/(d^2+h^2+t^2) over the segment (sigma_z ~ linear in distance;
    h = initial vertical mixing length). Two carriageway halves at +-off. Same integral as the noise intensity line source."""
    ux,uy=bx-ax,by-ay; L=np.hypot(ux,uy); ux/=L; uy/=L
    qx,qy=px-ax,py-ay; tA=-(qx*ux+qy*uy); tB=L+tA; d=np.maximum(np.abs(qx*uy-qy*ux),0.5)
    E=np.zeros(len(ax))
    for dd in (np.maximum(d-off,1.0),d+off):
        D3=np.hypot(dd,h); E+=0.5*q*(np.arctan(tB/D3)-np.arctan(tA/D3))/D3
    return E
def road_field(px,py,K,cut,hs,chunk=20000):
    """Sum kernel over census segments within `cut` m. K: projected GeoDataFrame with nox_gkmd, off. Returns dict h->array."""
    pts=shapely.points(px,py); tree=shapely.STRtree(pts)
    SG=segments(K.geometry.values,K.id.values); src=K.set_index('id')
    out={h:np.zeros(len(px)) for h in hs}
    q_all=src.nox_gkmd.loc[SG.id.values].values/86.4   # g/km/day -> mg/m/s *1e? (arbitrary consistent unit: g/km/day / 86.4 = mg/m/s *1e-3... scale fitted)
    off_all=src.off.loc[SG.id.values].values
    for i0 in range(0,len(SG),chunk):
        sg=SG.iloc[i0:i0+chunk]
        lines=shapely.linestrings(np.stack([np.c_[sg.ax,sg.ay],np.c_[sg.bx,sg.by]],axis=1))
        si,ri=tree.query(lines,predicate='dwithin',distance=cut)
        if len(si)==0: continue
        j=si+i0
        for h in hs:
            np.add.at(out[h],ri,seg_kernel(SG.ax.values[j],SG.ay.values[j],SG.bx.values[j],SG.by.values[j],px[ri],py[ri],q_all[j],off_all[j],h))
    return out
def utm_epsg(lon): return 6688+int((lon+180)//6)+1-51
def road_field_multi(px,py,K,qcols,cut=1000,h=20,chunk=3000):
    """As road_field (one h) but for several emission columns at once (the kernel is linear in q). Returns array (n, len(qcols))."""
    pts=shapely.points(px,py); tree=shapely.STRtree(pts)
    SG=segments(K.geometry.values,K.id.values); src=K.set_index('id')
    Q=src[qcols].loc[SG.id.values].values/86.4; off_all=src.off.loc[SG.id.values].values
    out=np.zeros((len(px),len(qcols)))
    for i0 in range(0,len(SG),chunk):
        sg=SG.iloc[i0:i0+chunk]
        lines=shapely.linestrings(np.stack([np.c_[sg.ax,sg.ay],np.c_[sg.bx,sg.by]],axis=1))
        si,ri=tree.query(lines,predicate='dwithin',distance=cut)
        if len(si)==0: continue
        j=si+i0; k=seg_kernel(SG.ax.values[j],SG.ay.values[j],SG.bx.values[j],SG.by.values[j],px[ri],py[ri],np.ones(len(j)),off_all[j],h)
        for c in range(len(qcols)): np.add.at(out[:,c],ri,k*Q[j,c])
    return out
def sections_multi(gp,efpath,years):
    """Sections of one census network with NOx emissions (g/km/day) for several emission-factor years: columns q{year}."""
    K=load_sections(gp,efpath,years[0])
    for y in years:
        K[f'q{y}']=K.v24s.fillna(0)*ef(K.Vs.values,'small',y,efpath)+K.v24l.fillna(0)*ef(K.Vl.values,'large',y,efpath)
    return K[K.ok].copy()
