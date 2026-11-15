"""TROPOMI-inferred surface NO2 (Cooper et al. 2022, Zenodo 5484305, asia 2019 annual mean, ~1 km, ppb) sampled as a 3x3-cell mean (~3 km)."""
import numpy as np, netCDF4
from scipy.ndimage import uniform_filter
def load(fn,k=3):
    d=netCDF4.Dataset(fn); lat=np.ravel(d['LAT_CENTER'][:]); lon=np.ravel(d['LON_CENTER'][:])
    li=np.where((lat>23)&(lat<46.5))[0]; lj=np.where((lon>122)&(lon<149))[0]
    Z=np.array(d['surface_no2_ppb'][li.min():li.max()+1,lj.min():lj.max()+1],float); Z[~np.isfinite(Z)|(Z<0)]=np.nan
    Z0=np.nan_to_num(Z); M=np.isfinite(Z).astype(float)
    Zk=uniform_filter(Z0,k,mode='nearest')/np.maximum(uniform_filter(M,k,mode='nearest'),1e-9); Zk[uniform_filter(M,k,mode='nearest')<1e-6]=np.nan
    return lat[li.min():li.max()+1],lon[lj.min():lj.max()+1],Zk
def sample(grid,lat,lon):
    glat,glon,Z=grid
    # nearest cell (regular grids, either orientation)
    i=np.clip(np.rint((lat-glat[0])/(glat[1]-glat[0])).astype(int),0,len(glat)-1); j=np.clip(np.rint((lon-glon[0])/(glon[1]-glon[0])).astype(int),0,len(glon)-1)
    return Z[i,j]
