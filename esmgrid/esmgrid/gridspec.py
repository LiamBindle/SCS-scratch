import os.path

import xarray as xr
import ESMF
from esmgrid.visualize import GridESMFFromMemory

def load_face(ds: xr.Dataset):
    grid_x_centers = ds.x[1::2, 1::2].values
    grid_y_centers = ds.y[1::2, 1::2].values
    grid_x_edges   = ds.x[0::2, 0::2].values
    grid_y_edges   = ds.y[0::2, 0::2].values
    grid = GridESMFFromMemory(48, 48, ESMF.CoordSys.SPH_DEG)
    grid.load_center_xy(grid_x_centers, grid_y_centers)
    grid.load_corner_xy(grid_x_edges, grid_y_edges)
    return grid

def open_mosaic(basedir: str, mosaic_filename: str):
    ds = xr.open_dataset(os.path.join(basedir, mosaic_filename))
    tile_files = [os.path.join(basedir, tile_file.item()) for tile_file in ds['tile_files'].astype(str)]
    ds_tiles = xr.open_mfdataset(tile_files, concat_dim='tile', combine='nested')
    ds_contacts = xr.open_dataset(os.path.join(basedir, ds['contact_files'].astype(str).item()))
    ds = xr.merge([ds, ds_tiles, ds_contacts])

    grid1 = load_face(ds.isel(tile=1))
    print(ds)





if __name__ == '__main__':
    basedir = '~/Downloads/mosaic_20090716/output/'
    filename = 'C48_mosaic.nc'
    open_mosaic(basedir, filename)
