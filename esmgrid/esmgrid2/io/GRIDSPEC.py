import os.path
import re
import pyproj
import xarray as xr
from esmgrid2.base import LRGTileBase, TileBase, TileMosaic


class GRIDSPEC:
    @staticmethod
    def read_tile(ds: xr.Dataset, crs: pyproj.Proj = pyproj.Proj(init='epsg:4326')) -> dict:
        tile = LRGTileBase()
        tile.xc = ds.x[1::2, 1::2].values
        tile.yc = ds.y[1::2, 1::2].values
        tile.xe = ds.x[0::2, 0::2].values
        tile.ye = ds.y[0::2, 0::2].values
        tile.crs = crs
        return {ds.tile.astype(str).item(): tile}

    @staticmethod
    def read_contacts(ds: xr.Dataset):
        contacts = []
        for contact_string, contact_index_string in zip(ds.contacts.astype(str), ds.contact_index.astype(str)):
            tile_search = re.search(r'^\w+:(\w+)::\w+:(\w+)$', contact_string.item())
            if tile_search is None:
                raise ValueError(f'Failed to parse contact string: {contact_string}')
            c1 = tile_search.group(1)
            c2 = tile_search.group(2)
            slice_search = re.search(r'^(\d+:\d+),(\d+:\d+)::(\d+:\d+),(\d+:\d+)$', contact_index_string.item())
            if slice_search is None:
                raise ValueError(f'Failed to parse slice: {contact_index_string.item()}')
            def parse_slice(s: str):
                index_search = re.search(r'^(\d+):(\d+)$', s)
                if tile_search is None:
                    raise ValueError(f'Failed to parse slice: {s}')
                return slice((int(index_search.group(1)) - 1) // 2, (int(index_search.group(2)) - 1) // 2 + 1)
            y1 = parse_slice(slice_search.group(1))
            x1 = parse_slice(slice_search.group(2))
            y2 = parse_slice(slice_search.group(3))
            x2 = parse_slice(slice_search.group(4))
            contacts.append({c1: (x1, y1), c2: (x2, y2)})
        return contacts

    @staticmethod
    def write_tile(tile: TileBase) -> xr.Dataset:
        pass

    @staticmethod
    def read_mosaic(path: str) -> TileMosaic:
        basepath = os.path.dirname(path)
        mosaic = xr.open_dataset(path)

        tile_mosaic = TileMosaic()
        for tile_filename in mosaic.tile_files.astype(str):
            tile_filepath = os.path.join(basepath, tile_filename.item())
            tile_mosaic.tiles.update(GRIDSPEC.read_tile(xr.open_dataset(tile_filepath)))

        contacts_filepath = os.path.join(basepath, mosaic.contact_files.astype(str).item())
        tile_mosaic.contacts = GRIDSPEC.read_contacts(xr.open_dataset(contacts_filepath))
        return tile_mosaic
