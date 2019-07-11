from abc import ABC, abstractmethod

import numpy as np
import pyproj
import shapely.geometry as shp
import matplotlib.pyplot as plt
import cartopy.crs


class TileBase(ABC):
    def __init__(self,
                 xc: np.ndarray = None, yc: np.ndarray = None,
                 xe: np.ndarray = None, ye: np.ndarray = None,
                 crs: pyproj.Proj = None,
                 is_uniform: bool = False,
                 ):
        self.xc = xc
        self.yc = yc
        self.xe = xe
        self.ye = ye
        self.crs = crs
        self.is_uniform = is_uniform
        self._areas = None
        self._polygons = None

    @property
    def has_edges(self) -> bool:
        return (self.xe is not None) and (self.ye is not None)

    @property
    def shape(self) -> tuple:
        return self.xc.shape

    @property
    def crs(self) -> pyproj.Proj:
        return self._crs

    @crs.setter
    def crs(self, crs: pyproj.Proj):
        self._crs = crs

    @property
    def xc(self) -> np.ndarray:
        return self._xc

    @xc.setter
    def xc(self, xc):
        self._xc = xc

    @property
    def yc(self) -> np.ndarray:
        return self._yc

    @yc.setter
    def yc(self, yc) -> np.ndarray:
        self._yc = yc

    @property
    def xe(self) -> np.ndarray:
        return self._xe

    @xe.setter
    def xe(self, xe) -> np.ndarray:
        self._xe = xe

    @property
    def ye(self) -> np.ndarray:
        return self._ye

    @ye.setter
    def ye(self, ye) -> np.ndarray:
        self._ye = ye

    @property
    @abstractmethod
    def polygons(self) -> np.ndarray:
        pass

    @abstractmethod
    def plot_lines(self, ax: plt.Axes, proj: cartopy.crs.CRS, **kwargs):
        pass

    def scatter_centers(self, ax: plt.Axes, proj: cartopy.crs.CRS, color='gray', **kwargs):
        x, y = pyproj.transform(self.crs, pyproj.Proj(proj.proj4_init), self.xc, self.yc)
        ax.scatter(x, y, color=color, **kwargs)

    def scatter_edges(self, ax: plt.Axes, proj: cartopy.crs.CRS, color='lightskyblue', **kwargs):
        x, y = pyproj.transform(self.crs, pyproj.Proj(proj.proj4_init), self.xe, self.ye)
        ax.scatter(x, y, color=color, **kwargs)

    @property
    def is_uniform(self) -> bool:
        return self._is_uniform

    @is_uniform.setter
    def is_uniform(self, value: bool):
        self._is_uniform = value

    @property
    def areas(self) -> np.ndarray:
        if self._areas is None:
            if self.is_uniform:
                self._areas = np.ones(self.shape) * self.polygons.flatten()[0].area
            else:
                f = np.vectorize(lambda polygon: polygon.area)
                self._areas = f(self.polygons)
        return self._areas

    def to_crs(self, crs: pyproj.Proj):
        xc, yc = pyproj.transform(self.crs, crs, self.xc, self.yc)
        if self.has_edges:
            xe, ye = pyproj.transform(self.crs, crs, self.xe, self.ye)
        else:
            xe, ye = (None, None)
        return type(self)(xc=xc, yc=yc, xe=xe, ye=ye, crs=crs)


class LRGTileBase(TileBase):
    @property
    def polygons(self) -> np.ndarray:
        if self._polygons is None:
            xy = np.array([self.xe, self.ye])
            xy = np.moveaxis(xy, 0, 2)
            bl = xy[:-1, :-1, :]
            br = xy[1:, :-1, :]
            ul = xy[:-1, 1:, :]
            ur = xy[1:, 1:, :]
            polygons_xy = np.array([bl, ul, ur, br])
            polygons_xy = np.moveaxis(polygons_xy, 0, 2)
            p = np.empty(polygons_xy.shape[:2], dtype=object)
            for i, polygons_y in enumerate(polygons_xy):
                p[i, :] = [shp.Polygon(poly) for poly in polygons_y]
            self._polygons = p
        return self._polygons

    def plot_lines(self, ax: plt.Axes, proj: cartopy.crs.CRS, color='steelblue', **kwargs):
        xx, yy = pyproj.transform(self.crs, pyproj.Proj(proj.proj4_init), self.xe, self.ye)
        for x, y in zip(xx, yy):
            ax.plot(x, y, color=color, **kwargs)
        for x, y in zip(xx.transpose(), yy.transpose()):
            ax.plot(x, y, color=color, **kwargs)


class TileMosaic:
    def __init__(self):
        self.tiles = {}
        self.contacts = []

    @property
    def tiles(self) -> dict:
        return self._tiles

    @tiles.setter
    def tiles(self, tiles: dict):
        self._tiles = tiles

    @property
    def contacts(self) -> list:
        return self._contacts

    @contacts.setter
    def contacts(self, contacts: list):
        self._contacts = contacts

    def plot_lines(self, ax: plt.Axes, proj: cartopy.crs.CRS, **kwargs):
        for tile in self.tiles.values():
            tile.plot_lines(ax, proj, **kwargs)

    def scatter_centers(self, ax: plt.Axes, proj: cartopy.crs.CRS, **kwargs):
        for tile in self.tiles.values():
            tile.scatter_centers(ax, proj, **kwargs)

    def scatter_edges(self, ax: plt.Axes, proj: cartopy.crs.CRS, **kwargs):
        for tile in self.tiles.values():
            tile.scatter_edges(ax, proj, **kwargs)




if __name__ == '__main__':
    grid = LRGTileBase(np.array([1,2]), np.array([2,3]), crs=pyproj.Proj(init='epsg:4326'))
    grid2 = grid.to_crs(pyproj.Proj(init='epsg:32643'))
    print(grid2)





