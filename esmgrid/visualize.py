from abc import ABC, abstractmethod
import functools

import numpy as np
import shapely.geometry as shp
import shapely.ops
import pyproj

import xarray as xr

import matplotlib.pyplot as plt
import descartes
import cartopy.crs as ccrs

import ESMF


class GridBase(ABC):
    @property
    def crs(self):
        pass

    def project_to(self, x, y, crs: pyproj.Proj):
        project = functools.partial(
            pyproj.transform,
            pyproj.Proj(**self.crs),    # src crs
            crs                         # dst crs
        )

    def get_xy_centers(self, crs: pyproj.Proj):
        x = self.x_centers
        y = self.y_centers
        src = pyproj.Proj(**self.crs)
        dst = crs
        x, y = pyproj.transform(src, dst, x, y)
        return x, y

    def get_xy_corners(self, crs: pyproj.Proj):
        x = self.x_corners
        y = self.y_corners
        src = pyproj.Proj(**self.crs)
        dst = crs
        x, y = pyproj.transform(src, dst, x, y)
        return x, y

    def get_xy_boxes(self, crs: pyproj.Proj):
        xy = np.array(self.get_xy_corners(crs))
        xy = np.moveaxis(xy, 0, 2)
        bl = xy[:-1, :-1, :]
        br = xy[1:, :-1, :]
        ul = xy[:-1, 1:, :]
        ur = xy[1:, 1:, :]
        xy_boxes = np.array([bl, ul, ur, br])
        xy_boxes = np.moveaxis(xy_boxes, 0, 2)
        return xy_boxes[..., 0], xy_boxes[..., 1]

    @staticmethod
    def xy_to_polygon(x, y):
        xy = np.array([x, y])
        xy = np.moveaxis(xy, 0, -1)
        p = np.empty(xy.shape[:2], dtype=object)
        for i, y_slice in enumerate(xy):
            p[i, :] = [shp.Polygon(poly) for poly in y_slice]
        return p

    @property
    def shape(self) -> np.ndarray:
        return np.array(self.x_centers.shape)

    @property
    @abstractmethod
    def x_centers(self) -> np.array:
        pass

    @property
    @abstractmethod
    def y_centers(self) -> np.array:
        pass

    @property
    @abstractmethod
    def x_corners(self) -> np.array:
        pass

    @property
    @abstractmethod
    def y_corners(self) -> np.array:
        pass

    def grid_centers(self) -> np.array:
        xy = np.array([self.x_centers, self.y_centers])
        xy = np.moveaxis(xy, 0, 2)
        points = np.empty(xy.shape[:2], dtype=object)
        for i, y in enumerate(xy):
            points[i, :] = [shp.Point(p) for p in y]
        return points

    def grid_corners(self) -> np.array:
        xy = np.array([self.x_corners, self.y_corners])
        xy = np.moveaxis(xy, 0, 2)
        points = np.empty(xy.shape[:2], dtype=object)
        for i, y in enumerate(xy):
            points[i, :] = [shp.Point(p) for p in y]
        return points

    def grid_lines(self) -> np.array:
        xy = np.array([self.x_corners, self.y_corners])
        xy = np.moveaxis(xy, 0, 2)

        lines_x = [shp.LineString(xy[i, :]) for i in range(xy.shape[0])]
        lines_y = [shp.LineString(xy[:, i]) for i in range(xy.shape[1])]

        return np.array(lines_x, dtype=object), np.array(lines_y, dtype=object)

    def grid_boxes(self) -> np.array:
        xy = np.array([self.x_corners, self.y_corners])
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
        return p

    def plot_centers(self, ax: plt.Axes, crs: pyproj.Proj, color='k', **kwargs):
        xy = np.array([self.x_centers, self.y_centers])
        src = pyproj.Proj(**self.crs)
        dst = crs
        xp, yp = pyproj.transform(src, dst, xy[0,:], xy[1,:])
        ax.scatter(xp, yp, color=color)

    def plot_corners(self, ax: plt.Axes, crs: pyproj.Proj, color='g', **kwargs):
        xy = np.array([self.x_corners, self.y_corners])
        src = pyproj.Proj(**self.crs)
        dst = crs
        xp, yp = pyproj.transform(src, dst, xy[0,:], xy[1,:])
        ax.scatter(xp, yp, color=color)

    def plot_lines(self, ax: plt.Axes, crs: pyproj.Proj, color='#6699cc', **kwargs):
        x_lines, y_lines = self.grid_lines()
        x_mp = shp.MultiLineString(x_lines.flatten().tolist())
        y_mp = shp.MultiLineString(y_lines.flatten().tolist())

        project = functools.partial(
            pyproj.transform,
            pyproj.Proj(**self.crs),    # src crs
            crs                         # dst crs
        )

        x_mp = shapely.ops.transform(project, x_mp)
        y_mp = shapely.ops.transform(project, y_mp)

        for line in x_mp:
            ax.plot(*line.xy, color=color, **kwargs)

        for line in y_mp:
            ax.plot(*line.xy, color=color, **kwargs)

    def plot_boxes(self, ax: plt.Axes, crs: pyproj.Proj, facecolor='none', edgecolor='#6699cc', **kwargs):
        boxes = self.grid_boxes()
        mp = shp.MultiPolygon(boxes.flatten().tolist())

        project = functools.partial(
            pyproj.transform,
            pyproj.Proj(**self.crs),    # src crs
            crs                         # dst crs
        )

        mp = shapely.ops.transform(project, mp)
        patch = descartes.PolygonPatch(mp, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
        ax.add_patch(patch)

    def to_SCRIP(self, filename: str):
        boxes = self.grid_boxes().flatten(order='F')

        x, y = self.get_xy_corners(pyproj.Proj(**self.crs))
        dx = np.diff(x, axis=0)[:, :-1]
        dy = np.diff(y, axis=1)[:-1, :]
        area = dx*dy * (np.pi/180)**2

        make_ccw = lambda polygon: np.array(shp.polygon.orient(polygon, 1).exterior)
        boxes_ccw = np.array([make_ccw(polygon) for polygon in boxes])
        boxes_ccw = boxes_ccw[:, :-1, :] # remove last point from boxes (which is copy of the first point)
        boxes_ccw = np.moveaxis(boxes_ccw, -1, 0) # bring xy dim to front

        ds = xr.Dataset(
            data_vars={
                'grid_dims': ('grid_rank', self.shape.astype(np.int32)),
                'grid_center_lat': ('grid_size', self.y_centers.flatten(order='F')),
                'grid_center_lon': ('grid_size', self.x_centers.flatten(order='F')),
                'grid_corner_lat': (('grid_size', 'grid_corners'), boxes_ccw[1, ...]),
                'grid_corner_lon': (('grid_size', 'grid_corners'), boxes_ccw[0, ...]),
                'grid_imask': ('grid_size', np.ones_like(self.x_centers.flatten()).astype(np.int32)),
                'grid_area' : ('grid_size', area.flatten())
            }
        )
        for varname in ['grid_center_lat', 'grid_center_lon', 'grid_corner_lat', 'grid_corner_lon']:
            ds[varname].attrs['units'] = 'degrees'
        for varname in ['grid_corner_lat', 'grid_corner_lon']:
            ds[varname].attrs['_FillValue'] = -9999.0
        ds['grid_area'].attrs.update({'units': 'radians^2', 'long_name': 'area weights'})
        ds.to_netcdf(filename)


class GridESMFBase(GridBase):
    _xdim=0
    _ydim=1

    @property
    def x_centers(self) -> np.array:
        return self._grid.get_coords(coord_dim=self._xdim, staggerloc=ESMF.StaggerLoc.CENTER)

    @property
    def y_centers(self) -> np.array:
        return self._grid.get_coords(coord_dim=self._ydim, staggerloc=ESMF.StaggerLoc.CENTER)

    @property
    def x_corners(self) -> np.array:
        return self._grid.get_coords(coord_dim=self._xdim, staggerloc=ESMF.StaggerLoc.CORNER)

    @property
    def y_corners(self) -> np.array:
        return self._grid.get_coords(coord_dim=self._ydim, staggerloc=ESMF.StaggerLoc.CORNER)

    @property
    def crs(self):
        if self._grid.coord_sys == ESMF.CoordSys.SPH_DEG:
            return {'init': 'epsg:4326', 'pm':-360}
        else:
            raise NotImplementedError(f'crs for {self._grid.coord_sys}')


class GridESMFFromMemory(GridESMFBase):
    def __init__(self, nx, ny, coord_sys, **kwargs):
        self._grid = ESMF.Grid(
            max_index=np.array([nx, ny]),
            staggerloc=ESMF.StaggerLoc.CENTER,
            coord_sys=coord_sys,
            **kwargs
        )

    def load_center_xy(self, x, y):
        x2d, y2d = np.meshgrid(x, y, indexing='ij')
        x2d = np.asfortranarray(x2d)
        y2d = np.asfortranarray(y2d)
        x_centers = self._grid.get_coords(coord_dim=self._xdim, staggerloc=ESMF.StaggerLoc.CENTER)
        y_centers = self._grid.get_coords(coord_dim=self._ydim, staggerloc=ESMF.StaggerLoc.CENTER)
        x_centers[...] = x2d
        y_centers[...] = y2d

    def load_corner_xy(self, x, y):
        self._grid.add_coords(staggerloc=ESMF.StaggerLoc.CORNER)
        x2d, y2d = np.meshgrid(x, y, indexing='ij')
        x2d = np.asfortranarray(x2d)
        y2d = np.asfortranarray(y2d)
        x_corners = self._grid.get_coords(coord_dim=self._xdim, staggerloc=ESMF.StaggerLoc.CORNER)
        y_corners = self._grid.get_coords(coord_dim=self._ydim, staggerloc=ESMF.StaggerLoc.CORNER)
        x_corners[...] = x2d
        y_corners[...] = y2d

class GridESMFFromFile(GridESMFBase):
    def __init__(self, filename: str, filetype: ESMF.FileFormat, add_corner_stagger=True, is_sphere=False, **kwargs):
        self._grid = ESMF.Grid(
            filename=filename,
            filetype=filetype,
            add_corner_stagger=add_corner_stagger,
            is_sphere=is_sphere,  # so that x_corners is Nx+1 (i.e. we don't want the longitude grid to wrap)
            **kwargs
        )

    @property
    def crs(self):
        return {'init': 'epsg:4326', 'pm': -360}


if __name__ == '__main__':
    lat = np.arange(0, 90.01, 1)
    lat_edge = lat - np.diff(lat)[0]/2
    lat_edge = np.append(lat_edge, lat[-1] + np.diff(lat)[0]/2)
    lon = np.arange(0, 360.01, 2)
    lon_edge = lon - np.diff(lon)[0]/2
    lon_edge = np.append(lon_edge, lon[-1] + np.diff(lon)[0]/2)

    #grid = GridESMF(len(lon), len(lat), ESMF.CoordSys.SPH_DEG)
    grid = GridESMFFromFile(
        filename='foo2.nc', #'/home/liam/Downloads/ESMPy_630r_01b/esmf/src/Infrastructure/Grid/examples/data/T42_grid.nc',
        filetype=ESMF.FileFormat.SCRIP,
    )
    #grid.to_SCRIP('foo2.nc')
    #grid.load_center_xy(lon, lat)
    #grid.load_corner_xy(lon_edge, lat_edge)

    plt.figure()
    crs = ccrs.Orthographic(-85, 35)
    ax = plt.axes(projection=crs)
    ax.set_global()
    ax.coastlines()
    grid.plot_lines(ax, pyproj.Proj(crs.proj4_init))
    grid.plot_centers(ax, pyproj.Proj(crs.proj4_init))
    grid.plot_corners(ax, pyproj.Proj(crs.proj4_init))
    plt.show()
