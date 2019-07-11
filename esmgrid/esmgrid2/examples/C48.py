import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from esmgrid2.io.GRIDSPEC import GRIDSPEC

filepath = 'datafiles/C48_mosaic.nc'
mosaic = GRIDSPEC.read_mosaic(filepath)

plt.figure()
crs = ccrs.Orthographic(-85, 35)
ax = plt.axes(projection=crs)
ax.set_global()
ax.coastlines()
mosaic.plot_lines(ax, crs)
plt.show()
