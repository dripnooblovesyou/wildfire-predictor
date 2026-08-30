import rasterio
import numpy as np

with rasterio.open("data/raw/fuel/LF2023_FBFM40_CONUS.tif") as src:
    print("CRS:", src.crs)
    print("Size:", src.width, "x", src.height)
    print("Bounds:", src.bounds)
    print("Data type:", src.dtypes[0])
    print("Nodata:", src.nodata)
    window = src.read(1, window=((0, 1000), (0, 1000)))
    print("Sample unique values:", np.unique(window)[:20])