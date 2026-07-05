import os
from dotenv import load_dotenv
import requests
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xrspatial import slope, aspect

load_dotenv()

API_KEY = os.environ.get("OT_API_KEY")

RAW = Path("data/raw/terrain")
RAW.mkdir(parents=True, exist_ok=True)

PNW_BBOX = {
    "min_lon": -124.8,
    "max_lon": -116.5,
    "min_lat":   42.0,
    "max_lat":   49.0,
}

# creates the parameters for the OpenTopography API request
url = "https://portal.opentopography.org/API/globaldem"
params = {
    "demtype": "SRTMGL3",
    "south": PNW_BBOX["min_lat"],
    "north": PNW_BBOX["max_lat"],
    "west":  PNW_BBOX["min_lon"],
    "east":  PNW_BBOX["max_lon"],
    "outputFormat": "GTiff",
    "API_Key": API_KEY,
}

out_path = RAW / "pnw_srtm30.tif"

if out_path.exists():
    print("Already downloaded:", out_path)
else:
    print("Downloading elevation data... (this may take a minute)")
    # stream = True downloads the large file in chunks
    resp = requests.get(url, params=params, stream=True, timeout=180)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

# reads the downloaded GeoTIFF file using rasterio
with rasterio.open(out_path) as src:
    print("Size:", src.width, "x", src.height, "pixels")
    print("Bounds:", src.bounds)
    print("CRS:", src.crs)
    elevation = src.read(1)
    bounds = src.bounds
    print("Elevation range:", elevation.min(), "to", elevation.max(), "meters")

# ignore values below -50 meters (which are likely errors or ocean areas)
elevation_masked = np.where(elevation < -50, np.nan, elevation)

# loads the elevation data and plots it using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(elevation_masked, cmap="terrain",
           extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
plt.colorbar(label="Elevation (m)")
plt.title("PNW Elevation (SRTM 90m)")
plt.savefig("elevation_map.png", dpi=120)
print("Saved elevation_map.png")

# wrap the elevation array as an xarray DataArray
dem = xr.DataArray(elevation, dims=["y", "x"])

slope_data = slope(dem)
aspect_data = aspect(dem)

print("Slope range:", float(slope_data.min()), "to", float(slope_data.max()), "degrees")
print("Aspect range:", float(aspect_data.min()), "to", float(aspect_data.max()), "degrees")

# save slope and aspect as GeoTIFFs, copying the geo-info from the source
with rasterio.open(out_path) as src:
    profile = src.profile
    profile.update(dtype="float32", count=1)

    with rasterio.open(RAW / "pnw_slope.tif", "w", **profile) as dst:
        dst.write(slope_data.values.astype("float32"), 1)

    with rasterio.open(RAW / "pnw_aspect.tif", "w", **profile) as dst:
        dst.write(aspect_data.values.astype("float32"), 1)

print("Saved slope and aspect")