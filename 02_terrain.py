import os
from dotenv import load_dotenv
import requests
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xrspatial import slope, aspect
from rasterio.warp import calculate_default_transform, reproject, Resampling

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

# reproject to UTM (meters) so slope math is correct
dst_crs = "EPSG:32610"  # UTM Zone 10N, units in meters
proj_path = RAW / "pnw_srtm30_utm.tif"

with rasterio.open(out_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    profile = src.profile.copy()
    profile.update(crs=dst_crs, transform=transform, width=width, height=height)

    with rasterio.open(proj_path, "w", **profile) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

print("Reprojected to UTM.")

# read reprojected elevation and compute slope + aspect
with rasterio.open(proj_path) as src:
    elevation_utm = src.read(1)
    utm_profile = src.profile.copy()

with rasterio.open(proj_path) as src:
    print("Pixel width (m):", src.transform[0])
    print("Pixel height (m):", -src.transform[4])

# compute slope and aspect using xrspatial
with rasterio.open(proj_path) as src:
    elevation_utm = src.read(1)
    utm_profile = src.profile.copy()
    transform = src.transform
    h, w = elevation_utm.shape

# build coordinate arrays in meters from the transform
xs = transform[2] + transform[0] * (np.arange(w) + 0.5)
ys = transform[5] + transform[4] * (np.arange(h) + 0.5)

dem = xr.DataArray(
    elevation_utm.astype("float32"),
    dims=["y", "x"],
    coords={"y": ys, "x": xs},
)

slope_data = slope(dem)
aspect_data = aspect(dem)

print("Slope range:", float(slope_data.min()), "to", float(slope_data.max()), "degrees")
print("Aspect range:", float(aspect_data.min()), "to", float(aspect_data.max()), "degrees")

sv = slope_data.values
sv = sv[~np.isnan(sv)]
print("Slope percentiles:")
print("  50th (median):", np.percentile(sv, 50))
print("  90th:", np.percentile(sv, 90))
print("  99th:", np.percentile(sv, 99))

# save slope and aspect with correct UTM geo-info
utm_profile.update(dtype="float32", count=1)

with rasterio.open(RAW / "pnw_slope.tif", "w", **utm_profile) as dst:
    dst.write(slope_data.values.astype("float32"), 1)

with rasterio.open(RAW / "pnw_aspect.tif", "w", **utm_profile) as dst:
    dst.write(aspect_data.values.astype("float32"), 1)

print("Saved slope and aspect")