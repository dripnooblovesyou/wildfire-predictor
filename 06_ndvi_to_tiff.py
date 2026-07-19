from pyhdf.SD import SD, SDC
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from datetime import datetime, timedelta
import re
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box as shapely_box
import geopandas as gpd

MODIS_TILE_SIZE = 1111950.5197665
MODIS_X_MIN = -20015109.354
MODIS_Y_MAX = 10007554.677

# MODIS sinusoidal projection definition
MODIS_CRS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"

def tile_bounds(h, v):
    """Return (left, bottom, right, top) in MODIS sinusoidal meters for tile hHHvVV."""
    left = MODIS_X_MIN + h * MODIS_TILE_SIZE
    top = MODIS_Y_MAX - v * MODIS_TILE_SIZE
    right = left + MODIS_TILE_SIZE
    bottom = top - MODIS_TILE_SIZE
    return left, bottom, right, top

def hdf_to_array(path):
    """Read the NDVI layer from a MODIS HDF file, return real NDVI values."""
    hdf = SD(str(path), SDC.READ)
    ds = hdf.select("1 km monthly NDVI")
    raw = ds.get()
    
    ndvi = raw.astype("float32")
    ndvi[raw == -3000] = np.nan       # mask fill values
    ndvi = ndvi / 10000.0             # apply scale factor
    return ndvi

def write_tile_tif(array, h, v, out_path):
    """Write a MODIS tile array as a GeoTIFF in sinusoidal projection."""
    left, bottom, right, top = tile_bounds(h, v)
    transform = from_bounds(left, bottom, right, top, array.shape[1], array.shape[0])
    
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=MODIS_CRS,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(array, 1)

# test on one file
test_file = list(Path("data/raw/ndvi").glob("*h08v04*"))[0]
print("Testing on:", test_file.name)
arr = hdf_to_array(test_file)
print("Shape:", arr.shape)
print("NDVI range:", np.nanmin(arr), "to", np.nanmax(arr))

# test it
TMP = Path("data/tmp")
TMP.mkdir(parents=True, exist_ok=True)

write_tile_tif(arr, 8, 4, TMP / "test_tile.tif")
print("Wrote test tile")

# verify it reads back correctly
with rasterio.open(TMP / "test_tile.tif") as src:
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Shape:", src.width, "x", src.height)

def parse_modis_date(filename):
    """Extract (year, month) from a MODIS filename like MOD13A3.A2000153.h08v04..."""
    m = re.search(r"A(\d{4})(\d{3})", filename)
    year = int(m.group(1))
    doy = int(m.group(2))
    date = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return date.year, date.month


# test it on a few files
files = sorted(Path("data/raw/ndvi").glob("*.hdf"))
for f in files[:6]:
    print(f.name, "→", parse_modis_date(f.name))

TARGET_CRS = "EPSG:32610"

def build_month_raster(year, month, all_files, out_dir):
    """Convert, mosaic, crop, and reproject all tiles for one month into a single UTM GeoTIFF."""
    out_path = out_dir / f"ndvi_{year}_{month:02d}.tif"
    if out_path.exists():
        return out_path

    month_files = [f for f in all_files if parse_modis_date(f.name) == (year, month)]
    if not month_files:
        return None

    # convert each tile to a temporary sinusoidal GeoTIFF
    tile_paths = []
    for f in month_files:
        h, v = re.search(r"h(\d{2})v(\d{2})", f.name).groups()
        arr = hdf_to_array(f)
        tp = TMP / f"tile_{year}_{month:02d}_h{h}v{v}.tif"
        write_tile_tif(arr, int(h), int(v), tp)
        tile_paths.append(tp)

    # mosaic the tiles together
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, mosaic_transform = merge(srcs)
    mosaic_meta = srcs[0].meta.copy()
    for s in srcs:
        s.close()

    mosaic_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": mosaic_transform,
    })

    mosaic_path = TMP / f"mosaic_{year}_{month:02d}.tif"
    with rasterio.open(mosaic_path, "w", **mosaic_meta) as dst:
        dst.write(mosaic)

    # PNW box in sinusoidal coordinates
    pnw_sinu = gpd.GeoDataFrame(
        geometry=[shapely_box(-124.8, 42.0, -116.5, 49.0)],
        crs="EPSG:4326"
    ).to_crs(MODIS_CRS)

    # CROP the mosaic to PNW before reprojecting
    with rasterio.open(mosaic_path) as src:
        cropped, cropped_transform = mask(src, pnw_sinu.geometry, crop=True)
        crop_meta = src.meta.copy()
        crop_meta.update({
            "height": cropped.shape[1],
            "width": cropped.shape[2],
            "transform": cropped_transform,
        })

    cropped_path = TMP / f"cropped_{year}_{month:02d}.tif"
    with rasterio.open(cropped_path, "w", **crop_meta) as dst:
        dst.write(cropped)

    # reproject the CROPPED raster to UTM
    with rasterio.open(cropped_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update({
            "crs": TARGET_CRS,
            "transform": transform,
            "width": width,
            "height": height,
        })

        with rasterio.open(out_path, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
            )

    # clean up temp files
    for p in tile_paths:
        p.unlink()
    mosaic_path.unlink()
    cropped_path.unlink()

    return out_path


# loop over every year/month in the fire season and build a raster for each
NDVI_OUT = Path("data/processed/ndvi")
NDVI_OUT.mkdir(parents=True, exist_ok=True)

for year in range(2000, 2026):
    for month in range(6, 10):
        print(f"Processing {year}-{month:02d}...")
        result = build_month_raster(year, month, files, NDVI_OUT)
        print("Created:", result)