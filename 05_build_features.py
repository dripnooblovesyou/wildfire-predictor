import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
import rasterio
from pyproj import Transformer


fires = gpd.read_file("data/raw/fire/pnw_fires_clean.geojson")
print("Fires loaded:", len(fires))

# reproject to UTM so distances are in meters
fires_utm = fires.to_crs("EPSG:32610")

# repair any invalid fire geometries (self-intersections etc.)
fires_utm["geometry"] = fires_utm.geometry.buffer(0)

# compute centroids in meter-space (geometrically correct)
fires_utm["centroid"] = fires_utm.geometry.centroid

print(fires_utm[["incidentname", "fireyear", "centroid"]].head())


def make_control_point(x, y, rng):
    """Given a fire centroid (x, y) in meters, return one control point 25-100km away."""
    distance = rng.uniform(25_000, 100_000)
    angle = rng.uniform(0, 2 * np.pi)
    new_x = x + distance * np.cos(angle)
    new_y = y + distance * np.sin(angle)
    return new_x, new_y


# study area boundary in UTM
pnw_box = gpd.GeoDataFrame(
    geometry=[box(-124.8, 42.0, -116.5, 49.0)],
    crs="EPSG:4326"
).to_crs("EPSG:32610")
study_area = pnw_box.geometry.iloc[0]
print("Study area bounds (UTM):", study_area.bounds)

def is_valid_control(x, y, year, fires_utm, study_area):
    """Check a control point is in the study area and not inside a fire that year."""
    pt = Point(x, y)
    
    if not study_area.contains(pt):
        return False
    
    # check against all fires from that year
    same_year = fires_utm[fires_utm["fireyear"] == year]
    if same_year.geometry.contains(pt).any():
        return False
    
    return True

rows = []
rng = np.random.default_rng(42)

N_CONTROLS = 5
MAX_ATTEMPTS = 50

# loop over all fires and generate control points
for idx, fire in fires_utm.iterrows():
    fx = fire["centroid"].x
    fy = fire["centroid"].y
    year = fire["fireyear"]
    
    # add the fire row (label = 1)
    rows.append({"x": fx, "y": fy, "year": year, "label": 1})
    
    #generate N_CONTROLS valid control points
    controls_made = 0
    attempts = 0
    
    # 
    while controls_made < N_CONTROLS and attempts < MAX_ATTEMPTS:
        cx, cy = make_control_point(fx, fy, rng)
        if is_valid_control(cx, cy, year, fires_utm, study_area):
        # if valid, append a dict with label = 0 and increment controls_made
            rows.append({"x": cx, "y": cy, "year": year, "label": 0})
            controls_made += 1
        # increment attempts either way
        attempts += 1

# 3. build the DataFrame
df = pd.DataFrame(rows)

print("Total rows:", len(df))
print("Label counts:")
print(df["label"].value_counts())

# build Point geometries from the x, y columns
df["geometry"] = [Point(x, y) for x, y in zip(df["x"], df["y"])]

# wrap df as a GeoDataFrame, specifying geometry="geometry" and crs="EPSG:32610"
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32610")

# make an output folder for processed data
from pathlib import Path
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# save it
gdf.to_file(PROC / "training_points.gpkg", driver="GPKG")
print("Saved", len(gdf), "points")

# extract elevation values
with rasterio.open("data/raw/terrain/pnw_srtm30_utm.tif") as src:
    # sample elevation values at each point
    coords = list(zip(gdf["x"], gdf["y"]))
    elevations = [val[0] for val in src.sample(coords)]

gdf["elevation"] = elevations
print(gdf[["x", "y", "label", "elevation"]].head())
print("Elevation range:", gdf["elevation"].min(), "to", gdf["elevation"].max())

# extract slope values
with rasterio.open("data/raw/terrain/pnw_slope.tif") as src:
    coords = list(zip(gdf["x"], gdf["y"]))
    slopes = [val[0] for val in src.sample(coords)]

gdf["slope"] = slopes
print(gdf[["x", "y", "label", "slope"]].head())
print("Slope range:", gdf["slope"].min(), "to", gdf["slope"].max())

# extract aspect values
with rasterio.open("data/raw/terrain/pnw_aspect.tif") as src:
    coords = list(zip(gdf["x"], gdf["y"]))
    aspects = [val[0] for val in src.sample(coords)]

gdf["aspect"] = aspects
print(gdf[["x", "y", "label", "aspect"]].head())
print("Aspect range:", gdf["aspect"].min(), "to", gdf["aspect"].max())

# convert aspect to northness and eastness
gdf["northness"] = np.cos(np.radians(gdf["aspect"]))
gdf["eastness"] = np.sin(np.radians(gdf["aspect"]))

print(gdf[["aspect", "northness", "eastness"]].head())

# extract NDVI values for each fire-season month (one column per month, e.g. ndvi_06)
NDVI_DIR = Path("data/processed/ndvi")

for month in [6, 7, 8, 9]:
    col = f"ndvi_{month:02d}"
    gdf[col] = np.nan

    # group points by year so each year's raster is only opened once
    for year in gdf["year"].unique():
        raster_path = NDVI_DIR / f"ndvi_{int(year)}_{month:02d}.tif"
        if not raster_path.exists():
            print(f"  Missing raster: {raster_path.name}")
            continue

        # filter to just this year's points, then sample them all at once
        year_mask = gdf["year"] == year
        coords = list(zip(gdf.loc[year_mask, "x"], gdf.loc[year_mask, "y"]))

        with rasterio.open(raster_path) as src:
            values = [val[0] for val in src.sample(coords)]

        gdf.loc[year_mask, col] = values

    print(f"{col} range:", gdf[col].min(), "to", gdf[col].max())

# NDVI anomaly = actual-year value minus each point's historical (2000-2025) mean for that month
coords = list(zip(gdf["x"], gdf["y"]))

for month in [6, 7, 8, 9]:
    # collect this month's NDVI at every point, for every year
    yearly_stack = []
    for yr in range(2000, 2026):
        raster_path = NDVI_DIR / f"ndvi_{yr}_{month:02d}.tif"
        if not raster_path.exists():
            continue
        with rasterio.open(raster_path) as src:
            vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
        yearly_stack.append(vals)

    # average across years → historical baseline for this month at each point
    baseline = np.nanmean(np.vstack(yearly_stack), axis=0)

    # anomaly = actual year value − baseline
    gdf[f"ndvi_anom_{month:02d}"] = gdf[f"ndvi_{month:02d}"] - baseline

    print(f"ndvi_anom_{month:02d} range:",
          gdf[f"ndvi_anom_{month:02d}"].min(), "to",
          gdf[f"ndvi_anom_{month:02d}"].max())

# reproject to lat/lon (weather lookups need lat/lon, not UTM meters)
gdf_latlon = gdf.to_crs("EPSG:4326")
gdf["lon"] = gdf_latlon.geometry.x
gdf["lat"] = gdf_latlon.geometry.y

print(gdf[["x", "y", "lon", "lat"]].head())

import xarray as xr
import zipfile

WEATHER_DIR = Path("data/raw/weather")
WEATHER_TMP = Path("data/tmp/weather")
WEATHER_TMP.mkdir(parents=True, exist_ok=True)

def open_weather_year(year):
    """Unzip (if not already done) and open the instant + accum NetCDF files for one year."""
    zip_path = WEATHER_DIR / f"era5_{year}_summer.nc"
    extract_dir = WEATHER_TMP / str(year)

    instant_path = extract_dir / "data_stream-oper_stepType-instant.nc"
    accum_path = extract_dir / "data_stream-oper_stepType-accum.nc"

    if not instant_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    inst = xr.open_dataset(instant_path)
    accum = xr.open_dataset(accum_path)
    return inst, accum

def sample_weather_var(ds, varname, month, lats, lons):
    """Average a variable over one month, then sample at each (lat, lon)."""
    # select this month's time steps
    monthly = ds[varname].sel(valid_time=ds["valid_time"].dt.month == month)
    # average over time → one value per grid cell
    monthly_mean = monthly.mean(dim="valid_time")
    # look up nearest cell for each point
    sampled = monthly_mean.sel(
        latitude=xr.DataArray(lats, dims="points"),
        longitude=xr.DataArray(lons, dims="points"),
        method="nearest",
    )
    return sampled.values

inst, accum = open_weather_year(2020)
mask_2020 = gdf["year"] == 2020
lats = gdf.loc[mask_2020, "lat"].values
lons = gdf.loc[mask_2020, "lon"].values

temps = sample_weather_var(inst, "t2m", 8, lats, lons)
print("August 2020 temp (Kelvin):", temps.min(), "to", temps.max())

# variables to sample directly (name in file → nice column prefix)
INSTANT_VARS = {"t2m": "temp", "d2m": "dewpoint", "u10": "wind_u", "v10": "wind_v"}

# initialize all the columns
for prefix in list(INSTANT_VARS.values()) + ["precip"]:
    for month in [6, 7, 8, 9]:
        gdf[f"{prefix}_{month:02d}"] = np.nan

# loop over years, open each year's weather once, sample all variables/months
for year in sorted(gdf["year"].unique()):
    zip_path = WEATHER_DIR / f"era5_{year}_summer.nc"
    if not zip_path.exists():
        print(f"  Missing weather: {year}")
        continue

    inst, accum = open_weather_year(year)

    mask = gdf["year"] == year
    lats = gdf.loc[mask, "lat"].values
    lons = gdf.loc[mask, "lon"].values

    for month in [6, 7, 8, 9]:
        # sample each instant variable and assign to gdf.loc[mask, f"{prefix}_{month:02d}"]
        for varname, prefix in INSTANT_VARS.items():
            sampled = sample_weather_var(inst, varname, month, lats, lons)
            gdf.loc[mask, f"{prefix}_{month:02d}"] = sampled
        # sample precip from accum the same way
        sampled = sample_weather_var(accum, "tp", month, lats, lons)
        gdf.loc[mask, f"precip_{month:02d}"] = sampled

    inst.close()
    accum.close()
    print(f"  Sampled {year}")

for month in [6, 7, 8, 9]:
    u = gdf[f"wind_u_{month:02d}"]
    v = gdf[f"wind_v_{month:02d}"]
    gdf[f"windspeed_{month:02d}"] = np.sqrt(u**2 + v**2)

print(gdf[[f"windspeed_{m:02d}" for m in [6,7,8,9]]].describe())

def relative_humidity(temp_k, dewpoint_k):
    """Compute RH (%) from temperature and dewpoint in Kelvin."""
    temp_c = temp_k - 273.15
    dew_c = dewpoint_k - 273.15
    # Magnus formula
    b, c = 17.625, 243.04
    e_temp = np.exp((b * temp_c) / (c + temp_c))
    e_dew = np.exp((b * dew_c) / (c + dew_c))
    return 100.0 * (e_dew / e_temp)

for month in [6, 7, 8, 9]:
    t = gdf[f"temp_{month:02d}"]
    d = gdf[f"dewpoint_{month:02d}"]
    gdf[f"humidity_{month:02d}"] = relative_humidity(t, d)

print(gdf[[f"humidity_{m:02d}" for m in [6,7,8,9]]].describe())

for month in [6, 7, 8, 9]:
    gdf[f"temp_{month:02d}"] = gdf[f"temp_{month:02d}"] - 273.15

print(gdf[[f"temp_{m:02d}" for m in [6,7,8,9]]].describe())

# ---- Spring precipitation (Jan-May total) ----
gdf["spring_precip"] = np.nan

for year in sorted(gdf["year"].unique()):
    spring_path = WEATHER_DIR / f"era5_{year}_spring.nc"
    if not spring_path.exists():
        print(f"  Missing spring: {year}")
        continue

    # spring files may also be zipped like the summer ones - handle both
    ds = xr.open_dataset(spring_path)

    mask = gdf["year"] == year
    lats = gdf.loc[mask, "lat"].values
    lons = gdf.loc[mask, "lon"].values

    # sum total precipitation across all Jan-May time steps, per grid cell
    total_precip = ds["tp"].sum(dim="valid_time")
    sampled = total_precip.sel(
        latitude=xr.DataArray(lats, dims="points"),
        longitude=xr.DataArray(lons, dims="points"),
        method="nearest",
    ).values

    gdf.loc[mask, "spring_precip"] = sampled
    ds.close()

print("Spring precip range:", gdf["spring_precip"].min(), "to", gdf["spring_precip"].max())

# vapor pressure deficit (kPa) - Tetens' equation, requires temp in Celsius
for month in [6, 7]:
    t = gdf[f"temp_{month:02d}"]          # already Celsius
    rh = gdf[f"humidity_{month:02d}"]     # percent
    sat_vp = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    actual_vp = sat_vp * (rh / 100)
    gdf[f"vpd_{month:02d}"] = sat_vp - actual_vp

print(gdf[[f"vpd_{m:02d}" for m in [6, 7]]].describe())

# fuel raster is in Albers EPSG:5070 — transform points from UTM to Albers
to_albers = Transformer.from_crs("EPSG:32610", "EPSG:5070", always_xy=True)
fuel_x, fuel_y = to_albers.transform(gdf["x"].values, gdf["y"].values)
fuel_coords = list(zip(fuel_x, fuel_y))

with rasterio.open("data/raw/fuel/LF2023_FBFM40_CONUS.tif") as src:
    fuel_codes = np.array([v[0] for v in src.sample(fuel_coords)], dtype=float)

gdf["fuel_code"] = fuel_codes

# check what we got
print("Fuel code counts:")
print(pd.Series(fuel_codes).value_counts().head(20))

def fuel_group(code):
    """Map FBFM40 code to broad fuel type."""
    if code in (91, 92, 93, 98, 99):
        return "nonburnable"
    elif 101 <= code <= 109:
        return "grass"
    elif 121 <= code <= 124:
        return "grass_shrub"
    elif 141 <= code <= 149:
        return "shrub"
    elif 161 <= code <= 165:
        return "timber_understory"
    elif 181 <= code <= 189:
        return "timber_litter"
    else:
        return "other"

gdf["fuel_group"] = gdf["fuel_code"].apply(fuel_group)
print(gdf["fuel_group"].value_counts())

# one-hot encode fuel groups into binary columns
fuel_dummies = pd.get_dummies(gdf["fuel_group"], prefix="fuel").astype(int)
gdf = pd.concat([gdf, fuel_dummies], axis=1)

print("Fuel columns added:", fuel_dummies.columns.tolist())

# drop the temporary lat/lon helper columns if you want, or keep them
# save the completed feature table
OUT = Path("data/processed")
gdf.to_file(OUT / "training_features.gpkg", driver="GPKG")

# also save as CSV (drop geometry for a flat table)
gdf.drop(columns="geometry").to_csv(OUT / "training_features.csv", index=False)

print("Saved training table:", gdf.shape)
print("Columns:", list(gdf.columns))