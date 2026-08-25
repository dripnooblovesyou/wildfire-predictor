import numpy as np
import joblib
import rasterio
import pandas as pd
import xarray as xr
import zipfile
from pathlib import Path
from pyproj import Transformer
import matplotlib.pyplot as plt
import requests
import geopandas as gpd

model = joblib.load("data/processed/fire_model.pkl")
print("Model loaded")

# same feature order the model was trained on (see 07_model.py, FEATURES_CLEAN)
FEATURES_CLEAN = [
    "elevation", "slope", "northness", "eastness",
    "ndvi_06", "ndvi_07",
    "ndvi_anom_06", "ndvi_anom_07",
    "temp_06", "temp_07",
    "precip_06", "precip_07",
    "windspeed_06", "windspeed_07",
    "humidity_06", "humidity_07",
]

# build a grid over the study area, 5km spacing
X_MIN, X_MAX = 350_000, 1_040_000
Y_MIN, Y_MAX = 4_650_000, 5_450_000
STEP = 5_000

x_coords = np.arange(X_MIN, X_MAX, STEP)
y_coords = np.arange(Y_MIN, Y_MAX, STEP)
xx, yy = np.meshgrid(x_coords, y_coords)

grid_x = xx.ravel()
grid_y = yy.ravel()

print("Grid points:", len(grid_x))
print("X range:", grid_x.min(), "to", grid_x.max())
print("Y range:", grid_y.min(), "to", grid_y.max())

coords = list(zip(grid_x, grid_y))

# sample terrain (static across years)
with rasterio.open("data/raw/terrain/pnw_srtm30_utm.tif") as src:
    elevation = np.array([v[0] for v in src.sample(coords)], dtype=float)

with rasterio.open("data/raw/terrain/pnw_slope.tif") as src:
    slope = np.array([v[0] for v in src.sample(coords)], dtype=float)

with rasterio.open("data/raw/terrain/pnw_aspect.tif") as src:
    aspect = np.array([v[0] for v in src.sample(coords)], dtype=float)

# convert aspect to northness/eastness
northness = np.cos(np.radians(aspect))
eastness = np.sin(np.radians(aspect))

print("Terrain sampled")
print("Elevation range:", np.nanmin(elevation), "to", np.nanmax(elevation))

NDVI_DIR = Path("data/processed/ndvi")

# historical NDVI baseline at each grid point (mean across all years, per month)
ndvi_baseline = {}
for month in [6, 7, 8, 9]:
    stack = []
    for yr in range(2000, 2026):
        rp = NDVI_DIR / f"ndvi_{yr}_{month:02d}.tif"
        if not rp.exists():
            continue
        with rasterio.open(rp) as src:
            stack.append(np.array([v[0] for v in src.sample(coords)], dtype=float))
    ndvi_baseline[month] = np.nanmean(np.vstack(stack), axis=0)

print("NDVI baselines computed")

WEATHER_DIR = Path("data/raw/weather")
WEATHER_TMP = Path("data/tmp/weather")
WEATHER_TMP.mkdir(parents=True, exist_ok=True)

# weather is stored by lat/lon, so reproject the grid once, up front (matches 05_build_features.py)
to_latlon = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
grid_lon, grid_lat = to_latlon.transform(grid_x, grid_y)

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
    monthly = ds[varname].sel(valid_time=ds["valid_time"].dt.month == month)
    monthly_mean = monthly.mean(dim="valid_time")
    sampled = monthly_mean.sel(
        latitude=xr.DataArray(lats, dims="points"),
        longitude=xr.DataArray(lons, dims="points"),
        method="nearest",
    )
    return sampled.values

def relative_humidity(temp_k, dewpoint_k):
    """Compute RH (%) from temperature and dewpoint in Kelvin."""
    temp_c = temp_k - 273.15
    dew_c = dewpoint_k - 273.15
    b, c = 17.625, 243.04
    e_temp = np.exp((b * temp_c) / (c + temp_c))
    e_dew = np.exp((b * dew_c) / (c + dew_c))
    return 100.0 * (e_dew / e_temp)

INSTANT_VARS = {"t2m": "temp", "d2m": "dewpoint", "u10": "wind_u", "v10": "wind_v"}

# accumulate risk predictions across years
risk_sum = np.zeros(len(grid_x))
years_counted = 0

for year in range(2000, 2026):
    weather_zip = WEATHER_DIR / f"era5_{year}_summer.nc"
    if not weather_zip.exists():
        print(f"  Missing weather: {year}")
        continue

    # sample NDVI for each month
    ndvi_cols = {}
    for month in [6, 7, 8, 9]:
        raster_path = NDVI_DIR / f"ndvi_{year}_{month:02d}.tif"
        if not raster_path.exists():
            print(f"  Missing NDVI: {raster_path.name}")
            ndvi_cols[month] = np.full(len(grid_x), np.nan)
            continue
        with rasterio.open(raster_path) as src:
            ndvi_cols[month] = np.array([v[0] for v in src.sample(coords)], dtype=float)

    # anomaly = this year's NDVI − historical baseline
    ndvi_anom = {}
    for month in [6, 7, 8, 9]:
        ndvi_anom[month] = ndvi_cols[month] - ndvi_baseline[month]

    # --- sample weather for each month ---
    inst, accum = open_weather_year(year)

    weather_cols = {}
    for month in [6, 7, 8, 9]:
        for varname, prefix in INSTANT_VARS.items():
            weather_cols[f"{prefix}_{month:02d}"] = sample_weather_var(
                inst, varname, month, grid_lat, grid_lon
            )
        weather_cols[f"precip_{month:02d}"] = sample_weather_var(
            accum, "tp", month, grid_lat, grid_lon
        )

    inst.close()
    accum.close()

    for month in [6, 7, 8, 9]:
        u = weather_cols[f"wind_u_{month:02d}"]
        v = weather_cols[f"wind_v_{month:02d}"]
        weather_cols[f"windspeed_{month:02d}"] = np.sqrt(u**2 + v**2)

        t = weather_cols[f"temp_{month:02d}"]
        d = weather_cols[f"dewpoint_{month:02d}"]
        weather_cols[f"humidity_{month:02d}"] = relative_humidity(t, d)

        weather_cols[f"temp_{month:02d}"] = t - 273.15

    # --- assemble feature table in EXACT FEATURES order ---
    grid_features = pd.DataFrame({
        "elevation": elevation,
        "slope": slope,
        "northness": northness,
        "eastness": eastness,
    })
    for month in [6, 7, 8, 9]:
        grid_features[f"ndvi_{month:02d}"] = ndvi_cols[month]
    for month in [6, 7, 8, 9]:
        grid_features[f"temp_{month:02d}"] = weather_cols[f"temp_{month:02d}"]
    for month in [6, 7, 8, 9]:
        grid_features[f"precip_{month:02d}"] = weather_cols[f"precip_{month:02d}"]
    for month in [6, 7, 8, 9]:
        grid_features[f"windspeed_{month:02d}"] = weather_cols[f"windspeed_{month:02d}"]
    for month in [6, 7, 8, 9]:
        grid_features[f"humidity_{month:02d}"] = weather_cols[f"humidity_{month:02d}"]
    for month in [6, 7, 8, 9]:
        grid_features[f"ndvi_anom_{month:02d}"] = ndvi_anom[month]
    grid_features = grid_features[FEATURES_CLEAN]

    # --- predict and accumulate ---
    probs = model.predict_proba(grid_features)[:, 1]
    risk_sum += probs
    years_counted += 1
    print(f"  Scored {year}")

# average across years
risk_avg = risk_sum / years_counted
print("Years counted:", years_counted)
print("Risk range:", risk_avg.min(), "to", risk_avg.max())

# reshape flat risk back into the 2D grid
risk_grid = risk_avg.reshape(xx.shape)

# mask out nodata areas (ocean, off-coverage) using elevation
elev_grid = elevation.reshape(xx.shape)
risk_grid = np.where(elev_grid <= 0, np.nan, risk_grid)

# download (and cache) Census state boundaries, then keep just WA/OR/ID
BOUNDARIES_DIR = Path("data/raw/boundaries")
BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
states_zip = BOUNDARIES_DIR / "cb_2018_us_state_20m.zip"

if not states_zip.exists():
    resp = requests.get(
        "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip",
        timeout=60,
    )
    resp.raise_for_status()
    states_zip.write_bytes(resp.content)

states = gpd.read_file(f"zip://{states_zip}")
pnw_states = states[states["STUSPS"].isin(["WA", "OR", "ID"])].to_crs("EPSG:32610")

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(
    risk_grid,
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower",
    cmap="YlOrRd",
    vmin=0, vmax=0.6,
    interpolation="bilinear",
)
pnw_states.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8)
fig.colorbar(im, ax=ax, label="Mean fire ignition risk (2000–2025)")
ax.set_title("PNW Wildfire Ignition Risk")
ax.set_xlabel("UTM Easting (m)")
ax.set_ylabel("UTM Northing (m)")
fig.savefig("risk_map.png", dpi=140, bbox_inches="tight")
print("Saved risk_map.png")