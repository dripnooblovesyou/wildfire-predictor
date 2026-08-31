"""
08_risk_map.py — Apply the final model across a PNW grid to produce
an average wildfire ignition risk map (2000–2025).

Final feature set (22 features):
  terrain:      elevation, slope, northness, eastness
  vegetation:   ndvi_06, ndvi_07, ndvi_anom_06, ndvi_anom_07
  moisture:     vpd_06, vpd_07
  weather:      precip_06, precip_07, windspeed_06, windspeed_07
  human:        dist_to_road
  seasonal:     spring_precip
  fuel:         fuel_grass, fuel_grass_shrub, fuel_shrub,
                fuel_timber_understory, fuel_timber_litter, fuel_nonburnable
"""

import numpy as np
import pandas as pd
import joblib
import rasterio
import xarray as xr
import zipfile
import geopandas as gpd
from pathlib import Path
from pyproj import Transformer

# ── Feature order MUST match 07_model.py exactly ──────────────────────────
FEATURES = [
    "elevation", "slope", "northness", "eastness",
    "ndvi_06", "ndvi_07",
    "ndvi_anom_06", "ndvi_anom_07",
    "vpd_06", "vpd_07",
    "precip_06", "precip_07",
    "windspeed_06", "windspeed_07",
    "dist_to_road",
    "spring_precip",
    "fuel_grass", "fuel_grass_shrub", "fuel_shrub",
    "fuel_timber_understory", "fuel_timber_litter", "fuel_nonburnable",
]

model = joblib.load("data/processed/fire_model.pkl")
print("Model loaded")

# ── Build grid over the study area, 5km spacing ───────────────────────────
X_MIN, X_MAX = 350_000, 1_040_000
Y_MIN, Y_MAX = 4_650_000, 5_450_000
STEP = 5_000

x_coords = np.arange(X_MIN, X_MAX, STEP)
y_coords = np.arange(Y_MIN, Y_MAX, STEP)
xx, yy = np.meshgrid(x_coords, y_coords)
grid_x = xx.ravel()
grid_y = yy.ravel()
coords = list(zip(grid_x, grid_y))
print("Grid points:", len(grid_x))

# ── Coordinate transforms for the grid ────────────────────────────────────
to_latlon = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
grid_lon, grid_lat = to_latlon.transform(grid_x, grid_y)

to_albers = Transformer.from_crs("EPSG:32610", "EPSG:5070", always_xy=True)
grid_ax, grid_ay = to_albers.transform(grid_x, grid_y)
albers_coords = list(zip(grid_ax, grid_ay))

# ── 1. Terrain (static) ───────────────────────────────────────────────────
with rasterio.open("data/raw/terrain/pnw_srtm30_utm.tif") as src:
    elevation = np.array([v[0] for v in src.sample(coords)], dtype=float)
with rasterio.open("data/raw/terrain/pnw_slope.tif") as src:
    slope = np.array([v[0] for v in src.sample(coords)], dtype=float)
with rasterio.open("data/raw/terrain/pnw_aspect.tif") as src:
    aspect = np.array([v[0] for v in src.sample(coords)], dtype=float)
northness = np.cos(np.radians(aspect))
eastness = np.sin(np.radians(aspect))
print("Terrain sampled")

# ── 2. Distance to road (static) ──────────────────────────────────────────
print("Sampling road distance...")
grid_pts = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(grid_x, grid_y), crs="EPSG:32610"
)
roads = []
for name in ["oregon", "washington"]:
    layer = f"zip://data/raw/roads/{name}.shp.zip!gis_osm_roads_free_1.shp"
    roads.append(gpd.read_file(layer))
roads = pd.concat(roads, ignore_index=True)
roads = gpd.GeoDataFrame(roads, crs="EPSG:4326").to_crs("EPSG:32610")
joined = gpd.sjoin_nearest(grid_pts, roads, how="left", distance_col="dist_to_road")
joined = joined[~joined.index.duplicated(keep="first")]
dist_to_road = joined["dist_to_road"].values
print("Road distance sampled")

# ── 3. Fuel type (static) ─────────────────────────────────────────────────
print("Sampling fuel type...")
with rasterio.open("data/raw/fuel/LF2023_FBFM40_CONUS.tif") as src:
    fuel_code = np.array([v[0] for v in src.sample(albers_coords)], dtype=float)

def fuel_group(code):
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

fuel_groups = pd.Series([fuel_group(c) for c in fuel_code])
# one-hot into the same columns the model expects
fuel_onehot = {}
for grp in ["grass", "grass_shrub", "shrub",
            "timber_understory", "timber_litter", "nonburnable"]:
    fuel_onehot[f"fuel_{grp}"] = (fuel_groups == grp).astype(int).values
print("Fuel sampled")

# ── 4. Spring precip baseline is per-year; sampled in loop below ───────────
NDVI_DIR = Path("data/processed/ndvi")
WEATHER_DIR = Path("data/raw/weather")
WEATHER_TMP = Path("data/tmp/weather")
INSTANT_VARS = {"t2m": "temp", "d2m": "dewpoint", "u10": "wind_u", "v10": "wind_v"}

# ── NDVI historical baseline (for anomaly), months 06 & 07 ────────────────
ndvi_baseline = {}
for month in [6, 7]:
    stack = []
    for yr in range(2000, 2026):
        rp = NDVI_DIR / f"ndvi_{yr}_{month:02d}.tif"
        if not rp.exists():
            continue
        with rasterio.open(rp) as src:
            stack.append(np.array([v[0] for v in src.sample(coords)], dtype=float))
    ndvi_baseline[month] = np.nanmean(np.vstack(stack), axis=0)
print("NDVI baselines computed")

# ── Weather helpers ───────────────────────────────────────────────────────
def open_weather_year(year):
    zip_path = WEATHER_DIR / f"era5_{year}_summer.nc"
    extract_dir = WEATHER_TMP / str(year)
    inst_p = extract_dir / "data_stream-oper_stepType-instant.nc"
    acc_p = extract_dir / "data_stream-oper_stepType-accum.nc"
    if not inst_p.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    return xr.open_dataset(inst_p), xr.open_dataset(acc_p)

def sample_var(ds, varname, month, lats, lons):
    monthly = ds[varname].sel(valid_time=ds["valid_time"].dt.month == month)
    mean = monthly.mean(dim="valid_time")
    return mean.sel(
        latitude=xr.DataArray(lats, dims="points"),
        longitude=xr.DataArray(lons, dims="points"),
        method="nearest",
    ).values

def rh(temp_k, dew_k):
    tc, dc = temp_k - 273.15, dew_k - 273.15
    b, c = 17.625, 243.04
    return 100.0 * (np.exp(b * dc / (c + dc)) / np.exp(b * tc / (c + tc)))

# ── Spring precip baseline (static across the map, per-year varies) ───────
# spring precip is per-year; we average predictions, so it's sampled in loop

# ── Loop over years, predict, accumulate ──────────────────────────────────
risk_sum = np.zeros(len(grid_x))
years_counted = 0

for year in range(2000, 2026):
    if not (WEATHER_DIR / f"era5_{year}_summer.nc").exists():
        continue

    # NDVI 06/07 + anomaly
    ndvi = {}
    for month in [6, 7]:
        rp = NDVI_DIR / f"ndvi_{year}_{month:02d}.tif"
        if not rp.exists():
            ndvi[month] = np.full(len(grid_x), np.nan)
            continue
        with rasterio.open(rp) as src:
            ndvi[month] = np.array([v[0] for v in src.sample(coords)], dtype=float)

    # weather
    inst, accum = open_weather_year(year)
    wx = {}
    for month in [6, 7]:
        for v, p in INSTANT_VARS.items():
            wx[f"{p}_{month:02d}"] = sample_var(inst, v, month, grid_lat, grid_lon)
        wx[f"precip_{month:02d}"] = sample_var(accum, "tp", month, grid_lat, grid_lon)
    inst.close(); accum.close()

    for month in [6, 7]:
        u, v = wx[f"wind_u_{month:02d}"], wx[f"wind_v_{month:02d}"]
        wx[f"windspeed_{month:02d}"] = np.sqrt(u**2 + v**2)
        t, d = wx[f"temp_{month:02d}"], wx[f"dewpoint_{month:02d}"]
        humidity = rh(t, d)
        tc = t - 273.15
        sat_vp = 0.6108 * np.exp(17.27 * tc / (tc + 237.3))
        wx[f"vpd_{month:02d}"] = sat_vp - sat_vp * (humidity / 100)

    # spring precip for this year
    spring_p = WEATHER_DIR / f"era5_{year}_spring.nc"
    if spring_p.exists():
        ds = xr.open_dataset(spring_p)
        total = ds["tp"].sum(dim="valid_time")
        spring_precip = total.sel(
            latitude=xr.DataArray(grid_lat, dims="points"),
            longitude=xr.DataArray(grid_lon, dims="points"),
            method="nearest",
        ).values
        ds.close()
    else:
        spring_precip = np.full(len(grid_x), np.nan)

    # assemble features in EXACT FEATURES order
    gf = pd.DataFrame({
        "elevation": elevation, "slope": slope,
        "northness": northness, "eastness": eastness,
        "ndvi_06": ndvi[6], "ndvi_07": ndvi[7],
        "ndvi_anom_06": ndvi[6] - ndvi_baseline[6],
        "ndvi_anom_07": ndvi[7] - ndvi_baseline[7],
        "vpd_06": wx["vpd_06"], "vpd_07": wx["vpd_07"],
        "precip_06": wx["precip_06"], "precip_07": wx["precip_07"],
        "windspeed_06": wx["windspeed_06"], "windspeed_07": wx["windspeed_07"],
        "dist_to_road": dist_to_road,
        "spring_precip": spring_precip,
    })
    for k, v in fuel_onehot.items():
        gf[k] = v
    gf = gf[FEATURES]

    probs = model.predict_proba(gf)[:, 1]
    risk_sum += probs
    years_counted += 1
    print(f"  Scored {year}")

risk_avg = risk_sum / years_counted
print("Years counted:", years_counted)
print("Risk range:", risk_avg.min(), "to", risk_avg.max())

# ── Render the map ────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

risk_grid = risk_avg.reshape(xx.shape)
elev_grid = elevation.reshape(xx.shape)
risk_grid = np.where(elev_grid <= 0, np.nan, risk_grid)

plt.figure(figsize=(10, 9))
plt.imshow(
    risk_grid,
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower", cmap="YlOrRd", vmin=0, vmax=0.6,
    interpolation="bilinear",
)
plt.colorbar(label="Mean fire ignition risk (2000–2025)")
plt.title("PNW Wildfire Ignition Risk")
plt.xlabel("UTM Easting (m)")
plt.ylabel("UTM Northing (m)")
plt.savefig("risk_map.png", dpi=140, bbox_inches="tight")
print("Saved risk_map.png")