import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box


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

# TODO: wrap df as a GeoDataFrame, specifying geometry="geometry" and crs="EPSG:32610"
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32610")

# make an output folder for processed data
from pathlib import Path
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# save it
gdf.to_file(PROC / "training_points.gpkg", driver="GPKG")
print("Saved", len(gdf), "points")