import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np

RAW = Path("data/raw/roads")
RAW.mkdir(parents=True, exist_ok=True)

URLS = {
    "oregon": "https://download.geofabrik.de/north-america/us/oregon-latest-free.shp.zip",
    "washington": "https://download.geofabrik.de/north-america/us/washington-latest-free.shp.zip",
}

# download both state extracts (cached)
for name, url in URLS.items():
    out = RAW / f"{name}.shp.zip"
    if out.exists():
        print(f"  Already have {name}")
        continue
    print(f"  Downloading {name}... (large file, be patient)")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    with open(out, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            f.write(chunk)
    print(f"  Saved {name}")

print("Downloads complete")

# read the roads layer from inside each zip
road_gdfs = []
for name in URLS:
    zip_path = RAW / f"{name}.shp.zip"
    # path into the zip to the roads shapefile
    layer_path = f"zip://{zip_path}!gis_osm_roads_free_1.shp"
    print(f"  Reading {name} roads...")
    roads = gpd.read_file(layer_path)
    print(f"    {len(roads)} road segments, columns: {roads.columns.tolist()}")
    road_gdfs.append(roads)

# combine both states
all_roads = pd.concat(road_gdfs, ignore_index=True)
all_roads = gpd.GeoDataFrame(all_roads, crs=road_gdfs[0].crs)
print("Total road segments:", len(all_roads))
print("CRS:", all_roads.crs)

# reproject roads to UTM
print("Reprojecting roads to UTM...")
all_roads_utm = all_roads.to_crs("EPSG:32610")

# load training points
points = gpd.read_file("data/processed/training_features.gpkg")
print("Points loaded:", len(points))

# nearest-road distance via spatial join
print("Computing nearest-road distances...")
joined = gpd.sjoin_nearest(points, all_roads_utm, how="left", distance_col="dist_to_road")

# sjoin_nearest can create duplicate rows if ties occur; keep first per point
joined = joined[~joined.index.duplicated(keep="first")]

print("Distance to road (m):")
print(joined["dist_to_road"].describe())

# attach distance back to points and save
points["dist_to_road"] = joined["dist_to_road"].values

# save updated feature table
points.to_file("data/processed/training_features.gpkg", driver="GPKG")
points.drop(columns="geometry").to_csv("data/processed/training_features.csv", index=False)
print("Saved training table with dist_to_road:", points.shape)