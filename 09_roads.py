import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path

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