from pathlib import Path
import requests
import geopandas as gpd
import json

# creates the raw data directory if it doesn't exist
RAW = Path("data/raw/fire")
RAW.mkdir(parents=True, exist_ok=True)

# boxes out the pnw region
PNW_BBOX = {
    "min_lon": -124.8,
    "max_lon": -116.5,
    "min_lat":   42.0,
    "max_lat":   49.0,
}

# sets up the parameters for the API request
spatial_filter = {
    "xmin": PNW_BBOX["min_lon"],
    "ymin": PNW_BBOX["min_lat"],
    "xmax": PNW_BBOX["max_lon"],
    "ymax": PNW_BBOX["max_lat"],
    "spatialReference": {"wkid": 4326},
}

params = {
    "where": "1=1",
    "geometry": json.dumps(spatial_filter),
    "geometryType": "esriGeometryEnvelope",
    "spatialRel": "esriSpatialRelIntersects",
    "outFields": "incidentname,fireyear,gisacres,state",
    "returnGeometry": "true",
    "f": "geojson",
    "resultRecordCount": 2000,
}

# sends a request to the ArcGIS API to get the fire data for the PNW region
BASE_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "Historic_Geomac_Perimeters_Combined_2000_2018/FeatureServer/0/query"
)

all_features = []
offset = 0

# loops through the API results until there are no more features to retrieve
while True:
    params["resultOffset"] = offset
    resp = requests.get(BASE_URL, params=params, timeout=60)
    data = resp.json()
    features = data.get("features", [])
    if not features:
        break
    all_features.extend(features)
    print(f"  Downloaded {len(all_features)} records so far...")
    offset += 2000

# saves the downloaded fire data as a GeoJSON file
geojson = {"type": "FeatureCollection", "features": all_features}
out_path = RAW / "pnw_fires_raw.geojson"
out_path.write_text(json.dumps(geojson))

# filter the fire data to only include fires in Oregon, Washington, and Idaho
fires = gpd.read_file(out_path)
fires = fires[fires["state"].isin(["OR", "WA", "ID"])]

# gets rid of smaller fires that are less than 10 acres in size
fires = fires[fires["gisacres"] >= 10]

# save the cleaned fire data as a GeoJSON file
clean_path = RAW / "pnw_fires_clean.geojson"
fires.to_file(clean_path, driver="GeoJSON")
print("Saved:", clean_path)