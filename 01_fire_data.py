from pathlib import Path
import requests
import geopandas as gpd
import json
import pandas as pd

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

params_2 = {
    "where": "1=1",
    "geometry": json.dumps(spatial_filter),
    "geometryType": "esriGeometryEnvelope",
    "spatialRel": "esriSpatialRelIntersects",
    "outFields": "poly_IncidentName,attr_FireDiscoveryDateTime,poly_GISAcres,attr_POOState",
    "returnGeometry": "true",
    "f": "geojson",
    "resultRecordCount": 2000,
}

# sends a request to the ArcGIS API to get the fire data for the PNW region
BASE_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "Historic_Geomac_Perimeters_Combined_2000_2018/FeatureServer/0/query"
)

# sends a request to the ArcGIS API to get the fire data for the WFIGS region
WFIGS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)

# function to download fire data from APIs
def download_fires(base_url, params):
    all_features = []
    offset = 0

    # loop to download all records in batches of 2000 until there are no more records left
    while True:
        params["resultOffset"] = offset
        resp = requests.get(base_url, params=params, timeout=60)
        data = resp.json()
        features = data.get("features", [])
        if not features:
            break
        all_features.extend(features)
        print(f"  Downloaded {len(all_features)} records so far...")
        offset += 2000
    return all_features

# function to get fire data, either from cache or by downloading
def get_fires(base_url, params, cache_path):
    if cache_path.exists():
        print(f"  Loading cached: {cache_path.name}")
        return json.loads(cache_path.read_text())["features"]
    features = download_fires(base_url, params)
    cache_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return features

all_features = get_fires(BASE_URL, params, RAW / "pnw_fires_raw.geojson")

features_2019 = get_fires(WFIGS_URL, params_2, RAW / "wfigs_raw.geojson")

# build the old-dataset GeoDataFrame from the cached features
fires = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")

# filter the fire data to only include fires in Oregon, Washington, and Idaho
fires = fires[fires["state"].isin(["OR", "WA", "ID"])]

# gets rid of smaller fires that are less than 10 acres in size
fires = fires[fires["gisacres"] >= 10]

# saves the 2019 fire data as a GeoJSON file
geojson_2019 = {"type": "FeatureCollection", "features": features_2019}
fires_2019 = gpd.GeoDataFrame.from_features(features_2019, crs="EPSG:4326")

# rename columns to match the old dataset
fires_2019 = fires_2019.rename(columns={
    "poly_IncidentName": "incidentname",
    "poly_GISAcres": "gisacres",
})

# convert millisecond timestamp to just the year
fires_2019["fireyear"] = pd.to_datetime(
    fires_2019["attr_FireDiscoveryDateTime"], unit="ms"
).dt.year

# strip the "US-" prefix from state
fires_2019["state"] = fires_2019["attr_POOState"].str.replace("US-", "", regex=False)

# keep only the shared columns in both datasets
cols = ["incidentname", "fireyear", "gisacres", "state", "geometry"]
fires_old = fires[cols]
fires_new = fires_2019[cols]

# combine into one dataset
all_fires = pd.concat([fires_old, fires_new], ignore_index=True)
all_fires = gpd.GeoDataFrame(all_fires, geometry="geometry")

# apply the same filters as before
all_fires = all_fires[all_fires["state"].isin(["OR", "WA", "ID"])]
all_fires = all_fires[all_fires["gisacres"] >= 10]

print(all_fires.shape)
print(all_fires["fireyear"].value_counts().sort_index())

clean_path = RAW / "pnw_fires_clean.geojson"
all_fires.to_file(clean_path, driver="GeoJSON")
print("Saved:", clean_path)