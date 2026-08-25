import geopandas as gpd

fires = gpd.read_file("data/raw/fire/pnw_fires_clean.geojson")
print("Columns:", fires.columns.tolist())
print(fires[["incidentname", "fireyear"]].head())