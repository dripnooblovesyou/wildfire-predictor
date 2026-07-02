import geopandas as gpd
import matplotlib.pyplot as plt

fires = gpd.read_file("data/raw/fire/pnw_fires_clean.geojson")

fig, ax = plt.subplots(figsize=(10, 8))
fires.plot(ax=ax, color="orangered", edgecolor="none", alpha=0.6)
ax.set_title("PNW Wildfires 2000-2026")
plt.savefig("fires_map.png", dpi=120)
print("Saved fires_map.png")