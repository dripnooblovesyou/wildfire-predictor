import rasterio
import matplotlib.pyplot as plt

with rasterio.open("data/processed/ndvi/ndvi_2015_07.tif") as src:
    ndvi = src.read(1)
    bounds = src.bounds
    print("Shape:", src.width, "x", src.height)
    print("Bounds:", bounds)

plt.figure(figsize=(10, 8))
plt.imshow(ndvi, cmap="RdYlGn",
           extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
plt.colorbar(label="NDVI")
plt.title("NDVI (2015-07)")
plt.savefig("ndvi_map.png", dpi=120)
print("Saved")