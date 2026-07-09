import rasterio
import matplotlib.pyplot as plt
import numpy as np

with rasterio.open("data/raw/terrain/pnw_slope.tif") as src:
    slope = src.read(1)
    bounds = src.bounds

plt.figure(figsize=(10, 8))
plt.imshow(slope, cmap="magma",
           extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
plt.colorbar(label="Slope (degrees)")
plt.title("PNW Slope")
plt.savefig("slope_map.png", dpi=120)
print("Saved slope_map.png")