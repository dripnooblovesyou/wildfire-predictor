from pyhdf.SD import SD, SDC
from pathlib import Path
import numpy as np

# grab the first NDVI file
f = list(Path("data/raw/ndvi").glob("*.hdf"))[0]
print("Opening:", f.name)

hdf = SD(str(f), SDC.READ)

ndvi_dataset = hdf.select("1 km monthly NDVI")
ndvi = ndvi_dataset.get()

print("Shape:", ndvi.shape)
print("Data type:", ndvi.dtype)
print("Raw value range:", ndvi.min(), "to", ndvi.max())

# check the scale factor and fill value stored in the file's attributes
attrs = ndvi_dataset.attributes()
print("\nScale factor:", attrs.get("scale_factor"))
print("Fill value:", attrs.get("_FillValue"))
print("Valid range:", attrs.get("valid_range"))

# convert raw integers to real NDVI values
ndvi_real = ndvi.astype(float)

# mask out fill values (missing data) before scaling
ndvi_real[ndvi == -3000] = np.nan

# apply scale factor
ndvi_real = ndvi_real / 10000.0

print("Real NDVI range:", np.nanmin(ndvi_real), "to", np.nanmax(ndvi_real))
print("Mean NDVI:", np.nanmean(ndvi_real))