import rasterio

with rasterio.Env() as env:
    drivers = env.drivers()
    hdf_drivers = [d for d in drivers if 'HDF' in d.upper()]
    print("HDF drivers available:", hdf_drivers)