import zipfile
import xarray as xr
from pathlib import Path

zip_path = Path("data/raw/weather/era5_2020_summer.nc")
extract_dir = Path("data/raw/weather/extracted")
extract_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path) as z:
    z.extractall(extract_dir)

# open the instantaneous variables
inst = xr.open_dataset(extract_dir / "data_stream-oper_stepType-instant.nc")
print("=== INSTANT ===")
print(inst)

print("\n=== ACCUM ===")
accum = xr.open_dataset(extract_dir / "data_stream-oper_stepType-accum.nc")
print(accum)