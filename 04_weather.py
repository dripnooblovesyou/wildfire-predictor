import cdsapi
from pathlib import Path

RAW = Path("data/raw/weather")
RAW.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "total_precipitation",
]

def download_year(year):
    out = RAW / f"era5_{year}_summer.nc"
    if out.exists():
        print(f"  Already have {out.name}")
        return

    print(f"  Requesting {year}... (this may queue for a while)")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year": str(year),
            "month": ["06", "07", "08", "09"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["12:00"],
            "area": [49.0, -124.8, 42.0, -116.5],  # N, W, S, E
            "format": "netcdf",
        },
        str(out),
    )
    print(f"  Saved {out.name}")

for year in range(2000, 2026):
    download_year(year)