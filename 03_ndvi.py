from pathlib import Path
import earthaccess

# creates the raw data directory if it doesn't exist
RAW = Path("data/raw/ndvi")
RAW.mkdir(parents=True, exist_ok=True)

# boxes out the pnw region
PNW_BBOX = {
    "min_lon": -124.8,
    "max_lon": -116.5,
    "min_lat":   42.0,
    "max_lat":   49.0,
}

# logs into earthdata (reads from ~/.netrc or prompts/stores credentials)
auth = earthaccess.login()
print("Logged in:", auth.authenticated)

# function to search for and download MOD13A3 NDVI granules
def get_ndvi():
    """
    Search NASA Earthdata for MODIS MOD13A3 (monthly NDVI, 1km) granules
    covering PNW_BBOX, restricted to the relevant date range, then download
    the matching granules into RAW.
    """
    all_results = []
    for year in range(2000, 2027):
        results = earthaccess.search_data(
            short_name="MOD13A3",
            bounding_box=(
                PNW_BBOX["min_lon"],
                PNW_BBOX["min_lat"],
                PNW_BBOX["max_lon"],
                PNW_BBOX["max_lat"],
            ),
            temporal=(f"{year}-06-01", f"{year}-09-30"),
        )
        print(f"  {year}: {len(results)} granules")
        all_results.extend(results)

    print(f"Total: {len(all_results)} granules")
    print("Downloading... (this will take a while)")
    files = earthaccess.download(all_results, str(RAW))
    print(f"Downloaded {len(files)} files to {RAW}")

    return files
    
results = get_ndvi()