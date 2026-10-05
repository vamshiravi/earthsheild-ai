from pathlib import Path
import earthaccess
from dotenv import load_dotenv

load_dotenv()

output = Path("data/raw/soil_moisture")
output.mkdir(parents=True, exist_ok=True)

print("Authenticating with NASA Earthdata...")

auth = earthaccess.login(strategy="environment")

if not auth.authenticated:
    raise RuntimeError("Earthdata authentication failed")

print("Searching for SMAP...")

results = earthaccess.search_data(
    short_name="SPL3SMP",
    temporal=("2025-09-30", "2025-09-30"),
    count=1,
)

if not results:
    raise RuntimeError("SMAP granule not found")

print("Downloading SMAP...")

files = earthaccess.download(results, local_path=output)

print("Downloaded:")
for file in files:
    print(file)
    
    