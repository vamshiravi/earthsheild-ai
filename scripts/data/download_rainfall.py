from pathlib import Path
import earthaccess
from dotenv import load_dotenv

load_dotenv()

output = Path("data/raw/rainfall")
output.mkdir(parents=True, exist_ok=True)

print("Authenticating with NASA Earthdata...")

auth = earthaccess.login(strategy="environment")

if not auth.authenticated:
    raise RuntimeError("Earthdata authentication failed")

print("Searching for IMERG...")

results = earthaccess.search_data(
    concept_id="C2723754864-GES_DISC",
    temporal=("2025-09-30", "2025-09-30"),
)

if not results:
    raise RuntimeError("IMERG granule not found")

print(f"Found {len(results)} granule(s)")
print("Downloading...")

files = earthaccess.download(results, local_path=output)

print("Downloaded:")
for file in files:
    print(file)