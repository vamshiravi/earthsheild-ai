import earthaccess
from pathlib import Path

earthaccess.login()

results = earthaccess.search_data(
    short_name="GPM_3IMERGDF",
    version="07",
    temporal=("2025-09-01", "2025-09-30"),
)

print(f"Granules found: {len(results)}")

output = Path("data/raw/rainfall")
output.mkdir(parents=True, exist_ok=True)

files = earthaccess.download(
    results,
    local_path=str(output)
)

print(f"Downloaded: {len(files)} files")