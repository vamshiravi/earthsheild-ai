import xarray as xr

path = "data/raw/rainfall/3B-DAY.MS.MRG.3IMERG.20250930-S000000-E235959.V07B.nc4"

ds = xr.open_dataset(path)

print(ds)
print("\nVariables:")
print(list(ds.data_vars))
