import h5py

path = "data/raw/soil_moisture/SMAP_L3_SM_P_20250930_R19240_001.h5"

with h5py.File(path, "r") as file:
    def show(name, obj):
        print(name)

    file.visititems(show)