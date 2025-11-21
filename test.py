import glob
import xarray as xr

def is_corrupt(path: str) -> bool:
    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            ds.load()
        return False
    except Exception as exc:
        print(f"[CORRUPT] {path} -> {exc}")
        return True

def main():
    patterns = ["data/raw/era5_data_*.nc"]
    corrupt = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            if is_corrupt(path):
                corrupt.append(path)
    if corrupt:
        print("\nArchivos corruptos encontrados:")
        for path in corrupt:
            print(f"  - {path}")
    else:
        print("Todos los NetCDF se abren correctamente.")

if __name__ == "__main__":
    main()