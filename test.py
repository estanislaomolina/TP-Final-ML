"""
inspeccionar_nc.py
Script para ver qué variables realmente contienen tus archivos NetCDF
"""

import xarray as xr
import zipfile
from pathlib import Path
import os


def inspeccionar_nc(filepath):
    """Inspecciona un archivo NetCDF y muestra su contenido"""
    
    try:
        ds = xr.open_dataset(filepath, engine='netcdf4')
        
        print(f"\n{'='*80}")
        print(f"Archivo: {os.path.basename(filepath)}")
        print(f"{'='*80}")
        
        # Dimensiones
        print("\n▶ DIMENSIONES:")
        for dim, size in ds.dims.items():
            print(f"  {dim}: {size}")
        
        # Coordenadas
        print("\n▶ COORDENADAS:")
        for coord in ds.coords:
            print(f"  {coord}: {ds.coords[coord].shape}")
        
        # Variables de datos
        print("\n▶ VARIABLES DE DATOS:")
        for var in ds.data_vars:
            attrs = ds[var].attrs
            long_name = attrs.get('long_name', 'Sin descripción')
            units = attrs.get('units', 'Sin unidades')
            print(f"  {var}:")
            print(f"    Descripción: {long_name}")
            print(f"    Unidades: {units}")
            print(f"    Shape: {ds[var].shape}")
        
        # Atributos globales
        print("\n▶ ATRIBUTOS GLOBALES:")
        for attr, value in ds.attrs.items():
            print(f"  {attr}: {value}")
        
        ds.close()
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    print("="*80)
    print("INSPECCIÓN DE ARCHIVOS ERA5")
    print("="*80)
    
    # Buscar archivos .nc
    archivos_nc = list(Path('data/raw').glob('*.nc'))
    
    if not archivos_nc:
        print("\n✗ No se encontraron archivos .nc")
        return
    
    print(f"\n✓ Encontrados {len(archivos_nc)} archivos")
    
    # Verificar si el primero es ZIP
    primer_archivo = archivos_nc[0]
    
    print(f"\nInspeccionando: {primer_archivo.name}")
    
    # Verificar si es ZIP
    try:
        with open(primer_archivo, 'rb') as f:
            es_zip = f.read(4) == b'PK\x03\x04'
    except:
        es_zip = False
    
    if es_zip:
        print("→ Es un archivo ZIP, descomprimiendo...")
        
        # Crear directorio temporal
        temp_dir = Path('data/raw/temp_inspect')
        temp_dir.mkdir(exist_ok=True)
        
        # Descomprimir
        with zipfile.ZipFile(primer_archivo, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Inspeccionar cada archivo .nc dentro
        archivos_dentro = list(temp_dir.glob('*.nc'))
        print(f"✓ {len(archivos_dentro)} archivos .nc dentro del ZIP")
        
        for nc_file in archivos_dentro:
            inspeccionar_nc(str(nc_file))
        
        # Limpiar
        import shutil
        shutil.rmtree(temp_dir)
        
    else:
        # Es un NetCDF directo
        inspeccionar_nc(str(primer_archivo))
    
    print("\n" + "="*80)
    print("INSPECCIÓN COMPLETADA")
    print("="*80)
    print("\nAhora puedes ajustar el mapeo de variables en procesar_era5_COMPLETO.py")


if __name__ == "__main__":
    main()