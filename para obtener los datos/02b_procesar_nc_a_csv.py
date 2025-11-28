"""
Script: Procesar ERA5 y generar datos HORARIOS (no agregados diarios)
"""

import os
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

def procesar_archivo_nc_horario(filepath):
    """
    Procesar archivo NetCDF y extraer datos HORA POR HORA
    """
    
    try:
        ds = xr.open_dataset(filepath, engine='h5netcdf')
        
        # Identificar dimensión de tiempo
        dim_tiempo = None
        for posible in ['time', 'valid_time', 'forecast_reference_time', 't']:
            if posible in ds.dims or posible in ds.coords:
                dim_tiempo = posible
                break
        
        if dim_tiempo is None:
            ds.close()
            return None
        
        # Obtener tiempos
        tiempos = pd.to_datetime(ds[dim_tiempo].values)
        
        datos_horarios = []
        
        # Variables ERA5
        var_mapping = {
            't2m': ('temp_2m', 273.15, 'kelvin'),
            '2t': ('temp_2m', 273.15, 'kelvin'),
            'ssrd': ('solar_rad', 3600, 'j_to_w'),
            'tcc': ('cloud_cover', 1, 'fraction'),
            'u10': ('wind_u', 1, 'ms'),
            'v10': ('wind_v', 1, 'ms'),
            'sp': ('pressure', 100, 'pa_to_hpa'),
            'blh': ('boundary_layer_height', 1, 'meter'),
            'cape': ('cape', 1, 'jkg'),
            'skt': ('skin_temp', 273.15, 'kelvin'),
            'tp': ('precipitation', 1000, 'm_to_mm'),
        }
        
        # Procesar cada timestep
        for i, tiempo in enumerate(tiempos):
            registro = {
                'datetime': tiempo,
                'fecha': tiempo.date(),
                'hora': tiempo.hour
            }
            
            # Extraer datos de cada variable para este timestep
            for var_era5, (var_nombre, factor, tipo_conv) in var_mapping.items():
                if var_era5 in ds:
                    try:
                        # Obtener valor en este timestep
                        valor = ds[var_era5].isel({dim_tiempo: i}).values
                        
                        # Convertir si es array
                        if isinstance(valor, np.ndarray):
                            valor = float(valor.flatten()[0])
                        else:
                            valor = float(valor)
                        
                        # Aplicar conversión
                        if tipo_conv == 'kelvin':
                            registro[var_nombre] = valor - factor
                        elif tipo_conv == 'j_to_w':
                            registro[var_nombre] = valor / factor
                        elif tipo_conv == 'pa_to_hpa':
                            registro[var_nombre] = valor / factor
                        elif tipo_conv == 'm_to_mm':
                            registro[var_nombre] = valor * factor
                        else:
                            registro[var_nombre] = valor
                            
                    except:
                        registro[var_nombre] = np.nan
            
            # Calcular wind_speed
            if 'wind_u' in registro and 'wind_v' in registro:
                if not np.isnan(registro['wind_u']) and not np.isnan(registro['wind_v']):
                    registro['wind_speed'] = np.sqrt(
                        registro['wind_u']**2 + registro['wind_v']**2
                    )
            
            datos_horarios.append(registro)
        
        ds.close()
        
        if datos_horarios:
            df = pd.DataFrame(datos_horarios)
            return df
        
        return None
        
    except Exception as e:
        print(f"    Error procesando archivo: {e}")
        return None


def procesar_lote_zip_horario(zip_path, lote_num):
    """
    Descomprimir y procesar un ZIP extrayendo datos HORARIOS
    """
    
    print(f"\n[Lote {lote_num}] {os.path.basename(zip_path)}")
    
    try:
        if not zipfile.is_zipfile(zip_path):
            print(f"  ✗ No es un archivo ZIP válido")
            return None
        
        # Crear directorio temporal
        temp_dir = f'data/raw/temp_lote_{lote_num}'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extraer
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            archivos_dentro = zip_ref.namelist()
            print(f"  Extrayendo {len(archivos_dentro)} archivos...")
            zip_ref.extractall(temp_dir)
        
        # Procesar cada archivo NetCDF
        dataframes_lote = []
        
        for nc_file in Path(temp_dir).glob('*.nc'):
            print(f"    Procesando: {nc_file.name}", end='')
            
            df = procesar_archivo_nc_horario(str(nc_file))
            
            if df is not None:
                dataframes_lote.append(df)
                print(f" → {len(df)} horas ✓")
            else:
                print(f" → Error ✗")
        
        # Limpiar
        import shutil
        shutil.rmtree(temp_dir)
        
        # Combinar datos de instant.nc y accum.nc por datetime
        if dataframes_lote:
            if len(dataframes_lote) > 1:
                # Merge por datetime
                df_lote = dataframes_lote[0]
                for df_extra in dataframes_lote[1:]:
                    df_lote = df_lote.merge(df_extra, on=['datetime', 'fecha', 'hora'], how='outer', suffixes=('', '_dup'))
                    # Eliminar columnas duplicadas
                    df_lote = df_lote[[c for c in df_lote.columns if not c.endswith('_dup')]]
            else:
                df_lote = dataframes_lote[0]
            
            df_lote = df_lote.sort_values('datetime').reset_index(drop=True)
            print(f"  ✓ Lote procesado: {len(df_lote)} horas únicas")
            return df_lote
        else:
            print(f"  ✗ No se pudo procesar ningún archivo")
            return None
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def main():
    """
    Script principal
    """
    
    print("="*70)
    print("PROCESAMIENTO ERA5: DATOS HORARIOS")
    print("="*70)
    
    # Buscar archivos .nc
    archivos_nc = sorted(Path('data/raw').glob('era5_data_*.nc'))
    
    if not archivos_nc:
        print("\n✗ No se encontraron archivos era5_data_*.nc en data/raw/")
        print("\nBuscando otros patrones...")
        archivos_nc = sorted(Path('data/raw').glob('*.nc'))
        
        if not archivos_nc:
            print("✗ No se encontró ningún archivo .nc")
            return
    
    print(f"\nArchivos encontrados: {len(archivos_nc)}\n")
    
    # Procesar cada archivo
    todos_dataframes = []
    
    for i, nc_path in enumerate(archivos_nc, 1):
        
        # Verificar si es ZIP
        if zipfile.is_zipfile(nc_path):
            df_lote = procesar_lote_zip_horario(str(nc_path), i)
        else:
            # Procesar directamente como NetCDF
            print(f"\n[Archivo {i}] {nc_path.name}")
            df_lote = procesar_archivo_nc_horario(str(nc_path))
            
            if df_lote is not None:
                print(f"  ✓ {len(df_lote)} horas procesadas")
        
        if df_lote is not None:
            todos_dataframes.append(df_lote)
    
    # Combinar
    if not todos_dataframes:
        print("\n✗ No se pudo procesar ningún archivo")
        return
    
    print("\n" + "="*70)
    print("COMBINACIÓN FINAL")
    print("="*70)
    
    df_final = pd.concat(todos_dataframes, ignore_index=True)
    
    print(f"\nFilas totales: {len(df_final)}")
    
    # Eliminar duplicados
    df_final = df_final.drop_duplicates(subset='datetime')
    df_final = df_final.sort_values('datetime').reset_index(drop=True)
    
    print(f"Horas únicas: {len(df_final)}")
    
    # Guardar
    output_path = 'data/raw/datos_meteorologicos_HORARIOS.csv'
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✓ CSV guardado: {output_path}")
    print(f"  Horas: {len(df_final)}")
    print(f"  Variables: {len(df_final.columns)}")
    
    if len(df_final) > 0:
        print(f"  Rango: {df_final['datetime'].min()} a {df_final['datetime'].max()}")
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    
    print(f"\nPrimeras 5 filas:")
    print(df_final.head())
    
    print(f"\n✓✓✓ COMPLETADO ✓✓✓")
    print("\nAhora ejecutá:")
    print("  python 03b_agregar_features_meteorologicas_completas.py")


if __name__ == "__main__":
    # Instalar h5netcdf si no está
    try:
        import h5netcdf
    except ImportError:
        print("Instalando h5netcdf...")
        os.system('pip install h5netcdf --quiet')
    
    main()