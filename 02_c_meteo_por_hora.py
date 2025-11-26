"""
02g_procesar_era5_HORARIO.py
Procesa archivos ERA5 ZIP manteniendo la dimensión temporal HORARIA
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
    Procesar archivo NetCDF manteniendo datos HORARIOS
    
    Returns:
        DataFrame con: fecha, hora, y todas las variables meteorológicas
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
        
        # Obtener timestamps horarios
        tiempos = pd.to_datetime(ds[dim_tiempo].values)
        
        datos_horarios = []
        
        # Procesar CADA timestamp (mantener granularidad horaria)
        for timestamp in tiempos:
            try:
                ds_hora = ds.sel({dim_tiempo: timestamp})
                
                hora_data = {
                    'datetime': timestamp,
                    'fecha': timestamp.date(),
                    'hora': timestamp.hour
                }
                
                # Mapeo de variables ERA5
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
                
                # Extraer valores ESPACIALES (promedio sobre área)
                for var_era5, (var_nombre, factor, tipo_conv) in var_mapping.items():
                    if var_era5 in ds_hora:
                        # Promediar sobre lat/lon (tomar valor representativo del área)
                        datos = ds_hora[var_era5].values
                        valor_mean = float(np.nanmean(datos))
                        
                        # Aplicar conversiones
                        if tipo_conv == 'kelvin':
                            hora_data[var_nombre] = valor_mean - factor
                        elif tipo_conv == 'j_to_w':
                            hora_data[var_nombre] = valor_mean / factor
                        elif tipo_conv == 'pa_to_hpa':
                            hora_data[var_nombre] = valor_mean / factor
                        elif tipo_conv == 'm_to_mm':
                            hora_data[var_nombre] = valor_mean * factor
                        else:
                            hora_data[var_nombre] = valor_mean
                
                # Features derivadas
                if 'wind_u' in hora_data and 'wind_v' in hora_data:
                    hora_data['wind_speed'] = np.sqrt(
                        hora_data['wind_u']**2 + hora_data['wind_v']**2
                    )
                
                datos_horarios.append(hora_data)
            
            except Exception as e:
                continue
        
        ds.close()
        
        if datos_horarios:
            df = pd.DataFrame(datos_horarios)
            return df
        
        return None
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def procesar_lote_zip_horario(zip_path, lote_num):
    """
    Descomprimir y procesar un archivo ZIP manteniendo datos horarios
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
                print(f" → {len(df)} registros horarios ✓")
            else:
                print(f" → Error ✗")
        
        # Limpiar temporales
        import shutil
        shutil.rmtree(temp_dir)
        
        # Combinar datos de este lote
        if dataframes_lote:
            df_lote = pd.concat(dataframes_lote, ignore_index=True)
            # Eliminar duplicados (mismo datetime)
            df_lote = df_lote.drop_duplicates(subset=['datetime'])
            print(f"  ✓ Lote procesado: {len(df_lote)} registros horarios únicos")
            return df_lote
        else:
            print(f"  ✗ No se pudo procesar ningún archivo del lote")
            return None
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def main():
    """
    Script principal
    """
    
    print("="*70)
    print("PROCESAMIENTO ERA5 CON DATOS HORARIOS")
    print("="*70)
    
    # Buscar archivos ZIP
    archivos_zip = sorted(Path('data/raw').glob('era5_data_*.nc'))
    
    if not archivos_zip:
        print("\n✗ No se encontraron archivos era5_data_*.nc")
        return
    
    print(f"\nArchivos ZIP encontrados: {len(archivos_zip)}\n")
    
    # Procesar cada lote
    todos_dataframes = []
    
    for i, zip_path in enumerate(archivos_zip, 1):
        df_lote = procesar_lote_zip_horario(str(zip_path), i)
        
        if df_lote is not None:
            todos_dataframes.append(df_lote)
    
    # Combinar todos los lotes
    if not todos_dataframes:
        print("\n✗ No se pudo procesar ningún lote")
        return
    
    print("\n" + "="*70)
    print("COMBINACIÓN FINAL")
    print("="*70)
    
    df_final = pd.concat(todos_dataframes, ignore_index=True)
    
    print(f"\nRegistros totales: {len(df_final)}")
    
    # Eliminar duplicados por datetime
    df_final = df_final.drop_duplicates(subset=['datetime'])
    df_final = df_final.sort_values('datetime').reset_index(drop=True)
    
    print(f"Registros únicos (fecha+hora): {len(df_final)}")
    
    # Guardar
    output_path = 'data/raw/datos_meteorologicos_HORARIOS.csv'
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✓ CSV guardado: {output_path}")
    print(f"  Registros: {len(df_final)}")
    print(f"  Columnas: {len(df_final.columns)}")
    print(f"  Rango temporal: {df_final['datetime'].min()} a {df_final['datetime'].max()}")
    
    # Resumen estadístico
    print("\n" + "="*70)
    print("RESUMEN DE DATOS HORARIOS")
    print("="*70)
    
    print(f"\nFechas únicas: {df_final['fecha'].nunique()}")
    print(f"Horas disponibles por día (promedio): {len(df_final) / df_final['fecha'].nunique():.1f}")
    
    # Mostrar ejemplo
    print("\nEjemplo de datos horarios (primer día):")
    primer_dia = df_final['fecha'].iloc[0]
    ejemplo = df_final[df_final['fecha'] == primer_dia][['datetime', 'hora', 'temp_2m', 'cape', 'solar_rad']]
    print(ejemplo.head(10))
    
    # Verificar cobertura horaria
    print("\nCobertura horaria (horas disponibles):")
    horas_disponibles = sorted(df_final['hora'].unique())
    print(f"Horas: {horas_disponibles}")
    
    print("\n" + "="*70)
    print("✓✓✓ PROCESAMIENTO HORARIO COMPLETADO ✓✓✓")
    print("="*70)
    print("\nSiguiente paso:")
    print("  python 03_combinar_datasets_HORARIO.py")
    print("="*70)


if __name__ == "__main__":
    # Instalar h5netcdf si no está
    try:
        import h5netcdf
    except ImportError:
        print("Instalando h5netcdf...")
        os.system('pip install h5netcdf --quiet')
        print()
    
    main()