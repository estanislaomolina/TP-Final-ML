"""
Script COMPLETO: Descomprimir y procesar inmediatamente cada lote
Para evitar que los archivos se sobrescriban
"""

import os
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

def procesar_archivo_nc(filepath, nombre_tiempo='valid_time'):
    """
    Procesar un archivo NetCDF ya abierto
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
        
        # Obtener fechas
        tiempos = pd.to_datetime(ds[dim_tiempo].values)
        fechas_disponibles = sorted(set(tiempos.date))
        
        datos_diarios = []
        
        for fecha in fechas_disponibles:
            fecha_str = str(fecha)
            
            try:
                ds_dia = ds.sel({dim_tiempo: fecha_str})
                stats = {'fecha': fecha_str}
                
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
                
                for var_era5, (var_nombre, factor, tipo_conv) in var_mapping.items():
                    if var_era5 in ds_dia:
                        datos = ds_dia[var_era5].values
                        
                        if tipo_conv == 'kelvin':
                            stats[f'{var_nombre}_max'] = float(np.nanmax(datos) - factor)
                            stats[f'{var_nombre}_min'] = float(np.nanmin(datos) - factor)
                            stats[f'{var_nombre}_mean'] = float(np.nanmean(datos) - factor)
                        elif tipo_conv == 'j_to_w':
                            stats[f'{var_nombre}_total'] = float(np.nansum(datos) / factor)
                            stats[f'{var_nombre}_max'] = float(np.nanmax(datos) / factor)
                        elif tipo_conv == 'pa_to_hpa':
                            stats[f'{var_nombre}_mean'] = float(np.nanmean(datos) / factor)
                        elif tipo_conv == 'm_to_mm':
                            stats[f'{var_nombre}_total'] = float(np.nansum(datos) * factor)
                        else:
                            stats[f'{var_nombre}_mean'] = float(np.nanmean(datos))
                            if var_era5 in ['blh', 'cape']:
                                stats[f'{var_nombre}_max'] = float(np.nanmax(datos))
                
                # Features derivadas
                if 'temp_2m_max' in stats and 'temp_2m_min' in stats:
                    stats['temp_differential'] = stats['temp_2m_max'] - stats['temp_2m_min']
                
                if 'wind_u_mean' in stats and 'wind_v_mean' in stats:
                    stats['wind_speed_mean'] = np.sqrt(
                        stats['wind_u_mean']**2 + stats['wind_v_mean']**2
                    )
                
                datos_diarios.append(stats)
            except:
                continue
        
        ds.close()
        
        if datos_diarios:
            df = pd.DataFrame(datos_diarios)
            df['fecha'] = pd.to_datetime(df['fecha'])
            return df
        
        return None
        
    except Exception as e:
        return None


def procesar_lote_zip(zip_path, lote_num):
    """
    Descomprimir y procesar un archivo ZIP inmediatamente
    """
    
    print(f"\n[Lote {lote_num}] {os.path.basename(zip_path)}")
    
    try:
        if not zipfile.is_zipfile(zip_path):
            print(f"  ✗ No es un archivo ZIP válido")
            return None
        
        # Crear directorio temporal para este lote
        temp_dir = f'data/raw/temp_lote_{lote_num}'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extraer
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            archivos_dentro = zip_ref.namelist()
            print(f"  Extrayendo {len(archivos_dentro)} archivos...")
            zip_ref.extractall(temp_dir)
        
        # Procesar inmediatamente cada archivo NetCDF extraído
        dataframes_lote = []
        
        for nc_file in Path(temp_dir).glob('*.nc'):
            print(f"    Procesando: {nc_file.name}", end='')
            
            df = procesar_archivo_nc(str(nc_file))
            
            if df is not None:
                dataframes_lote.append(df)
                print(f" → {len(df)} fechas ✓")
            else:
                print(f" → Error ✗")
        
        # Limpiar archivos temporales de este lote
        import shutil
        shutil.rmtree(temp_dir)
        
        # Combinar datos de este lote
        if dataframes_lote:
            df_lote = pd.concat(dataframes_lote, ignore_index=True)
            df_lote = df_lote.groupby('fecha', as_index=False).first()  # Eliminar duplicados
            print(f"  ✓ Lote procesado: {len(df_lote)} fechas únicas")
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
    print("PROCESAMIENTO COMPLETO ERA5: LOTE POR LOTE")
    print("="*70)
    
    # Buscar archivos ZIP
    archivos_zip = sorted(Path('data/raw').glob('era5_data_*.nc'))
    
    if not archivos_zip:
        print("\n✗ No se encontraron archivos era5_data_*.nc")
        return
    
    print(f"\nArchivos ZIP encontrados: {len(archivos_zip)}\n")
    
    # Procesar cada lote inmediatamente
    todos_dataframes = []
    
    for i, zip_path in enumerate(archivos_zip, 1):
        df_lote = procesar_lote_zip(str(zip_path), i)
        
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
    
    print(f"\nFilas totales: {len(df_final)}")
    
    # Eliminar duplicados por fecha (por si acaso)
    df_final = df_final.drop_duplicates(subset='fecha')
    df_final = df_final.sort_values('fecha').reset_index(drop=True)
    
    print(f"Fechas únicas: {len(df_final)}")
    
    # Guardar
    output_path = 'data/raw/datos_meteorologicos.csv'
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✓ CSV guardado: {output_path}")
    print(f"  Fechas: {len(df_final)}")
    print(f"  Variables: {len(df_final.columns)}")
    print(f"  Rango: {df_final['fecha'].min().date()} a {df_final['fecha'].max().date()}")
    
    # Resumen estadístico
    print("\n" + "="*70)
    print("RESUMEN ESTADÍSTICO")
    print("="*70)
    
    vars_clave = [col for col in ['temp_2m_max', 'solar_rad_max', 'cape_max', 
                  'boundary_layer_height_max', 'wind_speed_mean'] 
                  if col in df_final.columns]
    
    if vars_clave:
        print("\nVariables meteorológicas:")
        print(df_final[vars_clave].describe())
    
    # Verificar si hay datos faltantes
    print("\n\nVerificación de datos:")
    for col in df_final.columns:
        if col != 'fecha':
            null_count = df_final[col].isnull().sum()
            if null_count > 0:
                print(f"  ⚠️  {col}: {null_count} valores faltantes ({null_count/len(df_final)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✓✓✓ COMPLETADO ✓✓✓")
    print("="*70)
    print("\nSiguiente paso:")
    print("  python 03_combinar_datasets.py")
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