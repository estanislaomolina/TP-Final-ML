"""
procesar_era5_COMPLETO.py
========================================
Script ÚNICO para procesar archivos ERA5 .nc

OUTPUTS:
1. datos_meteorologicos_AGREGADOS_DIA.csv  → Features agregadas por día
2. datos_meteorologicos_HORARIOS.csv       → Todas las variables por cada hora

EJECUTAR: python procesar_era5_COMPLETO.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import zipfile
import shutil
import warnings
warnings.filterwarnings('ignore')


def es_zip(filepath):
    """Verifica si un archivo es un ZIP"""
    try:
        with open(filepath, 'rb') as f:
            return f.read(4) == b'PK\x03\x04'
    except:
        return False


def descomprimir_archivo(zip_path, extract_dir):
    """
    Descomprime un archivo ZIP y retorna lista de archivos .nc extraídos
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extraer todos los archivos
            zip_ref.extractall(extract_dir)
        
        # Buscar archivos .nc en el directorio extraído
        archivos_nc = list(Path(extract_dir).glob('*.nc'))
        return archivos_nc
    
    except Exception as e:
        print(f"    Error descomprimiendo: {e}")
        return []


def procesar_archivo_nc(filepath):
    """
    Procesa un archivo NetCDF y extrae datos horarios
    
    Returns:
        DataFrame con columnas: datetime, fecha, hora, y todas las variables meteorológicas
    """
    
    try:
        ds = xr.open_dataset(filepath, engine='netcdf4')
        
        # Detectar dimensión de tiempo - ahora es 'valid_time'
        dim_tiempo = None
        for posible in ['valid_time', 'time', 'forecast_reference_time', 't']:
            if posible in ds.dims or posible in ds.coords:
                dim_tiempo = posible
                break
        
        if dim_tiempo is None:
            ds.close()
            return None
        
        # Obtener timestamps
        tiempos = pd.to_datetime(ds[dim_tiempo].values)
        
        # Mapeo de variables ERA5 → nombres finales
        var_mapping = {
            't2m': ('temp_2m', 273.15, 'subtract'),       # Kelvin → Celsius
            'ssrd': ('solar_rad', 3600, 'divide'),        # J/m² → W/m²
            'tcc': ('cloud_cover', 1, 'none'),            # Fracción 0-1
            'u10': ('wind_u', 1, 'none'),                 # m/s
            'v10': ('wind_v', 1, 'none'),                 # m/s
            'sp': ('pressure', 100, 'divide'),            # Pa → hPa
            'blh': ('boundary_layer_height', 1, 'none'), # m
            'cape': ('cape', 1, 'none'),                  # J/kg
            'skt': ('skin_temp', 273.15, 'subtract'),    # Kelvin → Celsius
            'tp': ('precipitation', 1000, 'multiply'),    # m → mm
        }
        
        datos_horarios = []
        
        # Procesar cada timestamp
        for timestamp in tiempos:
            try:
                ds_hora = ds.sel({dim_tiempo: timestamp})
                
                registro = {
                    'datetime': timestamp,
                    'fecha': timestamp.date(),
                    'hora': timestamp.hour
                }
                
                # Extraer cada variable
                for var_era5, (var_nombre, factor, operacion) in var_mapping.items():
                    if var_era5 in ds_hora:
                        # Promediar sobre área geográfica (lat/lon)
                        datos = ds_hora[var_era5].values
                        valor = float(np.nanmean(datos))
                        
                        # Aplicar conversión
                        if operacion == 'subtract':
                            registro[var_nombre] = valor - factor
                        elif operacion == 'divide':
                            registro[var_nombre] = valor / factor
                        elif operacion == 'multiply':
                            registro[var_nombre] = valor * factor
                        else:
                            registro[var_nombre] = valor
                
                datos_horarios.append(registro)
                
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


def combinar_archivos_instant_accum(archivos_nc):
    """
    Combina archivos instant y accum que tienen variables diferentes
    
    Returns:
        DataFrame combinado con todas las variables
    """
    
    dfs = []
    
    for nc_file in archivos_nc:
        df = procesar_archivo_nc(str(nc_file))
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        return None
    
    if len(dfs) == 1:
        df_combined = dfs[0]
    else:
        # Combinar por datetime
        df_combined = dfs[0]
        for df in dfs[1:]:
            df_combined = df_combined.merge(df, on=['datetime', 'fecha', 'hora'], how='outer')
    
    # Calcular wind_speed si tenemos u y v
    if 'wind_u' in df_combined.columns and 'wind_v' in df_combined.columns:
        df_combined['wind_speed'] = np.sqrt(
            df_combined['wind_u']**2 + df_combined['wind_v']**2
        )
    
    return df_combined


def calcular_agregados_diarios(df_horarios):
    """
    Calcula agregados diarios a partir de datos horarios
    
    Returns:
        DataFrame con features agregadas por día
    """
    
    agregados = []
    
    for fecha in sorted(df_horarios['fecha'].unique()):
        dia_data = df_horarios[df_horarios['fecha'] == fecha]
        
        agg = {'fecha': fecha}
        
        # Solar radiation
        if 'solar_rad' in dia_data.columns:
            agg['solar_rad_total'] = dia_data['solar_rad'].sum()
            agg['solar_rad_max'] = dia_data['solar_rad'].max()
        
        # Precipitation
        if 'precipitation' in dia_data.columns:
            agg['precipitation_total'] = dia_data['precipitation'].sum()
        
        # Temperature 2m
        if 'temp_2m' in dia_data.columns:
            agg['temp_2m_max'] = dia_data['temp_2m'].max()
            agg['temp_2m_min'] = dia_data['temp_2m'].min()
            agg['temp_2m_mean'] = dia_data['temp_2m'].mean()
            agg['temp_differential'] = agg['temp_2m_max'] - agg['temp_2m_min']
        
        # Skin temperature
        if 'skin_temp' in dia_data.columns:
            agg['skin_temp_max'] = dia_data['skin_temp'].max()
            agg['skin_temp_min'] = dia_data['skin_temp'].min()
            agg['skin_temp_mean'] = dia_data['skin_temp'].mean()
        
        # Cloud cover
        if 'cloud_cover' in dia_data.columns:
            agg['cloud_cover_mean'] = dia_data['cloud_cover'].mean()
        
        # Wind
        if 'wind_u' in dia_data.columns:
            agg['wind_u_mean'] = dia_data['wind_u'].mean()
        if 'wind_v' in dia_data.columns:
            agg['wind_v_mean'] = dia_data['wind_v'].mean()
        if 'wind_speed' in dia_data.columns:
            agg['wind_speed_mean'] = dia_data['wind_speed'].mean()
        
        # Pressure
        if 'pressure' in dia_data.columns:
            agg['pressure_mean'] = dia_data['pressure'].mean()
        
        # Boundary layer height
        if 'boundary_layer_height' in dia_data.columns:
            agg['boundary_layer_height_mean'] = dia_data['boundary_layer_height'].mean()
            agg['boundary_layer_height_max'] = dia_data['boundary_layer_height'].max()
        
        # CAPE
        if 'cape' in dia_data.columns:
            agg['cape_mean'] = dia_data['cape'].mean()
            agg['cape_max'] = dia_data['cape'].max()
        
        agregados.append(agg)
    
    return pd.DataFrame(agregados)


def main():
    """
    Función principal
    """
    
    print("="*80)
    print("PROCESAMIENTO COMPLETO ERA5")
    print("="*80)
    print("\nEste script genera:")
    print("  1. datos_meteorologicos_AGREGADOS_DIA.csv  (features por día)")
    print("  2. datos_meteorologicos_HORARIOS.csv       (features por hora)")
    print("="*80)
    
    # Buscar archivos .nc
    print("\nBuscando archivos .nc...")
    archivos_nc = list(Path('data/raw').glob('*.nc'))
    
    if not archivos_nc:
        print("✗ No se encontraron archivos .nc en data/raw/")
        print("  Coloca tus archivos ERA5 (.nc) en data/raw/")
        return
    
    print(f"✓ Encontrados {len(archivos_nc)} archivos .nc")
    for f in archivos_nc:
        print(f"  - {f.name}")
    
    # Verificar si son ZIPs
    print("\n" + "="*80)
    print("VERIFICANDO TIPO DE ARCHIVOS")
    print("="*80)
    
    archivos_por_zip = []  # Lista de listas (cada ZIP tiene sus archivos .nc)
    temp_dir = Path('data/raw/temp_extracted')
    temp_dir.mkdir(exist_ok=True)
    
    for nc_file in archivos_nc:
        if es_zip(nc_file):
            print(f"\n{nc_file.name} → Es un archivo ZIP, descomprimiendo...")
            
            # Crear subdirectorio para este archivo
            extract_subdir = temp_dir / nc_file.stem
            extract_subdir.mkdir(exist_ok=True)
            
            # Descomprimir
            archivos_extraidos = descomprimir_archivo(str(nc_file), str(extract_subdir))
            
            if archivos_extraidos:
                print(f"  ✓ {len(archivos_extraidos)} archivos .nc extraídos")
                archivos_por_zip.append(archivos_extraidos)
            else:
                print(f"  ✗ No se encontraron archivos .nc dentro del ZIP")
        else:
            print(f"\n{nc_file.name} → Archivo NetCDF directo")
            archivos_por_zip.append([nc_file])
    
    if not archivos_por_zip:
        print("\n✗ No hay archivos .nc para procesar")
        return
    
    print(f"\n✓ Total de ZIPs/archivos a procesar: {len(archivos_por_zip)}")
    
    # Procesar cada grupo (cada ZIP)
    print("\n" + "="*80)
    print("PROCESANDO Y COMBINANDO ARCHIVOS")
    print("="*80)
    print("\nCada ZIP contiene 2 archivos (.nc):")
    print("  - instant.nc: temp, viento, cape, etc.")
    print("  - accum.nc: radiación solar, precipitación")
    print("Se combinarán automáticamente por timestamp")
    
    todos_dataframes = []
    
    for i, archivos_grupo in enumerate(archivos_por_zip, 1):
        print(f"\n[{i}/{len(archivos_por_zip)}] Procesando grupo con {len(archivos_grupo)} archivo(s)")
        
        # Combinar instant + accum de este grupo
        df = combinar_archivos_instant_accum(archivos_grupo)
        
        if df is not None:
            print(f"  ✓ {len(df)} registros horarios combinados")
            print(f"  ✓ Variables: {len([c for c in df.columns if c not in ['datetime','fecha','hora']])}")
            todos_dataframes.append(df)
        else:
            print(f"  ✗ No se pudo procesar este grupo")
    
    # Limpiar archivos temporales
    if temp_dir.exists():
        print("\nLimpiando archivos temporales...")
        shutil.rmtree(temp_dir)
        print("  ✓ Temporales eliminados")
    
    if not todos_dataframes:
        print("\n✗ No se pudo procesar ningún archivo")
        return
    
    # Combinar todos los datos horarios
    print("\n" + "="*80)
    print("COMBINANDO DATOS HORARIOS")
    print("="*80)
    
    df_horarios = pd.concat(todos_dataframes, ignore_index=True)
    
    # Eliminar duplicados por datetime
    df_horarios = df_horarios.drop_duplicates(subset=['datetime'])
    df_horarios = df_horarios.sort_values('datetime').reset_index(drop=True)
    
    print(f"\n✓ Total registros horarios únicos: {len(df_horarios)}")
    print(f"✓ Fechas: {df_horarios['fecha'].min()} a {df_horarios['fecha'].max()}")
    print(f"✓ Días únicos: {df_horarios['fecha'].nunique()}")
    print(f"✓ Horas por día (promedio): {len(df_horarios) / df_horarios['fecha'].nunique():.1f}")
    
    # Variables disponibles
    vars_disponibles = [col for col in df_horarios.columns if col not in ['datetime', 'fecha', 'hora']]
    print(f"\n✓ Variables meteorológicas disponibles: {len(vars_disponibles)}")
    for var in vars_disponibles:
        print(f"  - {var}")
    
    # Calcular agregados diarios
    print("\n" + "="*80)
    print("CALCULANDO AGREGADOS DIARIOS")
    print("="*80)
    
    df_agregados = calcular_agregados_diarios(df_horarios)
    
    print(f"\n✓ Agregados calculados para {len(df_agregados)} días")
    
    # Columnas en orden específico
    columnas_agregados = [
        'fecha', 'solar_rad_total', 'solar_rad_max', 'precipitation_total',
        'temp_2m_max', 'temp_2m_min', 'temp_2m_mean', 'cloud_cover_mean',
        'wind_u_mean', 'wind_v_mean', 'pressure_mean',
        'boundary_layer_height_mean', 'boundary_layer_height_max',
        'cape_mean', 'cape_max', 'skin_temp_max', 'skin_temp_min', 
        'skin_temp_mean', 'temp_differential', 'wind_speed_mean'
    ]
    
    # Reordenar columnas (solo las que existen)
    columnas_existentes = [col for col in columnas_agregados if col in df_agregados.columns]
    df_agregados = df_agregados[columnas_existentes]
    
    print(f"✓ Features agregadas: {len(df_agregados.columns) - 1}")  # -1 por 'fecha'
    for col in df_agregados.columns:
        if col != 'fecha':
            print(f"  - {col}")
    
    # Guardar archivos
    print("\n" + "="*80)
    print("GUARDANDO ARCHIVOS")
    print("="*80)
    
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Agregados diarios
    file_agregados = 'data/processed/datos_meteorologicos_AGREGADOS_DIA.csv'
    df_agregados.to_csv(file_agregados, index=False)
    print(f"\n✓ Archivo 1: {file_agregados}")
    print(f"  Registros: {len(df_agregados)}")
    print(f"  Columnas: {len(df_agregados.columns)}")
    
    # 2. Datos horarios
    file_horarios = 'data/processed/datos_meteorologicos_HORARIOS.csv'
    df_horarios.to_csv(file_horarios, index=False)
    print(f"\n✓ Archivo 2: {file_horarios}")
    print(f"  Registros: {len(df_horarios)}")
    print(f"  Columnas: {len(df_horarios.columns)}")
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE DATOS")
    print("="*80)
    
    print("\n▶ AGREGADOS DIARIOS")
    print(f"  Archivo: {file_agregados}")
    print(f"  Período: {df_agregados['fecha'].min()} a {df_agregados['fecha'].max()}")
    print(f"  Días: {len(df_agregados)}")
    print(f"\n  Ejemplo (primer día):")
    print(df_agregados.head(1).T)
    
    print("\n▶ DATOS HORARIOS")
    print(f"  Archivo: {file_horarios}")
    print(f"  Período: {df_horarios['fecha'].min()} a {df_horarios['fecha'].max()}")
    print(f"  Registros totales: {len(df_horarios)}")
    print(f"  Horas disponibles: {sorted(df_horarios['hora'].unique())}")
    
    print(f"\n  Ejemplo (primer día, primeras 5 horas):")
    primer_dia = df_horarios['fecha'].iloc[0]
    ejemplo = df_horarios[df_horarios['fecha'] == primer_dia].head(5)
    print(ejemplo[['fecha', 'hora', 'temp_2m', 'cape', 'solar_rad', 'wind_speed']])
    
    # Estadísticas
    print("\n" + "="*80)
    print("ESTADÍSTICAS")
    print("="*80)
    
    print("\nAgregados diarios:")
    print(df_agregados.describe().T[['mean', 'min', 'max']])
    
    print("\n" + "="*80)
    print("✓✓✓ PROCESAMIENTO COMPLETADO ✓✓✓")
    print("="*80)
    print("\nArchivos generados:")
    print(f"  1. {file_agregados}")
    print(f"  2. {file_horarios}")
    print("\nPróximos pasos:")
    print("  - Combinar con datos de vuelos")
    print("  - Análisis exploratorio")
    print("="*80)


if __name__ == "__main__":
    # Verificar dependencias
    try:
        import netCDF4
    except ImportError:
        print("Instalando netCDF4...")
        os.system('pip install netCDF4 --quiet --break-system-packages')
        print()
    
    try:
        import xarray
    except ImportError:
        print("Instalando xarray...")
        os.system('pip install xarray --quiet --break-system-packages')
        print()
    
    main()