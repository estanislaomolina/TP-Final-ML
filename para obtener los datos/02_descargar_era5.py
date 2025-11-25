"""
Script para descargar datos meteorológicos de ERA5
Requiere registro en Copernicus Climate Data Store (CDS)
https://cds.climate.copernicus.eu/
"""

import cdsapi
import pandas as pd
import numpy as np
import xarray as xr
import os
from datetime import datetime


def chunk_list(items, chunk_size):
    items_sorted = sorted(items)
    for i in range(0, len(items_sorted), chunk_size):
        yield items_sorted[i:i + chunk_size]


class ERA5Downloader:
    def __init__(self, output_dir='data/raw'):
        """
        IMPORTANTE: Antes de usar este script:
        1. Registrarse en: https://cds.climate.copernicus.eu/
        2. Obtener API key
        3. Crear archivo ~/.cdsapirc con:
           url: https://cds.climate.copernicus.eu/api/v2
           key: {UID}:{API_KEY}
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            self.client = cdsapi.Client()
        except Exception as e:
            print("ERROR: No se pudo inicializar cliente CDS API")
            print("Asegúrate de haber configurado ~/.cdsapirc")
            print(f"Detalles: {e}")
            self.client = None
    
    def descargar_era5_para_fechas(self, fechas, lat_range, lon_range, output_file):
        """
        Descargar datos ERA5 para fechas específicas
        
        Parameters:
        -----------
        fechas : list
            Lista de fechas en formato 'YYYY-MM-DD'
        lat_range : tuple
            (lat_min, lat_max) en grados
        lon_range : tuple
            (lon_min, lon_max) en grados
        output_file : str
            Nombre del archivo netCDF de salida
        """
        
        if self.client is None:
            print("ERROR: Cliente CDS API no disponible")
            return None
        
        # Convertir fechas a formato necesario
        fechas_dt = pd.to_datetime(fechas)
        años = sorted(list(set(fechas_dt.year)))
        meses = sorted(list(set(fechas_dt.month)))
        dias = sorted(list(set(fechas_dt.day)))
        
        # Convertir a strings con formato necesario
        años_str = [str(y) for y in años]
        meses_str = [f"{m:02d}" for m in meses]
        dias_str = [f"{d:02d}" for d in dias]
        
        # Horas del día (típicamente 09:00 a 18:00 para térmicas)
        horas = ['09:00', '10:00', '11:00', '12:00', '13:00', 
                 '14:00', '15:00', '16:00', '17:00', '18:00']
        
        # Variables a descargar
        variables = [
            '2m_temperature',                        # Temperatura a 2m
            'surface_solar_radiation_downwards',     # Radiación solar
            'total_cloud_cover',                     # Cobertura de nubes
            '10m_u_component_of_wind',              # Viento U (10m)
            '10m_v_component_of_wind',              # Viento V (10m)
            'surface_pressure',                      # Presión superficial
            'boundary_layer_height',                 # Altura capa límite
            'convective_available_potential_energy', # CAPE
            'skin_temperature',                      # Temp superficie
            'total_precipitation',                   # Precipitación
        ]
        
        # Área geográfica [Norte, Oeste, Sur, Este]
        area = [
            lat_range[1],  # Norte
            lon_range[0],  # Oeste
            lat_range[0],  # Sur
            lon_range[1],  # Este
        ]
        
        print(f"\nDescargando datos ERA5...")
        print(f"  Años: {años_str}")
        print(f"  Meses: {meses_str}")
        print(f"  Días únicos: {len(dias_str)}")
        print(f"  Área: {area}")
        
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            self.client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': variables,
                    'year': años_str,
                    'month': meses_str,
                    'day': dias_str,
                    'time': horas,
                    'area': area,
                    'format': 'netcdf',
                },
                output_path
            )
            
            print(f"✓ Datos ERA5 descargados en: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"✗ Error descargando ERA5: {e}")
            return None

    def descargar_era5_en_batches(self, fechas, lat_range, lon_range, output_prefix='era5_data', batch_size=10):
        if self.client is None:
            print("ERROR: Cliente CDS API no disponible")
            return []
        
        resultados = []
        for idx, fechas_lote in enumerate(chunk_list(fechas, batch_size), start=1):
            nombre_archivo = f"{output_prefix}_{idx:02d}.nc"
            ruta_archivo = os.path.join(self.output_dir, nombre_archivo)

            if os.path.exists(ruta_archivo):
                print(f"Archivo existente, se omite descarga: {ruta_archivo}")
                resultados.append({'file': ruta_archivo, 'fechas': fechas_lote})
                continue

            ruta = self.descargar_era5_para_fechas(
                fechas=fechas_lote,
                lat_range=lat_range,
                lon_range=lon_range,
                output_file=nombre_archivo
            )
            if ruta:
                resultados.append({'file': ruta, 'fechas': fechas_lote})
        return resultados
    
    def procesar_netcdf_a_csv(self, netcdf_file, fechas_vuelos, output_csv=None):
        """
        Procesar archivo netCDF y extraer datos diarios agregados
        
        Parameters:
        -----------
        netcdf_file : str
            Path al archivo .nc descargado
        fechas_vuelos : list
            Lista de fechas de vuelos
        output_csv : str | None
            Nombre del archivo CSV de salida
        """
        
        print(f"\nProcesando {netcdf_file}...")
        
        ds = xr.open_dataset(netcdf_file, engine='netcdf4')
        df_meteo = []
        
        for fecha in pd.to_datetime(fechas_vuelos).unique():
            fecha_str = fecha.strftime('%Y-%m-%d')
            ds_dia = ds.sel(time=fecha_str)
            
            stats = {
                'fecha': fecha_str,
                'temp_2m_max': float(ds_dia['t2m'].max().values - 273.15),
                'temp_2m_min': float(ds_dia['t2m'].min().values - 273.15),
                'temp_2m_mean': float(ds_dia['t2m'].mean().values - 273.15),
                'solar_rad_total': float(ds_dia['ssrd'].sum().values / 3600),
                'solar_rad_max': float(ds_dia['ssrd'].max().values / 3600),
                'cloud_cover_mean': float(ds_dia['tcc'].mean().values),
                'wind_u_mean': float(ds_dia['u10'].mean().values),
                'wind_v_mean': float(ds_dia['v10'].mean().values),
                'pressure_mean': float(ds_dia['sp'].mean().values / 100),
                'boundary_layer_height_max': float(ds_dia['blh'].max().values),
                'boundary_layer_height_mean': float(ds_dia['blh'].mean().values),
                'cape_max': float(ds_dia['cape'].max().values),
                'cape_mean': float(ds_dia['cape'].mean().values),
                'skin_temp_max': float(ds_dia['skt'].max().values - 273.15),
                'precipitation_total': float(ds_dia['tp'].sum().values * 1000),
            }
            
            stats['temp_differential'] = stats['temp_2m_max'] - stats['temp_2m_min']
            stats['wind_speed_mean'] = np.sqrt(stats['wind_u_mean']**2 + stats['wind_v_mean']**2)
            
            df_meteo.append(stats)
        
        ds.close()
        
        df = pd.DataFrame(df_meteo)
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        print(f"✓ Datos meteorológicos procesados ({len(df)} días, {len(df.columns)} variables)")
        
        if output_csv:
            output_path = os.path.join(self.output_dir, output_csv)
            df.to_csv(output_path, index=False)
            print(f"  Archivo: {output_path}")
        
        return df


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("DESCARGA DE DATOS ERA5")
    print("="*70)
    
    vuelos_file = None
    for posible_file in ['data/raw/vuelos_metadata_completo.csv', 'data/raw/vuelos_metadata.csv']:
        if os.path.exists(posible_file):
            vuelos_file = posible_file
            break
    
    if vuelos_file is None:
        print(f"ERROR: No se encuentra archivo de vuelos")
        print("Archivos buscados:")
        print("  - data/raw/vuelos_metadata_completo.csv")
        print("  - data/raw/vuelos_metadata.csv")
        print("\nPrimero ejecuta: python 01_procesar_igc_COMPLETO.py")
        exit(1)
    
    df_vuelos = pd.read_csv(vuelos_file)
    df_vuelos['fecha'] = pd.to_datetime(df_vuelos['fecha'])
    
    print(f"\nVuelos cargados: {len(df_vuelos)}")
    print(f"Fechas únicas: {df_vuelos['fecha'].nunique()}")
    
    print("\nCalculando área geográfica automáticamente...")
    
    lat_min_vuelos = df_vuelos['lat_despegue'].min()
    lat_max_vuelos = df_vuelos['lat_despegue'].max()
    lon_min_vuelos = df_vuelos['lon_despegue'].min()
    lon_max_vuelos = df_vuelos['lon_despegue'].max()
    
    print(f"\nUbicaciones de vuelos:")
    print(f"  Latitud:  {lat_min_vuelos:.2f}° a {lat_max_vuelos:.2f}°")
    print(f"  Longitud: {lon_min_vuelos:.2f}° a {lon_max_vuelos:.2f}°")
    
    MARGEN = 1.5
    
    LAT_MIN = np.floor(lat_min_vuelos - MARGEN)
    LAT_MAX = np.ceil(lat_max_vuelos + MARGEN)
    LON_MIN = np.floor(lon_min_vuelos - MARGEN)
    LON_MAX = np.ceil(lon_max_vuelos + MARGEN)
    
    print(f"\nÁrea de descarga ERA5 (con margen de {MARGEN}°):")
    print(f"  Latitud:  {LAT_MIN:.1f}° a {LAT_MAX:.1f}°")
    print(f"  Longitud: {LON_MIN:.1f}° a {LON_MAX:.1f}°")
    
    rango_lat = LAT_MAX - LAT_MIN
    rango_lon = LON_MAX - LON_MIN
    area_km2 = rango_lat * 111 * rango_lon * 92
    
    print(f"\nDimensiones:")
    print(f"  Rango latitudinal: {rango_lat:.1f}° (~{rango_lat*111:.0f} km)")
    print(f"  Rango longitudinal: {rango_lon:.1f}° (~{rango_lon*92:.0f} km)")
    print(f"  Área aproximada: {area_km2:,.0f} km²")
    
    downloader = ERA5Downloader(output_dir='data/raw')
    
    fechas_unicas = sorted(df_vuelos['fecha'].dt.strftime('%Y-%m-%d').unique().tolist())
    
    print(f"\n¿Descargar datos ERA5 para:")
    print(f"  • {len(fechas_unicas)} días")
    print(f"  • Región: Lat [{LAT_MIN:.1f}, {LAT_MAX:.1f}], Lon [{LON_MIN:.1f}, {LON_MAX:.1f}]")
    print(f"  • Área: ~{area_km2:,.0f} km²")
    
    puntos_grilla = ((LAT_MAX - LAT_MIN) / 0.25) * ((LON_MAX - LON_MIN) / 0.25)
    variables = 10
    horas = 10
    dias = len(fechas_unicas)
    
    tamaño_estimado_mb = (puntos_grilla * variables * horas * dias * 4) / (1024 * 1024)
    
    print(f"\n📦 Tamaño estimado de descarga: {tamaño_estimado_mb:.1f} MB")
    print(f"⏱️  Tiempo estimado: {max(10, int(tamaño_estimado_mb / 5))}-{max(15, int(tamaño_estimado_mb / 3))} minutos")
    
    print("\nNOTA: Esta descarga puede tardar. Asegúrate de:")
    print("  1. Tener configurado ~/.cdsapirc con tu API key de CDS")
    print("  2. Tener conexión estable a internet")
    print("  3. Tener espacio en disco (~{:.0f} MB)".format(tamaño_estimado_mb * 1.5))
    
    respuesta = input("\n¿Continuar con la descarga? (s/n): ")
    
    if respuesta.lower() == 's':
        lotes = downloader.descargar_era5_en_batches(
            fechas=fechas_unicas,
            lat_range=(LAT_MIN, LAT_MAX),
            lon_range=(LON_MIN, LON_MAX),
            batch_size=10
        )
        
        if lotes:
            df_total = []
            for idx, lote in enumerate(lotes, start=1):
                print(f"\nProcesando lote {idx}/{len(lotes)}...")
                df_chunk = downloader.procesar_netcdf_a_csv(
                    netcdf_file=lote['file'],
                    fechas_vuelos=lote['fechas'],
                    output_csv=None
                )
                if df_chunk is not None:
                    df_total.append(df_chunk)
            
            if df_total:
                df_meteo = (
                    pd.concat(df_total, ignore_index=True)
                      .drop_duplicates(subset='fecha')
                      .sort_values('fecha')
                )
                salida_final = os.path.join('data/raw', 'datos_meteorologicos.csv')
                df_meteo.to_csv(salida_final, index=False)
                
                print("\n" + "="*70)
                print("✓ Fase 2 completada: Datos meteorológicos descargados por lotes")
                print(f"  Archivo combinado: {salida_final}")
                print("="*70)
            else:
                print("No se generaron datos meteorológicos.")
        else:
            print("No se descargó ningún lote. Revisa los mensajes anteriores.")
    else:
        print("\nDescarga cancelada.")