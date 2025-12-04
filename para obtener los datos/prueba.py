"""
03b_agregar_features_meteorologicas_completas.py
Enriquece el dataset SOLO con la grilla horaria (9h-18h) y filtra estrictamente las columnas finales.
"""

import pandas as pd
import numpy as np
import os

# --- TU LISTA MAESTRA DE COLUMNAS ---
COLUMNAS_DESEADAS = [
    # Datos de Vuelo
    "fecha","pilot","glider","competition_id","altura_max_m","altura_min_m","altura_despegue_m","altura_aterrizaje_m",
    "ganancia_altura_m","rango_altura_m","duracion_min","duracion_horas","distancia_km","velocidad_promedio_kmh",
    "lat_despegue","lon_despegue","hora_despegue","hora_despegue_decimal","num_gps_fixes","frecuencia_muestreo_seg",
    "num_termicas","intensidad_termicas_mean_ms","intensidad_termicas_max_ms","intensidad_termicas_min_ms",
    "intensidad_termicas_std_ms","altura_base_termicas_mean_m","altura_tope_termicas_mean_m",
    "altura_base_termicas_min_m","altura_tope_termicas_max_m","ganancia_por_termica_mean_m",
    "ganancia_por_termica_max_m","duracion_termica_mean_seg","duracion_termica_max_seg","hora_primera_termica",
    "hora_ultima_termica","dispersion_termicas_lat","dispersion_termicas_lon","tiempo_en_termicas_min",
    "tiempo_en_planeo_min","porcentaje_tiempo_termicas","tasa_ascenso_mean_ms","tasa_descenso_mean_ms",
    "bearing_change_mean_deg","bearing_change_max_deg","bearing_change_std_deg","ground_speed_mean_kmh",
    "ground_speed_max_kmh","ground_speed_min_kmh","ground_speed_std_kmh","hora_inicio_decimal","hora_fin_decimal",
    "altura_mean_manana_m","altura_mean_mediodia_m","altura_mean_tarde1_m","altura_mean_tarde2_m",
    "lat_min","lat_max","lon_min","lon_max","lat_centro","lon_centro","rango_lat_deg","rango_lon_deg",
    "distancia_max_despegue_km","area_vuelo_km2","altura_std_m","altura_cv","cambio_altura_mean_m",
    "cambio_altura_std_m","filename","flight_id",
    
    # Grilla Meteorológica (09h - 18h)
    "solar_rad_09h","solar_rad_10h","solar_rad_11h","solar_rad_12h","solar_rad_13h","solar_rad_14h","solar_rad_15h","solar_rad_16h","solar_rad_17h","solar_rad_18h",
    "precipitation_09h","precipitation_10h","precipitation_11h","precipitation_12h","precipitation_13h","precipitation_14h","precipitation_15h","precipitation_16h","precipitation_17h","precipitation_18h",
    "temp_2m_09h","temp_2m_10h","temp_2m_11h","temp_2m_12h","temp_2m_13h","temp_2m_14h","temp_2m_15h","temp_2m_16h","temp_2m_17h","temp_2m_18h",
    "cloud_cover_09h","cloud_cover_10h","cloud_cover_11h","cloud_cover_12h","cloud_cover_13h","cloud_cover_14h","cloud_cover_15h","cloud_cover_16h","cloud_cover_17h","cloud_cover_18h",
    "wind_u_09h","wind_u_10h","wind_u_11h","wind_u_12h","wind_u_13h","wind_u_14h","wind_u_15h","wind_u_16h","wind_u_17h","wind_u_18h",
    "wind_v_09h","wind_v_10h","wind_v_11h","wind_v_12h","wind_v_13h","wind_v_14h","wind_v_15h","wind_v_16h","wind_v_17h","wind_v_18h",
    "pressure_09h","pressure_10h","pressure_11h","pressure_12h","pressure_13h","pressure_14h","pressure_15h","pressure_16h","pressure_17h","pressure_18h",
    "boundary_layer_height_09h","boundary_layer_height_10h","boundary_layer_height_11h","boundary_layer_height_12h","boundary_layer_height_13h","boundary_layer_height_14h","boundary_layer_height_15h","boundary_layer_height_16h","boundary_layer_height_17h","boundary_layer_height_18h",
    "cape_09h","cape_10h","cape_11h","cape_12h","cape_13h","cape_14h","cape_15h","cape_16h","cape_17h","cape_18h",
    "skin_temp_09h","skin_temp_10h","skin_temp_11h","skin_temp_12h","skin_temp_13h","skin_temp_14h","skin_temp_15h","skin_temp_16h","skin_temp_17h","skin_temp_18h",
    "wind_speed_09h","wind_speed_10h","wind_speed_11h","wind_speed_12h","wind_speed_13h","wind_speed_14h","wind_speed_15h","wind_speed_16h","wind_speed_17h","wind_speed_18h"
]

def generar_grilla_horaria_fija(df_meteo_horario):
    """
    Crea una estructura de columnas fijas para cada hora entre 9 y 18.
    """
    print("\nGenerando grilla horaria fija (9h - 18h)...")
    
    # Filtrar rango
    df_filtrado = df_meteo_horario[
        (df_meteo_horario['hora'] >= 9) & 
        (df_meteo_horario['hora'] <= 18)
    ].copy()
    
    df_filtrado['hora'] = df_filtrado['hora'].astype(int)
    
    # Pivotar
    cols_excluir = ['fecha', 'hora', 'datetime']
    variables = [c for c in df_filtrado.columns if c not in cols_excluir]
    
    df_pivot = df_filtrado.pivot(index='fecha', columns='hora', values=variables)
    
    # Aplanar nombres (temp_2m_09h)
    nuevas_columnas = []
    for var_name, hora in df_pivot.columns:
        nuevas_columnas.append(f"{var_name}_{hora:02d}h")
    
    df_pivot.columns = nuevas_columnas
    df_pivot = df_pivot.reset_index()
    
    return df_pivot

def enriquecer_dataset():
    print("="*70)
    print("ENRIQUECIMIENTO Y LIMPIEZA DE DATASET")
    print("="*70)
    
    dataset_file = 'data/processed/dataset_FINAL.csv'
    meteo_file = 'data/raw/datos_meteorologicos_HORARIOS.csv'
    
    if not os.path.exists(dataset_file) or not os.path.exists(meteo_file):
        print("ERROR: Faltan archivos.")
        return

    # 1. Cargar Vuelos
    print("Cargando vuelos...")
    df_vuelos = pd.read_csv(dataset_file)
    df_vuelos['fecha'] = pd.to_datetime(df_vuelos['fecha']).dt.date
    
    # --- LIMPIEZA INICIAL: Quedarse solo con columnas de vuelo válidas ---
    # Identificamos qué columnas de la LISTA MAESTRA son de "vuelo" (las que ya deberían estar en df_vuelos)
    # Asumimos que las columnas meteorológicas (las de _09h, _10h...) NO están en el archivo original de vuelos.
    cols_vuelo_deseadas = [c for c in COLUMNAS_DESEADAS if not (c.endswith('h') and ('solar_' in c or 'temp_' in c or 'wind_' in c or 'cape_' in c))]
    
    # Filtramos el dataframe original para quitar basura vieja
    cols_a_mantener = [c for c in cols_vuelo_deseadas if c in df_vuelos.columns]
    
    # Si falta la fecha, la forzamos a quedarse
    if 'fecha' not in cols_a_mantener: 
        cols_a_mantener.insert(0, 'fecha')
        
    df_vuelos = df_vuelos[cols_a_mantener]
    print(f"  ✓ Datos de vuelo limpiados. Columnas actuales: {len(df_vuelos.columns)}")

    # 2. Cargar Meteo y Generar Grilla
    print("Cargando y procesando meteo...")
    df_meteo = pd.read_csv(meteo_file)
    df_meteo['fecha'] = pd.to_datetime(df_meteo['fecha']).dt.date
    
    df_grilla = generar_grilla_horaria_fija(df_meteo)
    
    # 3. Merge (Unión)
    print("\nFusionando datos...")
    df_final = df_vuelos.merge(df_grilla, on='fecha', how='left')
    
    # 4. FILTRADO FINAL ESTRICTO
    print("\nAplicando filtro final de columnas...")
    
    # Verificamos qué columnas de TU LISTA existen realmente en el dataframe unido
    cols_finales_validas = [c for c in COLUMNAS_DESEADAS if c in df_final.columns]
    
    # Reporte de faltantes (para que sepas si algo de tu lista no se pudo calcular)
    faltantes = set(COLUMNAS_DESEADAS) - set(cols_finales_validas)
    if faltantes:
        print(f"  ⚠ Aviso: {len(faltantes)} columnas de tu lista no se encontraron y serán ignoradas.")
        # Opcional: imprimir las primeras 5 faltantes para debug
        print(f"    Ej: {list(faltantes)[:5]}")
    
    # Reordenamos y filtramos
    df_final = df_final[cols_finales_validas]
    
    # 5. Guardar
    if df_final.duplicated().any():
        df_final = df_final.drop_duplicates()
        
    output_file = 'data/processed/dataset_FINAL_GRID.csv'
    df_final.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print(f"✓ GUARDADO EXITOSO: {output_file}")
    print(f"  Columnas: {len(df_final.columns)}")
    print(f"  Filas: {len(df_final)}")
    print(f"  Primeras 3 columnas: {df_final.columns[:3].tolist()}")
    print(f"  Últimas 3 columnas: {df_final.columns[-3:].tolist()}")
    print("="*70)

    return df_final

if __name__ == "__main__":
    enriquecer_dataset()