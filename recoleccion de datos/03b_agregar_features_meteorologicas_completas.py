"""
03b_agregar_features_meteorologicas_completas.py
Enriquece el dataset con features meteorológicas:
- Hora específica del vuelo (ya existe)
- Agregados del día completo (max, min, mean, std)
- Ventanas temporales (2h antes, durante vuelo, tendencias)
"""

import pandas as pd
import numpy as np
import os


def calcular_agregados_diarios(df_meteo_horario):
    """
    Calcula estadísticas agregadas por día
    
    Returns:
        DataFrame con features: fecha, temp_max_dia, temp_min_dia, etc.
    """
    
    print("\nCalculando agregados diarios...")
    
    # Agrupar por fecha
    agregados = []
    
    for fecha in df_meteo_horario['fecha'].unique():
        dia_data = df_meteo_horario[df_meteo_horario['fecha'] == fecha]
        
        agg_dia = {'fecha': fecha}
        
        # Variables a agregar
        variables = [
            'temp_2m', 'solar_rad', 'precipitation', 'cloud_cover',
            'wind_u', 'wind_v', 'wind_speed', 'pressure',
            'boundary_layer_height', 'cape', 'skin_temp'
        ]
        
        for var in variables:
            if var in dia_data.columns:
                agg_dia[f'{var}_max_dia'] = dia_data[var].max()
                agg_dia[f'{var}_min_dia'] = dia_data[var].min()
                agg_dia[f'{var}_mean_dia'] = dia_data[var].mean()
                agg_dia[f'{var}_std_dia'] = dia_data[var].std()
        
        # Features derivadas diarias
        if 'temp_2m' in dia_data.columns:
            agg_dia['temp_range_dia'] = agg_dia['temp_2m_max_dia'] - agg_dia['temp_2m_min_dia']
        
        if 'cape' in dia_data.columns and dia_data['cape'].notna().any():
            # Hora de CAPE máximo
            idx_max_cape = dia_data['cape'].idxmax()
            if pd.notna(idx_max_cape):
                agg_dia['hora_cape_max'] = dia_data.loc[idx_max_cape, 'hora']
            else:
                agg_dia['hora_cape_max'] = np.nan
        
        if 'temp_2m' in dia_data.columns and dia_data['temp_2m'].notna().any():
            # Hora de temperatura máxima
            idx_max_temp = dia_data['temp_2m'].idxmax()
            if pd.notna(idx_max_temp):
                agg_dia['hora_temp_max'] = dia_data.loc[idx_max_temp, 'hora']
            else:
                agg_dia['hora_temp_max'] = np.nan
        
        agregados.append(agg_dia)
    
    df_agregados = pd.DataFrame(agregados)
    print(f"  ✓ Agregados calculados para {len(df_agregados)} días")
    
    return df_agregados


def calcular_ventanas_temporales(vuelo, df_meteo_horario):
    """
    Calcula features de ventanas temporales alrededor del vuelo
    
    Parameters:
    -----------
    vuelo : Series
        Fila del vuelo
    df_meteo_horario : DataFrame
        Datos meteorológicos horarios completos
    
    Returns:
    --------
    dict : Features de ventanas temporales
    """
    
    fecha = vuelo['fecha']
    hora_despegue = vuelo.get('hora_despegue_decimal', None)
    duracion_horas = vuelo.get('duracion_horas', 2)  # Default 2h si no está
    
    if pd.isna(hora_despegue):
        return {}
    
    # Filtrar datos del día
    dia_data = df_meteo_horario[df_meteo_horario['fecha'] == fecha].copy()
    
    if len(dia_data) == 0:
        return {}
    
    dia_data = dia_data.sort_values('hora')
    
    features_ventana = {}
    
    # Features para CADA hora previa (1h, 2h, 3h, 4h, 5h antes del despegue)
    variables_clave = ['temp_2m', 'cape', 'wind_speed', 'solar_rad', 'cloud_cover', 'boundary_layer_height']
    
    for horas_atras in [1, 2, 3, 4, 5]:
        hora_target = int(hora_despegue) - horas_atras
        
        if hora_target >= 9:  # Solo si está dentro del rango horario (9h-18h)
            datos_hora = dia_data[dia_data['hora'] == hora_target]
            
            if len(datos_hora) > 0:
                for var in variables_clave:
                    if var in datos_hora.columns:
                        valor = datos_hora[var].iloc[0]
                        features_ventana[f'{var}_pre{horas_atras}h'] = valor
    
    # Ventana DURANTE el vuelo
    hora_inicio_vuelo = int(hora_despegue)
    hora_fin_vuelo = min(18, int(hora_despegue + duracion_horas))
    datos_vuelo = dia_data[(dia_data['hora'] >= hora_inicio_vuelo) & (dia_data['hora'] <= hora_fin_vuelo)]
    
    if len(datos_vuelo) > 0:
        if 'temp_2m' in datos_vuelo.columns:
            features_ventana['temp_mean_durante_vuelo'] = datos_vuelo['temp_2m'].mean()
            features_ventana['temp_change_durante_vuelo'] = datos_vuelo['temp_2m'].iloc[-1] - datos_vuelo['temp_2m'].iloc[0]
        if 'cape' in datos_vuelo.columns:
            features_ventana['cape_mean_durante_vuelo'] = datos_vuelo['cape'].mean()
            features_ventana['cape_max_durante_vuelo'] = datos_vuelo['cape'].max()
            features_ventana['cape_min_durante_vuelo'] = datos_vuelo['cape'].min()
        if 'wind_speed' in datos_vuelo.columns:
            features_ventana['wind_mean_durante_vuelo'] = datos_vuelo['wind_speed'].mean()
            features_ventana['wind_max_durante_vuelo'] = datos_vuelo['wind_speed'].max()
        if 'solar_rad' in datos_vuelo.columns:
            features_ventana['solar_mean_durante_vuelo'] = datos_vuelo['solar_rad'].mean()
    
    # Diferencia con momento óptimo del día
    if 'cape' in dia_data.columns and len(dia_data) > 0 and dia_data['cape'].notna().any():
        cape_max_dia = dia_data['cape'].max()
        if pd.notna(cape_max_dia) and cape_max_dia > 0:
            idx_max = dia_data['cape'].idxmax()
            if pd.notna(idx_max):
                hora_cape_max = dia_data.loc[idx_max, 'hora']
                
                features_ventana['diff_hora_cape_optimo'] = hora_despegue - hora_cape_max
                features_ventana['ratio_cape_vs_optimo'] = vuelo.get('meteo_cape', 0) / cape_max_dia
    
    return features_ventana


def enriquecer_dataset():
    """
    Enriquece el dataset con features meteorológicas completas
    """
    
    print("="*70)
    print("ENRIQUECIMIENTO DE FEATURES METEOROLÓGICAS")
    print("="*70)
    
    # Cargar datasets
    print("\nCargando datos...")
    
    dataset_file = 'data/processed/dataset_FINAL.csv'
    meteo_file = 'data/raw/datos_meteorologicos_HORARIOS.csv'
    
    if not os.path.exists(dataset_file):
        print(f"ERROR: No se encuentra {dataset_file}")
        return None
    
    if not os.path.exists(meteo_file):
        print(f"ERROR: No se encuentra {meteo_file}")
        return None
    
    df_vuelos = pd.read_csv(dataset_file)
    df_meteo = pd.read_csv(meteo_file)
    
    print(f"  Vuelos: {len(df_vuelos)}")
    print(f"  Registros meteo horarios: {len(df_meteo)}")
    
    # Preparar fechas
    df_vuelos['fecha'] = pd.to_datetime(df_vuelos['fecha']).dt.date
    df_meteo['fecha'] = pd.to_datetime(df_meteo['fecha']).dt.date
    
    # 1. Calcular agregados diarios
    print("\n" + "="*70)
    print("PASO 1: AGREGADOS DIARIOS")
    print("="*70)
    
    df_agregados = calcular_agregados_diarios(df_meteo)
    
    cols_agregados = [c for c in df_agregados.columns if c != 'fecha']
    cols_duplicadas = [c for c in cols_agregados if c in df_vuelos.columns]
    if cols_duplicadas:
        print(f"  ⚠ Eliminando columnas duplicadas antes del merge: {len(cols_duplicadas)}")
        df_vuelos = df_vuelos.drop(columns=cols_duplicadas)
    
    df_enriquecido = df_vuelos.merge(df_agregados, on='fecha', how='left')
    
    print(f"  ✓ Dataset con agregados diarios: {df_enriquecido.shape}")
    
    # 2. Calcular ventanas temporales
    print("\n" + "="*70)
    print("PASO 2: VENTANAS TEMPORALES")
    print("="*70)
    
    ventanas_list = []
    
    for idx, vuelo in df_enriquecido.iterrows():
        ventanas = calcular_ventanas_temporales(vuelo, df_meteo)
        ventanas_list.append(ventanas)
        
        if (idx + 1) % 50 == 0:
            print(f"  Procesados: {idx + 1}/{len(df_enriquecido)} vuelos...")
    
    df_ventanas = pd.DataFrame(ventanas_list)
    
    # Combinar todo
    df_final = pd.concat([df_enriquecido.reset_index(drop=True), 
                          df_ventanas.reset_index(drop=True)], axis=1)
    
    if df_final.columns.duplicated().any():
        columnas_duplicadas = df_final.columns[df_final.columns.duplicated()].unique().tolist()
        print(f"\n  ⚠ Columnas duplicadas detectadas tras la concatenación: {len(columnas_duplicadas)}")
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]
        print(f"    → Se conservaron las primeras apariciones. Columnas actuales: {df_final.shape[1]}")
    
    print(f"\n  ✓ Dataset con ventanas temporales: {df_final.shape}")
    
    # 3. Features derivadas adicionales
    print("\n" + "="*70)
    print("PASO 3: FEATURES DERIVADAS")
    print("="*70)
    
    # Hora relativa al momento óptimo
    if 'hora_cape_max' in df_final.columns and 'hora_despegue_decimal' in df_final.columns:
        df_final['horas_desde_cape_max'] = df_final['hora_despegue_decimal'] - df_final['hora_cape_max']
    
    if 'hora_temp_max' in df_final.columns and 'hora_despegue_decimal' in df_final.columns:
        df_final['horas_desde_temp_max'] = df_final['hora_despegue_decimal'] - df_final['hora_temp_max']
    
    # Categorización de momento del día
    if 'hora_despegue_decimal' in df_final.columns:
        def categorizar_momento_dia(hora):
            if hora < 11:
                return 'mañana_temprano'
            elif hora < 13:
                return 'mediodia'
            elif hora < 15:
                return 'tarde_temprano'
            elif hora < 17:
                return 'tarde_media'
            else:
                return 'tarde_tarde'
        
        df_final['momento_dia'] = df_final['hora_despegue_decimal'].apply(categorizar_momento_dia)
    
    # Interacciones importantes
    if 'meteo_cape' in df_final.columns and 'meteo_solar_rad' in df_final.columns:
        df_final['cape_x_solar'] = df_final['meteo_cape'] * df_final['meteo_solar_rad']
    
    if 'meteo_temp_2m' in df_final.columns and 'meteo_boundary_layer_height' in df_final.columns:
        df_final['temp_x_blh'] = df_final['meteo_temp_2m'] * df_final['meteo_boundary_layer_height']
    
    print(f"  ✓ Features derivadas agregadas")
    print(f"\n  Dataset final: {df_final.shape}")
    
    # Guardar
    output_file = 'data/processed/dataset_FINAL.csv'
    df_final.to_csv(output_file, index=False)
    
    print(f"\n✓ Dataset enriquecido guardado: {output_file}")
    
    # Resumen de features
    print("\n" + "="*70)
    print("RESUMEN DE FEATURES METEOROLÓGICAS")
    print("="*70)
    
    # Categorizar features
    features_hora_especifica = [col for col in df_final.columns if col.startswith('meteo_') and not col.endswith('_dia')]
    features_agregados_dia = [col for col in df_final.columns if col.endswith('_dia')]
    features_ventanas = [col for col in df_final.columns if 'pre2h' in col or 'durante_vuelo' in col]
    features_derivadas = [col for col in df_final.columns if any(x in col for x in ['_x_', 'ratio_', 'horas_desde_', 'diff_hora_'])]
    
    print(f"\n1. Features de HORA ESPECÍFICA del vuelo: {len(features_hora_especifica)}")
    print("   (Condiciones exactas cuando despegó)")
    for feat in features_hora_especifica[:10]:
        print(f"     • {feat}")
    if len(features_hora_especifica) > 10:
        print(f"     ... y {len(features_hora_especifica) - 10} más")
    
    print(f"\n2. Features AGREGADAS del DÍA completo: {len(features_agregados_dia)}")
    print("   (Max, min, mean, std de todo el día)")
    for feat in features_agregados_dia[:10]:
        print(f"     • {feat}")
    if len(features_agregados_dia) > 10:
        print(f"     ... y {len(features_agregados_dia) - 10} más")
    
    print(f"\n3. Features de VENTANAS TEMPORALES: {len(features_ventanas)}")
    print("   (2h antes, durante vuelo, tendencias)")
    for feat in features_ventanas:
        print(f"     • {feat}")
    
    print(f"\n4. Features DERIVADAS e INTERACCIONES: {len(features_derivadas)}")
    print("   (Ratios, diferencias, productos)")
    for feat in features_derivadas:
        print(f"     • {feat}")
    
    total_features_meteo = (len(features_hora_especifica) + 
                           len(features_agregados_dia) + 
                           len(features_ventanas) + 
                           len(features_derivadas))
    
    print(f"\n{'='*70}")
    print(f"TOTAL FEATURES METEOROLÓGICAS: {total_features_meteo}")
    print(f"TOTAL FEATURES DATASET: {len(df_final.columns)}")
    print(f"{'='*70}")
    
    # Ejemplo de un vuelo
    print("\n" + "="*70)
    print("EJEMPLO DE FEATURES PARA UN VUELO")
    print("="*70)
    
    ejemplo = df_final.iloc[0]
    
    print(f"\nVuelo: {ejemplo.get('flight_id', 'N/A')}")
    print(f"Fecha: {ejemplo['fecha']}")
    print(f"Hora despegue: {ejemplo.get('hora_despegue_decimal', 'N/A'):.2f}")
    
    print(f"\n▶ HORA ESPECÍFICA (al despegue):")
    if 'meteo_temp_2m' in ejemplo:
        print(f"  Temperatura: {ejemplo['meteo_temp_2m']:.1f}°C")
    if 'meteo_cape' in ejemplo:
        print(f"  CAPE: {ejemplo['meteo_cape']:.0f} J/kg")
    if 'meteo_wind_speed' in ejemplo:
        print(f"  Viento: {ejemplo['meteo_wind_speed']:.1f} m/s")
    
    print(f"\n▶ AGREGADOS DEL DÍA:")
    if 'temp_2m_max_dia' in ejemplo:
        print(f"  Temp máx día: {ejemplo['temp_2m_max_dia']:.1f}°C")
    if 'cape_max_dia' in ejemplo:
        print(f"  CAPE máx día: {ejemplo['cape_max_dia']:.0f} J/kg")
    if 'temp_range_dia' in ejemplo:
        print(f"  Rango térmico día: {ejemplo['temp_range_dia']:.1f}°C")
    
    print(f"\n▶ VENTANAS TEMPORALES:")
    if 'cape_mean_durante_vuelo' in ejemplo:
        print(f"  CAPE promedio durante vuelo: {ejemplo['cape_mean_durante_vuelo']:.0f} J/kg")
    if 'temp_trend_pre2h' in ejemplo:
        print(f"  Tendencia temp (2h antes): {ejemplo['temp_trend_pre2h']:.1f}°C")
    
    print(f"\n▶ FEATURES DERIVADAS:")
    if 'ratio_cape_vs_optimo' in ejemplo:
        print(f"  Ratio CAPE vs óptimo día: {ejemplo['ratio_cape_vs_optimo']:.2f}")
    if 'horas_desde_cape_max' in ejemplo:
        print(f"  Horas desde CAPE máx: {ejemplo['horas_desde_cape_max']:.1f}h")
    
    print("\n" + "="*70)
    print("✓✓✓ ENRIQUECIMIENTO COMPLETADO ✓✓✓")
    print("="*70)
    print("\nPróximo paso:")
    print("  Usar dataset_COMPLETO_ENRIQUECIDO.csv para análisis y modelado")
    print("="*70)
    
    return df_final


if __name__ == "__main__":
    df = enriquecer_dataset()