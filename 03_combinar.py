"""
03c_features_por_segmentos_horarios.py
Agrega features meteorológicas divididas en segmentos horarios
Sin usar series temporales, solo features estáticas
"""

import pandas as pd
import numpy as np
import os


def obtener_condiciones_hora_especifica(fecha, hora, df_meteo_horario):
    """
    Obtiene condiciones meteorológicas de una hora específica
    
    Returns:
        dict con todas las variables meteorológicas de esa hora
    """
    
    # Filtrar datos del día
    dia_data = df_meteo_horario[df_meteo_horario['fecha'] == fecha]
    
    if len(dia_data) == 0:
        return {}
    
    # Encontrar hora más cercana
    horas_disponibles = sorted(dia_data['hora'].unique())
    hora_int = int(round(hora))
    
    # Buscar hora exacta o más cercana
    if hora_int in horas_disponibles:
        hora_match = hora_int
    else:
        diferencias = [abs(h - hora_int) for h in horas_disponibles]
        hora_match = horas_disponibles[diferencias.index(min(diferencias))]
    
    # Obtener datos de esa hora
    hora_data = dia_data[dia_data['hora'] == hora_match].iloc[0]
    
    # Extraer variables meteorológicas
    variables = {}
    for col in hora_data.index:
        if col not in ['fecha', 'hora', 'datetime']:
            variables[col] = hora_data[col]
    
    return variables


def calcular_segmentos_horas_absolutas(vuelo, df_meteo_horario, max_horas=6):
    """
    Calcula condiciones meteorológicas por hora absoluta desde despegue
    
    hora_0: Al despegue (14:30 → 14:00 o 15:00)
    hora_1: 1 hora después (15:30 → 15:00 o 16:00)
    hora_2: 2 horas después (16:30 → 16:00 o 17:00)
    ...
    
    Parameters:
    -----------
    max_horas : int
        Número máximo de horas a extraer (vuelos más largos se truncan)
    
    Returns:
    --------
    dict con features tipo: temp_hora_0, cape_hora_1, etc.
    """
    
    fecha = vuelo['fecha']
    hora_despegue = vuelo.get('hora_despegue_decimal', None)
    duracion_horas = vuelo.get('duracion_horas', 2)
    
    if pd.isna(hora_despegue):
        return {}
    
    features = {}
    
    # Para cada hora desde el despegue
    for h in range(max_horas + 1):  # 0, 1, 2, 3, 4, 5, 6
        hora_objetivo = hora_despegue + h
        
        # Si esta hora existe en el vuelo (no se pasó de la duración)
        if h <= duracion_horas:
            condiciones = obtener_condiciones_hora_especifica(
                fecha, hora_objetivo, df_meteo_horario
            )
            
            # Agregar cada variable con sufijo _hora_X
            for var, valor in condiciones.items():
                features[f'{var}_hora_{h}'] = valor
        else:
            # Hora fuera del vuelo → NaN
            # Obtener lista de variables de hora_0 para mantener consistencia
            if h == 0:
                pass  # No hay referencia aún
            else:
                # Usar variables de hora anterior como referencia
                vars_referencia = ['temp_2m', 'cape', 'solar_rad', 'wind_speed', 
                                 'cloud_cover', 'pressure', 'boundary_layer_height',
                                 'wind_u', 'wind_v', 'skin_temp', 'precipitation']
                for var in vars_referencia:
                    features[f'{var}_hora_{h}'] = np.nan
    
    return features


def calcular_segmentos_percentiles(vuelo, df_meteo_horario, percentiles=[0, 25, 50, 75, 100]):
    """
    Calcula condiciones meteorológicas en percentiles del vuelo
    
    0%: Inicio (despegue)
    25%: Primer cuarto del vuelo
    50%: Mitad del vuelo
    75%: Tres cuartos del vuelo
    100%: Final (aterrizaje)
    
    Esto normaliza la duración del vuelo
    
    Returns:
    --------
    dict con features tipo: temp_p0, cape_p25, temp_p50, etc.
    """
    
    fecha = vuelo['fecha']
    hora_despegue = vuelo.get('hora_despegue_decimal', None)
    duracion_horas = vuelo.get('duracion_horas', 2)
    
    if pd.isna(hora_despegue):
        return {}
    
    features = {}
    
    for pct in percentiles:
        # Calcular hora correspondiente a este percentil
        hora_objetivo = hora_despegue + (duracion_horas * pct / 100)
        
        condiciones = obtener_condiciones_hora_especifica(
            fecha, hora_objetivo, df_meteo_horario
        )
        
        # Agregar cada variable con sufijo _pX
        for var, valor in condiciones.items():
            features[f'{var}_p{pct}'] = valor
    
    # Features derivadas entre percentiles
    # Cambio de inicio a fin
    if 'temp_2m_p0' in features and 'temp_2m_p100' in features:
        features['temp_change_p0_to_p100'] = features['temp_2m_p100'] - features['temp_2m_p0']
    
    if 'cape_p0' in features and 'cape_p100' in features:
        features['cape_change_p0_to_p100'] = features['cape_p100'] - features['cape_p0']
    
    # Cambio de inicio a mitad
    if 'temp_2m_p0' in features and 'temp_2m_p50' in features:
        features['temp_change_p0_to_p50'] = features['temp_2m_p50'] - features['temp_2m_p0']
    
    if 'cape_p0' in features and 'cape_p50' in features:
        features['cape_change_p0_to_p50'] = features['cape_p50'] - features['cape_p0']
    
    # Cambio de mitad a fin
    if 'temp_2m_p50' in features and 'temp_2m_p100' in features:
        features['temp_change_p50_to_p100'] = features['temp_2m_p100'] - features['temp_2m_p50']
    
    if 'cape_p50' in features and 'cape_p100' in features:
        features['cape_change_p50_to_p100'] = features['cape_p100'] - features['cape_p50']
    
    return features


def agregar_segmentos_horarios():
    """
    Agrega features de segmentos horarios al dataset
    """
    
    print("="*70)
    print("FEATURES POR SEGMENTOS HORARIOS")
    print("="*70)
    
    # Cargar datasets
    print("\nCargando datos...")
    
    dataset_file = 'data/processed/dataset_COMPLETO_ENRIQUECIDO.csv'
    meteo_file = 'data/raw/datos_meteorologicos_HORARIOS.csv'
    
    if not os.path.exists(dataset_file):
        print(f"ERROR: No se encuentra {dataset_file}")
        print("Ejecuta primero: python 03b_agregar_features_meteorologicas_completas.py")
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
    
    # 1. Segmentos por horas absolutas
    print("\n" + "="*70)
    print("PASO 1: SEGMENTOS POR HORA ABSOLUTA (hora_0, hora_1, ...)")
    print("="*70)
    print("\nEsto crea features para cada hora desde el despegue:")
    print("  hora_0 = condiciones al despegue")
    print("  hora_1 = condiciones 1h después del despegue")
    print("  hora_2 = condiciones 2h después del despegue")
    print("  ... hasta hora_6")
    print("\nVuelos más cortos tendrán NaN en horas posteriores")
    
    segmentos_horas = []
    
    for idx, vuelo in df_vuelos.iterrows():
        segmentos = calcular_segmentos_horas_absolutas(
            vuelo, df_meteo, max_horas=6
        )
        segmentos_horas.append(segmentos)
        
        if (idx + 1) % 50 == 0:
            print(f"  Procesados: {idx + 1}/{len(df_vuelos)} vuelos...")
    
    df_horas = pd.DataFrame(segmentos_horas)
    
    print(f"\n  ✓ Features por hora absoluta: {len(df_horas.columns)} columnas")
    print(f"    Ejemplo: {list(df_horas.columns[:10])}")
    
    # 2. Segmentos por percentiles
    print("\n" + "="*70)
    print("PASO 2: SEGMENTOS POR PERCENTILES (p0, p25, p50, p75, p100)")
    print("="*70)
    print("\nEsto normaliza la duración del vuelo:")
    print("  p0 = inicio (0% del vuelo)")
    print("  p25 = primer cuarto (25% del vuelo)")
    print("  p50 = mitad (50% del vuelo)")
    print("  p75 = tres cuartos (75% del vuelo)")
    print("  p100 = final (100% del vuelo)")
    print("\nTodos los vuelos tienen estas 5 mediciones, sin importar duración")
    
    segmentos_pct = []
    
    for idx, vuelo in df_vuelos.iterrows():
        segmentos = calcular_segmentos_percentiles(
            vuelo, df_meteo, percentiles=[0, 25, 50, 75, 100]
        )
        segmentos_pct.append(segmentos)
        
        if (idx + 1) % 50 == 0:
            print(f"  Procesados: {idx + 1}/{len(df_vuelos)} vuelos...")
    
    df_pct = pd.DataFrame(segmentos_pct)
    
    print(f"\n  ✓ Features por percentiles: {len(df_pct.columns)} columnas")
    print(f"    Ejemplo: {list(df_pct.columns[:10])}")
    
    # 3. Combinar todo
    print("\n" + "="*70)
    print("PASO 3: COMBINACIÓN")
    print("="*70)
    
    df_final = pd.concat([
        df_vuelos.reset_index(drop=True),
        df_horas.reset_index(drop=True),
        df_pct.reset_index(drop=True)
    ], axis=1)
    
    print(f"\n  Dataset final: {df_final.shape}")
    
    # Guardar
    output_file = 'data/processed/dataset_FINAL_CON_SEGMENTOS.csv'
    df_final.to_csv(output_file, index=False)
    
    print(f"\n✓ Dataset guardado: {output_file}")
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE FEATURES POR SEGMENTOS")
    print("="*70)
    
    # Identificar features
    features_hora_absoluta = [col for col in df_final.columns if '_hora_' in col]
    features_percentiles = [col for col in df_final.columns if '_p0' in col or '_p25' in col or '_p50' in col or '_p75' in col or '_p100' in col]
    features_cambios = [col for col in df_final.columns if 'change_p' in col]
    
    print(f"\n1. SEGMENTOS POR HORA ABSOLUTA: {len(features_hora_absoluta)} features")
    print(f"   Variables por hora: temp_2m, cape, solar_rad, wind_speed, etc.")
    print(f"   Horas: 0, 1, 2, 3, 4, 5, 6")
    print(f"   Total: ~11 variables × 7 horas = ~77 features")
    
    # Mostrar ejemplo por variable
    print(f"\n   Ejemplo (temperatura):")
    temp_features = [col for col in features_hora_absoluta if col.startswith('temp_2m_hora_')]
    for feat in sorted(temp_features):
        print(f"     • {feat}")
    
    print(f"\n   Ejemplo (CAPE):")
    cape_features = [col for col in features_hora_absoluta if col.startswith('cape_hora_')]
    for feat in sorted(cape_features):
        print(f"     • {feat}")
    
    print(f"\n2. SEGMENTOS POR PERCENTILES: {len(features_percentiles)} features")
    print(f"   Variables: temp_2m, cape, solar_rad, wind_speed, etc.")
    print(f"   Percentiles: 0%, 25%, 50%, 75%, 100%")
    print(f"   Total: ~11 variables × 5 percentiles = ~55 features")
    
    # Mostrar ejemplo
    print(f"\n   Ejemplo (temperatura en percentiles):")
    temp_pct = [col for col in features_percentiles if col.startswith('temp_2m_p')]
    for feat in sorted(temp_pct):
        print(f"     • {feat}")
    
    print(f"\n3. CAMBIOS ENTRE PERCENTILES: {len(features_cambios)} features")
    print(f"   Cambios calculados:")
    for feat in sorted(features_cambios):
        print(f"     • {feat}")
    
    total_segmentos = len(features_hora_absoluta) + len(features_percentiles) + len(features_cambios)
    print(f"\n{'='*70}")
    print(f"TOTAL FEATURES DE SEGMENTOS: {total_segmentos}")
    print(f"TOTAL FEATURES DATASET: {len(df_final.columns)}")
    print(f"{'='*70}")
    
    # Ejemplo de un vuelo
    print("\n" + "="*70)
    print("EJEMPLO: EVOLUCIÓN METEOROLÓGICA DURANTE UN VUELO")
    print("="*70)
    
    # Buscar un vuelo con duración ~3 horas
    ejemplo_idx = None
    for idx, vuelo in df_final.iterrows():
        if 2.5 <= vuelo.get('duracion_horas', 0) <= 3.5:
            ejemplo_idx = idx
            break
    
    if ejemplo_idx is None:
        ejemplo_idx = 0
    
    ejemplo = df_final.iloc[ejemplo_idx]
    
    print(f"\nVuelo: {ejemplo.get('flight_id', 'N/A')}")
    print(f"Duración: {ejemplo.get('duracion_horas', 'N/A'):.2f} horas")
    print(f"Hora despegue: {ejemplo.get('hora_despegue_decimal', 'N/A'):.2f}")
    
    print(f"\n▶ ENFOQUE 1: Por hora ABSOLUTA desde despegue")
    print(f"\n  Hora | Temp (°C) | CAPE (J/kg) | Viento (m/s)")
    print(f"  " + "-"*50)
    
    for h in range(7):
        temp_col = f'temp_2m_hora_{h}'
        cape_col = f'cape_hora_{h}'
        wind_col = f'wind_speed_hora_{h}'
        
        temp = ejemplo.get(temp_col, np.nan)
        cape = ejemplo.get(cape_col, np.nan)
        wind = ejemplo.get(wind_col, np.nan)
        
        if pd.notna(temp):
            print(f"    {h}  | {temp:>7.1f}   | {cape:>10.0f}  | {wind:>10.1f}")
        else:
            print(f"    {h}  |    N/A    |     N/A    |     N/A")
    
    print(f"\n▶ ENFOQUE 2: Por PERCENTILES del vuelo")
    print(f"\n   %   | Temp (°C) | CAPE (J/kg) | Viento (m/s)")
    print(f"  " + "-"*50)
    
    for pct in [0, 25, 50, 75, 100]:
        temp_col = f'temp_2m_p{pct}'
        cape_col = f'cape_p{pct}'
        wind_col = f'wind_speed_p{pct}'
        
        temp = ejemplo.get(temp_col, np.nan)
        cape = ejemplo.get(cape_col, np.nan)
        wind = ejemplo.get(wind_col, np.nan)
        
        if pd.notna(temp):
            print(f"  {pct:>3}  | {temp:>7.1f}   | {cape:>10.0f}  | {wind:>10.1f}")
    
    # Mostrar cambios
    print(f"\n▶ CAMBIOS durante el vuelo:")
    if 'temp_change_p0_to_p100' in ejemplo:
        print(f"  Temperatura (inicio → fin): {ejemplo['temp_change_p0_to_p100']:+.1f}°C")
    if 'cape_change_p0_to_p100' in ejemplo:
        print(f"  CAPE (inicio → fin): {ejemplo['cape_change_p0_to_p100']:+.0f} J/kg")
    
    print("\n" + "="*70)
    print("✓✓✓ SEGMENTOS HORARIOS COMPLETADOS ✓✓✓")
    print("="*70)
    print("\nAhora tenés:")
    print("  ✓ Condiciones hora por hora (hora_0, hora_1, ...)")
    print("  ✓ Condiciones normalizadas por duración (p0, p25, p50, ...)")
    print("  ✓ Cambios entre etapas del vuelo")
    print("\nDataset listo para modelado!")
    print("="*70)
    
    # Análisis de missingness
    print("\n" + "="*70)
    print("ANÁLISIS DE VALORES FALTANTES EN SEGMENTOS")
    print("="*70)
    
    print("\nPor hora absoluta (esperado: vuelos cortos tienen NaN en horas altas):")
    for h in range(7):
        cols_hora = [col for col in df_final.columns if f'_hora_{h}' in col]
        if cols_hora:
            missing_pct = df_final[cols_hora[0]].isnull().sum() / len(df_final) * 100
            print(f"  hora_{h}: {missing_pct:.1f}% faltantes")
    
    print("\nPor percentiles (deberían tener muy pocos NaN):")
    for pct in [0, 25, 50, 75, 100]:
        cols_pct = [col for col in df_final.columns if f'_p{pct}' in col and 'change' not in col]
        if cols_pct:
            missing_pct = df_final[cols_pct[0]].isnull().sum() / len(df_final) * 100
            print(f"  p{pct}: {missing_pct:.1f}% faltantes")
    
    return df_final


if __name__ == "__main__":
    df = agregar_segmentos_horarios()