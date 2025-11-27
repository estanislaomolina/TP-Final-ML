"""
generar_dataset_FINAL.py
=======================
Genera UN SOLO dataset con TODO combinado:
- Datos de vuelos (IGC)
- Meteorología agregada del día
- Meteorología por hora (segmentos horarios del vuelo)

OUTPUT: dataset_FINAL.csv
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def obtener_condiciones_hora(fecha, hora, df_meteo_horarios):
    """
    Obtiene condiciones meteorológicas de una hora específica
    """
    
    # Filtrar por fecha y hora
    dato = df_meteo_horarios[
        (df_meteo_horarios['fecha'] == fecha) & 
        (df_meteo_horarios['hora'] == int(round(hora)))
    ]
    
    if len(dato) == 0:
        # Buscar hora más cercana
        dia_data = df_meteo_horarios[df_meteo_horarios['fecha'] == fecha]
        if len(dia_data) == 0:
            return {}
        
        horas_disponibles = dia_data['hora'].values
        hora_int = int(round(hora))
        diferencias = np.abs(horas_disponibles - hora_int)
        idx_min = np.argmin(diferencias)
        
        if diferencias[idx_min] <= 2:  # Máx 2 horas de diferencia
            dato = dia_data.iloc[idx_min:idx_min+1]
        else:
            return {}
    
    # Convertir a dict (funciona tanto para Series como DataFrame)
    if len(dato) > 0:
        registro = dato.iloc[0].to_dict()
    else:
        return {}
    
    # Extraer variables (excluir datetime, fecha, hora)
    variables = {}
    for col, valor in registro.items():
        if col not in ['datetime', 'fecha', 'hora']:
            variables[col] = valor
    
    return variables


def agregar_segmentos_horarios(vuelo, df_meteo_horarios, max_horas=6):
    """
    Agrega condiciones meteorológicas por hora del vuelo
    """
    
    fecha = vuelo['fecha']
    hora_despegue = vuelo.get('hora_despegue_decimal', None)
    duracion_horas = vuelo.get('duracion_horas', 2)
    
    if pd.isna(hora_despegue):
        return {}
    
    features = {}
    
    # Hora específica al despegue
    condiciones_despegue = obtener_condiciones_hora(fecha, hora_despegue, df_meteo_horarios)
    for var, valor in condiciones_despegue.items():
        features[f'meteo_{var}_despegue'] = valor
    
    # Por hora absoluta (hora_0, hora_1, ...)
    for h in range(max_horas + 1):
        hora_objetivo = hora_despegue + h
        
        if h <= duracion_horas:
            condiciones = obtener_condiciones_hora(fecha, hora_objetivo, df_meteo_horarios)
            for var, valor in condiciones.items():
                features[f'meteo_{var}_hora_{h}'] = valor
        else:
            # Marcar como NaN si el vuelo no llegó a esta hora
            for var in ['temp_2m', 'solar_rad', 'cloud_cover', 'wind_u', 'wind_v', 
                       'wind_speed', 'pressure', 'boundary_layer_height', 'cape', 
                       'skin_temp', 'precipitation']:
                features[f'meteo_{var}_hora_{h}'] = np.nan
    
    # Por percentiles (p0, p25, p50, p75, p100)
    for pct in [0, 25, 50, 75, 100]:
        hora_objetivo = hora_despegue + (duracion_horas * pct / 100)
        condiciones = obtener_condiciones_hora(fecha, hora_objetivo, df_meteo_horarios)
        
        for var, valor in condiciones.items():
            features[f'meteo_{var}_p{pct}'] = valor
    
    return features


def main():
    """
    Función principal
    """
    
    print("="*80)
    print("GENERACIÓN DE DATASET FINAL")
    print("="*80)
    print("\nCombinando:")
    print("  1. Datos de vuelos (IGC)")
    print("  2. Meteorología agregada del día")
    print("  3. Meteorología por hora (segmentos)")
    print("\nOutput: dataset_FINAL.csv")
    print("="*80)
    
    # 1. Cargar datos
    print("\n▶ PASO 1: Cargando archivos...")
    
    vuelos_file = 'data/raw/vuelos_metadata.csv'
    meteo_dia_file = 'data/processed/datos_meteorologicos_AGREGADOS_DIA.csv'
    meteo_hora_file = 'data/processed/datos_meteorologicos_HORARIOS.csv'
    
    if not os.path.exists(vuelos_file):
        print(f"✗ ERROR: No se encuentra {vuelos_file}")
        return
    
    if not os.path.exists(meteo_dia_file):
        print(f"✗ ERROR: No se encuentra {meteo_dia_file}")
        print("  Ejecuta primero: python procesar_era5_COMPLETO.py")
        return
    
    if not os.path.exists(meteo_hora_file):
        print(f"✗ ERROR: No se encuentra {meteo_hora_file}")
        print("  Ejecuta primero: python procesar_era5_COMPLETO.py")
        return
    
    df_vuelos = pd.read_csv(vuelos_file)
    df_meteo_dia = pd.read_csv(meteo_dia_file)
    df_meteo_hora = pd.read_csv(meteo_hora_file)
    
    print(f"  ✓ Vuelos: {len(df_vuelos)} registros")
    print(f"  ✓ Meteo día: {len(df_meteo_dia)} días")
    print(f"  ✓ Meteo hora: {len(df_meteo_hora)} registros horarios")
    
    # Preparar fechas
    df_vuelos['fecha'] = pd.to_datetime(df_vuelos['fecha']).dt.date
    df_meteo_dia['fecha'] = pd.to_datetime(df_meteo_dia['fecha']).dt.date
    df_meteo_hora['fecha'] = pd.to_datetime(df_meteo_hora['fecha']).dt.date
    
    # 2. Combinar vuelos con agregados del día
    print("\n▶ PASO 2: Combinando vuelos con meteorología del día...")
    
    df_combined = df_vuelos.merge(
        df_meteo_dia,
        on='fecha',
        how='left',
        suffixes=('', '_dia')
    )
    
    # Renombrar columnas de agregados del día
    rename_dict = {}
    for col in df_meteo_dia.columns:
        if col != 'fecha':
            rename_dict[col] = f'meteo_dia_{col}'
    
    df_combined = df_combined.rename(columns=rename_dict)
    
    print(f"  ✓ Combinado: {len(df_combined)} vuelos")
    
    vuelos_sin_meteo = df_combined[df_combined[[c for c in df_combined.columns if 'meteo_dia_' in c][0]].isnull()].shape[0]
    print(f"  ✓ Vuelos con meteorología: {len(df_combined) - vuelos_sin_meteo}")
    
    if vuelos_sin_meteo > 0:
        print(f"  ⚠ Vuelos sin meteorología: {vuelos_sin_meteo}")
    
    # 3. Agregar segmentos horarios
    print("\n▶ PASO 3: Agregando segmentos horarios...")
    print("  (Esto puede tardar unos minutos...)")
    
    segmentos_list = []
    
    for idx, vuelo in df_combined.iterrows():
        segmentos = agregar_segmentos_horarios(vuelo, df_meteo_hora, max_horas=6)
        segmentos_list.append(segmentos)
        
        if (idx + 1) % 50 == 0:
            print(f"    Procesados: {idx + 1}/{len(df_combined)} vuelos...")
    
    df_segmentos = pd.DataFrame(segmentos_list)
    
    print(f"  ✓ Segmentos horarios agregados: {len(df_segmentos.columns)} features")
    
    # 4. Combinar todo
    print("\n▶ PASO 4: Combinando todo en dataset final...")
    
    df_final = pd.concat([
        df_combined.reset_index(drop=True),
        df_segmentos.reset_index(drop=True)
    ], axis=1)
    
    # 5. Crear variable de clasificación si no existe
    if 'calidad_dia' not in df_final.columns:
        print("\n▶ PASO 5: Creando variable de clasificación...")
        
        def clasificar_dia(altura_max):
            if pd.isna(altura_max):
                return None
            elif altura_max > 2500:
                return 'Excelente'
            elif altura_max > 1500:
                return 'Bueno'
            elif altura_max > 1000:
                return 'Regular'
            else:
                return 'Malo'
        
        df_final['calidad_dia'] = df_final['altura_max_m'].apply(clasificar_dia)
        
        print(f"  ✓ Distribución de clases:")
        print(df_final['calidad_dia'].value_counts().sort_index())
    
    # 6. Features temporales
    print("\n▶ PASO 6: Agregando features temporales...")
    
    df_final['mes'] = pd.to_datetime(df_final['fecha']).dt.month
    df_final['dia_año'] = pd.to_datetime(df_final['fecha']).dt.dayofyear
    df_final['dia_semana'] = pd.to_datetime(df_final['fecha']).dt.dayofweek
    
    print(f"  ✓ Features temporales agregadas")
    
    # 7. Guardar
    print("\n▶ PASO 7: Guardando dataset final...")
    
    os.makedirs('data/processed', exist_ok=True)
    output_file = 'data/processed/dataset_FINAL.csv'
    
    df_final.to_csv(output_file, index=False)
    
    print(f"\n✓ Dataset guardado: {output_file}")
    
    # 8. Resumen
    print("\n" + "="*80)
    print("RESUMEN DEL DATASET FINAL")
    print("="*80)
    
    print(f"\n✓ Total de registros: {len(df_final)}")
    print(f"✓ Total de columnas: {len(df_final.columns)}")
    
    # Categorizar features
    features_vuelo = [c for c in df_final.columns if not c.startswith('meteo_') and c not in ['fecha', 'mes', 'dia_año', 'dia_semana', 'calidad_dia']]
    features_meteo_dia = [c for c in df_final.columns if c.startswith('meteo_dia_')]
    features_meteo_despegue = [c for c in df_final.columns if 'despegue' in c and c.startswith('meteo_')]
    features_meteo_hora = [c for c in df_final.columns if '_hora_' in c and c.startswith('meteo_')]
    features_meteo_percentil = [c for c in df_final.columns if '_p0' in c or '_p25' in c or '_p50' in c or '_p75' in c or '_p100' in c]
    
    print(f"\n▶ FEATURES DE VUELO (IGC): {len(features_vuelo)}")
    print(f"  Ejemplos: {features_vuelo[:5]}")
    
    print(f"\n▶ FEATURES METEOROLÓGICAS DEL DÍA: {len(features_meteo_dia)}")
    print(f"  Ejemplos: {features_meteo_dia[:5]}")
    
    print(f"\n▶ FEATURES METEOROLÓGICAS AL DESPEGUE: {len(features_meteo_despegue)}")
    print(f"  Ejemplos: {features_meteo_despegue[:5]}")
    
    print(f"\n▶ FEATURES METEOROLÓGICAS POR HORA: {len(features_meteo_hora)}")
    print(f"  Ejemplos: {features_meteo_hora[:5]}")
    
    print(f"\n▶ FEATURES METEOROLÓGICAS POR PERCENTIL: {len(features_meteo_percentil)}")
    print(f"  Ejemplos: {features_meteo_percentil[:5]}")
    
    total_meteo = len(features_meteo_dia) + len(features_meteo_despegue) + len(features_meteo_hora) + len(features_meteo_percentil)
    
    print(f"\n{'='*80}")
    print(f"TOTAL FEATURES METEOROLÓGICAS: {total_meteo}")
    print(f"{'='*80}")
    
    # Verificar valores faltantes
    print(f"\n▶ VALORES FALTANTES:")
    
    missing_meteo_dia = df_final[features_meteo_dia].isnull().sum().sum()
    missing_meteo_hora = df_final[features_meteo_hora].isnull().sum().sum()
    
    print(f"  Meteorología del día: {missing_meteo_dia / (len(df_final) * len(features_meteo_dia)) * 100:.1f}%")
    print(f"  Meteorología horaria: {missing_meteo_hora / (len(df_final) * len(features_meteo_hora)) * 100:.1f}%")
    print(f"    (Normal: vuelos cortos no tienen todas las horas)")
    
    # Ejemplo de un vuelo
    print("\n" + "="*80)
    print("EJEMPLO DE REGISTRO COMPLETO")
    print("="*80)
    
    ejemplo = df_final.iloc[0]
    
    print(f"\nVuelo: {ejemplo.get('flight_id', 'N/A')}")
    print(f"Fecha: {ejemplo['fecha']}")
    print(f"Duración: {ejemplo.get('duracion_horas', 'N/A'):.2f} horas")
    
    print(f"\n▶ DATOS DEL VUELO:")
    print(f"  Altura máxima: {ejemplo.get('altura_max_m', 'N/A'):.0f} m")
    print(f"  Distancia: {ejemplo.get('distancia_km', 'N/A'):.1f} km")
    print(f"  Calidad día: {ejemplo.get('calidad_dia', 'N/A')}")
    
    print(f"\n▶ METEOROLOGÍA DEL DÍA:")
    if 'meteo_dia_temp_2m_max' in ejemplo:
        print(f"  Temp máx día: {ejemplo['meteo_dia_temp_2m_max']:.1f}°C")
    if 'meteo_dia_cape_max' in ejemplo:
        print(f"  CAPE máx día: {ejemplo['meteo_dia_cape_max']:.0f} J/kg")
    
    print(f"\n▶ METEOROLOGÍA AL DESPEGUE:")
    if 'meteo_temp_2m_despegue' in ejemplo:
        print(f"  Temperatura: {ejemplo['meteo_temp_2m_despegue']:.1f}°C")
    if 'meteo_cape_despegue' in ejemplo:
        print(f"  CAPE: {ejemplo['meteo_cape_despegue']:.0f} J/kg")
    
    print(f"\n▶ EVOLUCIÓN HORARIA (primeras 3 horas):")
    for h in range(3):
        temp_col = f'meteo_temp_2m_hora_{h}'
        cape_col = f'meteo_cape_hora_{h}'
        if temp_col in ejemplo and pd.notna(ejemplo[temp_col]):
            print(f"  Hora {h}: T={ejemplo[temp_col]:.1f}°C, CAPE={ejemplo[cape_col]:.0f} J/kg")
    
    print(f"\n▶ EVOLUCIÓN POR PERCENTILES:")
    for pct in [0, 50, 100]:
        temp_col = f'meteo_temp_2m_p{pct}'
        cape_col = f'meteo_cape_p{pct}'
        if temp_col in ejemplo and pd.notna(ejemplo[temp_col]):
            print(f"  p{pct}: T={ejemplo[temp_col]:.1f}°C, CAPE={ejemplo[cape_col]:.0f} J/kg")
    
    print("\n" + "="*80)
    print("✓✓✓ DATASET FINAL GENERADO ✓✓✓")
    print("="*80)
    print(f"\nArchivo: {output_file}")
    print(f"Registros: {len(df_final)}")
    print(f"Features totales: {len(df_final.columns)}")
    print("\n¡Listo para análisis exploratorio y modelado!")
    print("="*80)


if __name__ == "__main__":
    main()