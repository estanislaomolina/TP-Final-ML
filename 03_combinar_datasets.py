"""
Script CORREGIDO para combinar datasets
Mantiene TODAS las features sin hacer splits
"""

import pandas as pd
import numpy as np
import os

def combinar_datasets_completo():
    """
    Combina vuelos y meteorología manteniendo TODAS las columnas
    """
    
    print("="*70)
    print("COMBINACIÓN DE DATASETS - TODAS LAS FEATURES")
    print("="*70)
    
    # Cargar datos
    print("\nCargando datos...")
    
    vuelos_file = 'data/raw/vuelos_metadata.csv'
    meteo_file = 'data/raw/datos_meteorologicos.csv'
    
    if not os.path.exists(vuelos_file):
        print(f"ERROR: No se encuentra {vuelos_file}")
        return None
    
    if not os.path.exists(meteo_file):
        print(f"ERROR: No se encuentra {meteo_file}")
        return None
    
    df_vuelos = pd.read_csv(vuelos_file)
    df_meteo = pd.read_csv(meteo_file)
    
    print(f"  Vuelos cargados: {len(df_vuelos)} filas, {len(df_vuelos.columns)} columnas")
    print(f"  Meteo cargados: {len(df_meteo)} filas, {len(df_meteo.columns)} columnas")
    
    # Convertir fechas
    df_vuelos['fecha'] = pd.to_datetime(df_vuelos['fecha']).dt.date
    df_meteo['fecha'] = pd.to_datetime(df_meteo['fecha']).dt.date
    
    print(f"\nFechas únicas en vuelos: {df_vuelos['fecha'].nunique()}")
    print(f"Fechas únicas en meteorología: {df_meteo['fecha'].nunique()}")
    
    # Combinar por fecha (INNER JOIN para tener solo fechas con ambos datos)
    print("\nCombinando por fecha (inner join)...")
    df_combined = pd.merge(
        df_vuelos,
        df_meteo,
        on='fecha',
        how='inner',
        suffixes=('_vuelo', '_meteo')  # Por si hay columnas repetidas
    )
    
    print(f"  ✓ Registros combinados: {len(df_combined)}")
    print(f"  ✓ Total de columnas: {len(df_combined.columns)}")
    
    # Crear variable de clasificación: calidad_dia
    print("\nCreando variable target 'calidad_dia'...")
    
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
    
    df_combined['calidad_dia'] = df_combined['altura_max_m'].apply(clasificar_dia)
    
    # Mostrar distribución
    print("\nDistribución de calidad de días:")
    print(df_combined['calidad_dia'].value_counts().sort_index())
    
    # Agregar features temporales útiles
    print("\nAgregando features temporales...")
    df_combined['fecha_dt'] = pd.to_datetime(df_combined['fecha'])
    df_combined['mes'] = df_combined['fecha_dt'].dt.month
    df_combined['dia_año'] = df_combined['fecha_dt'].dt.dayofyear
    df_combined['dia_semana'] = df_combined['fecha_dt'].dt.dayofweek
    df_combined['año'] = df_combined['fecha_dt'].dt.year
    
    # Eliminar filas con valores faltantes en targets principales
    print("\nLimpiando datos...")
    antes = len(df_combined)
    
    # Solo eliminar si faltan targets críticos
    targets_criticos = ['altura_max_m', 'duracion_min', 'distancia_km']
    df_combined = df_combined.dropna(subset=targets_criticos)
    
    despues = len(df_combined)
    print(f"  Filas eliminadas por NaN en targets: {antes - despues}")
    
    # Guardar dataset completo
    os.makedirs('data/processed', exist_ok=True)
    output_file = 'data/processed/dataset_completo.csv'
    df_combined.to_csv(output_file, index=False)
    
    print(f"\n✓ Dataset final guardado: {output_file}")
    print(f"  Vuelos: {len(df_combined)}")
    print(f"  Features totales: {len(df_combined.columns)}")
    
    # Mostrar categorías de features
    print("\n" + "="*70)
    print("CATEGORÍAS DE FEATURES EN EL DATASET")
    print("="*70)
    
    # Identificar categorías
    features_vuelo = [col for col in df_combined.columns if col in df_vuelos.columns and col != 'fecha']
    features_meteo = [col for col in df_combined.columns if col in df_meteo.columns and col != 'fecha']
    features_derivadas = ['calidad_dia', 'mes', 'dia_año', 'dia_semana', 'año', 'fecha_dt']
    
    print(f"\n📊 Features de VUELO (del IGC): {len(features_vuelo)}")
    print("   Categorías:")
    print("   • Básicas: altura, duración, distancia, velocidad")
    print("   • Térmicas: número, intensidad, altura, duración")
    print("   • Trayectoria: rumbo, velocidad ground")
    print("   • Temporales: franjas horarias")
    print("   • Espaciales: bounding box, área")
    print("   • Variabilidad: std, coeficientes")
    
    print(f"\n🌤️  Features METEOROLÓGICAS (de ERA5): {len(features_meteo)}")
    print("   Variables:")
    print("   • Temperatura: max, min, mean, differential")
    print("   • Radiación solar: total, max")
    print("   • Viento: componentes U/V, velocidad")
    print("   • CAPE: max, mean")
    print("   • Capa límite: altura max, mean")
    print("   • Nubes, presión, precipitación")
    
    print(f"\n🎯 Features DERIVADAS: {len(features_derivadas)}")
    print("   • calidad_dia (target clasificación)")
    print("   • mes, dia_año, dia_semana, año")
    
    print(f"\n📈 TOTAL FEATURES: {len(df_combined.columns)}")
    
    # Resumen estadístico
    print("\n" + "="*70)
    print("RESUMEN ESTADÍSTICO - TARGETS PRINCIPALES")
    print("="*70)
    
    targets = ['altura_max_m', 'duracion_min', 'distancia_km', 
               'num_termicas', 'intensidad_termicas_mean_ms',
               'temp_2m_max', 'cape_max', 'solar_rad_max']
    
    targets_disponibles = [t for t in targets if t in df_combined.columns]
    
    if targets_disponibles:
        print("\n")
        print(df_combined[targets_disponibles].describe())
    
    # Info sobre valores faltantes
    print("\n" + "="*70)
    print("VALORES FALTANTES")
    print("="*70)
    
    missing = df_combined.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) > 0:
        print(f"\nColumnas con valores faltantes: {len(missing)}")
        print("\nTop 10:")
        for col, count in missing.head(10).items():
            pct = count / len(df_combined) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print("\n✓ No hay valores faltantes")
    
    # Guardar lista de columnas para referencia
    with open('data/processed/columnas_dataset.txt', 'w') as f:
        f.write("COLUMNAS DEL DATASET COMPLETO\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"FEATURES DE VUELO ({len(features_vuelo)}):\n")
        for col in sorted(features_vuelo):
            f.write(f"  - {col}\n")
        
        f.write(f"\nFEATURES METEOROLÓGICAS ({len(features_meteo)}):\n")
        for col in sorted(features_meteo):
            f.write(f"  - {col}\n")
        
        f.write(f"\nFEATURES DERIVADAS ({len(features_derivadas)}):\n")
        for col in features_derivadas:
            f.write(f"  - {col}\n")
        
        f.write(f"\nTOTAL: {len(df_combined.columns)} columnas\n")
    
    print("\n✓ Lista de columnas guardada: data/processed/columnas_dataset.txt")
    
    print("\n" + "="*70)
    print("✓✓✓ DATASET COMPLETO CREADO ✓✓✓")
    print("="*70)
    print("\nArchivo generado:")
    print(f"  • data/processed/dataset_completo.csv")
    print(f"    {len(df_combined)} filas × {len(df_combined.columns)} columnas")
    print("\nEste dataset tiene TODAS las features de:")
    print("  ✓ Vuelos (IGC processing)")
    print("  ✓ Meteorología (ERA5)")
    print("  ✓ Features derivadas")
    print("\n¡Listo para análisis exploratorio y feature engineering!")
    print("="*70)
    
    return df_combined


if __name__ == "__main__":
    df = combinar_datasets_completo()