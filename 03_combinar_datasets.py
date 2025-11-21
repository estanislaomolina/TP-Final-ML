"""
Script para combinar datos de vuelos y meteorología en un dataset final
"""

import pandas as pd
import numpy as np
import os

def combinar_datasets():
    """
    Combina metadata de vuelos con datos meteorológicos
    """
    
    print("="*70)
    print("COMBINACIÓN DE DATASETS")
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
        print("Ejecuta primero: python 02_descargar_era5.py")
        return None
    
    df_vuelos = pd.read_csv(vuelos_file)
    df_meteo = pd.read_csv(meteo_file)
    
    # Convertir fechas
    df_vuelos['fecha'] = pd.to_datetime(df_vuelos['fecha']).dt.date
    df_meteo['fecha'] = pd.to_datetime(df_meteo['fecha']).dt.date
    
    print(f"  Vuelos: {len(df_vuelos)} registros")
    print(f"  Datos meteo: {len(df_meteo)} días")
    
    # Combinar por fecha
    print("\nCombinando por fecha...")
    df_combined = pd.merge(
        df_vuelos,
        df_meteo,
        on='fecha',
        how='inner'
    )
    
    print(f"  Registros combinados: {len(df_combined)}")
    
    # Crear variable de clasificación: calidad_dia
    print("\nCreando variable de clasificación...")
    
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
    
    # Agregar features derivadas adicionales
    print("\nAgregando features derivadas...")
    
    # Mes y día del año (para estacionalidad)
    df_combined['fecha_dt'] = pd.to_datetime(df_combined['fecha'])
    df_combined['mes'] = df_combined['fecha_dt'].dt.month
    df_combined['dia_año'] = df_combined['fecha_dt'].dt.dayofyear
    
    # Seleccionar columnas finales
    # Features meteorológicas
    features_meteo = [col for col in df_meteo.columns if col != 'fecha']
    
    # Targets
    targets = [
        'altura_max_m',
        'ganancia_altura_m',
        'duracion_min',
        'distancia_km',
        'calidad_dia'
    ]
    
    # Metadata útil
    metadata = [
        'flight_id',
        'fecha',
        'pilot',
        'glider',
        'mes',
        'dia_año'
    ]
    
    # Columnas finales
    columnas_finales = metadata + features_meteo + targets
    df_final = df_combined[columnas_finales].copy()
    
    # Eliminar filas con valores faltantes en targets importantes
    print("\nLimpiando datos...")
    antes = len(df_final)
    df_final = df_final.dropna(subset=targets[:-1])  # No dropear por calidad_dia
    despues = len(df_final)
    print(f"  Filas eliminadas por NaN en targets: {antes - despues}")
    
    # Guardar dataset procesado
    os.makedirs('data/processed', exist_ok=True)
    output_file = 'data/processed/dataset_completo.csv'
    df_final.to_csv(output_file, index=False)
    
    print(f"\n✓ Dataset final guardado: {output_file}")
    print(f"  Registros: {len(df_final)}")
    print(f"  Features: {len(features_meteo)}")
    print(f"  Targets: {len(targets)}")
    
    # Resumen estadístico
    print("\n" + "="*70)
    print("RESUMEN ESTADÍSTICO DEL DATASET FINAL")
    print("="*70)
    
    print("\nVariables Target:")
    print(df_final[targets[:-1]].describe())
    
    print("\nVariables Meteorológicas (primeras 5):")
    print(df_final[features_meteo[:5]].describe())
    
    return df_final


def dividir_train_test(df, test_size=0.2, random_state=42):
    """
    Dividir en conjunto de desarrollo y test
    """
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*70)
    print("DIVISIÓN EN DESARROLLO Y TEST")
    print("="*70)
    
    # Split estratificado por calidad_dia
    df_dev, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['calidad_dia']
    )
    
    print(f"\nDataset desarrollo: {len(df_dev)} ({(1-test_size)*100:.0f}%)")
    print(f"Dataset test: {len(df_test)} ({test_size*100:.0f}%)")
    
    # Verificar distribución de clases
    print("\nDistribución en desarrollo:")
    print(df_dev['calidad_dia'].value_counts(normalize=True).sort_index())
    
    print("\nDistribución en test:")
    print(df_test['calidad_dia'].value_counts(normalize=True).sort_index())
    
    # Guardar
    df_dev.to_csv('data/processed/vuelos_dev.csv', index=False)
    df_test.to_csv('data/processed/vuelos_test.csv', index=False)
    
    print("\n✓ Datasets guardados:")
    print("  - data/processed/vuelos_dev.csv")
    print("  - data/processed/vuelos_test.csv")
    
    return df_dev, df_test


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    # Combinar datasets
    df_final = combinar_datasets()
    
    if df_final is not None:
        # Dividir en dev y test
        df_dev, df_test = dividir_train_test(df_final, test_size=0.2)
        
        print("\n" + "="*70)
        print("✓ FASE 3 COMPLETADA: Dataset final creado")
        print("="*70)
        print("\nArchivos generados:")
        print("  1. data/processed/dataset_completo.csv")
        print("  2. data/processed/vuelos_dev.csv")
        print("  3. data/processed/vuelos_test.csv")
        print("\nSiguiente paso: Análisis exploratorio de datos (EDA)")
        print("  Ejecuta: jupyter notebook")
        print("  Y abre: notebooks/01_analisis_exploratorio.ipynb")
