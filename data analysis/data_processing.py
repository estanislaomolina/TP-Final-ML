"""
data_processing.py
==================
Funciones para limpieza, preprocesamiento y split de datos.

Autor: Estanislao
Fecha: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os


def cargar_dataset(filepath='data/processed/dataset_FINAL.csv'):
    """
    Carga el dataset final.
    
    Args:
        filepath: Ruta al archivo CSV
        
    Returns:
        DataFrame con el dataset
    """
    df = pd.read_csv(filepath)
    df['fecha'] = pd.to_datetime(df['fecha'])
    print(f"✓ Dataset cargado: {df.shape}")
    return df


def identificar_columnas(df):
    """
    Identifica y separa columnas por tipo.
    
    Args:
        df: DataFrame
        
    Returns:
        dict con listas de columnas por categoría
    """
    
    columnas = {
        'metadata': ['fecha', 'pilot', 'glider', 'competition_id', 'filename', 'flight_id'],
        
        'targets_regresion': [
            'altura_max_m',
            'duracion_horas', 
            'duracion_min',
            'distancia_km',
            'velocidad_promedio_kmh'
        ],
        
        'targets_clasificacion': [
            'calidad_dia'
        ],
        
        'features_vuelo': [],  # Se llenarán automáticamente
        
        'features_meteo_dia': [col for col in df.columns if col.startswith('meteo_dia_')],
        
        'features_meteo_despegue': [col for col in df.columns if 'despegue' in col and col.startswith('meteo_')],
        
        'features_meteo_hora': [col for col in df.columns if '_hora_' in col and col.startswith('meteo_')],
        
        'features_meteo_percentil': [col for col in df.columns if any(f'_p{p}' in col for p in [0,25,50,75,100]) and col.startswith('meteo_')],
        
        'features_temporales': ['mes', 'dia_año', 'dia_semana']
    }
    
    # Identificar features de vuelo (todo lo que no es meteo, target, metadata o temporal)
    usadas = (columnas['metadata'] + 
              columnas['targets_regresion'] + 
              columnas['targets_clasificacion'] +
              columnas['features_meteo_dia'] +
              columnas['features_meteo_despegue'] +
              columnas['features_meteo_hora'] +
              columnas['features_meteo_percentil'] +
              columnas['features_temporales'])
    
    columnas['features_vuelo'] = [col for col in df.columns if col not in usadas]
    
    return columnas


def analizar_missing_values(df):
    """
    Analiza valores faltantes en el dataset.
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame con resumen de missing values
    """
    missing = pd.DataFrame({
        'columna': df.columns,
        'n_missing': df.isnull().sum(),
        'pct_missing': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    missing = missing[missing['n_missing'] > 0].sort_values('pct_missing', ascending=False)
    
    return missing


def eliminar_columnas_innecesarias(df, columnas_dict, umbral_nan=50, targets_a_predecir=None):
    """
    Elimina columnas innecesarias para modelado.
    IMPORTANTE: NO elimina 'fecha' (se necesita para split temporal)
    
    Args:
        df: DataFrame
        columnas_dict: Dict con categorización de columnas
        umbral_nan: % de NaN para eliminar columna
        
    Returns:
        DataFrame limpio, lista de columnas eliminadas
    """
    
    df_clean = df.copy()
    eliminadas = []
    targets_a_predecir = set(targets_a_predecir or [])
    
    # 1. Eliminar metadata (excepto 'fecha' que se necesita para split temporal)
    metadata_sin_fecha = [col for col in columnas_dict['metadata'] if col != 'fecha']
    eliminadas.extend(metadata_sin_fecha)
    
    # 2. Eliminar features de vuelo (no disponibles en predicción)
    features_vuelo = [
        col for col in columnas_dict['features_vuelo']
        if col not in targets_a_predecir
    ]
    eliminadas.extend(features_vuelo)    
    # 3. Eliminar targets de regresión que no usaremos como target principal
    # (Mantener solo altura_max_m, duracion_horas, distancia_km)
    targets_reg = set(columnas_dict['targets_regresion'])
    targets_secundarios = list(targets_reg - targets_a_predecir)
    eliminadas.extend(targets_secundarios)
    
    # 4. Eliminar columnas con muchos NaN
    missing = analizar_missing_values(df_clean)
    cols_muchos_nan = missing[missing['pct_missing'] > umbral_nan]['columna'].tolist()
    eliminadas.extend(cols_muchos_nan)
    
    # Eliminar duplicados en la lista
    eliminadas = list(set(eliminadas))
    
    # Filtrar solo columnas que existen
    eliminadas_existentes = [col for col in eliminadas if col in df_clean.columns]
    
    df_clean = df_clean.drop(columns=eliminadas_existentes)
    
    print(f"✓ Columnas eliminadas: {len(eliminadas_existentes)}")
    print(f"✓ Columnas restantes: {len(df_clean.columns)}")
    print(f"✓ 'fecha' mantenida para split temporal")
    
    return df_clean, eliminadas_existentes


def imputar_missing_values(df, estrategia='median'):
    """
    Imputa valores faltantes.
    
    Args:
        df: DataFrame
        estrategia: 'mean', 'median', 'most_frequent'
        
    Returns:
        DataFrame con valores imputados, imputer ajustado
    """
    
    df_imputed = df.copy()
    
    # Separar columnas numéricas y categóricas
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_categoricas = df.select_dtypes(include=['object']).columns.tolist()
    
    # Imputer para numéricas
    if len(cols_numericas) > 0:
        imputer_num = SimpleImputer(strategy=estrategia)
        df_imputed[cols_numericas] = imputer_num.fit_transform(df[cols_numericas])
    else:
        imputer_num = None
    
    # Imputer para categóricas
    if len(cols_categoricas) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_imputed[cols_categoricas] = imputer_cat.fit_transform(df[cols_categoricas])
    else:
        imputer_cat = None
    
    n_imputados = df.isnull().sum().sum()
    print(f"✓ Valores imputados: {n_imputados}")
    
    return df_imputed, {'numerico': imputer_num, 'categorico': imputer_cat}


def separar_features_targets(df):
    """
    Separa features (X) y targets (y).
    
    IMPORTANTE: Elimina TODOS los targets posibles de X, no solo los que se van a predecir.
    Esto evita data leakage.
    
    Args:
        df: DataFrame limpio
        
    Returns:
        X (features), y_reg (targets regresión), y_clf (target clasificación)
    """
    
    # Targets que el usuario quiere predecir
    targets_reg_deseados = [
        'altura_max_m',
        'altura_min_m',
        'ganancia_altura_m',
        'duracion_min',
        'velocidad_promedio_kmh',
        'num_termicas',
        'intensidad_termicas_mean_ms',
        'tiempo_en_termicas_min',
        'tasa_ascenso_mean_ms'
    ]
    
    # TODOS los posibles targets (incluso los que no se van a predecir)
    # DEBEN eliminarse de X para evitar data leakage
    todos_los_targets_posibles = [
        'altura_max_m',
        'altura_min_m',
        'ganancia_altura_m',
        'duracion_horas',           # ← Aunque no lo predecimos, debe eliminarse
        'duracion_min',
        'distancia_km',             # ← Aunque no lo predecimos, debe eliminarse
        'velocidad_promedio_kmh',
        'num_termicas',
        'intensidad_termicas_mean_ms',
        'tiempo_en_termicas_min',
        'tasa_ascenso_mean_ms',
        'tasa_descenso_mean_ms',
        'ground_speed_mean_kmh',
        'variacion_altura_m'
    ]
    
    # Target de clasificación
    target_clf = 'calidad_dia'
    
    # Filtrar targets deseados que existen
    targets_reg_existentes = [t for t in targets_reg_deseados if t in df.columns]
    
    # Filtrar TODOS los targets posibles que existen (para eliminar de X)
    todos_targets_existentes = [t for t in todos_los_targets_posibles if t in df.columns]
    
    # Columnas a excluir de X: TODOS los targets posibles + fecha
    cols_excluir = list(set(todos_targets_existentes))  # Usar set para evitar duplicados
    
    if target_clf in df.columns:
        cols_excluir.append(target_clf)
    
    # CRÍTICO: Eliminar 'fecha' de features (no debe estar en X)
    if 'fecha' in df.columns:
        cols_excluir.append('fecha')
    
    # Features (todo lo demás)
    X = df.drop(columns=cols_excluir)
    
    # Targets (solo los deseados)
    y_reg = df[targets_reg_existentes] if targets_reg_existentes else None
    y_clf = df[target_clf] if target_clf in df.columns else None
    
    print(f"✓ Features (X): {X.shape}")
    if y_reg is not None:
        print(f"✓ Targets regresión (y_reg): {y_reg.shape} - {len(targets_reg_existentes)} targets")
        print(f"  Targets: {targets_reg_existentes}")
    if y_clf is not None:
        print(f"✓ Target clasificación (y_clf): {len(y_clf)} muestras")
    
    # Verificación
    targets_en_X = [col for col in X.columns if any(t in col for t in ['duracion', 'distancia', 'altura_max', 'altura_min'])]
    if len(targets_en_X) > 0:
        print(f"\n⚠ ADVERTENCIA: Posibles targets en X: {targets_en_X[:5]}")
    
    return X, y_reg, y_clf


def split_temporal(df, train_pct=0.70, val_pct=0.15, test_pct=0.15):
    """
    Split temporal del dataset (NO aleatorio).
    Importante para evitar data leakage temporal.
    
    CRÍTICO: Si una fecha aparece en Train, NO puede aparecer en Val/Test.
    
    Args:
        df: DataFrame con columna 'fecha'
        train_pct: % para train
        val_pct: % para validation
        test_pct: % para test
        
    Returns:
        dict con splits: {'train': df_train, 'val': df_val, 'test': df_test}
    """
    
    # Verificar que suman 1.0
    assert abs(train_pct + val_pct + test_pct - 1.0) < 0.01, "Los porcentajes deben sumar 1.0"
    
    # Ordenar por fecha
    df_sorted = df.sort_values('fecha').reset_index(drop=True)
    
    n = len(df_sorted)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)
    
    # Split inicial
    df_train = df_sorted.iloc[:n_train].copy()
    df_val = df_sorted.iloc[n_train:n_train+n_val].copy()
    df_test = df_sorted.iloc[n_train+n_val:].copy()
    
    # CRÍTICO: Eliminar fechas duplicadas
    # Si una fecha está en train, eliminarla de val
    fechas_train = set(df_train['fecha'].dt.date)
    df_val = df_val[~df_val['fecha'].dt.date.isin(fechas_train)].copy()
    
    # Si una fecha está en val (después de limpieza), eliminarla de test
    fechas_val = set(df_val['fecha'].dt.date)
    df_test = df_test[~df_test['fecha'].dt.date.isin(fechas_val)].copy()
    
    # También eliminar fechas de train de test
    df_test = df_test[~df_test['fecha'].dt.date.isin(fechas_train)].copy()
    
    print("="*60)
    print("SPLIT TEMPORAL")
    print("="*60)
    print(f"Train: {len(df_train)} muestras ({len(df_train)/n*100:.1f}%)")
    print(f"  Fechas: {df_train['fecha'].min().date()} a {df_train['fecha'].max().date()}")
    print(f"\nVal:   {len(df_val)} muestras ({len(df_val)/n*100:.1f}%)")
    if len(df_val) > 0:
        print(f"  Fechas: {df_val['fecha'].min().date()} a {df_val['fecha'].max().date()}")
    else:
        print(f"  ⚠ WARNING: Val vacío después de eliminar fechas duplicadas")
    print(f"\nTest:  {len(df_test)} muestras ({len(df_test)/n*100:.1f}%)")
    if len(df_test) > 0:
        print(f"  Fechas: {df_test['fecha'].min().date()} a {df_test['fecha'].max().date()}")
    else:
        print(f"  ⚠ WARNING: Test vacío después de eliminar fechas duplicadas")
    
    # Verificación final
    if len(df_val) > 0 and len(df_test) > 0:
        if df_train['fecha'].max() < df_val['fecha'].min() and df_val['fecha'].max() < df_test['fecha'].min():
            print(f"\n✓ Sin superposición temporal")
        else:
            print(f"\n⚠ ADVERTENCIA: Verificar fechas manualmente")
    
    # Verificar que no hay fechas compartidas
    fechas_train_final = set(df_train['fecha'].dt.date)
    fechas_val_final = set(df_val['fecha'].dt.date)
    fechas_test_final = set(df_test['fecha'].dt.date)
    
    overlap_train_val = fechas_train_final & fechas_val_final
    overlap_val_test = fechas_val_final & fechas_test_final
    overlap_train_test = fechas_train_final & fechas_test_final
    
    if len(overlap_train_val) == 0 and len(overlap_val_test) == 0 and len(overlap_train_test) == 0:
        print(f"✓ Sin fechas compartidas entre splits")
    else:
        if overlap_train_val:
            print(f"✗ Fechas compartidas Train-Val: {overlap_train_val}")
        if overlap_val_test:
            print(f"✗ Fechas compartidas Val-Test: {overlap_val_test}")
        if overlap_train_test:
            print(f"✗ Fechas compartidas Train-Test: {overlap_train_test}")
    
    print("="*60)
    
    return {
        'train': df_train,
        'val': df_val,
        'test': df_test
    }


def escalar_features(X_train, X_val, X_test):
    """
    Escala features numéricas usando StandardScaler.
    IMPORTANTE: Ajusta solo en train, aplica en val y test.
    
    Args:
        X_train, X_val, X_test: DataFrames con features
        
    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    
    # Identificar columnas numéricas
    cols_numericas = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(cols_numericas) == 0:
        print("⚠ No hay columnas numéricas para escalar")
        return X_train, X_val, X_test, None
    
    # Crear scaler
    scaler = StandardScaler()
    
    # Ajustar SOLO en train
    scaler.fit(X_train[cols_numericas])
    
    # Aplicar a todos
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_numericas] = scaler.transform(X_train[cols_numericas])
    X_val_scaled[cols_numericas] = scaler.transform(X_val[cols_numericas])
    X_test_scaled[cols_numericas] = scaler.transform(X_test[cols_numericas])
    
    print(f"✓ Features escaladas: {len(cols_numericas)}")
    print(f"  Media train: {X_train_scaled[cols_numericas].mean().mean():.6f} (debe ser ~0)")
    print(f"  Std train: {X_train_scaled[cols_numericas].std().mean():.6f} (debe ser ~1)")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def guardar_datasets_procesados(splits_dict, output_dir='data/processed'):
    """
    Guarda los datasets procesados.
    
    Args:
        splits_dict: Dict con 'train', 'val', 'test', cada uno con 'X', 'y_reg', 'y_clf'
        output_dir: Directorio de salida
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, data in splits_dict.items():
        # Features
        data['X'].to_csv(f"{output_dir}/X_{split_name}.csv", index=False)
        
        # Targets regresión
        if data['y_reg'] is not None:
            data['y_reg'].to_csv(f"{output_dir}/y_reg_{split_name}.csv", index=False)
        
        # Target clasificación
        if data['y_clf'] is not None:
            data['y_clf'].to_csv(f"{output_dir}/y_clf_{split_name}.csv", index=False)
    
    print(f"\n✓ Datasets guardados en {output_dir}/")
    print(f"  - X_{{train,val,test}}.csv")
    print(f"  - y_reg_{{train,val,test}}.csv")
    print(f"  - y_clf_{{train,val,test}}.csv")


def guardar_split_metadata(df_train, df_val, df_test, output_dir='data/processed'):
    """
    Guarda metadata del split temporal para validación posterior.
    
    Args:
        df_train, df_val, df_test: DataFrames con columna 'fecha'
        output_dir: Directorio de salida
    """
    
    import json
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        'train': {
            'n_samples': len(df_train),
            'fecha_min': df_train['fecha'].min().isoformat(),
            'fecha_max': df_train['fecha'].max().isoformat(),
            'fechas_unicas': sorted([d.isoformat() for d in df_train['fecha'].dt.date.unique()])
        },
        'val': {
            'n_samples': len(df_val),
            'fecha_min': df_val['fecha'].min().isoformat(),
            'fecha_max': df_val['fecha'].max().isoformat(),
            'fechas_unicas': sorted([d.isoformat() for d in df_val['fecha'].dt.date.unique()])
        },
        'test': {
            'n_samples': len(df_test),
            'fecha_min': df_test['fecha'].min().isoformat(),
            'fecha_max': df_test['fecha'].max().isoformat(),
            'fechas_unicas': sorted([d.isoformat() for d in df_test['fecha'].dt.date.unique()])
        },
        'split_timestamp': datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata del split guardada en {output_dir}/split_metadata.json")


def guardar_preprocesadores(imputers, scaler, output_dir='data/processed'):
    """
    Guarda los objetos de preprocesamiento (imputers, scaler).
    
    Args:
        imputers: Dict con imputers
        scaler: StandardScaler
        output_dir: Directorio de salida
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    if imputers['numerico'] is not None:
        joblib.dump(imputers['numerico'], f"{output_dir}/imputer_numerico.pkl")
    
    if imputers['categorico'] is not None:
        joblib.dump(imputers['categorico'], f"{output_dir}/imputer_categorico.pkl")
    
    if scaler is not None:
        joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    
    print(f"✓ Preprocesadores guardados en {output_dir}/")