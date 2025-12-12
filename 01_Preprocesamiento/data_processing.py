"""
Funciones para limpieza, preprocesamiento y split de datos.

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
    
    if len(missing) > 0:
        print("\nColumnas con valores faltantes:")
        (missing.head(20))
    else:
        print("No hay valores faltantes")
    return missing


def add_avg(df, column_prefix:list):
    """
    Agrega columnas de promedio para las columnas que comienzan con los prefijos dados.
    
    Args:
        df: DataFrame
        column_prefix: Lista de prefijos de columnas para calcular el promedio
        
    Returns:
        DataFrame con nuevas columnas de promedio agregadas
    """
    added_cols = []
    df_copy = df.copy()
    for prefix in column_prefix:
        cols = [col for col in df_copy.columns if col.startswith(prefix)]
        if cols:
            df_copy[f'{prefix}_avg'] = df_copy[cols].mean(axis=1)
            added_cols.append(f'{prefix}_avg')
    return df_copy, added_cols


def analizar_correlaciones(df, targets, top_n=5, corr_features=None):
    """
    Analiza y muestra las correlaciones de los targets con las features dadas.
    En caso de no especificar features, usa todas las columnas numéricas.
    
    Args:
        df: DataFrame
        targets: Lista de targets
        top_n: Número de top correlaciones a mostrar
        corr_features: Lista de features para calcular correlaciones
    """
    if corr_features is None:
        corr_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for target in targets:
        if target in df.columns and target not in corr_features:
            corr_target = df[corr_features + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
            print(f"\n▶ Correlaciones con {target} (Top {top_n}):")
            print(corr_target.head(top_n).to_string())


def split_data(df, test_size=0.2, random_state=42):
    """
    Divide el dataset en conjuntos de desarrollo y prueba.
    
    Args:
        df: DataFrame
        test_size: Proporción del conjunto de prueba
        random_state: Semilla para reproducibilidad
        
    Returns:
        dev: DataFrame de desarrollo
        test: DataFrame de prueba
    """
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_index = int(len(df) * (1 - test_size))
    dev = df_shuffled.iloc[:split_index].reset_index(drop=True)
    test = df_shuffled.iloc[split_index:].reset_index(drop=True)
    print(f"Datos divididos: Dev={dev.shape}, Test={test.shape}")
    return dev, test

def clip_outliers(df, target, lower_percentile=0.01, upper_percentile=0.99):
    """
    Recorta los valores atípicos en una columna específica del DataFrame.
    
    Args:
        df: DataFrame
        column: Nombre de la columna a recortar
        lower_percentile: Percentil inferior para el recorte
        upper_percentile: Percentil superior para el recorte
    """
    if target in df.columns:
        lower_bound = df[target].quantile(lower_percentile)
        upper_bound = df[target].quantile(upper_percentile)
        df[target] = df[target].clip(lower=lower_bound, upper=upper_bound)
        print(f"Valores atípicos recortados en '{target}' entre percentiles {lower_percentile} y {upper_percentile}.")
    return df