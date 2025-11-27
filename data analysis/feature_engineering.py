"""
feature_engineering.py
======================
Funciones para crear features adicionales.

Autor: Estanislao
Fecha: 2024
"""

import pandas as pd
import numpy as np
from scipy import stats


def tratar_outliers(df, columnas, metodo='clip', umbral=3):
    """
    Trata outliers en columnas numéricas.
    
    Args:
        df: DataFrame
        columnas: Lista de columnas a tratar
        metodo: 'clip' (reemplazar por límites), 'remove' (eliminar filas), 'winsorize'
        umbral: Para Z-score (default 3)
        
    Returns:
        DataFrame con outliers tratados, dict con info de tratamiento
    """
    
    df_clean = df.copy()
    info = {}


def crear_features_interacciones(df, prefijo='meteo_'):
    """
    Crea features de interacción entre variables meteorológicas.
    
    Args:
        df: DataFrame
        prefijo: Prefijo de columnas a considerar
        
    Returns:
        DataFrame con nuevas features
    """
    
    df_new = df.copy()
    features_creadas = []
    
    # Interacciones clave basadas en conocimiento del dominio
    
    # 1. Índice de convección: CAPE * temp_differential
    if 'meteo_dia_cape_max' in df.columns and 'meteo_dia_temp_differential' in df.columns:
        df_new['indice_conveccion'] = df['meteo_dia_cape_max'] * df['meteo_dia_temp_differential']
        features_creadas.append('indice_conveccion')
    
    # 2. Índice térmico: solar_rad * temp
    if 'meteo_dia_solar_rad_max' in df.columns and 'meteo_dia_temp_2m_max' in df.columns:
        df_new['indice_termico'] = df['meteo_dia_solar_rad_max'] * df['meteo_dia_temp_2m_max']
        features_creadas.append('indice_termico')
    
    # 3. Ratio CAPE / boundary layer height
    if 'meteo_dia_cape_max' in df.columns and 'meteo_dia_boundary_layer_height_max' in df.columns:
        df_new['ratio_cape_blh'] = df['meteo_dia_cape_max'] / (df['meteo_dia_boundary_layer_height_max'] + 1)
        features_creadas.append('ratio_cape_blh')
    
    # 4. Estabilidad atmosférica: temp_differential / boundary_layer_height
    if 'meteo_dia_temp_differential' in df.columns and 'meteo_dia_boundary_layer_height_mean' in df.columns:
        df_new['estabilidad_atmosferica'] = df['meteo_dia_temp_differential'] / (df['meteo_dia_boundary_layer_height_mean'] + 1)
        features_creadas.append('estabilidad_atmosferica')
    
    # 5. Potencial solar: solar_rad * (1 - cloud_cover)
    if 'meteo_dia_solar_rad_max' in df.columns and 'meteo_dia_cloud_cover_mean' in df.columns:
        df_new['potencial_solar'] = df['meteo_dia_solar_rad_max'] * (1 - df['meteo_dia_cloud_cover_mean'])
        features_creadas.append('potencial_solar')
    
    print(f"✓ Features de interacción creadas: {len(features_creadas)}")
    for feat in features_creadas:
        print(f"  - {feat}")
    
    return df_new


def crear_features_variacion(df):
    """
    Crea features de variación temporal (entre percentiles).
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame con nuevas features
    """
    
    df_new = df.copy()
    features_creadas = []
    
    # Variables meteorológicas a analizar
    variables = ['temp_2m', 'cape', 'solar_rad', 'wind_speed', 'cloud_cover']
    
    for var in variables:
        col_p0 = f'meteo_{var}_p0'
        col_p100 = f'meteo_{var}_p100'
        col_p50 = f'meteo_{var}_p50'
        
        # Variación total (p100 - p0)
        if col_p0 in df.columns and col_p100 in df.columns:
            df_new[f'variacion_{var}'] = df[col_p100] - df[col_p0]
            features_creadas.append(f'variacion_{var}')
        
        # Tendencia (pendiente aproximada)
        if col_p0 in df.columns and col_p50 in df.columns and col_p100 in df.columns:
            # Pendiente aproximada: (p100 - p0) / tiempo
            df_new[f'tendencia_{var}'] = (df[col_p100] - df[col_p0]) / 2  # Normalizado por tiempo relativo
            features_creadas.append(f'tendencia_{var}')
    
    print(f"✓ Features de variación creadas: {len(features_creadas)}")
    
    return df_new


def crear_features_agregados(df):
    """
    Crea features agregadas de percentiles.
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame con nuevas features
    """
    
    df_new = df.copy()
    features_creadas = []
    
    # Variables meteorológicas
    variables = ['temp_2m', 'cape', 'solar_rad', 'wind_speed']
    
    for var in variables:
        cols_percentiles = [f'meteo_{var}_p{p}' for p in [0, 25, 50, 75, 100]]
        cols_existentes = [c for c in cols_percentiles if c in df.columns]
        
        if len(cols_existentes) >= 3:
            # Media de percentiles
            df_new[f'mean_percentiles_{var}'] = df[cols_existentes].mean(axis=1)
            features_creadas.append(f'mean_percentiles_{var}')
            
            # Std de percentiles (variabilidad)
            df_new[f'std_percentiles_{var}'] = df[cols_existentes].std(axis=1)
            features_creadas.append(f'std_percentiles_{var}')
    
    print(f"✓ Features agregadas creadas: {len(features_creadas)}")
    
    return df_new


def crear_features_temporales_ciclicas(df):
    """
    Crea features cíclicas para mes y día del año.
    Útil para capturar estacionalidad.
    
    Args:
        df: DataFrame con columnas 'mes' y 'dia_año'
        
    Returns:
        DataFrame con features cíclicas
    """
    
    df_new = df.copy()
    features_creadas = []
    
    # Codificación cíclica del mes (sin, cos)
    if 'mes' in df.columns:
        df_new['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df_new['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        features_creadas.extend(['mes_sin', 'mes_cos'])
    
    # Codificación cíclica del día del año
    if 'dia_año' in df.columns:
        df_new['dia_año_sin'] = np.sin(2 * np.pi * df['dia_año'] / 365)
        df_new['dia_año_cos'] = np.cos(2 * np.pi * df['dia_año'] / 365)
        features_creadas.extend(['dia_año_sin', 'dia_año_cos'])
    
    # Codificación cíclica del día de la semana
    if 'dia_semana' in df.columns:
        df_new['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df_new['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        features_creadas.extend(['dia_semana_sin', 'dia_semana_cos'])
    
    print(f"✓ Features temporales cíclicas creadas: {len(features_creadas)}")
    
    return df_new


def aplicar_feature_engineering(df, incluir_interacciones=True, 
                                incluir_variacion=True, 
                                incluir_agregados=True,
                                incluir_ciclicas=True):
    """
    Aplica todos los pasos de feature engineering.
    
    Args:
        df: DataFrame
        incluir_*: Flags para incluir/excluir tipos de features
        
    Returns:
        DataFrame con nuevas features
    """
    
    print("="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df_engineered = df.copy()
    
    if incluir_interacciones:
        print("\n▶ Creando features de interacción...")
        df_engineered = crear_features_interacciones(df_engineered)
    
    if incluir_variacion:
        print("\n▶ Creando features de variación...")
        df_engineered = crear_features_variacion(df_engineered)
    
    if incluir_agregados:
        print("\n▶ Creando features agregadas...")
        df_engineered = crear_features_agregados(df_engineered)
    
    if incluir_ciclicas:
        print("\n▶ Creando features temporales cíclicas...")
        df_engineered = crear_features_temporales_ciclicas(df_engineered)
    
    n_nuevas = len(df_engineered.columns) - len(df.columns)
    print(f"\n{'='*80}")
    print(f"✓ Total de features nuevas: {n_nuevas}")
    print(f"✓ Total de features: {len(df.columns)} → {len(df_engineered.columns)}")
    print(f"{'='*80}")
    
    return df_engineered


def seleccionar_features_por_correlacion(X, y, umbral=0.05, top_n=50):
    """
    Selecciona features basándose en correlación con target.
    
    Args:
        X: DataFrame de features
        y: Serie o DataFrame de target(s)
        umbral: Correlación mínima absoluta
        top_n: Máximo número de features a seleccionar
        
    Returns:
        Lista de features seleccionadas
    """
    
    # Si y es DataFrame (múltiples targets), usar el primero
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    # Combinar para calcular correlaciones
    df_temp = pd.concat([X, y], axis=1)
    
    # Correlaciones con el target
    correlaciones = df_temp.corr()[y.name].drop(y.name).abs().sort_values(ascending=False)
    
    # Filtrar por umbral
    features_correlacionadas = correlaciones[correlaciones >= umbral].head(top_n).index.tolist()
    
    print(f"✓ Features seleccionadas por correlación (>{umbral}): {len(features_correlacionadas)}")
    print(f"  Top 10:")
    for feat in features_correlacionadas[:10]:
        print(f"    {feat}: {correlaciones[feat]:.3f}")
    
    return features_correlacionadas