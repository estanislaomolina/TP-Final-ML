"""
analisis_exploratorio.py
Funciones para análisis exploratorio de datos
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Tuple, Dict


def get_basic_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Obtiene estadísticas descriptivas básicas
    
    Returns:
        (numeric_stats, categorical_stats)
    """
    
    # Categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        categorical_stats = pd.DataFrame({
            'feature': categorical_cols,
            'n_unique': [df[col].nunique() for col in categorical_cols],
            'most_common': [df[col].mode()[0] if len(df[col].mode()) > 0 else None 
                            for col in categorical_cols],
            'most_common_freq': [df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
                                for col in categorical_cols]
        })
    else:
        categorical_stats = pd.DataFrame()
    
    return categorical_stats


def analyze_target_distributions(df: pd.DataFrame, targets: List[str]) -> Dict:
    """
    Analiza la distribución de variables target
    
    Returns:
        Dict con estadísticas y tests de normalidad
    """
    target_analysis = {}
    
    for target in targets:
        if target not in df.columns:
            continue
        
        data = df[target].dropna()
        analysis = {
            'count': len(data),
            'missing': df[target].isnull().sum(),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q1': data.quantile(0.25),
            'q3': data.quantile(0.75)
        }
        target_analysis[target] = analysis
        
    return target_analysis


def compute_correlation_matrix(df: pd.DataFrame, features: List[str] = None, method: str = 'pearson') -> pd.DataFrame:
    """
    Calcula matriz de correlación
    
    Parameters:
    -----------
    df : DataFrame
    features : List[str], optional
        Features específicas. Si None, usa todas las numéricas
    method : str
        'pearson', 'spearman', o 'kendall'
    
    Returns:
    --------
    Matriz de correlación
    """
    
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filtrar features que existen
    features = [f for f in features if f in df.columns]
    
    corr_matrix = df[features].corr(method=method)
    
    return corr_matrix


def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Encuentra pares de features con alta correlación
    
    Parameters:
    -----------
    corr_matrix : DataFrame
        Matriz de correlación
    threshold : float
        Umbral de correlación (en valor absoluto)
    
    Returns:
    --------
    DataFrame con pares altamente correlacionados
    """
    
    high_corr_pairs = []
    
    # Recorrer matriz triangular superior
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if len(high_corr_pairs) > 0:
        return pd.DataFrame(high_corr_pairs).sort_values(
            'correlation', 
            key=lambda x: abs(x), 
            ascending=False
        )
    else:
        return pd.DataFrame()


def analyze_feature_importance_by_correlation(df: pd.DataFrame, targets: List[str], top_n: int = 20) -> Dict:
    """
    Analiza importancia de features basándose en correlación con targets
    
    Returns:
        Dict con top features correlacionadas con cada target
    """
    
    feature_importance = {}
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # Excluir los targets de las features
    features = [f for f in numeric_features if f not in targets]
    
    for target in targets:
        if target not in df.columns:
            continue
        
        # Calcular correlaciones
        correlations = df[features + [target]].corr()[target].drop(target)
        
        # Ordenar por valor absoluto
        correlations_abs = correlations.abs().sort_values(ascending=False)
        
        feature_importance[target] = {
            'top_positive': correlations.sort_values(ascending=False).head(top_n),
            'top_negative': correlations.sort_values(ascending=True).head(top_n),
            'top_absolute': correlations_abs.head(top_n)
        }
    
    return feature_importance


def detect_outliers_multiple_methods(df: pd.DataFrame, feature: str) -> Dict:
    """
    Detecta outliers usando múltiples métodos
    
    Returns:
        Dict con resultados de diferentes métodos
    """
    
    data = df[feature].dropna()
    
    outliers_info = {}
    
    # 1. Método IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_iqr = Q1 - 1.5 * IQR
    upper_iqr = Q3 + 1.5 * IQR
    outliers_iqr = data[(data < lower_iqr) | (data > upper_iqr)]
    
    outliers_info['iqr'] = {
        'count': len(outliers_iqr),
        'percentage': (len(outliers_iqr) / len(data)) * 100,
        'lower_bound': lower_iqr,
        'upper_bound': upper_iqr
    }
    
    # 2. Método Z-score
    z_scores = np.abs(stats.zscore(data))
    outliers_zscore = data[z_scores > 3]
    
    outliers_info['zscore'] = {
        'count': len(outliers_zscore),
        'percentage': (len(outliers_zscore) / len(data)) * 100,
        'threshold': 3
    }
    
    # 3. Percentiles (más conservador)
    p1 = data.quantile(0.01)
    p99 = data.quantile(0.99)
    outliers_percentile = data[(data < p1) | (data > p99)]
    
    outliers_info['percentile'] = {
        'count': len(outliers_percentile),
        'percentage': (len(outliers_percentile) / len(data)) * 100,
        'p1': p1,
        'p99': p99
    }
    
    return outliers_info


def analyze_categorical_distribution(df: pd.DataFrame, feature: str, top_n: int = 10) -> Dict:
    """
    Analiza distribución de una feature categórica
    
    Returns:
        Dict con conteos, porcentajes y estadísticas
    """
    
    if feature not in df.columns:
        return {'error': f'Feature {feature} no encontrada'}
    
    value_counts = df[feature].value_counts()
    value_pcts = df[feature].value_counts(normalize=True) * 100
    
    analysis = {
        'n_unique': df[feature].nunique(),
        'n_missing': df[feature].isnull().sum(),
        'top_values': value_counts.head(top_n),
        'top_percentages': value_pcts.head(top_n),
        'entropy': stats.entropy(value_counts)  # Medida de dispersión
    }
    
    return analysis


def compare_distributions_by_class(df: pd.DataFrame, feature: str, class_col: str) -> Dict:
    """
    Compara distribución de una feature entre diferentes clases
    
    Returns:
        Dict con estadísticas por clase
    """
    
    if feature not in df.columns or class_col not in df.columns:
        return {'error': 'Features no encontradas'}
    
    comparison = {}
    
    for class_value in df[class_col].unique():
        if pd.notna(class_value):
            class_data = df[df[class_col] == class_value][feature].dropna()
            
            comparison[class_value] = {
                'count': len(class_data),
                'mean': class_data.mean() if len(class_data) > 0 else None,
                'median': class_data.median() if len(class_data) > 0 else None,
                'std': class_data.std() if len(class_data) > 0 else None,
                'min': class_data.min() if len(class_data) > 0 else None,
                'max': class_data.max() if len(class_data) > 0 else None
            }
    
    # Test estadístico (ANOVA si >2 clases, t-test si 2 clases)
    classes = [df[df[class_col] == c][feature].dropna() for c in df[class_col].unique() if pd.notna(c)]
    
    if len(classes) > 2:
        f_stat, p_value = stats.f_oneway(*classes)
        comparison['statistical_test'] = {
            'test': 'ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    elif len(classes) == 2:
        t_stat, p_value = stats.ttest_ind(*classes)
        comparison['statistical_test'] = {
            'test': 't-test',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return comparison


def get_feature_value_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene rangos de valores para todas las features numéricas
    
    Returns:
        DataFrame con min, max, range, mean, median por feature
    """
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    ranges = []
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 0:
            ranges.append({
                'feature': col,
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min(),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'cv': (data.std() / data.mean() * 100) if data.mean() != 0 else np.inf
            })
    
    return pd.DataFrame(ranges).sort_values('range', ascending=False)


def print_target_analysis(target_analysis: Dict):
    """Imprime análisis de targets de forma legible"""
    
    print("=" * 70)
    print("ANÁLISIS DE VARIABLES TARGET")
    print("=" * 70)
    
    for target, analysis in target_analysis.items():
        print(f"\n{target}:")
        print(f"  Observaciones: {analysis['count']}")
        print(f"  Faltantes: {analysis['missing']}")
        print(f"  Media: {analysis['mean']:.2f}")
        print(f"  Mediana: {analysis['median']:.2f}")
        print(f"  Std: {analysis['std']:.2f}")
        print(f"  Rango: [{analysis['min']:.2f}, {analysis['max']:.2f}]")
