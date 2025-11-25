"""
utils.py
Funciones de utilidad general para el proyecto
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def load_dataset(filepath: str) -> pd.DataFrame:
    """Carga el dataset desde CSV"""
    df = pd.read_csv(filepath)
    print(f"Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def check_data_consistency(df: pd.DataFrame) -> Dict:
    """
    Realiza chequeos de consistencia del dataset
    
    Returns:
        Dict con información sobre:
        - duplicados
        - valores faltantes
        - tipos de datos
        - valores únicos por columna
        - inconsistencias detectadas
    """
    
    consistency_report = {}
    
    # 1. Duplicados
    duplicates = df.duplicated().sum()
    consistency_report['duplicates'] = {
        'count': duplicates,
        'percentage': (duplicates / len(df)) * 100
    }
    
    # 2. Valores faltantes
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    consistency_report['missing_values'] = pd.DataFrame({
        'count': missing,
        'percentage': missing_pct
    }).sort_values('count', ascending=False)
    
    # 3. Tipos de datos
    consistency_report['data_types'] = df.dtypes.value_counts()
    
    # 4. Valores únicos (para detectar posibles IDs o categóricas con alta cardinalidad)
    unique_counts = df.nunique()
    high_cardinality = unique_counts[unique_counts > len(df) * 0.5]
    consistency_report['high_cardinality_features'] = high_cardinality
    
    # 5. Constantes (columnas con un solo valor)
    constant_features = [col for col in df.columns if df[col].nunique() <= 1]
    consistency_report['constant_features'] = constant_features
    
    # 6. Chequear valores infinitos en numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {col: np.isinf(df[col]).sum() for col in numeric_cols}
    inf_counts = {k: v for k, v in inf_counts.items() if v > 0}
    consistency_report['infinite_values'] = inf_counts
    
    # 7. Chequear rangos sospechosos en targets
    targets = ['altura_max_m', 'duracion_min', 'distancia_km']
    suspicious_ranges = {}
    for target in targets:
        if target in df.columns:
            suspicious = {}
            if (df[target] < 0).any():
                suspicious['negative_values'] = (df[target] < 0).sum()
            if (df[target] == 0).any():
                suspicious['zero_values'] = (df[target] == 0).sum()
            if len(suspicious) > 0:
                suspicious_ranges[target] = suspicious
    consistency_report['suspicious_ranges'] = suspicious_ranges
    
    return consistency_report


def identify_useless_features(df: pd.DataFrame) -> Dict:
    """
    Identifica features que probablemente no aporten valor predictivo
    
    Criterios:
    - IDs (alta cardinalidad, valores únicos ~= n_rows)
    - Nombres, pilotos, etc.
    - Constantes
    - Casi constantes (>95% mismo valor)
    """
    
    useless_features = {
        'identifiers': [],
        'high_cardinality': [],
        'constant': [],
        'almost_constant': [],
        'recommended_to_drop': []
    }
    
    n_rows = len(df)
    
    for col in df.columns:
        n_unique = df[col].nunique()
        
        # IDs: cardinalidad muy alta
        if n_unique > n_rows * 0.95:
            useless_features['identifiers'].append(col)
            useless_features['recommended_to_drop'].append(col)
        
        # Constantes
        elif n_unique == 1:
            useless_features['constant'].append(col)
            useless_features['recommended_to_drop'].append(col)
        
        # Casi constantes
        elif n_unique > 1:
            most_common_pct = df[col].value_counts().iloc[0] / n_rows
            if most_common_pct > 0.95:
                useless_features['almost_constant'].append(col)
                useless_features['recommended_to_drop'].append(col)
        
        # Alta cardinalidad en categóricas (pilot, glider, etc.)
        if df[col].dtype == 'object' and n_unique > 50:
            useless_features['high_cardinality'].append(col)
            # No siempre se dropean, depende del contexto
    
    # Añadir features conocidas que no aportan
    known_useless = ['flight_id', 'filename', 'fecha_dt']
    for col in known_useless:
        if col in df.columns and col not in useless_features['recommended_to_drop']:
            useless_features['recommended_to_drop'].append(col)
    
    return useless_features


def get_feature_ranges(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Obtiene rangos y estadísticas de features numéricas
    """
    
    stats = []
    
    for feature in features:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            stats.append({
                'feature': feature,
                'min': df[feature].min(),
                'max': df[feature].max(),
                'mean': df[feature].mean(),
                'median': df[feature].median(),
                'std': df[feature].std(),
                'range': df[feature].max() - df[feature].min(),
                'missing_pct': (df[feature].isnull().sum() / len(df)) * 100
            })
    
    return pd.DataFrame(stats)


def split_train_val_test(df: pd.DataFrame, 
                         stratify_col: str,
                         train_size: float = 0.7,
                         val_size: float = 0.15,
                         test_size: float = 0.15,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split estratificado en train/val/test
    
    Parameters:
    -----------
    df : DataFrame
    stratify_col : str
        Columna para estratificación
    train_size, val_size, test_size : float
        Proporciones (deben sumar 1.0)
    random_state : int
    
    Returns:
    --------
    train_df, val_df, test_df
    """
    
    from sklearn.model_selection import train_test_split
    
    # Verificar proporciones
    assert abs(train_size + val_size + test_size - 1.0) < 0.001, "Las proporciones deben sumar 1.0"
    
    # Eliminar NaNs en columna de estratificación
    df_clean = df.dropna(subset=[stratify_col])
    
    # Primer split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df_clean,
        test_size=(val_size + test_size),
        stratify=df_clean[stratify_col],
        random_state=random_state
    )
    
    # Segundo split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df[stratify_col],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def check_class_balance(df: pd.DataFrame, target_col: str) -> Dict:
    """
    Analiza el balance de clases para targets de clasificación
    
    Returns:
        Dict con conteos, porcentajes y métrica de desbalance
    """
    
    balance_info = {}
    
    if target_col not in df.columns:
        return {'error': f'Columna {target_col} no encontrada'}
    
    # Conteos
    value_counts = df[target_col].value_counts()
    value_pcts = df[target_col].value_counts(normalize=True) * 100
    
    balance_info['counts'] = value_counts
    balance_info['percentages'] = value_pcts
    
    # Métricas de desbalance
    min_class = value_counts.min()
    max_class = value_counts.max()
    balance_info['imbalance_ratio'] = max_class / min_class
    
    # Clasificación del desbalance
    if balance_info['imbalance_ratio'] < 1.5:
        balance_info['balance_status'] = 'Balanceado'
    elif balance_info['imbalance_ratio'] < 3:
        balance_info['balance_status'] = 'Levemente desbalanceado'
    elif balance_info['imbalance_ratio'] < 5:
        balance_info['balance_status'] = 'Moderadamente desbalanceado'
    else:
        balance_info['balance_status'] = 'Fuertemente desbalanceado'
    
    return balance_info


def print_consistency_report(report: Dict):
    """Imprime reporte de consistencia de forma legible"""
    
    print("=" * 70)
    print("REPORTE DE CONSISTENCIA DE DATOS")
    print("=" * 70)
    
    # Duplicados
    print(f"\n1. DUPLICADOS:")
    print(f"   Total: {report['duplicates']['count']}")
    print(f"   Porcentaje: {report['duplicates']['percentage']:.2f}%")
    
    # Valores faltantes
    print(f"\n2. VALORES FALTANTES:")
    missing_df = report['missing_values']
    missing_with_values = missing_df[missing_df['count'] > 0]
    if len(missing_with_values) > 0:
        print(f"   Columnas con valores faltantes: {len(missing_with_values)}")
        print(f"\n   Top 10:")
        for idx, row in missing_with_values.head(10).iterrows():
            print(f"   - {idx}: {row['count']} ({row['percentage']:.2f}%)")
    else:
        print("   ✓ No hay valores faltantes")
    
    # Tipos de datos
    print(f"\n3. TIPOS DE DATOS:")
    for dtype, count in report['data_types'].items():
        print(f"   - {dtype}: {count} columnas")
    
    # Alta cardinalidad
    print(f"\n4. FEATURES CON ALTA CARDINALIDAD (posibles IDs):")
    if len(report['high_cardinality_features']) > 0:
        for col, n_unique in report['high_cardinality_features'].items():
            print(f"   - {col}: {n_unique} valores únicos")
    else:
        print("   ✓ No detectadas")
    
    # Constantes
    print(f"\n5. FEATURES CONSTANTES:")
    if len(report['constant_features']) > 0:
        print(f"   {report['constant_features']}")
    else:
        print("   ✓ No detectadas")
    
    # Infinitos
    print(f"\n6. VALORES INFINITOS:")
    if len(report['infinite_values']) > 0:
        for col, count in report['infinite_values'].items():
            print(f"   - {col}: {count}")
    else:
        print("   ✓ No detectados")
    
    # Rangos sospechosos
    print(f"\n7. RANGOS SOSPECHOSOS EN TARGETS:")
    if len(report['suspicious_ranges']) > 0:
        for col, issues in report['suspicious_ranges'].items():
            print(f"   - {col}:")
            for issue_type, count in issues.items():
                print(f"     {issue_type}: {count}")
    else:
        print("   ✓ No detectados")


def print_useless_features_report(useless: Dict):
    """Imprime reporte de features inútiles"""
    
    print("\n" + "=" * 70)
    print("IDENTIFICACIÓN DE FEATURES SIN VALOR PREDICTIVO")
    print("=" * 70)
    
    print(f"\n1. IDENTIFICADORES (IDs):")
    if len(useless['identifiers']) > 0:
        for col in useless['identifiers']:
            print(f"   - {col}")
    else:
        print("   ✓ No detectados")
    
    print(f"\n2. ALTA CARDINALIDAD (categóricas con muchos valores):")
    if len(useless['high_cardinality']) > 0:
        for col in useless['high_cardinality']:
            print(f"   - {col}")
    else:
        print("   ✓ No detectadas")
    
    print(f"\n3. CONSTANTES:")
    if len(useless['constant']) > 0:
        for col in useless['constant']:
            print(f"   - {col}")
    else:
        print("   ✓ No detectadas")
    
    print(f"\n4. CASI CONSTANTES (>95% mismo valor):")
    if len(useless['almost_constant']) > 0:
        for col in useless['almost_constant']:
            print(f"   - {col}")
    else:
        print("   ✓ No detectadas")
    
    print(f"\n5. RECOMENDADAS PARA ELIMINAR:")
    if len(useless['recommended_to_drop']) > 0:
        print(f"   Total: {len(useless['recommended_to_drop'])}")
        for col in useless['recommended_to_drop']:
            print(f"   - {col}")
    else:
        print("   ✓ Ninguna")
