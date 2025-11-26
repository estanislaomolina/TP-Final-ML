"""
preprocessing.py
Funciones de preprocesamiento de datos con prevención de data leakage
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class PreprocessingPipeline:
    """
    Pipeline de preprocesamiento con funciones modulares
    Evita data leakage ajustando solo con datos de entrenamiento
    """
    
    def __init__(self):
        self.imputers = {}
        self.scalers = {}
        self.encoders = {}
        self.outlier_bounds = {}
        self.scaler_fill_values = {}
    
    def handle_missing_values(self, train_df, val_df, threshold=0.5):
        """
        Manejo de valores faltantes
        
        Parameters:
        -----------
        train_df : DataFrame
            Datos de entrenamiento
        val_df : DataFrame
            Datos de validación
        threshold : float
            Umbral para eliminar columnas (% de valores faltantes)
        
        Returns:
        --------
        train_clean, val_clean, info_dict
        """
        
        info = {
            'dropped_columns': [],
            'imputed_count': 0,
            'strategy': {}
        }
        
        # 1. Eliminar columnas con muchos faltantes (solo si > threshold)
        missing_pct = train_df.isnull().sum() / len(train_df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if len(cols_to_drop) > 0:
            train_df = train_df.drop(columns=cols_to_drop)
            val_df = val_df.drop(columns=cols_to_drop)
            info['dropped_columns'] = cols_to_drop
        
        # 2. Imputar valores faltantes
        # Numéricas: mediana (robusto a outliers)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols_with_na = [col for col in numeric_cols if train_df[col].isnull().any()]
        
        if len(numeric_cols_with_na) > 0:
            imputer_numeric = SimpleImputer(strategy='median')
            train_df[numeric_cols_with_na] = imputer_numeric.fit_transform(
                train_df[numeric_cols_with_na]
            )
            val_df[numeric_cols_with_na] = imputer_numeric.transform(
                val_df[numeric_cols_with_na]
            )
            
            self.imputers['numeric'] = imputer_numeric
            info['strategy']['numeric'] = 'median'
            info['imputed_count'] += len(numeric_cols_with_na)
        
        # Categóricas: moda
        categorical_cols = train_df.select_dtypes(include=['object']).columns
        categorical_cols_with_na = [col for col in categorical_cols if train_df[col].isnull().any()]
        
        if len(categorical_cols_with_na) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            train_df[categorical_cols_with_na] = imputer_cat.fit_transform(
                train_df[categorical_cols_with_na]
            )
            val_df[categorical_cols_with_na] = imputer_cat.transform(
                val_df[categorical_cols_with_na]
            )
            
            self.imputers['categorical'] = imputer_cat
            info['strategy']['categorical'] = 'most_frequent'
            info['imputed_count'] += len(categorical_cols_with_na)
        
        return train_df, val_df, info
    
    def handle_outliers(self, train_df, val_df, method='winsorize', 
                       lower_percentile=0.01, upper_percentile=0.99):
        """
        Manejo de outliers
        
        Parameters:
        -----------
        train_df, val_df : DataFrame
        method : str
            'winsorize' (capping) o 'remove'
        lower_percentile, upper_percentile : float
            Percentiles para winsorización
        
        Returns:
        --------
        train_processed, val_processed, info_dict
        """
        
        info = {
            'method': method,
            'n_features': 0,
            'n_modified_train': 0,
            'n_modified_val': 0
        }
        
        # Solo procesar numéricas
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        
        if method == 'winsorize':
            # Calcular bounds en train
            for col in numeric_cols:
                lower_bound = train_df[col].quantile(lower_percentile)
                upper_bound = train_df[col].quantile(upper_percentile)
                
                self.outlier_bounds[col] = (lower_bound, upper_bound)
                
                # Aplicar capping
                train_before = (train_df[col] < lower_bound) | (train_df[col] > upper_bound)
                train_df[col] = train_df[col].clip(lower_bound, upper_bound)
                
                val_before = (val_df[col] < lower_bound) | (val_df[col] > upper_bound)
                val_df[col] = val_df[col].clip(lower_bound, upper_bound)
                
                info['n_modified_train'] += train_before.sum()
                info['n_modified_val'] += val_before.sum()
            
            info['n_features'] = len(numeric_cols)
        
        return train_df, val_df, info
    
    def encode_categorical(self, train_df, val_df, categorical_features, method='onehot'):
        """
        Codificación de variables categóricas
        
        Parameters:
        -----------
        train_df, val_df : DataFrame
        categorical_features : list
            Lista de columnas categóricas
        method : str
            'onehot', 'label', o 'target'
        
        Returns:
        --------
        train_encoded, val_encoded, info_dict
        """
        
        info = {
            'method': method,
            'n_new_columns': 0
        }
        
        if method == 'onehot':
            # One-Hot Encoding
            train_encoded = pd.get_dummies(train_df, columns=categorical_features, 
                                          drop_first=True)
            
            # Asegurar que val tenga las mismas columnas
            val_encoded = pd.get_dummies(val_df, columns=categorical_features,
                                        drop_first=True)
            
            # Alinear columnas
            missing_cols = set(train_encoded.columns) - set(val_encoded.columns)
            for col in missing_cols:
                val_encoded[col] = 0
            
            val_encoded = val_encoded[train_encoded.columns]
            
            info['n_new_columns'] = len(train_encoded.columns) - len(train_df.columns)
            
            return train_encoded, val_encoded, info
        
        else:
            return train_df, val_df, info
    
    def scale_features(self, train_df, val_df, features_to_scale, method='robust'):
        """
        Normalización/Estandarización de features
        
        Parameters:
        -----------
        train_df, val_df : DataFrame
        features_to_scale : list
            Lista de columnas a escalar
        method : str
            'standard', 'robust', o 'minmax'
        
        Returns:
        --------
        train_scaled, val_scaled, scaler
        """
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Método '{method}' no reconocido")
        
        train_numeric = train_df[features_to_scale].apply(pd.to_numeric, errors='coerce')
        val_numeric = val_df[features_to_scale].apply(pd.to_numeric, errors='coerce')

        train_numeric = train_numeric.replace([np.inf, -np.inf], np.nan)
        val_numeric = val_numeric.replace([np.inf, -np.inf], np.nan)

        fill_values = train_numeric.median()
        train_numeric = train_numeric.fillna(fill_values)
        val_numeric = val_numeric.fillna(fill_values)

        train_df[features_to_scale] = scaler.fit_transform(train_numeric)
        val_df[features_to_scale] = scaler.transform(val_numeric)
        
        self.scalers[method] = scaler
        self.scaler_fill_values[method] = fill_values
        
        return train_df, val_df, scaler


def detect_outliers_zscore(data, column, threshold=3):
    """
    Detecta outliers usando Z-score
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    outliers = data[column][z_scores > threshold]
    return outliers


def detect_outliers_iqr(data, column, factor=1.5):
    """
    Detecta outliers usando método IQR
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    return outliers, lower_bound, upper_bound
