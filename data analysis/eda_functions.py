"""
eda_functions.py
================
Funciones para análisis exploratorio de datos (EDA).

Autor: Estanislao
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def configurar_visualizacion():
    """
    Configura el estilo de las visualizaciones.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def resumen_dataset(df):
    """
    Genera un resumen completo del dataset.
    
    Args:
        df: DataFrame
    """
    print("="*80)
    print("RESUMEN DEL DATASET")
    print("="*80)
    print(f"\nDimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"\nRango temporal: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    print(f"Días únicos: {df['fecha'].nunique()}")
    
    print("\n" + "="*80)
    print("TIPOS DE DATOS")
    print("="*80)
    print(df.dtypes.value_counts())
    
    print("\n" + "="*80)
    print("VALORES FALTANTES")
    print("="*80)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'n_missing': missing[missing > 0],
            'pct_missing': missing_pct[missing > 0]
        }).sort_values('pct_missing', ascending=False)
        print(missing_df)
    else:
        print("✓ No hay valores faltantes")


def analizar_targets(df, targets_reg, target_clf=None):
    """
    Analiza distribución de variables target.
    
    Args:
        df: DataFrame
        targets_reg: Lista de targets de regresión
        target_clf: Target de clasificación (opcional)
    """
    
    print("="*80)
    print("ANÁLISIS DE TARGETS")
    print("="*80)
    
    # Regresión
    if targets_reg:
        print("\n▶ TARGETS DE REGRESIÓN:")
        for target in targets_reg:
            if target in df.columns:
                print(f"\n{target}:")
                print(f"  Media: {df[target].mean():.2f}")
                print(f"  Mediana: {df[target].median():.2f}")
                print(f"  Std: {df[target].std():.2f}")
                print(f"  Min: {df[target].min():.2f}")
                print(f"  Max: {df[target].max():.2f}")
    
    # Clasificación
    if target_clf and target_clf in df.columns:
        print(f"\n▶ TARGET DE CLASIFICACIÓN ({target_clf}):")
        print(df[target_clf].value_counts().sort_index())
        print(f"\n Distribución (%):")
        print((df[target_clf].value_counts(normalize=True) * 100).round(2).sort_index())


def plot_distribucion_targets(df, targets_reg, target_clf=None):
    """
    Visualiza distribución de targets.
    
    Args:
        df: DataFrame
        targets_reg: Lista de targets de regresión
        target_clf: Target de clasificación (opcional)
    """
    
    n_plots = len(targets_reg) + (1 if target_clf else 0)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Targets de regresión
    for i, target in enumerate(targets_reg):
        if target in df.columns:
            axes[i].hist(df[target].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Distribución: {target}')
            axes[i].set_xlabel(target)
            axes[i].set_ylabel('Frecuencia')
            axes[i].axvline(df[target].mean(), color='red', linestyle='--', label='Media')
            axes[i].axvline(df[target].median(), color='green', linestyle='--', label='Mediana')
            axes[i].legend()
    
    # Target de clasificación
    if target_clf and target_clf in df.columns:
        counts = df[target_clf].value_counts().sort_index()
        axes[-1].bar(range(len(counts)), counts.values, tick_label=counts.index)
        axes[-1].set_title(f'Distribución: {target_clf}')
        axes[-1].set_xlabel(target_clf)
        axes[-1].set_ylabel('Frecuencia')
        
        # Agregar porcentajes
        for i, v in enumerate(counts.values):
            axes[-1].text(i, v + max(counts.values)*0.02, 
                         f'{v}\n({v/len(df)*100:.1f}%)', 
                         ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def analizar_correlaciones(df, targets, top_n=20):
    """
    Analiza correlaciones entre features y targets.
    
    Args:
        df: DataFrame
        targets: Lista de targets
        top_n: Top N correlaciones a mostrar
    """
    
    print("="*80)
    print("CORRELACIONES CON TARGETS")
    print("="*80)
    
    # Seleccionar solo columnas numéricas
    df_num = df.select_dtypes(include=[np.number])
    
    for target in targets:
        if target in df_num.columns:
            print(f"\n▶ TOP {top_n} CORRELACIONES CON {target}:")
            
            corr = df_num.corr()[target].drop(target).abs().sort_values(ascending=False)
            print(corr.head(top_n))


def plot_correlacion_heatmap(df, targets, top_n=15):
    """
    Visualiza heatmap de correlaciones.
    
    Args:
        df: DataFrame
        targets: Lista de targets
        top_n: Top N features a incluir
    """
    
    df_num = df.select_dtypes(include=[np.number])
    
    for target in targets:
        if target in df_num.columns:
            # Seleccionar top features correlacionadas
            corr_target = df_num.corr()[target].drop(target).abs().sort_values(ascending=False)
            top_features = corr_target.head(top_n).index.tolist()
            
            # Matriz de correlación
            cols_plot = top_features + [target]
            corr_matrix = df_num[cols_plot].corr()
            
            # Plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, vmin=-1, vmax=1, square=True)
            plt.title(f'Correlaciones con {target} (Top {top_n} features)')
            plt.tight_layout()
            plt.show()


def analizar_evolucion_temporal(df, targets):
    """
    Analiza evolución temporal de targets.
    
    Args:
        df: DataFrame con columna 'fecha'
        targets: Lista de targets
    """
    
    print("="*80)
    print("EVOLUCIÓN TEMPORAL")
    print("="*80)
    
    # Agrupar por fecha
    df_temporal = df.groupby('fecha')[targets].mean()
    
    # Estadísticas por mes
    df['mes'] = pd.to_datetime(df['fecha']).dt.month
    print("\nPromedio por mes:")
    print(df.groupby('mes')[targets].mean().round(2))


def plot_evolucion_temporal(df, targets):
    """
    Visualiza evolución temporal de targets.
    
    Args:
        df: DataFrame con columna 'fecha'
        targets: Lista de targets
    """
    
    n_targets = len(targets)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 4*n_targets))
    
    if n_targets == 1:
        axes = [axes]
    
    for i, target in enumerate(targets):
        if target in df.columns:
            # Agrupar por fecha
            df_temporal = df.groupby('fecha')[target].mean()
            
            axes[i].plot(df_temporal.index, df_temporal.values, marker='o', linestyle='-', alpha=0.7)
            axes[i].set_title(f'Evolución Temporal: {target}')
            axes[i].set_xlabel('Fecha')
            axes[i].set_ylabel(target)
            axes[i].grid(True, alpha=0.3)
            
            # Agregar media móvil
            if len(df_temporal) > 7:
                rolling_mean = df_temporal.rolling(window=7).mean()
                axes[i].plot(rolling_mean.index, rolling_mean.values, 
                           color='red', linewidth=2, label='Media móvil (7 días)')
                axes[i].legend()
    
    plt.tight_layout()
    plt.show()


def detectar_outliers(df, columnas, metodo='iqr', umbral=3):
    """
    Detecta outliers en columnas numéricas.
    
    Args:
        df: DataFrame
        columnas: Lista de columnas a analizar
        metodo: 'iqr' o 'zscore'
        umbral: Umbral para zscore (default 3)
        
    Returns:
        DataFrame con información de outliers
    """
    
    outliers_info = []
    
    for col in columnas:
        if col not in df.columns or df[col].dtype not in [np.float64, np.int64]:
            continue
        
        datos = df[col].dropna()
        
        if metodo == 'iqr':
            Q1 = datos.quantile(0.25)
            Q3 = datos.quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            outliers = datos[(datos < limite_inferior) | (datos > limite_superior)]
            
        elif metodo == 'zscore':
            z_scores = np.abs(stats.zscore(datos))
            outliers = datos[z_scores > umbral]
        
        if len(outliers) > 0:
            outliers_info.append({
                'columna': col,
                'n_outliers': len(outliers),
                'pct_outliers': len(outliers) / len(datos) * 100,
                'valores_outliers': outliers.values[:5]  # Primeros 5
            })
    
    return pd.DataFrame(outliers_info)


def plot_boxplots(df, columnas, n_cols=3):
    """
    Visualiza boxplots para detectar outliers.
    
    Args:
        df: DataFrame
        columnas: Lista de columnas
        n_cols: Número de columnas en el grid
    """
    
    n_plots = len(columnas)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columnas):
        if col in df.columns:
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'Boxplot: {col}')
            axes[i].set_ylabel(col)
    
    # Ocultar subplots vacíos
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()