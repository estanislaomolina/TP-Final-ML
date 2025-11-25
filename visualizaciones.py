"""
visualizaciones.py
Funciones para crear visualizaciones del análisis exploratorio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import os


def setup_plotting_style():
    """Configura el estilo de plotting"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_missing_values(missing_df: pd.DataFrame, 
                        save_path: str = None,
                        top_n: int = 20):
    """
    Visualiza valores faltantes
    
    Parameters:
    -----------
    missing_df : DataFrame
        DataFrame con columnas 'count' y 'percentage'
    save_path : str, optional
    top_n : int
        Número de features a mostrar
    """
    
    missing_with_values = missing_df[missing_df['count'] > 0]
    
    if len(missing_with_values) == 0:
        print("No hay valores faltantes para visualizar")
        return
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(missing_with_values[:top_n]) * 0.3)))
    
    data_to_plot = missing_with_values.head(top_n)
    data_to_plot['percentage'].plot(kind='barh', ax=ax, color='coral')
    
    ax.set_xlabel('Porcentaje de valores faltantes (%)')
    ax.set_title(f'Top {min(top_n, len(missing_with_values))} Features con Valores Faltantes')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_target_distributions(df: pd.DataFrame,
                              targets: List[str],
                              save_path: str = None):
    """
    Visualiza distribuciones de targets (histogramas + boxplots)
    
    Parameters:
    -----------
    df : DataFrame
    targets : List[str]
        Lista de columnas target
    save_path : str, optional
    """
    
    n_targets = len([t for t in targets if t in df.columns])
    
    if n_targets == 0:
        print("No hay targets para visualizar")
        return
    
    fig, axes = plt.subplots(2, n_targets, figsize=(5*n_targets, 10))
    
    if n_targets == 1:
        axes = axes.reshape(-1, 1)
    
    for i, target in enumerate(targets):
        if target not in df.columns:
            continue
        
        data = df[target].dropna()
        
        # Histograma
        axes[0, i].hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Media')
        axes[0, i].axvline(data.median(), color='green', linestyle='--', linewidth=2, label='Mediana')
        axes[0, i].set_xlabel(target)
        axes[0, i].set_ylabel('Frecuencia')
        axes[0, i].set_title(f'Distribución: {target}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Boxplot
        axes[1, i].boxplot(data, vert=False, patch_artist=True,
                          boxprops=dict(facecolor='lightblue'))
        axes[1, i].set_xlabel(target)
        axes[1, i].set_title(f'Boxplot: {target}')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_feature_distributions(df: pd.DataFrame,
                               features: List[str],
                               save_path: str = None,
                               n_cols: int = 3):
    """
    Visualiza distribuciones de múltiples features
    
    Parameters:
    -----------
    df : DataFrame
    features : List[str]
    save_path : str, optional
    n_cols : int
        Número de columnas en el grid
    """
    
    features = [f for f in features if f in df.columns]
    
    if len(features) == 0:
        print("No hay features para visualizar")
        return
    
    n_rows = int(np.ceil(len(features) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.ravel() if n_rows * n_cols > 1 else [axes]
    
    for i, feature in enumerate(features):
        raw_series = df[feature]
        clean_series = raw_series.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_series) == 0:
            axes[i].text(0.5, 0.5, 'Sin datos', ha='center', va='center')
            axes[i].set_title(feature)
            continue
        
        axes[i].hist(clean_series, bins=30, edgecolor='black', alpha=0.7)
        data = df[feature].dropna()
        
        if len(data) == 0:
            axes[i].text(0.5, 0.5, 'Sin datos', ha='center', va='center')
            axes[i].set_title(feature)
            continue
        
        axes[i].hist(data, bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frecuencia')
        axes[i].set_title(feature)
        axes[i].grid(True, alpha=0.3)
    
    # Ocultar axes sobrantes
    for i in range(len(features), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(corr_matrix: pd.DataFrame,
                            save_path: str = None,
                            figsize: Tuple[int, int] = (16, 14),
                            annot: bool = False):
    """
    Visualiza matriz de correlación como heatmap
    
    Parameters:
    -----------
    corr_matrix : DataFrame
        Matriz de correlación
    save_path : str, optional
    figsize : Tuple[int, int]
    annot : bool
        Si True, muestra valores en el heatmap
    """
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(corr_matrix, 
                annot=annot,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                fmt='.2f' if annot else '')
    
    plt.title('Matriz de Correlación Completa', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_correlation_with_target(df: pd.DataFrame,
                                 target: str,
                                 top_n: int = 20,
                                 save_path: str = None):
    """
    Visualiza top features correlacionadas con un target específico
    
    Parameters:
    -----------
    df : DataFrame
    target : str
        Columna target
    top_n : int
        Número de features a mostrar
    save_path : str, optional
    """
    
    if target not in df.columns:
        print(f"Target {target} no encontrado")
        return
    
    # Calcular correlaciones
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [col for col in numeric_cols if col != target]
    
    correlations = df[features + [target]].corr()[target].drop(target)
    correlations = correlations.dropna()
    
    # Top correlaciones (por valor absoluto)
    top_corr = correlations.abs().sort_values(ascending=False).head(top_n)
    top_corr_values = correlations[top_corr.index]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))
    
    colors = ['green' if x > 0 else 'red' for x in top_corr_values]
    top_corr_values.plot(kind='barh', ax=ax, color=colors)
    
    ax.set_xlabel('Correlación con ' + target)
    ax.set_title(f'Top {top_n} Features Correlacionadas con {target}')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_scatter_matrix_with_target(df: pd.DataFrame,
                                    target: str,
                                    top_features: List[str],
                                    save_path: str = None):
    """
    Crea scatter plots de top features vs target
    
    Parameters:
    -----------
    df : DataFrame
    target : str
    top_features : List[str]
        Features a plotear
    save_path : str, optional
    """
    
    top_features = [f for f in top_features if f in df.columns]
    
    if len(top_features) == 0 or target not in df.columns:
        print("Features o target no encontrados")
        return
    
    n_features = len(top_features)
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.ravel() if n_rows * n_cols > 1 else [axes]
    
    # Calcular correlaciones para títulos
    correlations = df[[target] + top_features].corr()[target]
    
    for i, feature in enumerate(top_features):
        axes[i].scatter(df[feature], df[target], alpha=0.5, s=20)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)
        
        corr_val = correlations[feature]
        axes[i].set_title(f'{feature}\n(r = {corr_val:.3f})')
        
        # Línea de tendencia
        if df[feature].notna().any() and df[target].notna().any():
            z = np.polyfit(df[feature].dropna(), 
                          df[target].dropna(), 1)
            p = np.poly1d(z)
            axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2)
        
        axes[i].grid(True, alpha=0.3)
    
    # Ocultar axes sobrantes
    for i in range(len(top_features), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_boxplots_by_class(df: pd.DataFrame,
                           features: List[str],
                           class_col: str,
                           save_path: str = None,
                           n_cols: int = 3):
    """
    Crea boxplots de features agrupados por clase
    
    Parameters:
    -----------
    df : DataFrame
    features : List[str]
    class_col : str
        Columna de clasificación
    save_path : str, optional
    n_cols : int
    """
    
    features = [f for f in features if f in df.columns]
    
    if len(features) == 0 or class_col not in df.columns:
        print("Features o columna de clase no encontrados")
        return
    
    n_rows = int(np.ceil(len(features) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.ravel() if n_rows * n_cols > 1 else [axes]
    
    for i, feature in enumerate(features):
        df.boxplot(column=feature, by=class_col, ax=axes[i], patch_artist=True)
        axes[i].set_xlabel('Clase')
        axes[i].set_ylabel(feature)
        axes[i].set_title(feature)
        axes[i].get_figure().suptitle('')  # Remover título automático
    
    # Ocultar axes sobrantes
    for i in range(len(features), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Comparación de Features por {class_col}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution(df: pd.DataFrame,
                            class_col: str,
                            save_path: str = None):
    """
    Visualiza distribución de clases (bar plot + pie chart)
    
    Parameters:
    -----------
    df : DataFrame
    class_col : str
    save_path : str, optional
    """
    
    if class_col not in df.columns:
        print(f"Columna {class_col} no encontrada")
        return
    
    value_counts = df[class_col].value_counts().sort_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    value_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_xlabel(class_col)
    ax1.set_ylabel('Frecuencia')
    ax1.set_title(f'Distribución de {class_col}')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for i, v in enumerate(value_counts):
        ax1.text(i, v + max(value_counts)*0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
           startangle=90, colors=sns.color_palette("husl", len(value_counts)))
    ax2.set_title(f'Proporción de {class_col}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_visualization_directory(base_path: str = 'visualizaciones'):
    """Crea directorio para guardar visualizaciones"""
    os.makedirs(base_path, exist_ok=True)
    return base_path
