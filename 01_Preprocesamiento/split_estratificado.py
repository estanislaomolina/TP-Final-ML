"""
Split estratificado para targets de regresión
Garantiza distribuciones similares en Dev y Test
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_estratificado_regresion(df, targets_reg, test_size=0.2, random_state=42):
    """
    Split estratificado basado en múltiples targets de regresión.
    
    Crea bins (cuartiles) para cada target y estratifica para garantizar
    que Dev y Test tengan distribuciones similares.
    
    Parameters:
    -----------
    df : DataFrame
        Dataset completo
    targets_reg : list
        Lista de targets de regresión para estratificar
    test_size : float
        Proporción de test (0.2 = 20%)
    random_state : int
        Semilla aleatoria
    
    Returns:
    --------
    dev, test : DataFrames
    """
    # Crear variable de estratificación combinada
    # Usamos cuartiles de cada target para crear bins
    
    strat_cols = []
    
    for target in targets_reg:
        # Crear 4 bins (cuartiles) por target
        bins = pd.qcut(df[target], q=4, labels=False, duplicates='drop')
        strat_cols.append(bins)
    
    # Combinar todos los targets en un string único
    # Cada combinación de cuartiles será un estrato
    df['_strat_key'] = strat_cols[0].astype(str)
    for i in range(1, len(strat_cols)):
        df['_strat_key'] += '_' + strat_cols[i].astype(str)
    
    # Split estratificado
    try:
        dev, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['_strat_key']
        )
    except ValueError:
        # Si hay estratos con muy pocas muestras, reducir número de bins
        print("⚠️ Demasiados estratos, reduciendo a 3 bins por target...")
        
        df['_strat_key'] = pd.qcut(df[targets_reg[0]], q=3, labels=False, duplicates='drop').astype(str)
        for target in targets_reg[1:]:
            bins = pd.qcut(df[target], q=3, labels=False, duplicates='drop')
            df['_strat_key'] += '_' + bins.astype(str)
        
        dev, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['_strat_key']
        )
    
    # Eliminar columna temporal
    dev = dev.drop(columns=['_strat_key'])
    test = test.drop(columns=['_strat_key'])
    
    return dev, test


def split_estratificado_simple(df, target_principal, test_size=0.2, random_state=42):
    """
    Split estratificado basado en UN solo target (más simple).
    
    Útil cuando hay un target principal o cuando el estratificado múltiple
    crea demasiadas combinaciones.
    
    Parameters:
    -----------
    df : DataFrame
    target_principal : str
        Target principal para estratificar
    test_size : float
    random_state : int
    
    Returns:
    --------
    dev, test : DataFrames
    """
    # Crear bins del target principal
    bins = pd.qcut(df[target_principal], q=4, labels=False, duplicates='drop')
    
    # Split
    dev, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=bins
    )
    
    return dev, test


def comparar_distribuciones(dev, test, targets):
    """
    Compara distribuciones de targets entre Dev y Test.
    
    Parameters:
    -----------
    dev, test : DataFrames
    targets : list de columnas a comparar
    """
    print("="*70)
    print("COMPARACIÓN DE DISTRIBUCIONES: Dev vs Test")
    print("="*70)
    
    for target in targets:
        media_dev = dev[target].mean()
        media_test = test[target].mean()
        std_dev = dev[target].std()
        std_test = test[target].std()
        
        diff_media = abs(media_dev - media_test)
        diff_std = abs(std_dev - std_test)
        
        # Calcular diferencia relativa
        diff_media_rel = diff_media / media_dev if media_dev != 0 else 0
        
        print(f"\n{target}:")
        print(f"  Media - Dev: {media_dev:.2f}, Test: {media_test:.2f} "
              f"(diff: {diff_media:.2f}, {diff_media_rel*100:.1f}%)")
        print(f"  Std   - Dev: {std_dev:.2f}, Test: {std_test:.2f} "
              f"(diff: {diff_std:.2f})")
        
        if diff_media_rel > 0.1:
            print(f"  ⚠️ Diferencia > 10% en media")
        else:
            print(f"  ✓ Distribuciones similares")


if __name__ == "__main__":
    # Ejemplo de uso
    print("Funciones de split estratificado listas para usar en preprocesamiento")
