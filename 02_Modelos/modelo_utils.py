"""
Funciones para entrenamiento y evaluación de modelos
Predicción de vuelo en planeador
"""

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans


def preparar_datos(dev, test, targets_reg, modo='simple'):
    """
    Separa features y targets, eliminando columnas no predictivas.
    
    Parameters:
    -----------
    dev : DataFrame
        Datos de desarrollo
    test : DataFrame
        Datos de test
    targets_reg : list
        Lista de targets de regresión
    modo : str
        'simple' = solo features promedio (recomendado para baseline)
        'completo' = todas las features horarias
    
    Returns:
    --------
    X_dev, y_dev, X_test, y_test
    """
    dev_proc = dev.copy()
    test_proc = test.copy()
    
    # Si modo simple, calcular promedios primero
    if modo == 'simple':
        cols = [col for col in dev_proc.columns if col.endswith('_avg')]
        dev_proc = dev_proc[targets_reg + ["hora_despegue_ajustada", "club_2", "club_1", "club_0"] + cols]
        test_proc = test_proc[targets_reg + ["hora_despegue_ajustada", "club_2", "club_1", "club_0"] + cols]
    
    # Separar features y targets
    y_dev = dev_proc[targets_reg]
    X_dev = dev_proc.drop(columns=targets_reg)
    X_dev.drop(columns=["fecha"], inplace=True, errors='ignore')
    y_test = test_proc[targets_reg]
    X_test = test_proc.drop(columns=targets_reg)
    X_test.drop(columns=["fecha"], inplace=True, errors='ignore')
    
    return X_dev, y_dev, X_test, y_test


def coordenadas_a_one_hot(dev, test):
    """
        Genera variables one-hot para latitud y longitud de despegue utilizando KMeans.
        Ya sabemos que hay 4 zonas principales de despegue. En base a eso, se crean 
        las columnas one-hot a los DataFrames dev y test, y elimina las columnas originales.
        
        Parameters
        ----------
        dev : pd.DataFrame
            DataFrame de desarrollo
        test : pd.DataFrame
            DataFrame de test
        
        Returns
        -------
        dev_one_hot : pd.DataFrame
            Dev con columnas one-hot agregadas
        test_one_hot : pd.DataFrame
            Test con columnas one-hot agregadas
    """
    #copia para no modificar originales
    dev = dev.copy()
    test = test.copy()
    
    if 'lat_despegue' in dev.columns and 'lon_despegue' in dev.columns:
        kmeans = KMeans(n_clusters=4, random_state=42)
        coords_dev = dev[['lat_despegue', 'lon_despegue']]
        kmeans.fit(coords_dev)

        dev_clusters = kmeans.predict(coords_dev)
        test_clusters = kmeans.predict(test[['lat_despegue', 'lon_despegue']])

        # Crear columnas one-hot SOLO para los primeros 3 clusters
        # Las agrega al principio del DataFrame
        for i in range(3):
            dev.insert(0, f'club_{i}', (dev_clusters == i).astype(int))
            test.insert(0, f'club_{i}', (test_clusters == i).astype(int))

        # Eliminar columnas originales
        dev = dev.drop(columns=['lat_despegue', 'lon_despegue'])
        test = test.drop(columns=['lat_despegue', 'lon_despegue'])

    return dev, test


def normalizar_columnas(dev, test, columns):
    """
    Normaliza con RobustScaler solo las columnas indicadas en `columns`,
    usando parámetros ajustados sobre `dev`, y aplica la misma
    transformación a `test`.

    Parameters
    ----------
    dev : pd.DataFrame
        DataFrame de desarrollo (fit del scaler).
    test : pd.DataFrame
        DataFrame de test (solo transform).
    columns : list[str]
        Lista de nombres de columnas a normalizar (numéricas).

    Returns
    -------
    dev_norm : pd.DataFrame
        Dev con las columnas en `columns` escaladas.
    test_norm : pd.DataFrame
        Test con las columnas en `columns` escaladas.
    scaler : RobustScaler
        Scaler ajustado, por si lo querés usar luego en nuevos datos.
    """
    scaler = RobustScaler()
    dev_norm = dev.copy()
    test_norm = test.copy()
    dev_norm[columns] = scaler.fit_transform(dev[columns])
    test_norm[columns] = scaler.transform(test[columns])

    return dev_norm, test_norm, scaler


def ajustar_hora_despegue(dev, test):
    """
    Ajusta la columna 'hora_despegue' para que esté en el rango [0, 1].
    Ademas le cambia el nombre a 'hora_despegue_ajustada' para evitar confusiones.
    Agrega la columna ajustada al principio del DataFrame y elimina la original.
    Parameters:
    -----------
    dev : DataFrame
        Datos de desarrollo
    test : DataFrame
        Datos de test
    
    Returns:
    --------
    dev_ajustado, test_ajustado : DataFrames con 'hora_despegue' ajustada
    """
    dev_ajustado = dev.copy()
    test_ajustado = test.copy()
    
    if 'hora_despegue' in dev.columns:
        dt_dev = pd.to_datetime(dev_ajustado['hora_despegue'])
        hora_decimal_dev = dt_dev.dt.hour + dt_dev.dt.minute / 60.0 + dt_dev.dt.second / 3600.0
        dev_ajustado.insert(0, 'hora_despegue_ajustada', hora_decimal_dev / 24.0)
        dev_ajustado = dev_ajustado.drop(columns=['hora_despegue'])
    if 'hora_despegue' in test.columns:
        dt_test = pd.to_datetime(test_ajustado['hora_despegue'])
        hora_decimal_test = dt_test.dt.hour + dt_test.dt.minute / 60.0 + dt_test.dt.second / 3600.0
        test_ajustado.insert(0, 'hora_despegue_ajustada', hora_decimal_test / 24.0)
        test_ajustado = test_ajustado.drop(columns=['hora_despegue'])
    
    return dev_ajustado, test_ajustado


def evaluar_modelo(y_true, y_pred, nombre_target):
    """
    Calcula métricas de regresión.
    
    Returns:
    --------
    dict con R², MAE, RMSE
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'target': nombre_target,
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse
    }


def entrenar_baseline_hibrido(X_dev, y_dev, X_test, y_test, targets_reg, modelo_principal, usar_cv=True):
    """
    Entrena modelo híbrido que usa DummyRegressor si el modelo principal da R² negativo.
    
    Parameters:
    -----------
    X_dev, y_dev, X_test, y_test : DataFrames/Series
    targets_reg : list de targets
    modelo_principal : callable que retorna instancia de modelo
    usar_cv : bool, si True usa CV para detectar R² negativo
    
    Returns:
    --------
    df_resultados : DataFrame con métricas
    modelos_seleccionados : dict con modelo elegido por target
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.dummy import DummyRegressor
    
    resultados = []
    modelos_seleccionados = {}
    
    print(f"MODELO HÍBRIDO - Anti R² Negativo")
    print("Estrategia: Si modelo da R² < 0 → usar DummyRegressor (R² = 0)\n")
    
    for target in targets_reg:
        # Entrenar modelo principal
        modelo_main = modelo_principal()
        modelo_main.fit(X_dev, y_dev[target])
        
        # Evaluar con CV
        if usar_cv:
            cv_scores = cross_val_score(modelo_main, X_dev, y_dev[target], 
                                       cv=5, scoring='r2', n_jobs=-1)
            cv_mean = cv_scores.mean()
        else:
            y_pred_dev = modelo_main.predict(X_dev)
            cv_mean = r2_score(y_dev[target], y_pred_dev)
        
        # Decidir qué modelo usar
        if cv_mean < 0:
            # Usar Dummy si principal da negativo
            modelo_final = DummyRegressor()
            modelo_final.fit(X_dev, y_dev[target])
            modelo_usado = "Dummy (R²<0)"
        else:
            modelo_final = modelo_main
            modelo_usado = "Ridge"
        
        modelos_seleccionados[target] = modelo_final
        
        # Predicciones con modelo final
        y_pred_dev = modelo_final.predict(X_dev)
        y_pred_test = modelo_final.predict(X_test)
        
        # Métricas
        metrics_dev = evaluar_modelo(y_dev[target], y_pred_dev, target)
        metrics_dev['split'] = 'Dev'
        metrics_dev['modelo'] = modelo_usado
        
        metrics_test = evaluar_modelo(y_test[target], y_pred_test, target)
        metrics_test['split'] = 'Test'
        metrics_test['modelo'] = modelo_usado
        
        resultados.append(metrics_dev)
        resultados.append(metrics_test)
        
        # Mostrar
        print(f"{target}:")
        print(f"  CV R²: {cv_mean:.4f} → {modelo_usado}")
        print(f"  Dev:  R² = {metrics_dev['R2']:.4f}")
        print(f"  Test: R² = {metrics_test['R2']:.4f}")
        
        if metrics_test['R2'] < 0:
            print(f"R² negativo detectado - REVISAR")
    
    return pd.DataFrame(resultados), modelos_seleccionados


def entrenar_baseline(X_dev, y_dev, X_test, y_test, targets_reg, modelos_dict):
    """
    Entrena modelos baseline y devuelve resultados.
    
    Parameters:
    -----------
    X_dev, y_dev, X_test, y_test : DataFrames/Series
    targets_reg : list de targets
    modelos_dict : dict con nombre_modelo: clase_sklearn o callable
    
    Returns:
    --------
    df_resultados : DataFrame con métricas
    modelos_entrenados : dict con modelos entrenados por target
    """
    resultados = []
    modelos_entrenados = {nombre: {} for nombre in modelos_dict.keys()}
    
    for nombre_modelo, modelo_class in modelos_dict.items():
        print(f"{nombre_modelo}")
        
        for target in targets_reg:
            # Entrenar (manejar tanto clases como callables)
            if callable(modelo_class):
                try:
                    modelo = modelo_class()
                except TypeError:
                    # Si falla, es una clase
                    modelo = modelo_class
            else:
                modelo = modelo_class()
            
            modelo.fit(X_dev, y_dev[target])
            modelos_entrenados[nombre_modelo][target] = modelo
            
            # Predicciones
            y_pred_dev = modelo.predict(X_dev)
            y_pred_test = modelo.predict(X_test)
            
            # Métricas
            metrics_dev = evaluar_modelo(y_dev[target], y_pred_dev, target)
            metrics_dev['split'] = 'Dev'
            metrics_dev['modelo'] = nombre_modelo
            
            metrics_test = evaluar_modelo(y_test[target], y_pred_test, target)
            metrics_test['split'] = 'Test'
            metrics_test['modelo'] = nombre_modelo
            
            resultados.append(metrics_dev)
            resultados.append(metrics_test)
            
            # Mostrar
            print(f"{target}:")
            print(f"  Dev:  R² = {metrics_dev['R2']:.4f}, "
                  f"MAE = {metrics_dev['MAE']:.2f}, RMSE = {metrics_dev['RMSE']:.2f}")
            print(f"  Test: R² = {metrics_test['R2']:.4f}, "
                  f"MAE = {metrics_test['MAE']:.2f}, RMSE = {metrics_test['RMSE']:.2f}")
    
    return pd.DataFrame(resultados), modelos_entrenados
