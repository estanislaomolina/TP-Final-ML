"""
Funciones para entrenamiento y evaluación de modelos
Predicción de vuelo en planeador
"""

import numpy as np
import pandas as pd
import re
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def calcular_features_promedio(df):
    """
    Calcula promedios de features meteorológicas horarias.
    Reduce ~100 features horarias a ~11 promedios.
    
    Parameters:
    -----------
    df : DataFrame
        Dataset con features horarias (09h-18h)
    
    Returns:
    --------
    DataFrame con features promedio agregadas
    """
    df_out = df.copy()
    
    # Variables meteorológicas con datos horarios
    variables_meteo = [
        'solar_rad', 'precipitation', 'temp_2m', 'cloud_cover',
        'wind_u', 'wind_v', 'pressure', 'boundary_layer_height',
        'cape', 'skin_temp', 'wind_speed'
    ]
    
    for var in variables_meteo:
        # Buscar todas las columnas horarias de esta variable
        pattern = f'{var}_\\d{{2}}h'
        cols_horarias = [col for col in df.columns if re.match(pattern, col)]
        
        if cols_horarias:
            # Calcular promedio
            df_out[f'{var}_avg'] = df[cols_horarias].mean(axis=1)
    
    return df_out


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
        dev_proc = calcular_features_promedio(dev_proc)
        test_proc = calcular_features_promedio(test_proc)
    
    # Columnas a eliminar (no son features predictivas)
    cols_no_features = [
        'fecha', 'pilot', 'glider', 'competition_id', 
        'filename', 'flight_id', 'calidad_dia',
        'hora_despegue'  # string - usamos hora_despegue_decimal
    ]
    
    # Filtrar solo las que existen
    cols_no_features = [col for col in cols_no_features if col in dev_proc.columns]
    
    # Features que son derivadas del vuelo (no disponibles en predicción)
    # IMPORTANTE: Conservar lat_despegue, lon_despegue, hora_despegue_decimal
    flight_features = [
        'altura_min_m', 'altura_despegue_m', 'altura_aterrizaje_m',
        'rango_altura_m', 'duracion_horas', 'num_gps_fixes',
        'frecuencia_muestreo_seg', 'intensidad_termicas_max_ms',
        'intensidad_termicas_min_ms', 'intensidad_termicas_std_ms',
        'altura_base_termicas_mean_m', 'altura_tope_termicas_mean_m',
        'altura_base_termicas_min_m', 'altura_tope_termicas_max_m',
        'ganancia_por_termica_mean_m', 'ganancia_por_termica_max_m',
        'duracion_termica_mean_seg', 'duracion_termica_max_seg',
        'hora_primera_termica', 'hora_ultima_termica',
        'dispersion_termicas_lat', 'dispersion_termicas_lon',
        'tiempo_en_planeo_min', 'porcentaje_tiempo_termicas',
        'tasa_descenso_mean_ms', 'bearing_change_mean_deg',
        'bearing_change_max_deg', 'bearing_change_std_deg',
        'ground_speed_mean_kmh', 'ground_speed_max_kmh',
        'ground_speed_min_kmh', 'ground_speed_std_kmh',
        'hora_inicio_decimal', 'hora_fin_decimal',
        'altura_mean_manana_m', 'altura_mean_mediodia_m',
        'altura_mean_tarde1_m', 'altura_mean_tarde2_m',
        'lat_min', 'lat_max', 'lon_min', 'lon_max',
        'lat_centro', 'lon_centro', 'rango_lat_deg', 'rango_lon_deg',
        'distancia_max_despegue_km', 'area_vuelo_km2',
        'altura_std_m', 'altura_cv', 'cambio_altura_mean_m',
        'cambio_altura_std_m'
    ]
    
    flight_features = [col for col in flight_features if col in dev_proc.columns]
    
    # Si modo simple, eliminar también todas las features horarias
    if modo == 'simple':
        import re
        horarias_pattern = re.compile(r'_\d{2}h$')
        cols_horarias = [col for col in dev_proc.columns if horarias_pattern.search(col)]
        flight_features.extend(cols_horarias)
    
    # Todas las columnas a eliminar
    cols_to_drop = list(set(cols_no_features + flight_features + targets_reg))
    
    # Separar X e y
    X_dev = dev_proc.drop(columns=cols_to_drop)
    y_dev = dev_proc[targets_reg]
    X_test = test_proc.drop(columns=cols_to_drop)
    y_test = test_proc[targets_reg]
    
    return X_dev, y_dev, X_test, y_test


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
    
    print(f"\n{'='*70}")
    print(f"MODELO HÍBRIDO - Anti R² Negativo")
    print('='*70)
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
            print(f"  ⚠️ R² negativo detectado - REVISAR")
    
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
        print(f"\n{'='*70}")
        print(f"{nombre_modelo}")
        print('='*70)
        
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
