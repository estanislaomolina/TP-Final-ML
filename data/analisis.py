"""
analizar_distribucion_calidad.py
=================================
Analiza la distribución actual de calidad_dia y sugiere umbrales óptimos.

Ejecutar: python analizar_distribucion_calidad.py
"""

import pandas as pd
import numpy as np


def analizar_distribucion():
    """Analiza la distribución de altura_max y calidad_dia actual."""
    
    print("="*80)
    print("ANÁLISIS DE DISTRIBUCIÓN DE CALIDAD DEL DÍA")
    print("="*80)
    
    # Cargar dataset
    df = pd.read_csv('processed/dataset_FINAL.csv')
    
    print(f"\n▶ Total de vuelos: {len(df)}")
    
    # Distribución de altura_max
    print(f"\n▶ Estadísticas de altura_max:")
    print(f"  Min:    {df['altura_max_m'].min():.1f}m")
    print(f"  Q1:     {df['altura_max_m'].quantile(0.25):.1f}m")
    print(f"  Mediana: {df['altura_max_m'].median():.1f}m")
    print(f"  Q3:     {df['altura_max_m'].quantile(0.75):.1f}m")
    print(f"  Max:    {df['altura_max_m'].max():.1f}m")
    print(f"  Media:  {df['altura_max_m'].mean():.1f}m")
    print(f"  Std:    {df['altura_max_m'].std():.1f}m")
    
    # Percentiles útiles
    print(f"\n▶ Percentiles de altura_max:")
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
        val = df['altura_max_m'].quantile(p/100)
        print(f"  P{p:2d}: {val:.1f}m")
    
    # Distribución actual de clases
    print(f"\n▶ Distribución ACTUAL de calidad_dia:")
    dist_actual = df['calidad_dia'].value_counts().sort_index()
    total = len(df)
    
    for clase in ['Malo', 'Regular', 'Bueno', 'Excelente']:
        if clase in dist_actual.index:
            count = dist_actual[clase]
            pct = (count / total) * 100
            print(f"  {clase:10s}: {count:3d} vuelos ({pct:5.1f}%)")
    
    # Calcular desbalance
    max_clase = dist_actual.max()
    min_clase = dist_actual.min()
    ratio = max_clase / min_clase
    
    print(f"\n▶ Desbalance actual:")
    print(f"  Clase más frecuente: {dist_actual.idxmax()} ({max_clase} vuelos)")
    print(f"  Clase menos frecuente: {dist_actual.idxmin()} ({min_clase} vuelos)")
    print(f"  Ratio: {ratio:.2f}:1")
    
    if ratio > 3:
        print(f"  ⚠ DESBALANCE SIGNIFICATIVO (ideal < 3:1)")
    else:
        print(f"  ✓ Desbalance aceptable")
    
    # Sugerir umbrales óptimos (cuartiles)
    print(f"\n" + "="*80)
    print("SUGERENCIAS DE UMBRALES")
    print("="*80)
    
    q25 = df['altura_max_m'].quantile(0.25)
    q50 = df['altura_max_m'].quantile(0.50)
    q75 = df['altura_max_m'].quantile(0.75)
    
    print(f"\n▶ Opción 1: BALANCEADO (cuartiles - 25% cada clase)")
    print(f"  Malo:      altura_max <= {q25:.0f}m")
    print(f"  Regular:   {q25:.0f}m < altura_max <= {q50:.0f}m")
    print(f"  Bueno:     {q50:.0f}m < altura_max <= {q75:.0f}m")
    print(f"  Excelente: altura_max > {q75:.0f}m")
    
    p20 = df['altura_max_m'].quantile(0.20)
    p40 = df['altura_max_m'].quantile(0.40)
    p70 = df['altura_max_m'].quantile(0.70)
    
    print(f"\n▶ Opción 2: MODERADO (20-20-30-30)")
    print(f"  Malo:      altura_max <= {p20:.0f}m")
    print(f"  Regular:   {p20:.0f}m < altura_max <= {p40:.0f}m")
    print(f"  Bueno:     {p40:.0f}m < altura_max <= {p70:.0f}m")
    print(f"  Excelente: altura_max > {p70:.0f}m")
    
    p10 = df['altura_max_m'].quantile(0.10)
    p30 = df['altura_max_m'].quantile(0.30)
    p60 = df['altura_max_m'].quantile(0.60)
    
    print(f"\n▶ Opción 3: RIGUROSO (10-20-30-40)")
    print(f"  Malo:      altura_max <= {p10:.0f}m")
    print(f"  Regular:   {p10:.0f}m < altura_max <= {p30:.0f}m")
    print(f"  Bueno:     {p30:.0f}m < altura_max <= {p60:.0f}m")
    print(f"  Excelente: altura_max > {p60:.0f}m")
    
    # Umbrales redondeados
    print(f"\n▶ Opción 4: REDONDEADOS BALANCEADOS")
    t1 = round(q25 / 100) * 100  # Redondear a centenas
    t2 = round(q50 / 100) * 100
    t3 = round(q75 / 100) * 100
    print(f"  Malo:      altura_max <= {t1:.0f}m")
    print(f"  Regular:   {t1:.0f}m < altura_max <= {t2:.0f}m")
    print(f"  Bueno:     {t2:.0f}m < altura_max <= {t3:.0f}m")
    print(f"  Excelente: altura_max > {t3:.0f}m")
    
    print(f"\n" + "="*80)
    print("RECOMENDACIÓN:")
    print("="*80)
    print(f"\nUsa la Opción 1 (cuartiles) para máximo balance,")
    print(f"o la Opción 4 (redondeados) para umbrales más interpretables.")
    print(f"\nLuego ejecuta: python reclasificar_calidad_dia.py --opcion 1")
    print(f"             o: python reclasificar_calidad_dia.py --umbral1 {t1:.0f} --umbral2 {t2:.0f} --umbral3 {t3:.0f}")
    print("="*80)


if __name__ == "__main__":
    analizar_distribucion()