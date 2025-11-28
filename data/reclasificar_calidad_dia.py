"""
reclasificar_calidad_dia.py
============================
Reclasifica calidad_dia en TODOS los archivos (raw y processed) sin recombinar.

Uso:
  python reclasificar_calidad_dia.py --opcion 1  (usa cuartiles)
  python reclasificar_calidad_dia.py --umbral1 1200 --umbral2 1800 --umbral3 2800
  python reclasificar_calidad_dia.py --custom  (te pregunta interactivamente)

Modifica:
  - data/raw/*.csv (archivos individuales)
  - data/processed/dataset_FINAL.csv
  - data/processed/y_clf_train.csv
  - data/processed/y_clf_val.csv
  - data/processed/y_clf_test.csv
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
from datetime import datetime
import shutil


def clasificar_dia_nuevo(altura_max, umbral1, umbral2, umbral3):
    """
    Nueva función de clasificación con umbrales personalizados.
    
    Args:
        altura_max: Altura máxima del vuelo
        umbral1: Umbral Malo/Regular
        umbral2: Umbral Regular/Bueno
        umbral3: Umbral Bueno/Excelente
    
    Returns:
        str: Clase ('Malo', 'Regular', 'Bueno', 'Excelente')
    """
    if pd.isna(altura_max):
        return None
    elif altura_max > umbral3:
        return 'Excelente'
    elif altura_max > umbral2:
        return 'Bueno'
    elif altura_max > umbral1:
        return 'Regular'
    else:
        return 'Malo'


def obtener_umbrales(opcion=None, u1=None, u2=None, u3=None):
    """Obtiene umbrales según la opción elegida."""
    
    # Cargar dataset para calcular percentiles
    df = pd.read_csv('processed/dataset_FINAL.csv')
    
    if opcion == 1:
        # Cuartiles (25% cada clase)
        q25 = df['altura_max_m'].quantile(0.25)
        q50 = df['altura_max_m'].quantile(0.50)
        q75 = df['altura_max_m'].quantile(0.75)
        return q25, q50, q75
    
    elif opcion == 2:
        # Moderado (20-20-30-30)
        p20 = df['altura_max_m'].quantile(0.20)
        p40 = df['altura_max_m'].quantile(0.40)
        p70 = df['altura_max_m'].quantile(0.70)
        return p20, p40, p70
    
    elif opcion == 3:
        # Riguroso (10-20-30-40)
        p10 = df['altura_max_m'].quantile(0.10)
        p30 = df['altura_max_m'].quantile(0.30)
        p60 = df['altura_max_m'].quantile(0.60)
        return p10, p30, p60
    
    elif opcion == 4:
        # Redondeados balanceados
        q25 = df['altura_max_m'].quantile(0.25)
        q50 = df['altura_max_m'].quantile(0.50)
        q75 = df['altura_max_m'].quantile(0.75)
        t1 = round(q25 / 100) * 100
        t2 = round(q50 / 100) * 100
        t3 = round(q75 / 100) * 100
        return t1, t2, t3
    
    elif u1 is not None and u2 is not None and u3 is not None:
        # Umbrales manuales
        return u1, u2, u3
    
    else:
        raise ValueError("Debes especificar --opcion o --umbral1/2/3")


def hacer_backup(archivos):
    """Hace backup de archivos antes de modificarlos."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_reclasificacion_{timestamp}"
    
    print(f"\n▶ Creando backup en: {backup_dir}/")
    os.makedirs(backup_dir, exist_ok=True)
    
    for archivo in archivos:
        if os.path.exists(archivo):
            dest = os.path.join(backup_dir, os.path.basename(archivo))
            shutil.copy2(archivo, dest)
    
    print(f"  ✓ {len(archivos)} archivos respaldados")
    return backup_dir


def mostrar_distribucion(df, nombre="Dataset"):
    """Muestra distribución de clases."""
    
    dist = df['calidad_dia'].value_counts().sort_index()
    total = len(df)
    
    print(f"\n  {nombre}:")
    for clase in ['Malo', 'Regular', 'Bueno', 'Excelente']:
        if clase in dist.index:
            count = dist[clase]
            pct = (count / total) * 100
            bar = '█' * int(pct / 2)
            print(f"    {clase:10s}: {count:3d} ({pct:5.1f}%) {bar}")


def reclasificar_archivos_raw(umbral1, umbral2, umbral3):
    """Reclasifica archivos en data/raw/."""
    
    print(f"\n" + "="*80)
    print("1. RECLASIFICANDO ARCHIVOS RAW")
    print("="*80)
    
    archivos_raw = glob.glob('raw/*.csv')
    print(f"\n▶ Encontrados {len(archivos_raw)} archivos raw")
    
    modificados = 0
    
    for archivo in archivos_raw:
        df = pd.read_csv(archivo)
        
        if 'altura_max_m' in df.columns and 'calidad_dia' in df.columns:
            # Reclasificar
            df['calidad_dia'] = df['altura_max_m'].apply(
                lambda x: clasificar_dia_nuevo(x, umbral1, umbral2, umbral3)
            )
            
            # Guardar
            df.to_csv(archivo, index=False)
            modificados += 1
    
    print(f"  ✓ {modificados} archivos modificados")


def reclasificar_dataset_final(umbral1, umbral2, umbral3):
    """Reclasifica dataset_FINAL.csv."""
    
    print(f"\n" + "="*80)
    print("2. RECLASIFICANDO DATASET_FINAL.CSV")
    print("="*80)
    
    archivo = 'processed/dataset_FINAL.csv'
    df = pd.read_csv(archivo)
    
    print(f"\n▶ Distribución ANTES:")
    mostrar_distribucion(df, "dataset_FINAL.csv")
    
    # Reclasificar
    df['calidad_dia'] = df['altura_max_m'].apply(
        lambda x: clasificar_dia_nuevo(x, umbral1, umbral2, umbral3)
    )
    
    print(f"\n▶ Distribución DESPUÉS:")
    mostrar_distribucion(df, "dataset_FINAL.csv")
    
    # Guardar
    df.to_csv(archivo, index=False)
    print(f"\n  ✓ dataset_FINAL.csv modificado")


def reclasificar_splits_clf(umbral1, umbral2, umbral3):
    """Reclasifica y_clf_train/val/test.csv."""
    
    print(f"\n" + "="*80)
    print("3. RECLASIFICANDO SPLITS DE CLASIFICACIÓN")
    print("="*80)
    
    # Cargar dataset_FINAL para obtener altura_max actualizada
    df_final = pd.read_csv('processed/dataset_FINAL.csv')
    
    # Para cada split
    for split_name in ['train', 'val', 'test']:
        archivo_y = f'processed/y_clf_{split_name}.csv'
        archivo_X = f'processed/X_{split_name}.csv'
        
        if not os.path.exists(archivo_y):
            print(f"  ⚠ {archivo_y} no existe, saltando...")
            continue
        
        # Cargar y_clf
        df_y = pd.read_csv(archivo_y)
        
        print(f"\n▶ {split_name.upper()}:")
        print(f"  Distribución ANTES:")
        mostrar_distribucion(df_y, f"y_clf_{split_name}")
        
        # Necesitamos altura_max_m para reclasificar
        # La manera más segura es leer de dataset_FINAL usando el orden
        # Asumir que los splits mantienen el orden temporal
        
        # Cargar X para saber el tamaño
        df_X = pd.read_csv(archivo_X)
        n_samples = len(df_X)
        
        # Obtener las primeras n_samples alturas de dataset_FINAL (ordenado por fecha)
        df_final_sorted = df_final.sort_values('fecha').reset_index(drop=True)
        
        if split_name == 'train':
            alturas = df_final_sorted['altura_max_m'].iloc[:n_samples]
        elif split_name == 'val':
            # Val viene después de train
            n_train = len(pd.read_csv('processed/X_train.csv'))
            alturas = df_final_sorted['altura_max_m'].iloc[n_train:n_train+n_samples]
        else:  # test
            n_train = len(pd.read_csv('processed/X_train.csv'))
            n_val = len(pd.read_csv('processed/X_val.csv'))
            alturas = df_final_sorted['altura_max_m'].iloc[n_train+n_val:n_train+n_val+n_samples]
        
        # Reclasificar
        df_y['calidad_dia'] = alturas.apply(
            lambda x: clasificar_dia_nuevo(x, umbral1, umbral2, umbral3)
        ).values
        
        print(f"  Distribución DESPUÉS:")
        mostrar_distribucion(df_y, f"y_clf_{split_name}")
        
        # Guardar
        df_y.to_csv(archivo_y, index=False)
        print(f"  ✓ y_clf_{split_name}.csv modificado")


def main():
    parser = argparse.ArgumentParser(
        description='Reclasificar calidad_dia en todos los archivos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python reclasificar_calidad_dia.py --opcion 1
  python reclasificar_calidad_dia.py --opcion 4
  python reclasificar_calidad_dia.py --umbral1 1200 --umbral2 1800 --umbral3 2800
        """
    )
    
    parser.add_argument('--opcion', type=int, choices=[1, 2, 3, 4],
                        help='Opción predefinida (1=cuartiles, 2=moderado, 3=riguroso, 4=redondeado)')
    parser.add_argument('--umbral1', type=float, help='Umbral Malo/Regular (metros)')
    parser.add_argument('--umbral2', type=float, help='Umbral Regular/Bueno (metros)')
    parser.add_argument('--umbral3', type=float, help='Umbral Bueno/Excelente (metros)')
    parser.add_argument('--custom', action='store_true', help='Modo interactivo')
    
    args = parser.parse_args()
    
    # Obtener umbrales
    if args.custom:
        print("="*80)
        print("MODO INTERACTIVO")
        print("="*80)
        print("\nIngresa los umbrales en metros:")
        u1 = float(input("  Umbral Malo/Regular: "))
        u2 = float(input("  Umbral Regular/Bueno: "))
        u3 = float(input("  Umbral Bueno/Excelente: "))
        umbral1, umbral2, umbral3 = u1, u2, u3
    else:
        umbral1, umbral2, umbral3 = obtener_umbrales(
            opcion=args.opcion,
            u1=args.umbral1,
            u2=args.umbral2,
            u3=args.umbral3
        )
    
    # Mostrar umbrales
    print("="*80)
    print("RECLASIFICACIÓN DE CALIDAD DEL DÍA")
    print("="*80)
    print(f"\n▶ Umbrales a usar:")
    print(f"  Malo:      altura_max <= {umbral1:.1f}m")
    print(f"  Regular:   {umbral1:.1f}m < altura_max <= {umbral2:.1f}m")
    print(f"  Bueno:     {umbral2:.1f}m < altura_max <= {umbral3:.1f}m")
    print(f"  Excelente: altura_max > {umbral3:.1f}m")
    
    # Confirmar
    print(f"\n⚠ ADVERTENCIA: Esto modificará TODOS los archivos con calidad_dia")
    respuesta = input("\n¿Continuar? (s/n): ")
    
    if respuesta.lower() != 's':
        print("\n❌ Operación cancelada")
        return
    
    # Hacer backup
    archivos_backup = [
        'processed/dataset_FINAL.csv',
        'processed/y_clf_train.csv',
        'processed/y_clf_val.csv',
        'processed/y_clf_test.csv'
    ] + glob.glob('raw/*.csv')
    
    backup_dir = hacer_backup(archivos_backup)
    
    # Reclasificar
    reclasificar_archivos_raw(umbral1, umbral2, umbral3)
    reclasificar_dataset_final(umbral1, umbral2, umbral3)
    reclasificar_splits_clf(umbral1, umbral2, umbral3)
    
    # Resumen final
    print(f"\n" + "="*80)
    print("✓ RECLASIFICACIÓN COMPLETADA")
    print("="*80)
    print(f"\nArchivos modificados:")
    print(f"  - raw/*.csv ({len(glob.glob('raw/*.csv'))} archivos)")
    print(f"  - processed/dataset_FINAL.csv")
    print(f"  - processed/y_clf_{{train,val,test}}.csv")
    print(f"\nBackup guardado en: {backup_dir}/")
    print(f"\nPróximos pasos:")
    print(f"  1. Verifica la distribución con: python analizar_distribucion_calidad.py")
    print(f"  2. Si estás conforme, continúa con el modelado")
    print(f"  3. Si no, restaura el backup y prueba otros umbrales")
    print("="*80)


if __name__ == "__main__":
    main()