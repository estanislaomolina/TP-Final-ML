"""
Visualizador de Ubicaciones de Vuelos
Muestra dónde están tus vuelos para verificar el área de descarga ERA5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def visualizar_ubicaciones():
    """
    Visualizar ubicaciones de despegue de todos los vuelos
    """
    
    print("="*70)
    print("VISUALIZACIÓN DE UBICACIONES DE VUELOS")
    print("="*70)
    
    # Cargar vuelos
    vuelos_file = None
    for posible_file in ['data/raw/vuelos_metadata_completo.csv', 'data/raw/vuelos_metadata.csv']:
        if os.path.exists(posible_file):
            vuelos_file = posible_file
            break
    
    if vuelos_file is None:
        print("\n✗ No se encuentra archivo de vuelos")
        print("Ejecuta primero: python 01_procesar_igc_COMPLETO.py")
        return
    
    df = pd.read_csv(vuelos_file)
    
    print(f"\nVuelos cargados: {len(df)}")
    
    # Ubicaciones
    lats = df['lat_despegue']
    lons = df['lon_despegue']
    
    # Calcular área
    lat_min = lats.min()
    lat_max = lats.max()
    lon_min = lons.min()
    lon_max = lons.max()
    
    # Margen ERA5
    MARGEN = 1.5
    era5_lat_min = np.floor(lat_min - MARGEN)
    era5_lat_max = np.ceil(lat_max + MARGEN)
    era5_lon_min = np.floor(lon_min - MARGEN)
    era5_lon_max = np.ceil(lon_max + MARGEN)
    
    print(f"\nUbicaciones de vuelos:")
    print(f"  Latitud:  {lat_min:.3f}° a {lat_max:.3f}°")
    print(f"  Longitud: {lon_min:.3f}° a {lon_max:.3f}°")
    
    print(f"\nÁrea de descarga ERA5 (con margen):")
    print(f"  Latitud:  {era5_lat_min:.1f}° a {era5_lat_max:.1f}°")
    print(f"  Longitud: {era5_lon_min:.1f}° a {era5_lon_max:.1f}°")
    
    # Identificar clusters (posibles clubes)
    from scipy.cluster.hierarchy import fclusterdata
    
    coords = np.column_stack([lats, lons])
    
    try:
        # Clustering jerárquico
        clusters = fclusterdata(coords, t=0.5, criterion='distance', method='complete')
        num_clusters = len(np.unique(clusters))
        
        print(f"\nClusters detectados (posibles clubes): {num_clusters}")
        
        for i in range(1, num_clusters + 1):
            cluster_lats = lats[clusters == i]
            cluster_lons = lons[clusters == i]
            cluster_center_lat = cluster_lats.mean()
            cluster_center_lon = cluster_lons.mean()
            
            print(f"\nCluster {i}: {len(cluster_lats)} vuelos")
            print(f"  Centro: {cluster_center_lat:.3f}°, {cluster_center_lon:.3f}°")
            print(f"  Rango lat: {cluster_lats.min():.3f}° a {cluster_lats.max():.3f}°")
            print(f"  Rango lon: {cluster_lons.min():.3f}° a {cluster_lons.max():.3f}°")
    except:
        clusters = np.ones(len(lats))
        num_clusters = 1
    
    # Crear visualización
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Mapa de fondo simple
    ax.set_facecolor('#E8F4F8')
    
    # Área de descarga ERA5 (rectángulo)
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (era5_lon_min, era5_lat_min),
        era5_lon_max - era5_lon_min,
        era5_lat_max - era5_lat_min,
        linewidth=3,
        edgecolor='red',
        facecolor='none',
        linestyle='--',
        label='Área descarga ERA5'
    )
    ax.add_patch(rect)
    
    # Puntos de despegue
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    
    for i in range(1, num_clusters + 1):
        mask = clusters == i
        ax.scatter(
            lons[mask],
            lats[mask],
            c=[colors[i-1]],
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=f'Club {i} ({mask.sum()} vuelos)'
        )
        
        # Centro del cluster
        if mask.sum() > 1:
            center_lat = lats[mask].mean()
            center_lon = lons[mask].mean()
            ax.scatter(
                center_lon,
                center_lat,
                c=[colors[i-1]],
                s=300,
                marker='*',
                edgecolors='black',
                linewidth=2
            )
    
    # Configuración
    ax.set_xlabel('Longitud', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitud', fontsize=12, fontweight='bold')
    ax.set_title('Ubicaciones de Vuelos y Área de Descarga ERA5', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Ajustar límites con padding
    padding = 0.5
    ax.set_xlim(era5_lon_min - padding, era5_lon_max + padding)
    ax.set_ylim(era5_lat_min - padding, era5_lat_max + padding)
    
    # Agregar anotaciones
    ax.text(
        0.02, 0.98,
        f'Total vuelos: {len(df)}\nClubes detectados: {num_clusters}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10
    )
    
    plt.tight_layout()
    
    # Guardar
    output_file = 'visualizacion_ubicaciones.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualización guardada: {output_file}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("INTERPRETACIÓN:")
    print("="*70)
    print("• Puntos de colores: Ubicaciones de despegue")
    print("• Estrellas: Centros de cada club")
    print("• Rectángulo rojo punteado: Área que cubrirá ERA5")
    print("="*70)

if __name__ == "__main__":
    visualizar_ubicaciones()
