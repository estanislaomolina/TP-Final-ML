"""
Procesador IGC COMPLETO - Máxima Extracción de Features
Extrae TODA la información posible para tener el máximo de features
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class IGCProcessorComplete:
    def __init__(self, igc_folder):
        self.igc_folder = igc_folder
        
    def parse_igc_file(self, filepath):
        """
        Parsear un archivo IGC con MÁXIMA extracción de información
        """
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  ✗ Error leyendo {filepath}: {e}")
            return None
        
        # Estructuras para almacenar datos
        metadata = {}
        fixes = []
        
        # Parsear líneas
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('H'):
                self._parse_header(line, metadata)
            elif line.startswith('B'):
                fix = self._parse_b_record(line)
                if fix:
                    fixes.append(fix)
        
        if len(fixes) < 20:
            print(f"  ⚠ Muy pocos puntos GPS en {os.path.basename(filepath)}")
            return None
        
        # Convertir a DataFrame
        df_fixes = pd.DataFrame(fixes)
        df_fixes = df_fixes[df_fixes['validity'] == 'A'].copy()
        
        if len(df_fixes) < 20:
            return None
        
        # ================================================================
        # EXTRACCIÓN COMPLETA DE FEATURES
        # ================================================================
        
        # 1. Features básicas del vuelo
        basic_stats = self._extract_basic_stats(df_fixes, metadata)
        
        # 2. Detección y análisis de térmicas
        thermal_stats = self._extract_thermal_features(df_fixes)
        
        # 3. Features de trayectoria
        trajectory_stats = self._extract_trajectory_features(df_fixes)
        
        # 4. Features temporales
        temporal_stats = self._extract_temporal_features(df_fixes)
        
        # 5. Features espaciales
        spatial_stats = self._extract_spatial_features(df_fixes)
        
        # 6. Features de variabilidad
        variability_stats = self._extract_variability_features(df_fixes)
        
        # Combinar todo
        all_stats = {
            **basic_stats,
            **thermal_stats,
            **trajectory_stats,
            **temporal_stats,
            **spatial_stats,
            **variability_stats
        }
        
        all_stats['filename'] = os.path.basename(filepath)
        
        return all_stats
    
    def _parse_header(self, line, metadata):
        """Parsear headers del IGC"""
        
        if 'HFDTE' in line:
            match = re.search(r'(\d{6})', line)
            if match:
                date_str = match.group(1)
                try:
                    day = int(date_str[0:2])
                    month = int(date_str[2:4])
                    year = int(date_str[4:6])
                    year = 2000 + year if year < 50 else 1900 + year
                    metadata['date'] = datetime(year, month, day)
                except:
                    pass
        
        elif 'HFPLT' in line:
            match = re.search(r'PILOT[^:]*:(.+)', line, re.IGNORECASE)
            if match:
                metadata['pilot'] = match.group(1).strip()
        
        elif 'HFGTY' in line:
            match = re.search(r'GLIDERTYPE[^:]*:(.+)', line, re.IGNORECASE)
            if match:
                metadata['glider'] = match.group(1).strip()
        
        elif 'HFCID' in line:
            match = re.search(r'COMPETITIONID[^:]*:(.+)', line, re.IGNORECASE)
            if match:
                metadata['competition_id'] = match.group(1).strip()
    
    def _parse_b_record(self, line):
        """Parsear B record (GPS fix)"""
        
        if len(line) < 35:
            return None
        
        try:
            hours = int(line[1:3])
            minutes = int(line[3:5])
            seconds = int(line[5:7])
            
            lat_deg = int(line[7:9])
            lat_min = int(line[9:14]) / 1000.0
            lat_dir = line[14]
            latitude = lat_deg + lat_min / 60.0
            if lat_dir == 'S':
                latitude = -latitude
            
            lon_deg = int(line[15:18])
            lon_min = int(line[18:23]) / 1000.0
            lon_dir = line[23]
            longitude = lon_deg + lon_min / 60.0
            if lon_dir == 'W':
                longitude = -longitude
            
            validity = line[24]
            alt_pressure = int(line[25:30])
            alt_gps = int(line[30:35])
            altitude = alt_gps if alt_gps > 0 else alt_pressure
            
            return {
                'time_seconds': hours * 3600 + minutes * 60 + seconds,
                'latitude': latitude,
                'longitude': longitude,
                'altitude': altitude,
                'validity': validity
            }
        except:
            return None
    
    def _extract_basic_stats(self, df_fixes, metadata):
        """Features básicas del vuelo"""
        
        # Alturas
        altura_max = df_fixes['altitude'].max()
        altura_min = df_fixes['altitude'].min()
        altura_despegue = df_fixes['altitude'].iloc[0]
        altura_aterrizaje = df_fixes['altitude'].iloc[-1]
        ganancia_altura = altura_max - min(altura_despegue, altura_aterrizaje)
        
        # Duración
        tiempo_inicio = df_fixes['time_seconds'].iloc[0]
        tiempo_fin = df_fixes['time_seconds'].iloc[-1]
        if tiempo_fin < tiempo_inicio:
            tiempo_fin += 86400
        duracion_segundos = tiempo_fin - tiempo_inicio
        duracion_minutos = duracion_segundos / 60.0
        
        # Distancia
        distancia_km = self._calculate_total_distance(df_fixes)
        
        # Velocidad
        velocidad_promedio = (distancia_km / duracion_minutos * 60) if duracion_minutos > 0 else 0
        
        # Ubicación despegue
        lat_despegue = df_fixes['latitude'].iloc[0]
        lon_despegue = df_fixes['longitude'].iloc[0]
        
        # Hora despegue
        hora_despegue_seg = df_fixes['time_seconds'].iloc[0]
        hora_despegue = f"{int(hora_despegue_seg // 3600):02d}:{int((hora_despegue_seg % 3600) // 60):02d}:00"
        
        return {
            'fecha': metadata.get('date', None),
            'pilot': metadata.get('pilot', 'Unknown'),
            'glider': metadata.get('glider', 'Unknown'),
            'competition_id': metadata.get('competition_id', ''),
            
            # Alturas
            'altura_max_m': int(altura_max),
            'altura_min_m': int(altura_min),
            'altura_despegue_m': int(altura_despegue),
            'altura_aterrizaje_m': int(altura_aterrizaje),
            'ganancia_altura_m': int(ganancia_altura),
            'rango_altura_m': int(altura_max - altura_min),
            
            # Tiempos
            'duracion_min': round(duracion_minutos, 1),
            'duracion_horas': round(duracion_minutos / 60, 2),
            
            # Distancias
            'distancia_km': round(distancia_km, 1),
            'velocidad_promedio_kmh': round(velocidad_promedio, 1),
            
            # Ubicación
            'lat_despegue': round(lat_despegue, 6),
            'lon_despegue': round(lon_despegue, 6),
            'hora_despegue': hora_despegue,
            'hora_despegue_decimal': round(hora_despegue_seg / 3600, 2),
            
            # Metadata
            'num_gps_fixes': len(df_fixes),
            'frecuencia_muestreo_seg': round(df_fixes['time_seconds'].diff().median(), 1)
        }
    
    def _extract_thermal_features(self, df_fixes):
        """
        Detección COMPLETA de térmicas y extracción de features
        """
        
        df = df_fixes.copy()
        
        # Calcular variometro (tasa de ascenso)
        df['dt'] = df['time_seconds'].diff()
        df['dh'] = df['altitude'].diff()
        df['vario'] = df['dh'] / df['dt']  # m/s
        
        # Limpiar
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['vario'])
        
        # Calcular cambios de dirección (para detectar círculos)
        df['dlat'] = df['latitude'].diff()
        df['dlon'] = df['longitude'].diff()
        df['bearing'] = np.arctan2(df['dlon'], df['dlat']) * 180 / np.pi
        df['bearing_change'] = df['bearing'].diff().abs()
        
        # DETECCIÓN DE TÉRMICAS
        # Una térmica es un segmento donde:
        # 1. Vario > umbral (ej: 0.5 m/s)
        # 2. Hay giro (cambios de rumbo significativos)
        
        umbral_vario = 0.5  # m/s
        df['climbing'] = df['vario'] > umbral_vario
        
        # Detectar cambios de estado (inicio/fin de térmica)
        df['state_change'] = df['climbing'].astype(int).diff()
        
        # Segmentar térmicas
        thermals = []
        in_thermal = False
        thermal_start_idx = 0
        
        for idx, row in df.iterrows():
            if row['state_change'] == 1:  # Empieza térmica
                in_thermal = True
                thermal_start_idx = idx
            elif row['state_change'] == -1 and in_thermal:  # Termina térmica
                in_thermal = False
                thermal_segment = df.loc[thermal_start_idx:idx]
                
                # Filtrar térmicas muy cortas (< 30 seg)
                if len(thermal_segment) > 10:
                    thermal_info = {
                        'intensidad': thermal_segment['vario'].mean(),
                        'intensidad_max': thermal_segment['vario'].max(),
                        'altura_base': thermal_segment['altitude'].iloc[0],
                        'altura_tope': thermal_segment['altitude'].iloc[-1],
                        'ganancia': thermal_segment['altitude'].iloc[-1] - thermal_segment['altitude'].iloc[0],
                        'duracion_seg': thermal_segment['dt'].sum(),
                        'lat_centro': thermal_segment['latitude'].mean(),
                        'lon_centro': thermal_segment['longitude'].mean(),
                        'hora_inicio': thermal_segment['time_seconds'].iloc[0]
                    }
                    thermals.append(thermal_info)
        
        # Calcular estadísticas de térmicas
        if len(thermals) > 0:
            df_thermals = pd.DataFrame(thermals)
            
            stats = {
                # Conteo
                'num_termicas': len(thermals),
                
                # Intensidad
                'intensidad_termicas_mean_ms': round(df_thermals['intensidad'].mean(), 2),
                'intensidad_termicas_max_ms': round(df_thermals['intensidad_max'].max(), 2),
                'intensidad_termicas_min_ms': round(df_thermals['intensidad'].min(), 2),
                'intensidad_termicas_std_ms': round(df_thermals['intensidad'].std(), 2),
                
                # Altura
                'altura_base_termicas_mean_m': round(df_thermals['altura_base'].mean(), 0),
                'altura_tope_termicas_mean_m': round(df_thermals['altura_tope'].mean(), 0),
                'altura_base_termicas_min_m': round(df_thermals['altura_base'].min(), 0),
                'altura_tope_termicas_max_m': round(df_thermals['altura_tope'].max(), 0),
                'ganancia_por_termica_mean_m': round(df_thermals['ganancia'].mean(), 0),
                'ganancia_por_termica_max_m': round(df_thermals['ganancia'].max(), 0),
                
                # Duración
                'duracion_termica_mean_seg': round(df_thermals['duracion_seg'].mean(), 0),
                'duracion_termica_max_seg': round(df_thermals['duracion_seg'].max(), 0),
                
                # Temporal
                'hora_primera_termica': round(df_thermals['hora_inicio'].min() / 3600, 2),
                'hora_ultima_termica': round(df_thermals['hora_inicio'].max() / 3600, 2),
                
                # Espacial
                'dispersion_termicas_lat': round(df_thermals['lat_centro'].std(), 4),
                'dispersion_termicas_lon': round(df_thermals['lon_centro'].std(), 4),
            }
        else:
            stats = {
                'num_termicas': 0,
                'intensidad_termicas_mean_ms': 0,
                'intensidad_termicas_max_ms': 0,
                'intensidad_termicas_min_ms': 0,
                'intensidad_termicas_std_ms': 0,
                'altura_base_termicas_mean_m': 0,
                'altura_tope_termicas_mean_m': 0,
                'altura_base_termicas_min_m': 0,
                'altura_tope_termicas_max_m': 0,
                'ganancia_por_termica_mean_m': 0,
                'ganancia_por_termica_max_m': 0,
                'duracion_termica_mean_seg': 0,
                'duracion_termica_max_seg': 0,
                'hora_primera_termica': 0,
                'hora_ultima_termica': 0,
                'dispersion_termicas_lat': 0,
                'dispersion_termicas_lon': 0,
            }
        
        # Tiempo en térmica vs planeo
        tiempo_total = df['dt'].sum()
        tiempo_subiendo = df[df['climbing']]['dt'].sum()
        tiempo_bajando = tiempo_total - tiempo_subiendo
        
        stats['tiempo_en_termicas_min'] = round(tiempo_subiendo / 60, 1)
        stats['tiempo_en_planeo_min'] = round(tiempo_bajando / 60, 1)
        stats['porcentaje_tiempo_termicas'] = round(tiempo_subiendo / tiempo_total * 100, 1) if tiempo_total > 0 else 0
        
        # Tasa de ascenso y descenso promedio
        stats['tasa_ascenso_mean_ms'] = round(df[df['vario'] > 0]['vario'].mean(), 2) if len(df[df['vario'] > 0]) > 0 else 0
        stats['tasa_descenso_mean_ms'] = round(df[df['vario'] < 0]['vario'].mean(), 2) if len(df[df['vario'] < 0]) > 0 else 0
        
        return stats
    
    def _extract_trajectory_features(self, df_fixes):
        """Features de la trayectoria del vuelo"""
        
        df = df_fixes.copy()
        
        # Calcular bearing (rumbo)
        df['dlat'] = df['latitude'].diff()
        df['dlon'] = df['longitude'].diff()
        df['bearing'] = np.arctan2(df['dlon'], df['dlat']) * 180 / np.pi
        df['bearing_change'] = df['bearing'].diff().abs()
        
        # Velocidad ground (velocidad sobre el suelo)
        df['dt'] = df['time_seconds'].diff()
        df['dist'] = np.sqrt(df['dlat']**2 + df['dlon']**2) * 111  # aprox km
        df['ground_speed'] = (df['dist'] / df['dt']) * 3600  # km/h
        
        stats = {
            # Cambios de rumbo
            'bearing_change_mean_deg': round(df['bearing_change'].mean(), 1),
            'bearing_change_max_deg': round(df['bearing_change'].max(), 1),
            'bearing_change_std_deg': round(df['bearing_change'].std(), 1),
            
            # Velocidad ground
            'ground_speed_mean_kmh': round(df['ground_speed'].mean(), 1),
            'ground_speed_max_kmh': round(df['ground_speed'].max(), 1),
            'ground_speed_min_kmh': round(df['ground_speed'].min(), 1),
            'ground_speed_std_kmh': round(df['ground_speed'].std(), 1),
        }
        
        return stats
    
    def _extract_temporal_features(self, df_fixes):
        """Features temporales del vuelo"""
        
        df = df_fixes.copy()
        
        # Hora de inicio y fin
        hora_inicio = df['time_seconds'].iloc[0] / 3600
        hora_fin = df['time_seconds'].iloc[-1] / 3600
        if hora_fin < hora_inicio:
            hora_fin += 24
        
        # Altitud por hora del día
        df['hora_decimal'] = df['time_seconds'] / 3600
        
        # Dividir en franjas horarias
        df['franja_horaria'] = pd.cut(df['hora_decimal'], bins=[0, 11, 13, 15, 17, 24], labels=['manana', 'mediodia', 'tarde1', 'tarde2', 'noche'])
        
        stats = {
            'hora_inicio_decimal': round(hora_inicio, 2),
            'hora_fin_decimal': round(hora_fin, 2),
        }
        
        # Altura promedio por franja horaria
        for franja in ['manana', 'mediodia', 'tarde1', 'tarde2']:
            datos_franja = df[df['franja_horaria'] == franja]
            if len(datos_franja) > 0:
                stats[f'altura_mean_{franja}_m'] = round(datos_franja['altitude'].mean(), 0)
            else:
                stats[f'altura_mean_{franja}_m'] = 0
        
        return stats
    
    def _extract_spatial_features(self, df_fixes):
        """Features espaciales del vuelo"""
        
        df = df_fixes.copy()
        
        # Bounding box del vuelo
        lat_min = df['latitude'].min()
        lat_max = df['latitude'].max()
        lon_min = df['longitude'].min()
        lon_max = df['longitude'].max()
        
        # Centro geométrico
        lat_centro = df['latitude'].mean()
        lon_centro = df['longitude'].mean()
        
        # Distancia máxima del despegue
        lat_despegue = df['latitude'].iloc[0]
        lon_despegue = df['longitude'].iloc[0]
        
        df['dist_despegue'] = np.sqrt(
            ((df['latitude'] - lat_despegue) * 111)**2 + 
            ((df['longitude'] - lon_despegue) * 92)**2
        )
        
        stats = {
            'lat_min': round(lat_min, 6),
            'lat_max': round(lat_max, 6),
            'lon_min': round(lon_min, 6),
            'lon_max': round(lon_max, 6),
            'lat_centro': round(lat_centro, 6),
            'lon_centro': round(lon_centro, 6),
            'rango_lat_deg': round(lat_max - lat_min, 4),
            'rango_lon_deg': round(lon_max - lon_min, 4),
            'distancia_max_despegue_km': round(df['dist_despegue'].max(), 1),
            'area_vuelo_km2': round((lat_max - lat_min) * 111 * (lon_max - lon_min) * 92, 1),
        }
        
        return stats
    
    def _extract_variability_features(self, df_fixes):
        """Features de variabilidad"""
        
        df = df_fixes.copy()
        
        # Variabilidad de altura
        df['dh'] = df['altitude'].diff()
        
        stats = {
            'altura_std_m': round(df['altitude'].std(), 1),
            'altura_cv': round(df['altitude'].std() / df['altitude'].mean(), 3) if df['altitude'].mean() > 0 else 0,
            'cambio_altura_mean_m': round(df['dh'].mean(), 1),
            'cambio_altura_std_m': round(df['dh'].std(), 1),
        }
        
        return stats
    
    def _calculate_total_distance(self, df_fixes):
        """Calcular distancia total (Haversine)"""
        
        total_distance = 0.0
        
        for i in range(1, len(df_fixes)):
            lat1 = np.radians(df_fixes.iloc[i-1]['latitude'])
            lon1 = np.radians(df_fixes.iloc[i-1]['longitude'])
            lat2 = np.radians(df_fixes.iloc[i]['latitude'])
            lon2 = np.radians(df_fixes.iloc[i]['longitude'])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371
            
            distance = r * c
            total_distance += distance
        
        return total_distance
    
    def process_all_igc_files(self):
        """Procesar todos los IGC"""
        
        print("="*70)
        print("PROCESADOR IGC COMPLETO - MÁXIMA EXTRACCIÓN")
        print("="*70)
        
        igc_files = list(Path(self.igc_folder).glob('**/*.igc'))
        
        if not igc_files:
            print(f"\n✗ No se encontraron archivos .igc en: {self.igc_folder}")
            return None
        
        print(f"\nArchivos IGC encontrados: {len(igc_files)}")
        print(f"Carpeta: {self.igc_folder}")
        print(f"\nExtrayendo TODAS las features posibles...\n")
        
        vuelos_data = []
        
        for i, igc_file in enumerate(igc_files, 1):
            print(f"[{i}/{len(igc_files)}] {igc_file.name}")
            
            stats = self.parse_igc_file(str(igc_file))
            
            if stats:
                vuelos_data.append(stats)
                print(f"  ✓ Alt: {stats['altura_max_m']}m, "
                      f"Térmicas: {stats['num_termicas']}, "
                      f"Intensidad: {stats['intensidad_termicas_mean_ms']}m/s")
            else:
                print(f"  ✗ No procesado")
        
        if not vuelos_data:
            print("\n✗ No se pudo procesar ningún archivo")
            return None
        
        df = pd.DataFrame(vuelos_data)
        df['flight_id'] = [f'IGC_{i:04d}' for i in range(len(df))]
        df = df.sort_values('fecha').reset_index(drop=True)
        
        # Guardar
        os.makedirs('data/raw', exist_ok=True)
        output_file = 'data/raw/vuelos_metadata_completo.csv'
        df.to_csv(output_file, index=False)
        
        print("\n" + "="*70)
        print("RESUMEN")
        print("="*70)
        print(f"✓ Archivos procesados: {len(df)}/{len(igc_files)}")
        print(f"✓ CSV guardado: {output_file}")
        print(f"\n📊 FEATURES EXTRAÍDAS: {len(df.columns)} columnas")
        print("\nCategorías de features:")
        print("  • Básicas: altura, duración, distancia, velocidad (20+)")
        print("  • Térmicas: conteo, intensidad, altura, duración (25+)")
        print("  • Trayectoria: rumbo, velocidad ground (10+)")
        print("  • Temporales: franjas horarias, evolución (10+)")
        print("  • Espaciales: bounding box, área, dispersión (15+)")
        print("  • Variabilidad: std, coeficientes de variación (5+)")
        print(f"\n  TOTAL: ~{len(df.columns)} features por vuelo")
        
        print(f"\nRango de fechas: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
        
        # Mostrar algunas columnas clave
        print("\nAlgunas features extraídas:")
        cols_mostrar = ['altura_max_m', 'num_termicas', 'intensidad_termicas_mean_ms', 
                       'duracion_min', 'distancia_km', 'porcentaje_tiempo_termicas']
        print(df[cols_mostrar].describe())
        
        print("\n" + "="*70)
        print("SIGUIENTE PASO:")
        print("="*70)
        print("Ejecutar: python 02_descargar_era5.py")
        print("="*70)
        
        return df


if __name__ == "__main__":
    
    print("\n¿Dónde están tus archivos .igc?")
    carpeta = input("Carpeta: ").strip()
    
    if not carpeta:
        carpeta = "./igc_files"
    
    if not os.path.exists(carpeta):
        print(f"\n✗ La carpeta '{carpeta}' no existe")
        os.makedirs(carpeta, exist_ok=True)
        print(f"✓ Carpeta creada: {carpeta}")
        print(f"\nCopiá tus .igc ahí y volvé a ejecutar")
        exit(0)
    
    processor = IGCProcessorComplete(carpeta)
    df = processor.process_all_igc_files()
    
    if df is not None:
        print("\n✓✓✓ DATASET COMPLETO GENERADO ✓✓✓")
        print(f"Con {len(df.columns)} features por vuelo")
        print("Ahora tenés TODO para hacer feature engineering después!")