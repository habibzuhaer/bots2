# bots2/engine/levels.py
import pandas as pd
import numpy as np

class LevelCalculator:
    """Вычисляет уровни поддержки и сопротивления по свечам."""
    
    def calculate_support_resistance(self, df: pd.DataFrame, 
                                     window: int = 20, 
                                     pivot_strength: int = 3):
        highs = df['high']
        lows = df['low']
        
        # Находим пивоты (локальные максимумы и минимумы)
        resistances = []
        supports = []
        
        for i in range(window, len(df)-window):
            if highs.iloc[i] == highs.iloc[i-window:i+window].max():
                resistances.append(highs.iloc[i])
            if lows.iloc[i] == lows.iloc[i-window:i+window].min():
                supports.append(lows.iloc[i])
        
        # Группируем близкие уровни (кластеризация)
        cluster_threshold = (highs.max() - lows.min()) * 0.005  # 0.5% от диапазона
        
        def cluster_levels(levels):
            if not levels:
                return []
            levels_sorted = sorted(levels)
            clusters = []
            current_cluster = [levels_sorted[0]]
            
            for price in levels_sorted[1:]:
                if price - current_cluster[-1] <= cluster_threshold:
                    current_cluster.append(price)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [price]
            if current_cluster:
                clusters.append(np.mean(current_cluster))
            return clusters
        
        return {
            'resistances': sorted(cluster_levels(resistances), reverse=True),
            'supports': sorted(cluster_levels(supports))
        }