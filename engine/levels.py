#!/usr/bin/env python3
"""
Модуль расчета уровней поддержки и сопротивления.
Использует алгоритмы: кластеризация, volume profile, фибоначчи.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats, signal
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Level:
    """Структура для хранения информации об уровне."""
    price: float
    strength: float  # 0-1
    type: str  # 'support' или 'resistance'
    touches: int  # количество касаний
    volume: float  # объем на уровне
    timeframe: str  # таймфрейм
    broken: bool = False  # был ли уровень пробит

class LevelCalculator:
    """
    Продвинутый калькулятор уровней поддержки и сопротивления.
    Комбинирует несколько методов для точного определения.
    """
    
    def __init__(self, 
                 cluster_threshold: float = 0.005,  # 0.5% для кластеризации
                 min_touches: int = 2,  # минимальное количество касаний
                 volume_weight: float = 0.3,  # вес объема в силе уровня
                 time_weight: float = 0.7):  # вес времени в силе уровня
        
        self.cluster_threshold = cluster_threshold
        self.min_touches = min_touches
        self.volume_weight = volume_weight
        self.time_weight = time_weight
        
        # Кеш расчетов
        self.level_cache = {}
    
    def calculate(self, df: pd.DataFrame, timeframe: str = '1h') -> Dict[str, List[Level]]:
        """
        Основной метод расчета уровней.
        Возвращает словарь с поддержками и сопротивлениями.
        """
        if df.empty:
            return {'supports': [], 'resistances': []}
        
        cache_key = f"{timeframe}_{len(df)}_{df.index[-1].timestamp()}"
        if cache_key in self.level_cache:
            return self.level_cache[cache_key]
        
        logger.info(f"🧮 Расчет уровней для {timeframe} ({len(df)} свечей)")
        
        # 1. Базовые уровни по экстремумам
        basic_levels = self._calculate_basic_levels(df, timeframe)
        
        # 2. Уровни по Volume Profile
        volume_levels = self._calculate_volume_levels(df, timeframe)
        
        # 3. Уровни по скользящим средним
        ma_levels = self._calculate_ma_levels(df, timeframe)
        
        # 4. Уровни Фибоначчи
        fibo_levels = self._calculate_fibonacci_levels(df, timeframe)
        
        # 5. Объединение и кластеризация всех уровней
        all_supports = (
            basic_levels['supports'] + 
            volume_levels['supports'] + 
            ma_levels['supports'] + 
            fibo_levels['supports']
        )
        
        all_resistances = (
            basic_levels['resistances'] + 
            volume_levels['resistances'] + 
            ma_levels['resistances'] + 
            fibo_levels['resistances']
        )
        
        # Кластеризуем уровни
        clustered_supports = self._cluster_levels(all_supports, 'support', timeframe)
        clustered_resistances = self._cluster_levels(all_resistances, 'resistance', timeframe)
        
        # Фильтруем слабые уровни
        strong_supports = [lvl for lvl in clustered_supports if lvl.strength >= 0.5]
        strong_resistances = [lvl for lvl in clustered_resistances if lvl.strength >= 0.5]
        
        # Сортируем по цене
        strong_supports.sort(key=lambda x: x.price)
        strong_resistances.sort(key=lambda x: x.price, reverse=True)
        
        result = {
            'supports': strong_supports[:10],  # топ-10 поддержек
            'resistances': strong_resistances[:10]  # топ-10 сопротивлений
        }
        
        # Кешируем результат
        self.level_cache[cache_key] = result
        
        logger.info(f"✅ Найдено {len(strong_supports)} поддержек, {len(strong_resistances)} сопротивлений")
        
        return result
    
    def _calculate_basic_levels(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Рассчитывает базовые уровни по экстремумам."""
        supports = []
        resistances = []
        
        window = 20  # окно для поиска экстремумов
        highs = df['high'].values
        lows = df['low'].values
        
        # Ищем локальные максимумы
        for i in range(window, len(highs) - window):
            if highs[i] == np.max(highs[i-window:i+window+1]):
                # Рассчитываем силу уровня
                touches = self._count_touches(highs[i], df, 'resistance')
                strength = self._calculate_strength(touches, df, highs[i])
                
                level = Level(
                    price=float(highs[i]),
                    strength=strength,
                    type='resistance',
                    touches=touches,
                    volume=df.iloc[i]['volume'],
                    timeframe=timeframe
                )
                resistances.append(level)
        
        # Ищем локальные минимумы
        for i in range(window, len(lows) - window):
            if lows[i] == np.min(lows[i-window:i+window+1]):
                touches = self._count_touches(lows[i], df, 'support')
                strength = self._calculate_strength(touches, df, lows[i])
                
                level = Level(
                    price=float(lows[i]),
                    strength=strength,
                    type='support',
                    touches=touches,
                    volume=df.iloc[i]['volume'],
                    timeframe=timeframe
                )
                supports.append(level)
        
        return {'supports': supports, 'resistances': resistances}
    
    def _calculate_volume_levels(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Рассчитывает уровни на основе Volume Profile."""
        supports = []
        resistances = []
        
        if len(df) < 50:
            return {'supports': supports, 'resistances': resistances}
        
        # Разбиваем ценовой диапазон на уровни
        price_range = np.linspace(df['low'].min(), df['high'].max(), 100)
        
        # Считаем объем на каждом уровне
        volume_at_price = np.zeros(len(price_range))
        
        for _, row in df.iterrows():
            # Распределяем объем между low и high свечи
            low_idx = np.searchsorted(price_range, row['low'])
            high_idx = np.searchsorted(price_range, row['high'])
            
            if high_idx > low_idx:
                volume_per_level = row['volume'] / (high_idx - low_idx)
                volume_at_price[low_idx:high_idx] += volume_per_level
        
        # Ищем пики объема
        peaks, properties = signal.find_peaks(volume_at_price, 
                                            height=np.mean(volume_at_price) * 1.5,
                                            distance=10)
        
        for peak in peaks:
            price = price_range[peak]
            volume = volume_at_price[peak]
            
            # Определяем тип уровня
            current_price = df['close'].iloc[-1]
            level_type = 'support' if price < current_price else 'resistance'
            
            touches = self._count_touches(price, df, level_type)
            strength = min(volume / np.max(volume_at_price), 1.0)
            
            level = Level(
                price=float(price),
                strength=strength,
                type=level_type,
                touches=touches,
                volume=float(volume),
                timeframe=timeframe
            )
            
            if level_type == 'support':
                supports.append(level)
            else:
                resistances.append(level)
        
        return {'supports': supports, 'resistances': resistances}
    
    def _calculate_ma_levels(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Рассчитывает уровни на основе скользящих средних."""
        supports = []
        resistances = []
        
        # Используем основные MA
        ma_periods = [20, 50, 100, 200]
        
        for period in ma_periods:
            ma = df['close'].rolling(window=period).mean().iloc[-1]
            
            if not np.isnan(ma):
                # Проверяем, действует ли MA как поддержка/сопротивление
                price_diff_pct = abs(df['close'].iloc[-1] - ma) / ma
                
                if price_diff_pct < 0.02:  # В пределах 2%
                    touches = self._count_touches(ma, df, 'dynamic')
                    strength = 0.7  # MA имеют высокую силу
                    
                    # Определяем тип по позиции относительно цены
                    level_type = 'support' if ma < df['close'].iloc[-1] else 'resistance'
                    
                    level = Level(
                        price=float(ma),
                        strength=strength,
                        type=level_type,
                        touches=touches,
                        volume=0,
                        timeframe=f"MA{period}"
                    )
                    
                    if level_type == 'support':
                        supports.append(level)
                    else:
                        resistances.append(level)
        
        return {'supports': supports, 'resistances': resistances}
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Рассчитывает уровни Фибоначчи."""
        supports = []
        resistances = []
        
        if len(df) < 100:
            return {'supports': supports, 'resistances': resistances}
        
        # Находим максимум и минимум за период
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        
        # Основные уровни Фибоначчи
        fibo_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        
        for level in fibo_levels:
            price = high - (diff * level)
            
            # Определяем тип уровня
            current_price = df['close'].iloc[-1]
            level_type = 'support' if price < current_price else 'resistance'
            
            touches = self._count_touches(price, df, level_type)
            strength = 0.8 if level in [0.382, 0.5, 0.618] else 0.6
            
            fibo_level = Level(
                price=float(price),
                strength=strength,
                type=level_type,
                touches=touches,
                volume=0,
                timeframe="FIBO"
            )
            
            if level_type == 'support':
                supports.append(fibo_level)
            else:
                resistances.append(fibo_level)
        
        return {'supports': supports, 'resistances': resistances}
    
    def _count_touches(self, price: float, df: pd.DataFrame, level_type: str) -> int:
        """Считает количество касаний уровня."""
        threshold = price * 0.005  # 0.5%
        touches = 0
        
        for _, row in df.iterrows():
            if level_type == 'support':
                if abs(row['low'] - price) < threshold:
                    touches += 1
            elif level_type == 'resistance':
                if abs(row['high'] - price) < threshold:
                    touches += 1
            else:  # dynamic
                if abs(row['close'] - price) < threshold:
                    touches += 1
        
        return touches
    
    def _calculate_strength(self, touches: int, df: pd.DataFrame, price: float) -> float:
        """Рассчитывает силу уровня от 0 до 1."""
        # Базовый вес от количества касаний
        touch_strength = min(touches / 10, 1.0)
        
        # Вес от времени (новые уровни слабее)
        time_factor = 0.5 if len(df) < 100 else 0.7
        
        # Вес от объема
        volume_idx = np.argmin(abs(df['close'] - price))
        volume = df.iloc[volume_idx]['volume'] if volume_idx < len(df) else 0
        avg_volume = df['volume'].mean()
        volume_strength = min(volume / (avg_volume * 2), 1.0)
        
        # Итоговая сила
        strength = (
            touch_strength * 0.4 +  # 40% от касаний
            time_factor * 0.3 +     # 30% от времени
            volume_strength * 0.3    # 30% от объема
        )
        
        return round(strength, 2)
    
    def _cluster_levels(self, levels: List[Level], level_type: str, timeframe: str) -> List[Level]:
        """Кластеризует близкие уровни."""
        if not levels:
            return []
        
        # Сортируем по цене
        levels.sort(key=lambda x: x.price)
        
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            # Проверяем, попадает ли уровень в текущий кластер
            price_diff = abs(level.price - current_cluster[-1].price)
            price_diff_pct = price_diff / current_cluster[-1].price
            
            if price_diff_pct <= self.cluster_threshold:
                current_cluster.append(level)
            else:
                # Создаем объединенный уровень из кластера
                clusters.append(self._merge_cluster(current_cluster, level_type, timeframe))
                current_cluster = [level]
        
        # Добавляем последний кластер
        if current_cluster:
            clusters.append(self._merge_cluster(current_cluster, level_type, timeframe))
        
        return clusters
    
    def _merge_cluster(self, cluster: List[Level], level_type: str, timeframe: str) -> Level:
        """Объединяет уровни в кластере в один уровень."""
        if len(cluster) == 1:
            return cluster[0]
        
        # Средневзвешенная цена
        total_strength = sum(lvl.strength for lvl in cluster)
        weighted_price = sum(lvl.price * lvl.strength for lvl in cluster) / total_strength
        
        # Суммируем касания и объем
        total_touches = sum(lvl.touches for lvl in cluster)
        total_volume = sum(lvl.volume for lvl in cluster)
        
        # Средняя сила
        avg_strength = total_strength / len(cluster)
        
        return Level(
            price=round(weighted_price, 2),
            strength=min(avg_strength * 1.2, 1.0),  # Усиливаем объединенный уровень
            type=level_type,
            touches=total_touches,
            volume=total_volume,
            timeframe=timeframe
        )
    
    def visualize_levels(self, df: pd.DataFrame, levels: Dict, save_path: str = None):
        """
        Визуализирует уровни на графике.
        Требует установки matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Рисуем свечи
            ax.plot(df.index, df['close'], label='Close Price', alpha=0.7)
            
            # Рисуем поддержки
            for level in levels['supports']:
                ax.axhline(y=level.price, color='green', alpha=level.strength, 
                          linestyle='--', label='Support' if level == levels['supports'][0] else "")
            
            # Рисуем сопротивления
            for level in levels['resistances']:
                ax.axhline(y=level.price, color='red', alpha=level.strength, 
                          linestyle='--', label='Resistance' if level == levels['resistances'][0] else "")
            
            ax.set_title('Support and Resistance Levels')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Форматируем оси времени
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 График сохранен: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib не установлен. Визуализация недоступна.")
    
    def clear_cache(self):
        """Очищает кеш уровней."""
        self.level_cache.clear()
        logger.info("🧹 Кеш уровней очищен")

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    # Создаем тестовые данные
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    prices = 50000 + np.cumsum(np.random.randn(500) * 100)
    
    df = pd.DataFrame({
        'open': prices - np.random.rand(500) * 100,
        'high': prices + np.random.rand(500) * 150,
        'low': prices - np.random.rand(500) * 150,
        'close': prices,
        'volume': np.random.rand(500) * 1000 + 500
    }, index=dates)
    
    # Тестируем калькулятор
    calculator = LevelCalculator()
    levels = calculator.calculate(df, '1h')
    
    print(f"✅ Поддержки: {len(levels['supports'])}")
    for support in levels['supports'][:3]:
        print(f"   ${support.price:.2f} (сила: {support.strength}, касаний: {support.touches})")
    
    print(f"\n✅ Сопротивления: {len(levels['resistances'])}")
    for resistance in levels['resistances'][:3]:
        print(f"   ${resistance.price:.2f} (сила: {resistance.strength}, касаний: {resistance.touches})")
    
    # Пробуем визуализацию
    try:
        calculator.visualize_levels(df.tail(100), levels)
    except:
        print("\n⚠️  Matplotlib не установлен, пропускаем визуализацию")