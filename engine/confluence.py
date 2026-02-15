#!/usr/bin/env python3
"""
ПОЛНЫЙ МОДУЛЬ АНАЛИЗА КОНФЛЮЭНСА
Версия: 2.0
Алгоритмы: Мульти-таймфрейм анализ, Согласованность сигналов, Взвешенная оценка
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import traceback
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ============================================================================
# ОПРЕДЕЛЕНИЯ КЛАССОВ
# ============================================================================

class ConfluenceLevel(Enum):
    """Уровни конфлюэнса."""
    VERY_WEAK = "very_weak"        # 0-20
    WEAK = "weak"                  # 20-40
    MEDIUM = "medium"              # 40-60
    STRONG = "strong"              # 60-80
    VERY_STRONG = "very_strong"    # 80-100

class ConfluenceFactor(Enum):
    """Факторы, влияющие на конфлюэнс."""
    MULTI_TIMEFRAME = "multi_timeframe"      # Мульти-таймфрейм согласованность
    VOLUME_PROFILE = "volume_profile"        # Объемный профиль
    TECHNICAL_INDICATORS = "technical"       # Технические индикаторы
    PATTERN_RECOGNITION = "patterns"         # Графические паттерны
    PRICE_ACTION = "price_action"            # Price Action
    MARKET_STRUCTURE = "market_structure"    # Структура рынка
    SENTIMENT = "sentiment"                  # Рыночные настроения

@dataclass
class ConfluenceScore:
    """Структура оценки конфлюэнса."""
    total_score: float  # 0-100
    level: ConfluenceLevel
    factors: Dict[ConfluenceFactor, float]
    timeframes: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Конвертирует в словарь."""
        return {
            'total_score': self.total_score,
            'level': self.level.value,
            'factors': {k.value: v for k, v in self.factors.items()},
            'timeframes': self.timeframes,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ConfluenceAnalysisResult:
    """Результат анализа конфлюэнса."""
    symbol: str
    primary_timeframe: str
    timestamp: datetime
    confluence_score: ConfluenceScore
    aligned_levels: Dict[str, List[float]]
    best_entries: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Конвертирует в словарь."""
        return {
            'symbol': self.symbol,
            'primary_timeframe': self.primary_timeframe,
            'timestamp': self.timestamp.isoformat(),
            'confluence_score': self.confluence_score.to_dict(),
            'aligned_levels': self.aligned_levels,
            'best_entries': self.best_entries,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }

# ============================================================================
# ОСНОВНОЙ КЛАСС CONFLUENCE CALCULATOR
# ============================================================================

class ConfluenceCalculator:
    """
    Продвинутый калькулятор конфлюэнса.
    
    Анализирует согласованность сигналов по нескольким измерениям:
    1. Мульти-таймфрейм анализ (1m, 5m, 15m, 1h, 4h, 1d, 1w)
    2. Различные типы индикаторов
    3. Объемный профиль
    4. Графические паттерны
    5. Структура рынка
    """
    
    VERSION = "2.0.0"
    
    def __init__(self,
                 min_timeframes: int = 2,
                 weight_timeframes: bool = True,
                 use_volume_profile: bool = True,
                 use_indicators: bool = True,
                 use_patterns: bool = True,
                 use_price_action: bool = True,
                 use_market_structure: bool = True,
                 alignment_threshold: float = 0.7):
        """
        Инициализация калькулятора конфлюэнса.
        
        Args:
            min_timeframes: Минимальное количество таймфреймов для конфлюэнса
            weight_timeframes: Взвешивать по старшинству таймфрейма
            use_volume_profile: Использовать объемный профиль
            use_indicators: Использовать технические индикаторы
            use_patterns: Использовать графические паттерны
            use_price_action: Использовать Price Action
            use_market_structure: Использовать структуру рынка
            alignment_threshold: Порог согласованности (0-1)
        """
        
        self.min_timeframes = min_timeframes
        self.weight_timeframes = weight_timeframes
        self.use_volume_profile = use_volume_profile
        self.use_indicators = use_indicators
        self.use_patterns = use_patterns
        self.use_price_action = use_price_action
        self.use_market_structure = use_market_structure
        self.alignment_threshold = alignment_threshold
        
        # Веса для разных таймфреймов
        self.timeframe_weights = {
            '1m': 1.0,
            '5m': 1.2,
            '15m': 1.5,
            '30m': 1.8,
            '1h': 2.0,
            '2h': 2.2,
            '4h': 2.5,
            '6h': 2.7,
            '8h': 2.8,
            '12h': 2.9,
            '1d': 3.0,
            '3d': 3.2,
            '1w': 3.5,
            '1M': 4.0
        }
        
        # Веса для разных факторов
        self.factor_weights = {
            ConfluenceFactor.MULTI_TIMEFRAME: 0.25,
            ConfluenceFactor.VOLUME_PROFILE: 0.15,
            ConfluenceFactor.TECHNICAL_INDICATORS: 0.20,
            ConfluenceFactor.PATTERN_RECOGNITION: 0.10,
            ConfluenceFactor.PRICE_ACTION: 0.15,
            ConfluenceFactor.MARKET_STRUCTURE: 0.10,
            ConfluenceFactor.SENTIMENT: 0.05
        }
        
        # Пороги для уровней конфлюэнса
        self.level_thresholds = {
            ConfluenceLevel.VERY_WEAK: (0, 20),
            ConfluenceLevel.WEAK: (20, 40),
            ConfluenceLevel.MEDIUM: (40, 60),
            ConfluenceLevel.STRONG: (60, 80),
            ConfluenceLevel.VERY_STRONG: (80, 100)
        }
        
        # Кеш расчетов
        self.cache = {}
        self.cache_max_size = 100
        
        # Статистика
        self.stats = {
            'total_analyses': 0,
            'cache_hits': 0,
            'average_scores': [],
            'errors': []
        }
        
        logger.info(f"✅ ConfluenceCalculator v{self.VERSION} инициализирован")
        logger.info(f"   Минимум таймфреймов: {min_timeframes}")
        logger.info(f"   Взвешивание ТФ: {weight_timeframes}")
        logger.info(f"   Порог согласованности: {alignment_threshold}")
    
    def analyze(self, data_frames: Dict[str, pd.DataFrame],
                symbol: str = "UNKNOWN",
                primary_timeframe: str = "1h") -> ConfluenceAnalysisResult:
        """
        Основной метод анализа конфлюэнса.
        
        Args:
            data_frames: Словарь {таймфрейм: DataFrame} с данными
            symbol: Идентификатор символа
            primary_timeframe: Основной таймфрейм для анализа
            
        Returns:
            ConfluenceAnalysisResult с результатами анализа
        """
        
        # Проверка входных данных
        if not data_frames or len(data_frames) < self.min_timeframes:
            logger.warning(f"⚠️  Недостаточно таймфреймов: {len(data_frames)}, требуется {self.min_timeframes}")
            return self._create_empty_result(symbol, primary_timeframe)
        
        # Генерация ключа кеша
        cache_key = f"{symbol}_{len(data_frames)}_{max(len(df) for df in data_frames.values())}"
        
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            logger.debug(f"🎯 Кеш попадание для {symbol}")
            return self.cache[cache_key]
        
        self.stats['total_analyses'] += 1
        
        logger.info(f"🧮 Анализ конфлюэнса для {symbol} ({len(data_frames)} таймфреймов)")
        start_time = datetime.now()
        
        try:
            # 1. Расчет факторов конфлюэнса
            factor_scores = self._calculate_factor_scores(data_frames, primary_timeframe)
            
            # 2. Расчет таймфрейм согласованности
            timeframe_scores = self._calculate_timeframe_alignment(data_frames)
            
            # 3. Общий счет конфлюэнса
            total_score = self._calculate_total_score(factor_scores, timeframe_scores)
            
            # 4. Определение уровня конфлюэнса
            level = self._determine_confluence_level(total_score)
            
            # 5. Поиск согласованных уровней
            aligned_levels = self._find_aligned_levels(data_frames)
            
            # 6. Лучшие точки входа
            best_entries = self._find_best_entries(data_frames, aligned_levels)
            
            # 7. Предупреждения и рекомендации
            warnings = self._generate_warnings(factor_scores, timeframe_scores)
            recommendations = self._generate_recommendations(total_score, aligned_levels, best_entries)
            
            # 8. Формирование результата
            result = ConfluenceAnalysisResult(
                symbol=symbol,
                primary_timeframe=primary_timeframe,
                timestamp=datetime.now(),
                confluence_score=ConfluenceScore(
                    total_score=total_score,
                    level=level,
                    factors=factor_scores,
                    timeframes=timeframe_scores,
                    details={
                        'processing_time': (datetime.now() - start_time).total_seconds(),
                        'data_points': sum(len(df) for df in data_frames.values())
                    }
                ),
                aligned_levels=aligned_levels,
                best_entries=best_entries,
                warnings=warnings,
                recommendations=recommendations
            )
            
            # 9. Обновление статистики
            self.stats['average_scores'].append(total_score)
            
            # 10. Логирование результатов
            self._log_results(result)
            
            # 11. Сохранение в кеш
            self.cache[cache_key] = result
            self._clean_cache()
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Ошибка анализа конфлюэнса для {symbol}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.stats['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'error': str(e)
            })
            
            return self._create_empty_result(symbol, primary_timeframe)
    
    def _calculate_factor_scores(self, data_frames: Dict[str, pd.DataFrame],
                                primary_timeframe: str) -> Dict[ConfluenceFactor, float]:
        """Рассчитывает оценки по каждому фактору конфлюэнса."""
        factor_scores = {}
        
        # Получаем данные для основного таймфрейма
        primary_df = data_frames.get(primary_timeframe)
        if primary_df is None:
            primary_df = list(data_frames.values())[0]
        
        # 1. Мульти-таймфрейм фактор (рассчитывается отдельно)
        mtf_score = self._calculate_mtf_factor(data_frames, primary_df)
        factor_scores[ConfluenceFactor.MULTI_TIMEFRAME] = mtf_score
        
        # 2. Объемный профиль
        if self.use_volume_profile:
            volume_score = self._calculate_volume_profile_factor(primary_df, data_frames)
            factor_scores[ConfluenceFactor.VOLUME_PROFILE] = volume_score
        
        # 3. Технические индикаторы
        if self.use_indicators:
            technical_score = self._calculate_technical_factor(primary_df, data_frames)
            factor_scores[ConfluenceFactor.TECHNICAL_INDICATORS] = technical_score
        
        # 4. Графические паттерны
        if self.use_patterns:
            pattern_score = self._calculate_pattern_factor(primary_df, data_frames)
            factor_scores[ConfluenceFactor.PATTERN_RECOGNITION] = pattern_score
        
        # 5. Price Action
        if self.use_price_action:
            pa_score = self._calculate_price_action_factor(primary_df, data_frames)
            factor_scores[ConfluenceFactor.PRICE_ACTION] = pa_score
        
        # 6. Структура рынка
        if self.use_market_structure:
            structure_score = self._calculate_market_structure_factor(primary_df, data_frames)
            factor_scores[ConfluenceFactor.MARKET_STRUCTURE] = structure_score
        
        # 7. Сентимент (если доступен)
        sentiment_score = self._calculate_sentiment_factor()
        if sentiment_score > 0:
            factor_scores[ConfluenceFactor.SENTIMENT] = sentiment_score
        
        return factor_scores
    
    def _calculate_mtf_factor(self, data_frames: Dict[str, pd.DataFrame],
                             primary_df: pd.DataFrame) -> float:
        """Рассчитывает мульти-таймфрейм фактор."""
        if len(data_frames) < 2:
            return 0.0
        
        try:
            current_price = primary_df['close'].iloc[-1]
            alignments = []
            
            for tf, df in data_frames.items():
                if df is primary_df:
                    continue
                
                if len(df) < 20:
                    continue
                
                # Определяем тренд на каждом таймфрейме
                df_trend = self._determine_trend(df)
                
                # Определяем ключевые уровни
                df_support, df_resistance = self._find_key_levels(df)
                
                # Оцениваем согласованность с основным ТФ
                if df_trend == self._determine_trend(primary_df):
                    # Тренды совпадают
                    alignments.append(1.0)
                else:
                    # Тренды расходятся
                    alignments.append(0.0)
                
                # Проверяем близость к уровням
                price_to_support = abs(current_price - df_support) / current_price if df_support else 1.0
                price_to_resistance = abs(df_resistance - current_price) / current_price if df_resistance else 1.0
                
                if price_to_support < 0.01 or price_to_resistance < 0.01:
                    # Цена близка к уровню на другом ТФ
                    alignments[-1] += 0.5
            
            if not alignments:
                return 0.0
            
            # Средняя согласованность с весами
            avg_alignment = np.mean(alignments)
            
            # Нормализуем к 0-100
            mtf_score = min(avg_alignment * 70, 100)
            
            # Бонус за количество таймфреймов
            timeframe_bonus = min(len(data_frames) * 5, 30)
            mtf_score = min(mtf_score + timeframe_bonus, 100)
            
            return mtf_score
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета MTF фактора: {e}")
            return 0.0
    
    def _calculate_volume_profile_factor(self, primary_df: pd.DataFrame,
                                        data_frames: Dict[str, pd.DataFrame]) -> float:
        """Рассчитывает фактор объемного профиля."""
        try:
            if len(primary_df) < 50:
                return 0.0
            
            current_price = primary_df['close'].iloc[-1]
            
            # Создаем Volume Profile для основного ТФ
            volume_levels = self._create_volume_profile(primary_df)
            
            if not volume_levels:
                return 0.0
            
            # Находим POC (Point of Control)
            poc_price = max(volume_levels, key=volume_levels.get)
            poc_volume = volume_levels[poc_price]
            
            # Оцениваем положение цены относительно POC
            price_distance = abs(current_price - poc_price) / current_price
            
            # Чем ближе цена к POC, тем выше конфлюэнс
            if price_distance < 0.01:
                poc_score = 90
            elif price_distance < 0.02:
                poc_score = 70
            elif price_distance < 0.03:
                poc_score = 50
            elif price_distance < 0.05:
                poc_score = 30
            else:
                poc_score = 10
            
            # Проверяем объемную поддержку на других ТФ
            volume_consensus = 0
            for tf, df in data_frames.items():
                if df is primary_df:
                    continue
                
                if len(df) < 30:
                    continue
                
                tf_volume_levels = self._create_volume_profile(df)
                if tf_volume_levels:
                    # Проверяем, есть ли пик объема около текущей цены
                    for price, volume in tf_volume_levels.items():
                        if abs(price - current_price) / current_price < 0.01:
                            volume_consensus += 1
                            break
            
            # Бонус за согласованность по объемам
            consensus_bonus = min(volume_consensus * 10, 30)
            
            total_score = poc_score + consensus_bonus
            
            return min(total_score, 100)
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета Volume Profile фактора: {e}")
            return 0.0
    
    def _create_volume_profile(self, df: pd.DataFrame, num_levels: int = 50) -> Dict[float, float]:
        """Создает объемный профиль для DataFrame."""
        volume_profile = {}
        
        try:
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_step = (price_max - price_min) / num_levels
            
            for i in range(num_levels):
                level_low = price_min + i * price_step
                level_high = level_low + price_step
                level_mid = (level_low + level_high) / 2
                
                volume_at_level = 0
                
                for _, row in df.iterrows():
                    # Проверяем, перекрывается ли свеча с уровнем
                    if row['high'] >= level_low and row['low'] <= level_high:
                        # Пропорциональное распределение объема
                        overlap = min(row['high'], level_high) - max(row['low'], level_low)
                        candle_range = row['high'] - row['low']
                        if candle_range > 0:
                            volume_at_level += row['volume'] * (overlap / candle_range)
                
                if volume_at_level > 0:
                    volume_profile[level_mid] = volume_at_level
            
            return volume_profile
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания Volume Profile: {e}")
            return {}
    
    def _calculate_technical_factor(self, primary_df: pd.DataFrame,
                                   data_frames: Dict[str, pd.DataFrame]) -> float:
        """Рассчитывает фактор технических индикаторов."""
        try:
            if len(primary_df) < 50:
                return 0.0
            
            current_price = primary_df['close'].iloc[-1]
            
            # Рассчитываем индикаторы для основного ТФ
            rsi = self._calculate_rsi(primary_df['close'])
            macd, signal = self._calculate_macd(primary_df['close'])
            bb_upper, bb_lower = self._calculate_bollinger_bands(primary_df['close'])
            
            # Оценка RSI
            rsi_score = 0
            if rsi < 30:
                rsi_score = 80  # Перепроданность, потенциал роста
            elif rsi > 70:
                rsi_score = 80  # Перекупленность, потенциал падения
            elif 40 <= rsi <= 60:
                rsi_score = 50  # Нейтрально
            
            # Оценка MACD
            macd_score = 0
            if macd > signal:
                macd_score = 70  # Бычий сигнал
            elif macd < signal:
                macd_score = 70  # Медвежий сигнал
            else:
                macd_score = 50  # Нейтрально
            
            # Оценка Bollinger Bands
            bb_score = 0
            if current_price <= bb_lower:
                bb_score = 80  # Нижняя полоса
            elif current_price >= bb_upper:
                bb_score = 80  # Верхняя полоса
            elif bb_lower < current_price < bb_upper:
                bb_score = 50  # Внутри канала
            
            # Усредняем с весами
            technical_score = (rsi_score * 0.3 + macd_score * 0.4 + bb_score * 0.3)
            
            return technical_score
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета Technical фактора: {e}")
            return 0.0
    
    def _calculate_pattern_factor(self, primary_df: pd.DataFrame,
                                 data_frames: Dict[str, pd.DataFrame]) -> float:
        """Рассчитывает фактор графических паттернов."""
        try:
            if len(primary_df) < 50:
                return 0.0
            
            # Поиск паттернов на основном ТФ
            patterns_primary = self._detect_patterns(primary_df)
            
            # Поиск паттернов на других ТФ
            patterns_total = len(patterns_primary)
            
            for tf, df in data_frames.items():
                if tf == "1h":  # Уже обработали
                    continue
                
                if len(df) >= 50:
                    patterns_other = self._detect_patterns(df)
                    patterns_total += len(patterns_other)
            
            # Чем больше паттернов, тем выше конфлюэнс
            # Но не более 10 паттернов для максимального счета
            pattern_score = min(patterns_total * 10, 100)
            
            return pattern_score
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета Pattern фактора: {e}")
            return 0.0
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """Обнаруживает графические паттерны в DataFrame."""
        patterns = []
        
        try:
            closes = df['close'].values[-30:]
            highs = df['high'].values[-30:]
            lows = df['low'].values[-30:]
            
            # Проверка Doji
            if len(closes) >= 1:
                body = abs(closes[-1] - df['open'].iloc[-1])
                range_price = highs[-1] - lows[-1]
                if range_price > 0 and body / range_price < 0.1:
                    patterns.append("doji")
            
            # Проверка Hammer/Shooting Star
            if len(closes) >= 2:
                body = abs(closes[-1] - df['open'].iloc[-1])
                lower_shadow = min(df['open'].iloc[-1], closes[-1]) - lows[-1]
                upper_shadow = highs[-1] - max(df['open'].iloc[-1], closes[-1])
                
                if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                    patterns.append("hammer")
                elif upper_shadow > body * 2 and lower_shadow < body * 0.5:
                    patterns.append("shooting_star")
            
            # Проверка Engulfing
            if len(closes) >= 2:
                if closes[-2] < df['open'].iloc[-2] and closes[-1] > df['open'].iloc[-1]:
                    if closes[-1] > df['open'].iloc[-2] and df['open'].iloc[-1] < closes[-2]:
                        patterns.append("bullish_engulfing")
                
                if closes[-2] > df['open'].iloc[-2] and closes[-1] < df['open'].iloc[-1]:
                    if closes[-1] < df['open'].iloc[-2] and df['open'].iloc[-1] > closes[-2]:
                        patterns.append("bearish_engulfing")
            
            # Простые трендовые паттерны
            if len(closes) >= 10:
                # Восходящий тренд
                if closes[-1] > closes[-5] > closes[-10]:
                    patterns.append("uptrend")
                
                # Нисходящий тренд
                if closes[-1] < closes[-5] < closes[-10]:
                    patterns.append("downtrend")
                
                # Флэт
                high_10 = max(highs[-10:])
                low_10 = min(lows[-10:])
                if (high_10 - low_10) / low_10 < 0.05:
                    patterns.append("consolidation")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обнаружения паттернов: {e}")
        
        return patterns
    
    def _calculate_price_action_factor(self, primary_df: pd.DataFrame,
                                      data_frames: Dict[str, pd.DataFrame]) -> float:
        """Рассчитывает фактор Price Action."""
        try:
            if len(primary_df) < 20:
                return 0.0
            
            current_price = primary_df['close'].iloc[-1]
            
            # Анализ свечных формаций на основном ТФ
            candle_score = 0
            
            # Последние 5 свечей
            for i in range(1, 6):
                if len(primary_df) <= i:
                    break
                
                idx = -i
                
                # Бычья свеча
                if primary_df['close'].iloc[idx] > primary_df['open'].iloc[idx]:
                    candle_score += 2
                
                # Медвежья свеча
                else:
                    candle_score -= 2
                
                # Длинная свеча (большой диапазон)
                candle_range = (primary_df['high'].iloc[idx] - primary_df['low'].iloc[idx]) / current_price
                if candle_range > 0.02:
                    if primary_df['close'].iloc[idx] > primary_df['open'].iloc[idx]:
                        candle_score += 3  # Сильная бычья свеча
                    else:
                        candle_score -= 3  # Сильная медвежья свеча
            
            # Нормализуем счет
            max_possible_score = 5 * 5  # 5 свечей * максимальный балл 5
            candle_score_normalized = (candle_score + max_possible_score) / (2 * max_possible_score) * 100
            
            # Проверка на разрывы (гэпы)
            gap_score = 0
            if len(primary_df) >= 2:
                prev_close = primary_df['close'].iloc[-2]
                current_open = primary_df['open'].iloc[-1]
                
                gap_up = (current_open - prev_close) / prev_close
                gap_down = (prev_close - current_open) / prev_close
                
                if gap_up > 0.01:
                    gap_score = 70  # Гэп вверх
                elif gap_down > 0.01:
                    gap_score = 70  # Гэп вниз
                else:
                    gap_score = 50  # Без гэпа
            
            # Усредняем
            pa_score = (candle_score_normalized * 0.7 + gap_score * 0.3)
            
            return pa_score
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета Price Action фактора: {e}")
            return 0.0
    
    def _calculate_market_structure_factor(self, primary_df: pd.DataFrame,
                                          data_frames: Dict[str, pd.DataFrame]) -> float:
        """Рассчитывает фактор структуры рынка."""
        try:
            if len(primary_df) < 50:
                return 0.0
            
            # Определяем структуру на основном ТФ
            structure_score = 0
            
            # Определяем тренд
            trend = self._determine_trend(primary_df)
            
            # Определяем ключевые уровни
            support, resistance = self._find_key_levels(primary_df)
            
            current_price = primary_df['close'].iloc[-1]
            
            # Оценка структуры
            if trend in ["strong_bullish", "bullish"]:
                structure_score += 40
            elif trend in ["strong_bearish", "bearish"]:
                structure_score += 40
            else:
                structure_score += 20
            
            # Оценка положения относительно уровней
            if support and current_price > support:
                distance_to_support = (current_price - support) / current_price
                if distance_to_support < 0.02:
                    structure_score += 30  # У поддержки
                elif distance_to_support < 0.05:
                    structure_score += 20  # Близко к поддержке
            
            if resistance and current_price < resistance:
                distance_to_resistance = (resistance - current_price) / current_price
                if distance_to_resistance < 0.02:
                    structure_score += 30  # У сопротивления
                elif distance_to_resistance < 0.05:
                    structure_score += 20  # Близко к сопротивлению
            
            # Проверяем структуру на других ТФ
            tf_agreement = 0
            for tf, df in data_frames.items():
                if df is primary_df:
                    continue
                
                if len(df) >= 50:
                    tf_trend = self._determine_trend(df)
                    if (trend in ["bullish", "strong_bullish"] and 
                        tf_trend in ["bullish", "strong_bullish"]):
                        tf_agreement += 1
                    elif (trend in ["bearish", "strong_bearish"] and 
                          tf_trend in ["bearish", "strong_bearish"]):
                        tf_agreement += 1
                    elif trend == "ranging" and tf_trend == "ranging":
                        tf_agreement += 1
            
            structure_score += min(tf_agreement * 10, 30)
            
            return min(structure_score, 100)
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета Market Structure фактора: {e}")
            return 0.0
    
    def _calculate_sentiment_factor(self) -> float:
        """Рассчитывает фактор рыночных настроений."""
        # В реальной системе здесь будет интеграция с API сентимента
        # Пока возвращаем нейтральное значение
        return 50.0
    
    def _calculate_timeframe_alignment(self, data_frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Рассчитывает согласованность по таймфреймам."""
        timeframe_scores = {}
        
        try:
            for tf, df in data_frames.items():
                if len(df) < 20:
                    timeframe_scores[tf] = 0.0
                    continue
                
                # Оцениваем качество тренда на каждом ТФ
                trend_strength = self._calculate_trend_strength(df)
                
                # Оцениваем волатильность
                volatility = self._calculate_volatility(df)
                
                # Оцениваем четкость уровней
                level_clarity = self._calculate_level_clarity(df)
                
                # Комбинированная оценка
                tf_score = (trend_strength * 0.4 + volatility * 0.3 + level_clarity * 0.3)
                
                # Применяем вес таймфрейма
                if self.weight_timeframes:
                    weight = self.timeframe_weights.get(tf, 1.0)
                    tf_score = min(tf_score * weight / 2, 100)
                
                timeframe_scores[tf] = tf_score
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета таймфрейм согласованности: {e}")
        
        return timeframe_scores
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Рассчитывает силу тренда."""
        try:
            closes = df['close'].values[-20:]
            
            if len(closes) < 5:
                return 50.0
            
            # Линейная регрессия для определения наклона
            x = np.arange(len(closes))
            slope, _, r_value, _, _ = np.polyfit(x, closes, 1, full=False)
            
            # Нормализуем наклон
            slope_normalized = abs(slope) / np.mean(closes) * 1000
            
            # R-squared показывает силу тренда
            r_squared = r_value ** 2 if r_value is not None else 0
            
            # Комбинируем
            strength = (min(slope_normalized, 50) * 0.5 + r_squared * 50)
            
            return min(strength, 100)
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета силы тренда: {e}")
            return 50.0
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Рассчитывает волатильность."""
        try:
            closes = df['close'].values[-20:]
            
            if len(closes) < 5:
                return 50.0
            
            # Стандартное отклонение доходностей
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * 100  # В процентах
            
            # Нормализуем к 0-100
            # Высокая волатильность = 100, низкая = 0
            volatility_score = min(volatility * 10, 100)
            
            return volatility_score
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета волатильности: {e}")
            return 50.0
    
    def _calculate_level_clarity(self, df: pd.DataFrame) -> float:
        """Рассчитывает четкость уровней."""
        try:
            if len(df) < 30:
                return 50.0
            
            highs = df['high'].values[-30:]
            lows = df['low'].values[-30:]
            
            # Ищем повторяющиеся уровни
            level_counts = defaultdict(int)
            
            for high in highs:
                # Округляем до 2 знаков
                level = round(high, 2)
                level_counts[level] += 1
            
            for low in lows:
                level = round(low, 2)
                level_counts[level] += 1
            
            if not level_counts:
                return 50.0
            
            # Находим максимальное количество касаний
            max_touches = max(level_counts.values())
            
            # Чем больше касаний уровня, тем выше четкость
            clarity = min(max_touches * 10, 100)
            
            return clarity
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета четкости уровней: {e}")
            return 50.0
    
    def _calculate_total_score(self, factor_scores: Dict[ConfluenceFactor, float],
                              timeframe_scores: Dict[str, float]) -> float:
        """Рассчитывает общий счет конфлюэнса."""
        # Взвешенная сумма факторных оценок
        factor_total = 0
        total_weight = 0
        
        for factor, score in factor_scores.items():
            weight = self.factor_weights.get(factor, 0)
            factor_total += score * weight
            total_weight += weight
        
        factor_weighted = factor_total / total_weight if total_weight > 0 else 0
        
        # Средняя оценка по таймфреймам
        tf_avg = np.mean(list(timeframe_scores.values())) if timeframe_scores else 0
        
        # Комбинируем
        total_score = (factor_weighted * 0.7 + tf_avg * 0.3)
        
        return min(max(total_score, 0), 100)
    
    def _determine_confluence_level(self, score: float) -> ConfluenceLevel:
        """Определяет уровень конфлюэнса по счету."""
        for level, (low, high) in self.level_thresholds.items():
            if low <= score < high:
                return level
        
        return ConfluenceLevel.MEDIUM  # По умолчанию
    
    def _find_aligned_levels(self, data_frames: Dict[str, pd.DataFrame]) -> Dict[str, List[float]]:
        """Находит согласованные уровни на всех таймфреймах."""
        aligned_levels = {
            'supports': [],
            'resistances': []
        }
        
        try:
            # Собираем все уровни
            all_supports = []
            all_resistances = []
            
            for tf, df in data_frames.items():
                if len(df) < 20:
                    continue
                
                support, resistance = self._find_key_levels(df)
                
                if support:
                    all_supports.append(support)
                if resistance:
                    all_resistances.append(resistance)
            
            # Кластеризуем уровни
            if all_supports:
                clustered_supports = self._cluster_levels(all_supports)
                aligned_levels['supports'] = [float(level) for level in clustered_supports]
            
            if all_resistances:
                clustered_resistances = self._cluster_levels(all_resistances)
                aligned_levels['resistances'] = [float(level) for level in clustered_resistances]
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска согласованных уровней: {e}")
        
        return aligned_levels
    
    def _find_key_levels(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Находит ключевые уровни поддержки и сопротивления."""
        try:
            if len(df) < 20:
                return None, None
            
            highs = df['high'].values[-20:]
            lows = df['low'].values[-20:]
            current_price = df['close'].iloc[-1]
            
            # Поиск ключевой поддержки
            support_candidates = []
            for low in lows:
                if low < current_price:
                    support_candidates.append(low)
            
            support = max(support_candidates) if support_candidates else None
            
            # Поиск ключевого сопротивления
            resistance_candidates = []
            for high in highs:
                if high > current_price:
                    resistance_candidates.append(high)
            
            resistance = min(resistance_candidates) if resistance_candidates else None
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска ключевых уровней: {e}")
            return None, None
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.01) -> List[float]:
        """Кластеризует близкие уровни."""
        if not levels:
            return []
        
        levels_sorted = sorted(levels)
        clusters = []
        current_cluster = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= threshold:
                current_cluster.append(level)
            else:
                # Среднее арифметическое кластера
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def _find_best_entries(self, data_frames: Dict[str, pd.DataFrame],
                          aligned_levels: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Находит лучшие точки входа на основе конфлюэнса."""
        best_entries = []
        
        try:
            primary_df = list(data_frames.values())[0]
            current_price = primary_df['close'].iloc[-1]
            
            # Анализируем точки входа
            entries = []
            
            # Точки входа у поддержек
            for support in aligned_levels.get('supports', []):
                if support < current_price:
                    distance = (current_price - support) / current_price
                    
                    if distance < 0.02:  # В пределах 2%
                        entry = {
                            'price': support,
                            'type': 'support',
                            'direction': 'BUY',
                            'distance_pct': distance * 100,
                            'confidence': self._calculate_entry_confidence(support, 'support', data_frames)
                        }
                        entries.append(entry)
            
            # Точки входа у сопротивлений
            for resistance in aligned_levels.get('resistances', []):
                if resistance > current_price:
                    distance = (resistance - current_price) / current_price
                    
                    if distance < 0.02:  # В пределах 2%
                        entry = {
                            'price': resistance,
                            'type': 'resistance',
                            'direction': 'SELL',
                            'distance_pct': distance * 100,
                            'confidence': self._calculate_entry_confidence(resistance, 'resistance', data_frames)
                        }
                        entries.append(entry)
            
            # Сортируем по уверенности
            entries.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Берем топ-3
            best_entries = entries[:3]
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска лучших точек входа: {e}")
        
        return best_entries
    
    def _calculate_entry_confidence(self, price: float, level_type: str,
                                   data_frames: Dict[str, pd.DataFrame]) -> float:
        """Рассчитывает уверенность в точке входа."""
        confidence = 50.0  # Базовая уверенность
        
        try:
            # Проверяем присутствие уровня на разных ТФ
            tf_count = 0
            
            for tf, df in data_frames.items():
                if len(df) < 20:
                    continue
                
                support, resistance = self._find_key_levels(df)
                
                if level_type == 'support' and support:
                    if abs(support - price) / price < 0.005:  # В пределах 0.5%
                        tf_count += 1
                        # Бонус за старшие ТФ
                        confidence += self.timeframe_weights.get(tf, 1.0) * 5
                
                if level_type == 'resistance' and resistance:
                    if abs(resistance - price) / price < 0.005:
                        tf_count += 1
                        confidence += self.timeframe_weights.get(tf, 1.0) * 5
            
            # Бонус за количество ТФ
            confidence += tf_count * 5
            
            # Проверяем объемную поддержку
            primary_df = list(data_frames.values())[0]
            if len(primary_df) >= 50:
                volume_profile = self._create_volume_profile(primary_df)
                for level_price, volume in volume_profile.items():
                    if abs(level_price - price) / price < 0.005:
                        # Высокий объем увеличивает уверенность
                        volume_percentile = volume / max(volume_profile.values())
                        confidence += volume_percentile * 20
                        break
            
            return min(confidence, 100)
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета уверенности входа: {e}")
            return 50.0
    
    def _generate_warnings(self, factor_scores: Dict[ConfluenceFactor, float],
                          timeframe_scores: Dict[str, float]) -> List[str]:
        """Генерирует предупреждения на основе анализа."""
        warnings = []
        
        # Проверка низких факторных оценок
        for factor, score in factor_scores.items():
            if score < 30:
                warnings.append(f"Низкая оценка фактора: {factor.value} ({score:.1f})")
        
        # Проверка расхождения таймфреймов
        if timeframe_scores:
            tf_scores_list = list(timeframe_scores.values())
            if len(tf_scores_list) >= 2:
                max_score = max(tf_scores_list)
                min_score = min(tf_scores_list)
                
                if max_score - min_score > 50:
                    warnings.append("Сильное расхождение между таймфреймами")
        
        # Проверка общего счета
        total_score = np.mean(list(factor_scores.values())) if factor_scores else 0
        if total_score < 40:
            warnings.append("Общий счет конфлюэнса низкий")
        
        return warnings
    
    def _generate_recommendations(self, total_score: float,
                                  aligned_levels: Dict[str, List[float]],
                                  best_entries: List[Dict[str, Any]]) -> List[str]:
        """Генерирует рекомендации на основе анализа."""
        recommendations = []
        
        # Рекомендации на основе общего счета
        if total_score >= 80:
            recommendations.append("Высокий конфлюэнс - отличные условия для торговли")
        elif total_score >= 60:
            recommendations.append("Хороший конфлюэнс - можно рассматривать сделки")
        elif total_score >= 40:
            recommendations.append("Средний конфлюэнс - соблюдайте осторожность")
        else:
            recommendations.append("Низкий конфлюэнс - лучше воздержаться от торговли")
        
        # Рекомендации на основе уровней
        if aligned_levels.get('supports'):
            support_str = ', '.join([f"${s:.2f}" for s in aligned_levels['supports'][:3]])
            recommendations.append(f"Ключевые поддержки: {support_str}")
        
        if aligned_levels.get('resistances'):
            resistance_str = ', '.join([f"${r:.2f}" for r in aligned_levels['resistances'][:3]])
            recommendations.append(f"Ключевые сопротивления: {resistance_str}")
        
        # Рекомендации на основе точек входа
        if best_entries:
            entry = best_entries[0]
            recommendations.append(
                f"Лучшая точка входа: {entry['direction']} @ ${entry['price']:.2f} "
                f"(уверенность: {entry['confidence']:.1f}%)"
            )
        
        return recommendations
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Определяет тренд на DataFrame."""
        try:
            if len(df) < 20:
                return "ranging"
            
            closes = df['close'].values[-20:]
            highs = df['high'].values[-20:]
            lows = df['low'].values[-20:]
            
            # Скользящие средние
            ma_short = np.mean(closes[-5:])
            ma_medium = np.mean(closes[-10:])
            ma_long = np.mean(closes)
            
            # Определяем тренд
            if ma_short > ma_medium > ma_long and closes[-1] > ma_short:
                return "strong_bullish"
            elif ma_short > ma_medium and closes[-1] > ma_short:
                return "bullish"
            elif ma_short < ma_medium < ma_long and closes[-1] < ma_short:
                return "strong_bearish"
            elif ma_short < ma_medium and closes[-1] < ma_short:
                return "bearish"
            else:
                return "ranging"
                
        except Exception as e:
            logger.error(f"❌ Ошибка определения тренда: {e}")
            return "ranging"
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Рассчитывает RSI."""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean().iloc[-1]
            
            if loss == 0:
                return 100.0
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета RSI: {e}")
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float]:
        """Рассчитывает MACD."""
        try:
            if len(prices) < 26:
                return 0.0, 0.0
            
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            return macd.iloc[-1], signal.iloc[-1]
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета MACD: {e}")
            return 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[float, float]:
        """Рассчитывает Bollinger Bands."""
        try:
            if len(prices) < 20:
                return prices.iloc[-1] * 1.02, prices.iloc[-1] * 0.98
            
            sma = prices.rolling(window=20).mean().iloc[-1]
            std = prices.rolling(window=20).std().iloc[-1]
            
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            return upper, lower
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета Bollinger Bands: {e}")
            return prices.iloc[-1] * 1.02, prices.iloc[-1] * 0.98
    
    def _log_results(self, result: ConfluenceAnalysisResult):
        """Логирует результаты анализа."""
        confluence = result.confluence_score
        
        logger.info(f"✅ Конфлюэнс для {result.symbol}:")
        logger.info(f"   🎯 Общий счет: {confluence.total_score:.1f} ({confluence.level.value})")
        logger.info(f"   📊 Факторы:")
        
        for factor, score in confluence.factors.items():
            logger.info(f"     • {factor.value}: {score:.1f}")
        
        if result.aligned_levels:
            supports = result.aligned_levels.get('supports', [])
            resistances = result.aligned_levels.get('resistances', [])
            logger.info(f"   🛡️  Согласованные уровни: {len(supports)}S / {len(resistances)}R")
        
        if result.best_entries:
            logger.info(f"   🎯 Лучшие точки входа: {len(result.best_entries)}")
            for entry in result.best_entries:
                logger.info(f"     • {entry['direction']} @ ${entry['price']:.2f} "
                           f"(уверенность: {entry['confidence']:.1f}%)")
        
        if result.warnings:
            logger.info(f"   ⚠️  Предупреждения: {len(result.warnings)}")
            for warning in result.warnings:
                logger.info(f"     • {warning}")
    
    def _create_empty_result(self, symbol: str, primary_timeframe: str) -> ConfluenceAnalysisResult:
        """Создает пустой результат при недостатке данных."""
        return ConfluenceAnalysisResult(
            symbol=symbol,
            primary_timeframe=primary_timeframe,
            timestamp=datetime.now(),
            confluence_score=ConfluenceScore(
                total_score=0,
                level=ConfluenceLevel.VERY_WEAK,
                factors={},
                timeframes={},
                details={'error': 'Недостаточно данных для анализа'}
            ),
            aligned_levels={},
            best_entries=[],
            warnings=["Недостаточно данных для анализа конфлюэнса"],
            recommendations=["Соберите больше данных для анализа"]
        )
    
    def _clean_cache(self):
        """Очищает старые записи из кеша."""
        if len(self.cache) > self.cache_max_size:
            keys_to_remove = list(self.cache.keys())[:len(self.cache) - self.cache_max_size]
            for key in keys_to_remove:
                del self.cache[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы калькулятора."""
        avg_score = np.mean(self.stats['average_scores']) if self.stats['average_scores'] else 0
        
        return {
            'version': self.VERSION,
            'total_analyses': self.stats['total_analyses'],
            'cache_hits': self.stats['cache_hits'],
            'cache_size': len(self.cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_analyses'], 1),
            'average_confluence_score': avg_score,
            'errors_count': len(self.stats['errors']),
            'recent_errors': self.stats['errors'][-5:] if self.stats['errors'] else [],
            'configuration': {
                'min_timeframes': self.min_timeframes,
                'weight_timeframes': self.weight_timeframes,
                'use_volume_profile': self.use_volume_profile,
                'use_indicators': self.use_indicators,
                'use_patterns': self.use_patterns,
                'use_price_action': self.use_price_action,
                'use_market_structure': self.use_market_structure,
                'alignment_threshold': self.alignment_threshold
            }
        }

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    # Тестирование модуля конфлюэнса
    print("🧪 Тестирование ConfluenceCalculator...")
    
    # Создаем тестовые данные для разных таймфреймов
    np.random.seed(42)
    
    data_frames = {}
    
    # 15 минут
    dates_15m = pd.date_range('2024-01-01', periods=300, freq='15min')
    prices_15m = 50000 + np.cumsum(np.random.randn(300) * 50)
    
    data_frames['15m'] = pd.DataFrame({
        'open': prices_15m - np.random.rand(300) * 50,
        'high': prices_15m + np.random.rand(300) * 75,
        'low': prices_15m - np.random.rand(300) * 75,
        'close': prices_15m,
        'volume': np.random.rand(300) * 1000 + 500
    }, index=dates_15m)
    
    # 1 час
    dates_1h = pd.date_range('2024-01-01', periods=200, freq='1h')
    prices_1h = 50000 + np.cumsum(np.random.randn(200) * 100)
    
    data_frames['1h'] = pd.DataFrame({
        'open': prices_1h - np.random.rand(200) * 100,
        'high': prices_1h + np.random.rand(200) * 150,
        'low': prices_1h - np.random.rand(200) * 150,
        'close': prices_1h,
        'volume': np.random.rand(200) * 1000 + 500
    }, index=dates_1h)
    
    # 4 часа
    dates_4h = pd.date_range('2024-01-01', periods=100, freq='4h')
    prices_4h = 50000 + np.cumsum(np.random.randn(100) * 200)
    
    data_frames['4h'] = pd.DataFrame({
        'open': prices_4h - np.random.rand(100) * 200,
        'high': prices_4h + np.random.rand(100) * 300,
        'low': prices_4h - np.random.rand(100) * 300,
        'close': prices_4h,
        'volume': np.random.rand(100) * 1000 + 500
    }, index=dates_4h)
    
    # 1 день
    dates_1d = pd.date_range('2024-01-01', periods=50, freq='1d')
    prices_1d = 50000 + np.cumsum(np.random.randn(50) * 500)
    
    data_frames['1d'] = pd.DataFrame({
        'open': prices_1d - np.random.rand(50) * 400,
        'high': prices_1d + np.random.rand(50) * 600,
        'low': prices_1d - np.random.rand(50) * 600,
        'close': prices_1d,
        'volume': np.random.rand(50) * 1000 + 500
    }, index=dates_1d)
    
    # Создаем и тестируем калькулятор
    calculator = ConfluenceCalculator(
        min_timeframes=2,
        weight_timeframes=True,
        use_volume_profile=True,
        use_indicators=True,
        use_patterns=True,
        use_price_action=True,
        use_market_structure=True,
        alignment_threshold=0.7
    )
    
    result = calculator.analyze(data_frames, "BTC/USDT", "1h")
    
    print(f"\n📊 Результаты анализа конфлюэнса:")
    print(f"   Общий счет: {result.confluence_score.total_score:.1f}")
    print(f"   Уровень: {result.confluence_score.level.value}")
    
    print(f"\n   Факторы:")
    for factor, score in result.confluence_score.factors.items():
        print(f"     • {factor.value}: {score:.1f}")
    
    if result.aligned_levels:
        print(f"\n   Согласованные уровни:")
        supports = result.aligned_levels.get('supports', [])
        resistances = result.aligned_levels.get('resistances', [])
        if supports:
            print(f"     Поддержки: {', '.join([f'${s:.2f}' for s in supports])}")
        if resistances:
            print(f"     Сопротивления: {', '.join([f'${r:.2f}' for r in resistances])}")
    
    if result.best_entries:
        print(f"\n   Лучшие точки входа:")
        for entry in result.best_entries:
            print(f"     • {entry['direction']} @ ${entry['price']:.2f} "
                  f"(уверенность: {entry['confidence']:.1f}%, дистанция: {entry['distance_pct']:.2f}%)")
    
    if result.warnings:
        print(f"\n   ⚠️ Предупреждения:")
        for warning in result.warnings:
            print(f"     • {warning}")
    
    if result.recommendations:
        print(f"\n   💡 Рекомендации:")
        for rec in result.recommendations:
            print(f"     • {rec}")
    
    # Статистика калькулятора
    stats = calculator.get_statistics()
    print(f"\n📈 Статистика калькулятора:")
    print(f"   Всего анализов: {stats['total_analyses']}")
    print(f"   Средний счет конфлюэнса: {stats['average_confluence_score']:.1f}")
    print(f"   Попаданий в кеш: {stats['cache_hits']}")
    print(f"   Размер кеша: {stats['cache_size']}")
    
    print("\n✅ Тестирование завершено!")
