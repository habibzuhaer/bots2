#!/usr/bin/env python3
"""
ПОЛНЫЙ МОДУЛЬ ГЕНЕРАЦИИ ТОРГОВЫХ СИГНАЛОВ
Версия: 2.0
Алгоритмы: Технический анализ, Анализ объемов, Паттерны, Конфлюэнс
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import traceback
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ============================================================================
# ОПРЕДЕЛЕНИЯ КЛАССОВ
# ============================================================================

class SignalType(Enum):
    """Типы торговых сигналов."""
    BREAKOUT = "breakout"          # Пробой уровня
    REVERSAL = "reversal"          # Разворот от уровня
    TREND_FOLLOWING = "trend_following"  # Следование тренду
    DIVERGENCE = "divergence"      # Дивергенция
    PATTERN = "pattern"            # Графический паттерн
    VOLUME_SPIKE = "volume_spike"  # Скачок объема

class SignalDirection(Enum):
    """Направление сигнала."""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

class SignalStrength(Enum):
    """Сила сигнала."""
    WEAK = "weak"          # 0.0-0.33
    MEDIUM = "medium"      # 0.34-0.66
    STRONG = "strong"      # 0.67-1.0
    VERY_STRONG = "very_strong"  # 0.9+

@dataclass
class Signal:
    """Структура торгового сигнала."""
    symbol: str
    signal_type: SignalType
    direction: SignalDirection
    strength: SignalStrength
    price: float
    confidence: float  # 0.0-1.0
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    timeframe: str = "1h"
    indicators: Dict[str, float] = field(default_factory=dict)
    levels: Dict[str, Any] = field(default_factory=dict)
    confluence: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Конвертирует в словарь."""
        return {
            'symbol': self.symbol,
            'type': self.signal_type.value,
            'direction': self.direction.value,
            'strength': self.strength.value,
            'price': self.price,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward_ratio': self.risk_reward_ratio,
            'timeframe': self.timeframe,
            'indicators': self.indicators,
            'levels': self.levels,
            'confluence': self.confluence,
            'description': self.description,
            'metadata': self.metadata
        }

@dataclass
class SignalAnalysisResult:
    """Результат анализа сигналов."""
    symbol: str
    timeframe: str
    timestamp: datetime
    signals: List[Signal]
    market_condition: Dict[str, Any]
    statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Конвертирует в словарь."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'signals': [signal.to_dict() for signal in self.signals],
            'market_condition': self.market_condition,
            'statistics': self.statistics
        }

# ============================================================================
# ОСНОВНОЙ КЛАСС SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """
    Продвинутый генератор торговых сигналов.
    
    Использует комбинацию методов:
    1. Технические индикаторы (RSI, MACD, MA, Bollinger Bands)
    2. Анализ объемов
    3. Графические паттерны
    4. Дивергенции
    5. Конфлюэнс сигналов
    6. Управление рисками
    """
    
    VERSION = "2.0.0"
    
    def __init__(self,
                 min_confidence: float = 0.6,
                 use_rsi: bool = True,
                 use_macd: bool = True,
                 use_ma: bool = True,
                 use_bollinger: bool = True,
                 use_volume: bool = True,
                 use_divergence: bool = True,
                 use_patterns: bool = True):
        """
        Инициализация генератора сигналов.
        
        Args:
            min_confidence: Минимальная уверенность для сигнала
            use_rsi: Использовать RSI
            use_macd: Использовать MACD
            use_ma: Использовать скользящие средние
            use_bollinger: Использовать Bollinger Bands
            use_volume: Использовать анализ объемов
            use_divergence: Использовать дивергенции
            use_patterns: Использовать графические паттерны
        """
        
        self.min_confidence = min_confidence
        
        # Флаги использования методов
        self.use_rsi = use_rsi
        self.use_macd = use_macd
        self.use_ma = use_ma
        self.use_bollinger = use_bollinger
        self.use_volume = use_volume
        self.use_divergence = use_divergence
        self.use_patterns = use_patterns
        
        # Параметры индикаторов
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.ma_short_period = 9
        self.ma_medium_period = 21
        self.ma_long_period = 50
        self.bollinger_period = 20
        self.bollinger_std = 2
        
        # Пороги для сигналов
        self.breakout_threshold = 0.01  # 1%
        self.reversal_threshold = 0.005  # 0.5%
        self.volume_spike_multiplier = 2.0
        
        # Веса для разных факторов
        self.weights = {
            'technical': 0.30,
            'levels': 0.25,
            'volume': 0.15,
            'confluence': 0.20,
            'risk': 0.10
        }
        
        # Кеш расчетов
        self.cache = {}
        self.cache_max_size = 100
        
        # Статистика
        self.stats = {
            'total_analyses': 0,
            'signals_generated': 0,
            'cache_hits': 0,
            'errors': []
        }
        
        logger.info(f"✅ SignalGenerator v{self.VERSION} инициализирован")
        logger.info(f"   Минимальная уверенность: {min_confidence}")
        logger.info(f"   Методы: RSI={use_rsi}, MACD={use_macd}, MA={use_ma}, "
                   f"BB={use_bollinger}, Volume={use_volume}")
    
    def analyze(self, df: pd.DataFrame, levels: Dict, confluence: Dict,
                symbol: str = "UNKNOWN", timeframe: str = "1h") -> SignalAnalysisResult:
        """
        Основной метод анализа и генерации сигналов.
        
        Args:
            df: DataFrame с ценовыми данными
            levels: Уровни поддержки/сопротивления
            confluence: Данные конфлюэнса
            symbol: Идентификатор символа
            timeframe: Таймфрейм данных
            
        Returns:
            SignalAnalysisResult с сигналами и анализом
        """
        
        # Проверка входных данных
        if df.empty or len(df) < 50:
            logger.warning(f"⚠️  Недостаточно данных: {len(df)} свечей")
            return SignalAnalysisResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                signals=[],
                market_condition={},
                statistics={}
            )
        
        # Генерация ключа кеша
        cache_key = f"{symbol}_{timeframe}_{len(df)}_{df.index[-1].timestamp()}"
        
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            logger.debug(f"🎯 Кеш попадание для {symbol} {timeframe}")
            return self.cache[cache_key]
        
        self.stats['total_analyses'] += 1
        
        logger.info(f"🧮 Анализ сигналов для {symbol} {timeframe} ({len(df)} свечей)")
        start_time = datetime.now()
        
        try:
            # 1. Расчет технических индикаторов
            indicators = self._calculate_indicators(df)
            
            # 2. Анализ рыночных условий
            market_condition = self._analyze_market_condition(df, indicators)
            
            # 3. Генерация сигналов по разным стратегиям
            all_signals = []
            
            # Сигналы по индикаторам
            if self.use_rsi:
                rsi_signals = self._generate_rsi_signals(df, indicators, levels, confluence, symbol, timeframe)
                all_signals.extend(rsi_signals)
            
            if self.use_macd:
                macd_signals = self._generate_macd_signals(df, indicators, levels, confluence, symbol, timeframe)
                all_signals.extend(macd_signals)
            
            if self.use_ma:
                ma_signals = self._generate_ma_signals(df, indicators, levels, confluence, symbol, timeframe)
                all_signals.extend(ma_signals)
            
            if self.use_bollinger:
                bollinger_signals = self._generate_bollinger_signals(df, indicators, levels, confluence, symbol, timeframe)
                all_signals.extend(bollinger_signals)
            
            if self.use_volume:
                volume_signals = self._generate_volume_signals(df, indicators, levels, confluence, symbol, timeframe)
                all_signals.extend(volume_signals)
            
            if self.use_divergence:
                divergence_signals = self._generate_divergence_signals(df, indicators, levels, confluence, symbol, timeframe)
                all_signals.extend(divergence_signals)
            
            if self.use_patterns:
                pattern_signals = self._generate_pattern_signals(df, indicators, levels, confluence, symbol, timeframe)
                all_signals.extend(pattern_signals)
            
            # 4. Сигналы на основе уровней
            level_signals = self._generate_level_signals(df, indicators, levels, confluence, symbol, timeframe)
            all_signals.extend(level_signals)
            
            # 5. Фильтрация и ранжирование сигналов
            filtered_signals = self._filter_and_rank_signals(all_signals)
            
            # 6. Добавление стоп-лоссов и тейк-профитов
            signals_with_risk = self._add_risk_management(filtered_signals, df, levels)
            
            # 7. Формирование результата
            result = SignalAnalysisResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                signals=signals_with_risk,
                market_condition=market_condition,
                statistics=self._calculate_statistics(start_time, datetime.now(), signals_with_risk)
            )
            
            # 8. Логирование результатов
            self._log_results(result)
            
            # 9. Сохранение в кеш
            self.cache[cache_key] = result
            self._clean_cache()
            
            self.stats['signals_generated'] += len(signals_with_risk)
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Ошибка анализа сигналов для {symbol}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.stats['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'error': str(e)
            })
            
            return SignalAnalysisResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                signals=[],
                market_condition={},
                statistics={}
            )
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Рассчитывает все технические индикаторы."""
        indicators = {}
        
        # RSI
        if self.use_rsi:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        if self.use_macd:
            exp1 = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
            indicators['macd'] = exp1 - exp2
            indicators['macd_signal'] = indicators['macd'].ewm(span=self.macd_signal, adjust=False).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Скользящие средние
        if self.use_ma:
            indicators['ma_short'] = df['close'].rolling(window=self.ma_short_period).mean()
            indicators['ma_medium'] = df['close'].rolling(window=self.ma_medium_period).mean()
            indicators['ma_long'] = df['close'].rolling(window=self.ma_long_period).mean()
        
        # Bollinger Bands
        if self.use_bollinger:
            indicators['bb_middle'] = df['close'].rolling(window=self.bollinger_period).mean()
            bb_std = df['close'].rolling(window=self.bollinger_period).std()
            indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * self.bollinger_std)
            indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * self.bollinger_std)
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
        
        # Объемные индикаторы
        if self.use_volume:
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
            indicators['obv'] = self._calculate_obv(df)
        
        # Дополнительные индикаторы
        indicators['atr'] = self._calculate_atr(df, 14)
        indicators['stochastic'] = self._calculate_stochastic(df, 14, 3)
        
        logger.debug(f"📈 Рассчитано индикаторов: {list(indicators.keys())}")
        
        return indicators
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Рассчитывает On-Balance Volume (OBV)."""
        obv = pd.Series(0, index=df.index)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Рассчитывает Average True Range (ATR)."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.Series:
        """Рассчитывает Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_d
    
    def _analyze_market_condition(self, df: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Анализирует текущие рыночные условия."""
        condition = {
            'trend': 'neutral',
            'volatility': 'low',
            'momentum': 'neutral',
            'volume_trend': 'neutral',
            'overall_bias': 'neutral'
        }
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Определение тренда
            if 'ma_short' in indicators and 'ma_medium' in indicators:
                ma_short = indicators['ma_short'].iloc[-1]
                ma_medium = indicators['ma_medium'].iloc[-1]
                
                if current_price > ma_short > ma_medium:
                    condition['trend'] = 'strong_bullish'
                elif current_price > ma_short and ma_short > ma_medium:
                    condition['trend'] = 'bullish'
                elif current_price < ma_short < ma_medium:
                    condition['trend'] = 'strong_bearish'
                elif current_price < ma_short and ma_short < ma_medium:
                    condition['trend'] = 'bearish'
                else:
                    condition['trend'] = 'ranging'
            
            # Определение волатильности
            if 'atr' in indicators and current_price > 0:
                atr = indicators['atr'].iloc[-1]
                atr_percent = atr / current_price
                
                if atr_percent > 0.05:
                    condition['volatility'] = 'high'
                elif atr_percent > 0.02:
                    condition['volatility'] = 'medium'
                else:
                    condition['volatility'] = 'low'
            
            # Определение момента
            if 'rsi' in indicators:
                rsi = indicators['rsi'].iloc[-1]
                
                if rsi > 70:
                    condition['momentum'] = 'overbought'
                elif rsi > 55:
                    condition['momentum'] = 'bullish'
                elif rsi < 30:
                    condition['momentum'] = 'oversold'
                elif rsi < 45:
                    condition['momentum'] = 'bearish'
                else:
                    condition['momentum'] = 'neutral'
            
            # Тренд объема
            if 'volume_ratio' in indicators:
                volume_ratio = indicators['volume_ratio'].iloc[-1]
                
                if volume_ratio > 1.5:
                    condition['volume_trend'] = 'high'
                elif volume_ratio > 1.0:
                    condition['volume_trend'] = 'rising'
                elif volume_ratio < 0.5:
                    condition['volume_trend'] = 'low'
                else:
                    condition['volume_trend'] = 'normal'
            
            # Общий байас
            bias_score = 0
            
            if condition['trend'] in ['strong_bullish', 'bullish']:
                bias_score += 1
            elif condition['trend'] in ['strong_bearish', 'bearish']:
                bias_score -= 1
            
            if condition['momentum'] in ['overbought', 'bullish']:
                bias_score += 0.5
            elif condition['momentum'] in ['oversold', 'bearish']:
                bias_score -= 0.5
            
            if condition['volume_trend'] in ['high', 'rising']:
                if condition['trend'] in ['strong_bullish', 'bullish']:
                    bias_score += 0.5
                elif condition['trend'] in ['strong_bearish', 'bearish']:
                    bias_score -= 0.5
            
            if bias_score > 1:
                condition['overall_bias'] = 'bullish'
            elif bias_score < -1:
                condition['overall_bias'] = 'bearish'
            else:
                condition['overall_bias'] = 'neutral'
                
        except Exception as e:
            logger.error(f"❌ Ошибка анализа рыночных условий: {e}")
        
        return condition
    
    def _generate_rsi_signals(self, df: pd.DataFrame, indicators: Dict, levels: Dict,
                             confluence: Dict, symbol: str, timeframe: str) -> List[Signal]:
        """Генерирует сигналы на основе RSI."""
        signals = []
        
        if 'rsi' not in indicators:
            return signals
        
        try:
            current_price = df['close'].iloc[-1]
            rsi = indicators['rsi'].iloc[-1]
            rsi_prev = indicators['rsi'].iloc[-2] if len(indicators['rsi']) > 1 else rsi
            
            # Перепроданность (потенциальный сигнал на покупку)
            if rsi < self.rsi_oversold:
                confidence = 0.6
                
                # Дополнительные факторы для увеличения уверенности
                if rsi < 25:
                    confidence = 0.8
                
                # Проверка на дивергенцию
                if self.use_divergence:
                    bullish_div = self._check_bullish_divergence(df, indicators['rsi'])
                    if bullish_div:
                        confidence = 0.85
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.REVERSAL,
                    direction=SignalDirection.BUY,
                    strength=SignalStrength.STRONG if confidence > 0.75 else SignalStrength.MEDIUM,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'rsi': rsi, 'rsi_prev': rsi_prev},
                    levels=levels,
                    confluence=confluence,
                    description=f"RSI oversold ({rsi:.1f}), potential reversal"
                )
                signals.append(signal)
            
            # Перекупленность (потенциальный сигнал на продажу)
            elif rsi > self.rsi_overbought:
                confidence = 0.6
                
                if rsi > 75:
                    confidence = 0.8
                
                if self.use_divergence:
                    bearish_div = self._check_bearish_divergence(df, indicators['rsi'])
                    if bearish_div:
                        confidence = 0.85
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.REVERSAL,
                    direction=SignalDirection.SELL,
                    strength=SignalStrength.STRONG if confidence > 0.75 else SignalStrength.MEDIUM,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'rsi': rsi, 'rsi_prev': rsi_prev},
                    levels=levels,
                    confluence=confluence,
                    description=f"RSI overbought ({rsi:.1f}), potential reversal"
                )
                signals.append(signal)
            
            # Выход из зоны перепроданности/перекупленности
            elif rsi_prev < self.rsi_oversold and rsi > self.rsi_oversold:
                # Выход из перепроданности - подтверждение бычьего движения
                confidence = 0.65
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.TREND_FOLLOWING,
                    direction=SignalDirection.BUY,
                    strength=SignalStrength.MEDIUM,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'rsi': rsi, 'rsi_prev': rsi_prev},
                    levels=levels,
                    confluence=confluence,
                    description=f"RSI exiting oversold zone ({rsi_prev:.1f} -> {rsi:.1f})"
                )
                signals.append(signal)
            
            elif rsi_prev > self.rsi_overbought and rsi < self.rsi_overbought:
                # Выход из перекупленности - подтверждение медвежьего движения
                confidence = 0.65
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.TREND_FOLLOWING,
                    direction=SignalDirection.SELL,
                    strength=SignalStrength.MEDIUM,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'rsi': rsi, 'rsi_prev': rsi_prev},
                    levels=levels,
                    confluence=confluence,
                    description=f"RSI exiting overbought zone ({rsi_prev:.1f} -> {rsi:.1f})"
                )
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"❌ Ошибка генерации RSI сигналов: {e}")
        
        return signals
    
    def _generate_macd_signals(self, df: pd.DataFrame, indicators: Dict, levels: Dict,
                              confluence: Dict, symbol: str, timeframe: str) -> List[Signal]:
        """Генерирует сигналы на основе MACD."""
        signals = []
        
        if 'macd' not in indicators or 'macd_signal' not in indicators:
            return signals
        
        try:
            current_price = df['close'].iloc[-1]
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            macd_histogram = indicators['macd_histogram'].iloc[-1] if 'macd_histogram' in indicators else macd - macd_signal
            
            macd_prev = indicators['macd'].iloc[-2] if len(indicators['macd']) > 1 else macd
            signal_prev = indicators['macd_signal'].iloc[-2] if len(indicators['macd_signal']) > 1 else macd_signal
            
            # Пересечение MACD и сигнальной линии
            if macd_prev <= signal_prev and macd > macd_signal:
                # Бычье пересечение
                confidence = 0.7
                
                # Увеличиваем уверенность если гистограмма растет
                if macd_histogram > 0 and abs(macd_histogram) > abs(macd - macd_signal) * 0.5:
                    confidence = 0.8
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.TREND_FOLLOWING,
                    direction=SignalDirection.BUY,
                    strength=SignalStrength.MEDIUM,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'macd': macd, 'macd_signal': macd_signal, 'macd_histogram': macd_histogram},
                    levels=levels,
                    confluence=confluence,
                    description=f"MACD bullish crossover (MACD: {macd:.4f}, Signal: {macd_signal:.4f})"
                )
                signals.append(signal)
            
            elif macd_prev >= signal_prev and macd < macd_signal:
                # Медвежье пересечение
                confidence = 0.7
                
                if macd_histogram < 0 and abs(macd_histogram) > abs(macd - macd_signal) * 0.5:
                    confidence = 0.8
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.TREND_FOLLOWING,
                    direction=SignalDirection.SELL,
                    strength=SignalStrength.MEDIUM,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'macd': macd, 'macd_signal': macd_signal, 'macd_histogram': macd_histogram},
                    levels=levels,
                    confluence=confluence,
                    description=f"MACD bearish crossover (MACD: {macd:.4f}, Signal: {macd_signal:.4f})"
                )
                signals.append(signal)
            
            # Дивергенция MACD
            if self.use_divergence:
                bullish_div = self._check_macd_bullish_divergence(df, indicators['macd'])
                if bullish_div:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.DIVERGENCE,
                        direction=SignalDirection.BUY,
                        strength=SignalStrength.STRONG,
                        price=current_price,
                        confidence=0.75,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'macd': macd, 'macd_signal': macd_signal},
                        levels=levels,
                        confluence=confluence,
                        description="MACD bullish divergence detected"
                    )
                    signals.append(signal)
                
                bearish_div = self._check_macd_bearish_divergence(df, indicators['macd'])
                if bearish_div:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.DIVERGENCE,
                        direction=SignalDirection.SELL,
                        strength=SignalStrength.STRONG,
                        price=current_price,
                        confidence=0.75,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'macd': macd, 'macd_signal': macd_signal},
                        levels=levels,
                        confluence=confluence,
                        description="MACD bearish divergence detected"
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"❌ Ошибка генерации MACD сигналов: {e}")
        
        return signals
    
    def _generate_ma_signals(self, df: pd.DataFrame, indicators: Dict, levels: Dict,
                            confluence: Dict, symbol: str, timeframe: str) -> List[Signal]:
        """Генерирует сигналы на основе скользящих средних."""
        signals = []
        
        if 'ma_short' not in indicators or 'ma_medium' not in indicators or 'ma_long' not in indicators:
            return signals
        
        try:
            current_price = df['close'].iloc[-1]
            ma_short = indicators['ma_short'].iloc[-1]
            ma_medium = indicators['ma_medium'].iloc[-1]
            ma_long = indicators['ma_long'].iloc[-1]
            
            ma_short_prev = indicators['ma_short'].iloc[-2] if len(indicators['ma_short']) > 1 else ma_short
            ma_medium_prev = indicators['ma_medium'].iloc[-2] if len(indicators['ma_medium']) > 1 else ma_medium
            
            # Золотой крест (бычий сигнал)
            if ma_short_prev <= ma_medium_prev and ma_short > ma_medium:
                confidence = 0.75
                
                # Дополнительное подтверждение если цена выше средних
                if current_price > ma_short:
                    confidence = 0.85
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.TREND_FOLLOWING,
                    direction=SignalDirection.BUY,
                    strength=SignalStrength.STRONG,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'ma_short': ma_short, 'ma_medium': ma_medium, 'ma_long': ma_long},
                    levels=levels,
                    confluence=confluence,
                    description=f"Golden cross detected (Short MA: {ma_short:.2f}, Medium MA: {ma_medium:.2f})"
                )
                signals.append(signal)
            
            # Мертвый крест (медвежий сигнал)
            elif ma_short_prev >= ma_medium_prev and ma_short < ma_medium:
                confidence = 0.75
                
                if current_price < ma_short:
                    confidence = 0.85
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.TREND_FOLLOWING,
                    direction=SignalDirection.SELL,
                    strength=SignalStrength.STRONG,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'ma_short': ma_short, 'ma_medium': ma_medium, 'ma_long': ma_long},
                    levels=levels,
                    confluence=confluence,
                    description=f"Death cross detected (Short MA: {ma_short:.2f}, Medium MA: {ma_medium:.2f})"
                )
                signals.append(signal)
            
            # Отскок от скользящей средней
            ma_distance_pct = abs(current_price - ma_medium) / current_price
            
            if ma_distance_pct < 0.01:  # В пределах 1%
                # Определяем направление
                if current_price > ma_medium and ma_short > ma_medium:
                    # Отскок вверх от поддержки (MA)
                    confidence = 0.65
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.REVERSAL,
                        direction=SignalDirection.BUY,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'ma_short': ma_short, 'ma_medium': ma_medium, 'distance_pct': ma_distance_pct},
                        levels=levels,
                        confluence=confluence,
                        description=f"Bounce from MA support (Price: {current_price:.2f}, MA: {ma_medium:.2f})"
                    )
                    signals.append(signal)
                
                elif current_price < ma_medium and ma_short < ma_medium:
                    # Отскок вниз от сопротивления (MA)
                    confidence = 0.65
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.REVERSAL,
                        direction=SignalDirection.SELL,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'ma_short': ma_short, 'ma_medium': ma_medium, 'distance_pct': ma_distance_pct},
                        levels=levels,
                        confluence=confluence,
                        description=f"Rejection from MA resistance (Price: {current_price:.2f}, MA: {ma_medium:.2f})"
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"❌ Ошибка генерации MA сигналов: {e}")
        
        return signals
    
    def _generate_bollinger_signals(self, df: pd.DataFrame, indicators: Dict, levels: Dict,
                                   confluence: Dict, symbol: str, timeframe: str) -> List[Signal]:
        """Генерирует сигналы на основе Bollinger Bands."""
        signals = []
        
        if 'bb_upper' not in indicators or 'bb_lower' not in indicators or 'bb_middle' not in indicators:
            return signals
        
        try:
            current_price = df['close'].iloc[-1]
            bb_upper = indicators['bb_upper'].iloc[-1]
            bb_lower = indicators['bb_lower'].iloc[-1]
            bb_middle = indicators['bb_middle'].iloc[-1]
            bb_width = indicators['bb_width'].iloc[-1] if 'bb_width' in indicators else 0
            
            # Касание верхней полосы (потенциальный сигнал на продажу)
            if current_price >= bb_upper * 0.995:  # В пределах 0.5%
                confidence = 0.6
                
                # Увеличиваем уверенность при высокой волатильности
                if bb_width > 0.05:  # Широкие полосы
                    confidence = 0.7
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.REVERSAL,
                    direction=SignalDirection.SELL,
                    strength=SignalStrength.MEDIUM,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_middle': bb_middle, 'bb_width': bb_width},
                    levels=levels,
                    confluence=confluence,
                    description=f"Price at Bollinger upper band (Price: {current_price:.2f}, Upper: {bb_upper:.2f})"
                )
                signals.append(signal)
            
            # Касание нижней полосы (потенциальный сигнал на покупку)
            elif current_price <= bb_lower * 1.005:  # В пределах 0.5%
                confidence = 0.6
                
                if bb_width > 0.05:
                    confidence = 0.7
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.REVERSAL,
                    direction=SignalDirection.BUY,
                    strength=SignalStrength.MEDIUM,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_middle': bb_middle, 'bb_width': bb_width},
                    levels=levels,
                    confluence=confluence,
                    description=f"Price at Bollinger lower band (Price: {current_price:.2f}, Lower: {bb_lower:.2f})"
                )
                signals.append(signal)
            
            # Сжатие полос (предвестник пробоя)
            if bb_width < 0.02:  # Очень узкие полосы
                confidence = 0.5
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BREAKOUT,
                    direction=SignalDirection.NEUTRAL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_middle': bb_middle, 'bb_width': bb_width},
                    levels=levels,
                    confluence=confluence,
                    description=f"Bollinger squeeze detected (Width: {bb_width:.4f}), expecting breakout"
                )
                signals.append(signal)
            
            # Отскок от средней линии
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            
            if 0.45 < bb_position < 0.55:  # Вблизи средней линии
                # Анализируем направление
                if df['close'].iloc[-1] > df['close'].iloc[-2] and bb_position > 0.5:
                    confidence = 0.6
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.TREND_FOLLOWING,
                        direction=SignalDirection.BUY,
                        strength=SignalStrength.WEAK,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'bb_position': bb_position, 'bb_middle': bb_middle},
                        levels=levels,
                        confluence=confluence,
                        description=f"Bounce from BB middle line, bullish bias (Position: {bb_position:.2f})"
                    )
                    signals.append(signal)
                elif df['close'].iloc[-1] < df['close'].iloc[-2] and bb_position < 0.5:
                    confidence = 0.6
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.TREND_FOLLOWING,
                        direction=SignalDirection.SELL,
                        strength=SignalStrength.WEAK,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'bb_position': bb_position, 'bb_middle': bb_middle},
                        levels=levels,
                        confluence=confluence,
                        description=f"Rejection from BB middle line, bearish bias (Position: {bb_position:.2f})"
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"❌ Ошибка генерации Bollinger сигналов: {e}")
        
        return signals
    
    def _generate_volume_signals(self, df: pd.DataFrame, indicators: Dict, levels: Dict,
                                confluence: Dict, symbol: str, timeframe: str) -> List[Signal]:
        """Генерирует сигналы на основе анализа объемов."""
        signals = []
        
        if 'volume_ratio' not in indicators:
            return signals
        
        try:
            current_price = df['close'].iloc[-1]
            volume_ratio = indicators['volume_ratio'].iloc[-1]
            volume_ratio_prev = indicators['volume_ratio'].iloc[-2] if len(indicators['volume_ratio']) > 1 else volume_ratio
            
            # Скачок объема
            if volume_ratio > self.volume_spike_multiplier:
                confidence = 0.7
                
                # Определяем направление по движению цены
                price_change = df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) > 1 else 0
                
                if price_change > 0:
                    # Объемный рост на повышение
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.VOLUME_SPIKE,
                        direction=SignalDirection.BUY,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'volume_ratio': volume_ratio, 'price_change': price_change},
                        levels=levels,
                        confluence=confluence,
                        description=f"Volume spike on uptick (Ratio: {volume_ratio:.2f}, Change: {price_change:.4f})"
                    )
                    signals.append(signal)
                elif price_change < 0:
                    # Объемный рост на понижение
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.VOLUME_SPIKE,
                        direction=SignalDirection.SELL,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'volume_ratio': volume_ratio, 'price_change': price_change},
                        levels=levels,
                        confluence=confluence,
                        description=f"Volume spike on downtick (Ratio: {volume_ratio:.2f}, Change: {price_change:.4f})"
                    )
                    signals.append(signal)
            
            # Дивергенция объема
            if self.use_divergence and 'obv' in indicators:
                # Бычья дивергенция OBV
                obv_bullish_div = self._check_obv_bullish_divergence(df, indicators['obv'])
                if obv_bullish_div:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.DIVERGENCE,
                        direction=SignalDirection.BUY,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=0.65,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'volume_ratio': volume_ratio, 'obv': indicators['obv'].iloc[-1]},
                        levels=levels,
                        confluence=confluence,
                        description="OBV bullish divergence detected"
                    )
                    signals.append(signal)
                
                # Медвежья дивергенция OBV
                obv_bearish_div = self._check_obv_bearish_divergence(df, indicators['obv'])
                if obv_bearish_div:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.DIVERGENCE,
                        direction=SignalDirection.SELL,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=0.65,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'volume_ratio': volume_ratio, 'obv': indicators['obv'].iloc[-1]},
                        levels=levels,
                        confluence=confluence,
                        description="OBV bearish divergence detected"
                    )
                    signals.append(signal)
            
            # Подтверждение тренда объемом
            price_trend = self._get_price_trend(df)
            volume_trend = "increasing" if volume_ratio > volume_ratio_prev else "decreasing"
            
            if price_trend == "up" and volume_trend == "increasing":
                # Восходящий тренд подтверждается растущим объемом
                confidence = 0.6
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.TREND_FOLLOWING,
                    direction=SignalDirection.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'volume_ratio': volume_ratio, 'volume_trend': volume_trend},
                    levels=levels,
                    confluence=confluence,
                    description=f"Uptrend confirmed by volume (Ratio: {volume_ratio:.2f}, Trend: {volume_trend})"
                )
                signals.append(signal)
            elif price_trend == "down" and volume_trend == "increasing":
                # Нисходящий тренд подтверждается растущим объемом
                confidence = 0.6
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.TREND_FOLLOWING,
                    direction=SignalDirection.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    timeframe=timeframe,
                    indicators={'volume_ratio': volume_ratio, 'volume_trend': volume_trend},
                    levels=levels,
                    confluence=confluence,
                    description=f"Downtrend confirmed by volume (Ratio: {volume_ratio:.2f}, Trend: {volume_trend})"
                )
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"❌ Ошибка генерации Volume сигналов: {e}")
        
        return signals
    
    def _generate_divergence_signals(self, df: pd.DataFrame, indicators: Dict, levels: Dict,
                                    confluence: Dict, symbol: str, timeframe: str) -> List[Signal]:
        """Генерирует сигналы на основе дивергенций."""
        signals = []
        
        if not self.use_divergence:
            return signals
        
        try:
            current_price = df['close'].iloc[-1]
            
            # RSI дивергенция (уже обрабатывается в _generate_rsi_signals)
            # Здесь можно добавить дополнительные проверки
            
            # MACD дивергенция (уже обрабатывается в _generate_macd_signals)
            # Здесь можно добавить дополнительные проверки
            
            # Ценовая дивергенция (не по индикатору)
            price_div_signals = self._check_price_divergence(df)
            
            for div_type, div_direction in price_div_signals:
                if div_direction == "bullish":
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.DIVERGENCE,
                        direction=SignalDirection.BUY,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=0.6,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'divergence_type': div_type},
                        levels=levels,
                        confluence=confluence,
                        description=f"Price {div_type} divergence detected"
                    )
                    signals.append(signal)
                elif div_direction == "bearish":
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.DIVERGENCE,
                        direction=SignalDirection.SELL,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=0.6,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'divergence_type': div_type},
                        levels=levels,
                        confluence=confluence,
                        description=f"Price {div_type} divergence detected"
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"❌ Ошибка генерации Divergence сигналов: {e}")
        
        return signals
    
    def _generate_pattern_signals(self, df: pd.DataFrame, indicators: Dict, levels: Dict,
                                 confluence: Dict, symbol: str, timeframe: str) -> List[Signal]:
        """Генерирует сигналы на основе графических паттернов."""
        signals = []
        
        if not self.use_patterns:
            return signals
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Обнаружение паттернов
            patterns = self._detect_chart_patterns(df)
            
            for pattern, pattern_direction in patterns:
                confidence = 0.6
                
                # Увеличиваем уверенность для определенных паттернов
                if pattern in ['double_bottom', 'head_shoulders_bottom']:
                    confidence = 0.7
                elif pattern in ['double_top', 'head_shoulders_top']:
                    confidence = 0.7
                
                if pattern_direction == "bullish":
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.PATTERN,
                        direction=SignalDirection.BUY,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'pattern': pattern},
                        levels=levels,
                        confluence=confluence,
                        description=f"{pattern.replace('_', ' ').title()} pattern detected"
                    )
                    signals.append(signal)
                elif pattern_direction == "bearish":
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.PATTERN,
                        direction=SignalDirection.SELL,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'pattern': pattern},
                        levels=levels,
                        confluence=confluence,
                        description=f"{pattern.replace('_', ' ').title()} pattern detected"
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"❌ Ошибка генерации Pattern сигналов: {e}")
        
        return signals
    
    def _generate_level_signals(self, df: pd.DataFrame, indicators: Dict, levels: Dict,
                               confluence: Dict, symbol: str, timeframe: str) -> List[Signal]:
        """Генерирует сигналы на основе уровней поддержки/сопротивления."""
        signals = []
        
        try:
            current_price = df['close'].iloc[-1]
            
            if not levels or 'supports' not in levels or 'resistances' not in levels:
                return signals
            
            # Извлекаем уровни из структуры
            supports = []
            resistances = []
            
            # Обрабатываем разные форматы уровней
            if isinstance(levels.get('supports'), list):
                for level in levels['supports']:
                    if hasattr(level, 'price'):
                        supports.append(level.price)
                    elif isinstance(level, dict) and 'price' in level:
                        supports.append(level['price'])
                    elif isinstance(level, (int, float)):
                        supports.append(float(level))
            
            if isinstance(levels.get('resistances'), list):
                for level in levels['resistances']:
                    if hasattr(level, 'price'):
                        resistances.append(level.price)
                    elif isinstance(level, dict) and 'price' in level:
                        resistances.append(level['price'])
                    elif isinstance(level, (int, float)):
                        resistances.append(float(level))
            
            # Находим ближайшие уровни
            nearest_support = None
            nearest_resistance = None
            min_support_distance = float('inf')
            min_resistance_distance = float('inf')
            
            for support in supports:
                if support < current_price:
                    distance = current_price - support
                    if distance < min_support_distance:
                        min_support_distance = distance
                        nearest_support = support
            
            for resistance in resistances:
                if resistance > current_price:
                    distance = resistance - current_price
                    if distance < min_resistance_distance:
                        min_resistance_distance = distance
                        nearest_resistance = resistance
            
            # Сигналы на основе уровней
            if nearest_support:
                support_distance_pct = min_support_distance / current_price
                
                # Отскок от поддержки
                if support_distance_pct < self.reversal_threshold:
                    confidence = 0.7
                    
                    # Проверяем бычье свечное формирование
                    last_candle_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
                    if last_candle_bullish:
                        confidence = 0.8
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.REVERSAL,
                        direction=SignalDirection.BUY,
                        strength=SignalStrength.STRONG if confidence > 0.75 else SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'support_price': nearest_support, 'distance_pct': support_distance_pct},
                        levels=levels,
                        confluence=confluence,
                        description=f"Bounce from support at {nearest_support:.2f} (Distance: {support_distance_pct:.2%})"
                    )
                    signals.append(signal)
                
                # Пробой поддержки
                elif support_distance_pct > self.breakout_threshold and df['close'].iloc[-1] < nearest_support:
                    confidence = 0.65
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BREAKOUT,
                        direction=SignalDirection.SELL,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'support_price': nearest_support, 'distance_pct': support_distance_pct},
                        levels=levels,
                        confluence=confluence,
                        description=f"Breakdown below support at {nearest_support:.2f} (Distance: {support_distance_pct:.2%})"
                    )
                    signals.append(signal)
            
            if nearest_resistance:
                resistance_distance_pct = min_resistance_distance / current_price
                
                # Отскок от сопротивления
                if resistance_distance_pct < self.reversal_threshold:
                    confidence = 0.7
                    
                    last_candle_bearish = df['close'].iloc[-1] < df['open'].iloc[-1]
                    if last_candle_bearish:
                        confidence = 0.8
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.REVERSAL,
                        direction=SignalDirection.SELL,
                        strength=SignalStrength.STRONG if confidence > 0.75 else SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'resistance_price': nearest_resistance, 'distance_pct': resistance_distance_pct},
                        levels=levels,
                        confluence=confluence,
                        description=f"Rejection from resistance at {nearest_resistance:.2f} (Distance: {resistance_distance_pct:.2%})"
                    )
                    signals.append(signal)
                
                # Пробой сопротивления
                elif resistance_distance_pct > self.breakout_threshold and df['close'].iloc[-1] > nearest_resistance:
                    confidence = 0.65
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BREAKOUT,
                        direction=SignalDirection.BUY,
                        strength=SignalStrength.MEDIUM,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'resistance_price': nearest_resistance, 'distance_pct': resistance_distance_pct},
                        levels=levels,
                        confluence=confluence,
                        description=f"Breakout above resistance at {nearest_resistance:.2f} (Distance: {resistance_distance_pct:.2%})"
                    )
                    signals.append(signal)
            
            # Консолидация между уровнями
            if nearest_support and nearest_resistance:
                range_pct = (nearest_resistance - nearest_support) / current_price
                
                if range_pct < 0.03:  # Узкий диапазон
                    confidence = 0.5
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BREAKOUT,
                        direction=SignalDirection.NEUTRAL,
                        strength=SignalStrength.WEAK,
                        price=current_price,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        timeframe=timeframe,
                        indicators={'support': nearest_support, 'resistance': nearest_resistance, 'range_pct': range_pct},
                        levels=levels,
                        confluence=confluence,
                        description=f"Consolidation between {nearest_support:.2f} and {nearest_resistance:.2f} (Range: {range_pct:.2%})"
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"❌ Ошибка генерации Level сигналов: {e}")
        
        return signals
    
    def _check_bullish_divergence(self, df: pd.DataFrame, indicator: pd.Series) -> bool:
        """Проверяет бычью дивергенцию."""
        if len(df) < 10 or len(indicator) < 10:
            return False
        
        try:
            # Ищем более низкие минимумы цены и более высокие минимумы индикатора
            prices = df['close'].values[-10:]
            indicator_values = indicator.values[-10:]
            
            # Находим два последних минимума цены
            min_idx1 = np.argmin(prices[:5])
            min_idx2 = 5 + np.argmin(prices[5:])
            
            if min_idx2 > min_idx1 and prices[min_idx2] < prices[min_idx1]:
                # Цена делает более низкий минимум
                if indicator_values[min_idx2] > indicator_values[min_idx1]:
                    # Индикатор делает более высокий минимум
                    return True
        except:
            pass
        
        return False
    
    def _check_bearish_divergence(self, df: pd.DataFrame, indicator: pd.Series) -> bool:
        """Проверяет медвежью дивергенцию."""
        if len(df) < 10 or len(indicator) < 10:
            return False
        
        try:
            # Ищем более высокие максимумы цены и более низкие максимумы индикатора
            prices = df['close'].values[-10:]
            indicator_values = indicator.values[-10:]
            
            # Находим два последних максимума цены
            max_idx1 = np.argmax(prices[:5])
            max_idx2 = 5 + np.argmax(prices[5:])
            
            if max_idx2 > max_idx1 and prices[max_idx2] > prices[max_idx1]:
                # Цена делает более высокий максимум
                if indicator_values[max_idx2] < indicator_values[max_idx1]:
                    # Индикатор делает более низкий максимум
                    return True
        except:
            pass
        
        return False
    
    def _check_macd_bullish_divergence(self, df: pd.DataFrame, macd: pd.Series) -> bool:
        """Проверяет бычью дивергенцию MACD."""
        return self._check_bullish_divergence(df, macd)
    
    def _check_macd_bearish_divergence(self, df: pd.DataFrame, macd: pd.Series) -> bool:
        """Проверяет медвежью дивергенцию MACD."""
        return self._check_bearish_divergence(df, macd)
    
    def _check_obv_bullish_divergence(self, df: pd.DataFrame, obv: pd.Series) -> bool:
        """Проверяет бычью дивергенцию OBV."""
        return self._check_bullish_divergence(df, obv)
    
    def _check_obv_bearish_divergence(self, df: pd.DataFrame, obv: pd.Series) -> bool:
        """Проверяет медвежью дивергенцию OBV."""
        return self._check_bearish_divergence(df, obv)
    
    def _check_price_divergence(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Проверяет ценовые дивергенции."""
        divergences = []
        
        if len(df) < 20:
            return divergences
        
        try:
            # Простая логика для демонстрации
            prices = df['close'].values[-20:]
            
            # Скользящие средние для сглаживания
            ma_short = pd.Series(prices).rolling(window=5).mean().values
            ma_long = pd.Series(prices).rolling(window=10).mean().values
            
            # Проверяем расхождения между MA
            if len(ma_short) > 10 and len(ma_long) > 10:
                # Бычья дивергенция: цена падает, но короткая MA отстает от длинной
                if (prices[-1] < prices[-10] and 
                    ma_short[-1] > ma_short[-10] and 
                    ma_long[-1] < ma_long[-10]):
                    divergences.append(("hidden_bullish", "bullish"))
                
                # Медвежья дивергенция: цена растет, но короткая MA отстает от длинной
                if (prices[-1] > prices[-10] and 
                    ma_short[-1] < ma_short[-10] and 
                    ma_long[-1] > ma_long[-10]):
                    divergences.append(("hidden_bearish", "bearish"))
        except:
            pass
        
        return divergences
    
    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Обнаруживает графические паттерны."""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        try:
            prices = df['close'].values[-20:]
            highs = df['high'].values[-20:]
            lows = df['low'].values[-20:]
            
            # Простая логика для демонстрации
            # В реальном проекте используйте библиотеку для распознавания паттернов
            
            # Проверяем Double Bottom
            if self._check_double_bottom(lows):
                patterns.append(("double_bottom", "bullish"))
            
            # Проверяем Double Top
            if self._check_double_top(highs):
                patterns.append(("double_top", "bearish"))
            
            # Проверяем Head and Shoulders
            if self._check_head_shoulders(highs, lows):
                patterns.append(("head_shoulders_top", "bearish"))
            
            # Проверяем Inverse Head and Shoulders
            if self._check_inverse_head_shoulders(highs, lows):
                patterns.append(("head_shoulders_bottom", "bullish"))
                
        except Exception as e:
            logger.error(f"❌ Ошибка обнаружения паттернов: {e}")
        
        return patterns
    
    def _check_double_bottom(self, lows: np.ndarray) -> bool:
        """Проверяет паттерн Double Bottom."""
        if len(lows) < 10:
            return False
        
        try:
            # Ищем два минимума примерно на одном уровне
            min1_idx = np.argmin(lows[:5])
            min2_idx = 5 + np.argmin(lows[5:])
            
            if abs(lows[min1_idx] - lows[min2_idx]) / lows[min1_idx] < 0.02:  # В пределах 2%
                # Проверяем, что между минимумами есть отскок
                middle_prices = lows[min1_idx+1:min2_idx]
                if len(middle_prices) > 0:
                    middle_avg = np.mean(middle_prices)
                    if middle_avg > lows[min1_idx] * 1.01:  # Отскок хотя бы на 1%
                        return True
        except:
            pass
        
        return False
    
    def _check_double_top(self, highs: np.ndarray) -> bool:
        """Проверяет паттерн Double Top."""
        if len(highs) < 10:
            return False
        
        try:
            # Ищем два максимума примерно на одном уровне
            max1_idx = np.argmax(highs[:5])
            max2_idx = 5 + np.argmax(highs[5:])
            
            if abs(highs[max1_idx] - highs[max2_idx]) / highs[max1_idx] < 0.02:  # В пределах 2%
                # Проверяем, что между максимумами есть откат
                middle_prices = highs[max1_idx+1:max2_idx]
                if len(middle_prices) > 0:
                    middle_avg = np.mean(middle_prices)
                    if middle_avg < highs[max1_idx] * 0.99:  # Откат хотя бы на 1%
                        return True
        except:
            pass
        
        return False
    
    def _check_head_shoulders(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Проверяет паттерн Head and Shoulders."""
        if len(highs) < 15 or len(lows) < 15:
            return False
        
        # Упрощенная проверка
        try:
            # Делим на три части
            part1 = highs[:5]
            part2 = highs[5:10]
            part3 = highs[10:15]
            
            if len(part1) > 0 and len(part2) > 0 and len(part3) > 0:
                max1 = np.max(part1)
                max2 = np.max(part2)
                max3 = np.max(part3)
                
                # Голова должна быть выше плеч
                if max2 > max1 and max2 > max3:
                    # Плечи примерно на одном уровне
                    if abs(max1 - max3) / max1 < 0.03:  # В пределах 3%
                        # Линия шеи (по минимумам)
                        neckline = min(lows[5:10])
                        if neckline < max1 * 0.98:  # Пробитие линии шеи
                            return True
        except:
            pass
        
        return False
    
    def _check_inverse_head_shoulders(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Проверяет паттерн Inverse Head and Shoulders."""
        if len(highs) < 15 or len(lows) < 15:
            return False
        
        # Упрощенная проверка
        try:
            # Делим на три части
            part1 = lows[:5]
            part2 = lows[5:10]
            part3 = lows[10:15]
            
            if len(part1) > 0 and len(part2) > 0 and len(part3) > 0:
                min1 = np.min(part1)
                min2 = np.min(part2)
                min3 = np.min(part3)
                
                # Голова должна быть ниже плеч
                if min2 < min1 and min2 < min3:
                    # Плечи примерно на одном уровне
                    if abs(min1 - min3) / min1 < 0.03:  # В пределах 3%
                        # Линия шеи (по максимумам)
                        neckline = max(highs[5:10])
                        if neckline > min1 * 1.02:  # Пробитие линии шеи
                            return True
        except:
            pass
        
        return False
    
    def _get_price_trend(self, df: pd.DataFrame) -> str:
        """Определяет тренд цены."""
        if len(df) < 10:
            return "neutral"
        
        try:
            # Простой анализ тренда
            prices = df['close'].values[-10:]
            
            # Линейная регрессия
            x = np.arange(len(prices))
            slope, _, _, _, _ = stats.linregress(x, prices)
            
            if slope > 0.001:  # Положительный наклон
                return "up"
            elif slope < -0.001:  # Отрицательный наклон
                return "down"
            else:
                return "neutral"
        except:
            return "neutral"
    
    def _filter_and_rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """Фильтрует и ранжирует сигналы."""
        if not signals:
            return []
        
        try:
            # Убираем нейтральные сигналы с низкой уверенностью
            filtered_signals = []
            for signal in signals:
                if signal.direction == SignalDirection.NEUTRAL and signal.confidence < 0.7:
                    continue
                filtered_signals.append(signal)
            
            # Группируем по направлению
            buy_signals = [s for s in filtered_signals if s.direction == SignalDirection.BUY]
            sell_signals = [s for s in filtered_signals if s.direction == SignalDirection.SELL]
            neutral_signals = [s for s in filtered_signals if s.direction == SignalDirection.NEUTRAL]
            
            # Сортируем по уверенности
            buy_signals.sort(key=lambda x: x.confidence, reverse=True)
            sell_signals.sort(key=lambda x: x.confidence, reverse=True)
            neutral_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            # Объединяем, оставляя только лучшие сигналы каждого типа
            result = []
            
            # Оставляем топ-2 сигнала каждого направления
            for signal_list in [buy_signals, sell_signals, neutral_signals]:
                result.extend(signal_list[:2])
            
            # Дополнительная фильтрация конфликтующих сигналов
            if buy_signals and sell_signals:
                # Если есть оба направления, оставляем только те, у которых разница в уверенности > 0.2
                best_buy = buy_signals[0] if buy_signals else None
                best_sell = sell_signals[0] if sell_signals else None
                
                if best_buy and best_sell:
                    confidence_diff = abs(best_buy.confidence - best_sell.confidence)
                    if confidence_diff < 0.2:
                        # Слишком близкие уверенности, оставляем только нейтральные
                        result = [s for s in result if s.direction == SignalDirection.NEUTRAL]
            
            # Фильтруем по минимальной уверенности
            result = [s for s in result if s.confidence >= self.min_confidence]
            
            logger.debug(f"📊 Фильтрация сигналов: {len(signals)} -> {len(result)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка фильтрации сигналов: {e}")
            return signals
    
    def _add_risk_management(self, signals: List[Signal], df: pd.DataFrame, levels: Dict) -> List[Signal]:
        """Добавляет стоп-лоссы и тейк-профиты к сигналам."""
        if not signals:
            return signals
        
        try:
            atr = self._calculate_atr(df, 14).iloc[-1] if len(df) >= 14 else df['close'].iloc[-1] * 0.02
            
            for signal in signals:
                if signal.direction == SignalDirection.NEUTRAL:
                    continue
                
                current_price = signal.price
                
                # Базовые значения на основе ATR
                atr_multiplier = 2.0
                base_sl_distance = atr * atr_multiplier
                base_tp_distance = base_sl_distance * 2  # Риск-прибыль 1:2
                
                # Настройка для разных типов сигналов
                if signal.signal_type == SignalType.BREAKOUT:
                    # Для пробоев используем более широкие стопы
                    base_sl_distance = atr * 3.0
                    base_tp_distance = base_sl_distance * 3  # Риск-прибыль 1:3
                elif signal.signal_type == SignalType.REVERSAL:
                    # Для разворотов используем более узкие стопы
                    base_sl_distance = atr * 1.5
                    base_tp_distance = base_sl_distance * 1.5  # Риск-прибыль 1:1.5
                
                # Корректировка на основе уровней
                if signal.direction == SignalDirection.BUY:
                    # Для покупок: стоп-лосс ниже, тейк-профит выше
                    
                    # Ищем ближайшую поддержку для стоп-лосса
                    if levels and 'supports' in levels:
                        supports = []
                        for level in levels['supports']:
                            if hasattr(level, 'price'):
                                supports.append(level.price)
                            elif isinstance(level, dict) and 'price' in level:
                                supports.append(level['price'])
                            elif isinstance(level, (int, float)):
                                supports.append(float(level))
                        
                        if supports:
                            supports_below = [s for s in supports if s < current_price]
                            if supports_below:
                                nearest_support = max(supports_below)
                                support_distance = current_price - nearest_support
                                
                                # Используем поддержку если она не слишком далеко
                                if support_distance < base_sl_distance * 2:
                                    signal.stop_loss = nearest_support * 0.995  # Немного ниже поддержки
                                else:
                                    signal.stop_loss = current_price - base_sl_distance
                            else:
                                signal.stop_loss = current_price - base_sl_distance
                        else:
                            signal.stop_loss = current_price - base_sl_distance
                    else:
                        signal.stop_loss = current_price - base_sl_distance
                    
                    # Тейк-профит
                    signal.take_profit = current_price + base_tp_distance
                    
                    # Корректируем тейк-профит по сопротивлениям
                    if levels and 'resistances' in levels:
                        resistances = []
                        for level in levels['resistances']:
                            if hasattr(level, 'price'):
                                resistances.append(level.price)
                            elif isinstance(level, dict) and 'price' in level:
                                resistances.append(level['price'])
                            elif isinstance(level, (int, float)):
                                resistances.append(float(level))
                        
                        if resistances:
                            resistances_above = [r for r in resistances if r > current_price]
                            if resistances_above:
                                nearest_resistance = min(resistances_above)
                                # Используем сопротивление если оно не слишком далеко
                                if nearest_resistance < signal.take_profit:
                                    signal.take_profit = nearest_resistance * 0.995  # Немного ниже сопротивления
                
                elif signal.direction == SignalDirection.SELL:
                    # Для продаж: стоп-лосс выше, тейк-профит ниже
                    
                    # Ищем ближайшее сопротивление для стоп-лосса
                    if levels and 'resistances' in levels:
                        resistances = []
                        for level in levels['resistances']:
                            if hasattr(level, 'price'):
                                resistances.append(level.price)
                            elif isinstance(level, dict) and 'price' in level:
                                resistances.append(level['price'])
                            elif isinstance(level, (int, float)):
                                resistances.append(float(level))
                        
                        if resistances:
                            resistances_above = [r for r in resistances if r > current_price]
                            if resistances_above:
                                nearest_resistance = min(resistances_above)
                                resistance_distance = nearest_resistance - current_price
                                
                                if resistance_distance < base_sl_distance * 2:
                                    signal.stop_loss = nearest_resistance * 1.005  # Немного выше сопротивления
                                else:
                                    signal.stop_loss = current_price + base_sl_distance
                            else:
                                signal.stop_loss = current_price + base_sl_distance
                        else:
                            signal.stop_loss = current_price + base_sl_distance
                    else:
                        signal.stop_loss = current_price + base_sl_distance
                    
                    # Тейк-профит
                    signal.take_profit = current_price - base_tp_distance
                    
                    # Корректируем тейк-профит по поддержкам
                    if levels and 'supports' in levels:
                        supports = []
                        for level in levels['supports']:
                            if hasattr(level, 'price'):
                                supports.append(level.price)
                            elif isinstance(level, dict) and 'price' in level:
                                supports.append(level['price'])
                            elif isinstance(level, (int, float)):
                                supports.append(float(level))
                        
                        if supports:
                            supports_below = [s for s in supports if s < current_price]
                            if supports_below:
                                nearest_support = max(supports_below)
                                # Используем поддержку если она не слишком далеко
                                if nearest_support > signal.take_profit:
                                    signal.take_profit = nearest_support * 1.005  # Немного выше поддержки
                
                # Расчет соотношения риск/прибыль
                if signal.stop_loss and signal.take_profit:
                    if signal.direction == SignalDirection.BUY:
                        risk = current_price - signal.stop_loss
                        reward = signal.take_profit - current_price
                    else:  # SELL
                        risk = signal.stop_loss - current_price
                        reward = current_price - signal.take_profit
                    
                    if risk > 0:
                        signal.risk_reward_ratio = reward / risk
                    else:
                        signal.risk_reward_ratio = 0
                
                # Добавляем информацию о риск-менеджменте в описание
                if signal.stop_loss and signal.take_profit and signal.risk_reward_ratio:
                    signal.description += f" | SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}, R/R: {signal.risk_reward_ratio:.2f}"
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления риск-менеджмента: {e}")
            return signals
    
    def _calculate_statistics(self, start_time: datetime, end_time: datetime,
                             signals: List[Signal]) -> Dict[str, Any]:
        """Рассчитывает статистику анализа."""
        processing_time = (end_time - start_time).total_seconds()
        
        # Распределение по типам
        type_dist = {}
        for signal in signals:
            signal_type = signal.signal_type.value
            type_dist[signal_type] = type_dist.get(signal_type, 0) + 1
        
        # Распределение по направлениям
        direction_dist = {
            'BUY': len([s for s in signals if s.direction == SignalDirection.BUY]),
            'SELL': len([s for s in signals if s.direction == SignalDirection.SELL]),
            'NEUTRAL': len([s for s in signals if s.direction == SignalDirection.NEUTRAL])
        }
        
        # Средняя уверенность
        avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0
        
        # Распределение по силе
        strength_dist = {
            'very_strong': len([s for s in signals if s.strength == SignalStrength.VERY_STRONG]),
            'strong': len([s for s in signals if s.strength == SignalStrength.STRONG]),
            'medium': len([s for s in signals if s.strength == SignalStrength.MEDIUM]),
            'weak': len([s for s in signals if s.strength == SignalStrength.WEAK])
        }
        
        return {
            'processing_time_seconds': processing_time,
            'total_signals': len(signals),
            'average_confidence': avg_confidence,
            'type_distribution': type_dist,
            'direction_distribution': direction_dist,
            'strength_distribution': strength_dist
        }
    
    def _log_results(self, result: SignalAnalysisResult):
        """Логирует результаты анализа."""
        signals = result.signals
        market_condition = result.market_condition
        stats = result.statistics
        
        logger.info(f"✅ Анализ сигналов для {result.symbol} {result.timeframe}:")
        logger.info(f"   📊 Всего сигналов: {len(signals)}")
        logger.info(f"   ⚡ Обработка: {stats['processing_time_seconds']:.3f}с")
        logger.info(f"   🎯 Средняя уверенность: {stats['average_confidence']:.1%}")
        logger.info(f"   📈 Рыночные условия: {market_condition.get('trend', 'unknown')}, "
                   f"{market_condition.get('volatility', 'unknown')}, {market_condition.get('overall_bias', 'unknown')}")
        
        if signals:
            # Группируем по направлению
            buy_signals = [s for s in signals if s.direction == SignalDirection.BUY]
            sell_signals = [s for s in signals if s.direction == SignalDirection.SELL]
            neutral_signals = [s for s in signals if s.direction == SignalDirection.NEUTRAL]
            
            if buy_signals:
                logger.info(f"   🟢 Сигналы на покупку: {len(buy_signals)}")
                for i, signal in enumerate(buy_signals[:3], 1):  # Только топ-3
                    logger.info(f"     {i}. {signal.signal_type.value} @ ${signal.price:.2f} "
                               f"(уверенность: {signal.confidence:.1%}, сила: {signal.strength.value})")
            
            if sell_signals:
                logger.info(f"   🔴 Сигналы на продажу: {len(sell_signals)}")
                for i, signal in enumerate(sell_signals[:3], 1):
                    logger.info(f"     {i}. {signal.signal_type.value} @ ${signal.price:.2f} "
                               f"(уверенность: {signal.confidence:.1%}, сила: {signal.strength.value})")
            
            if neutral_signals:
                logger.info(f"   ⚪ Нейтральные сигналы: {len(neutral_signals)}")
                for i, signal in enumerate(neutral_signals[:3], 1):
                    logger.info(f"     {i}. {signal.signal_type.value} @ ${signal.price:.2f} "
                               f"(уверенность: {signal.confidence:.1%})")
        else:
            logger.info(f"   📭 Сигналы не обнаружены")
    
    def _clean_cache(self):
        """Очищает старые записи из кеша."""
        if len(self.cache) > self.cache_max_size:
            keys_to_remove = list(self.cache.keys())[:len(self.cache) - self.cache_max_size]
            for key in keys_to_remove:
                del self.cache[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы генератора."""
        return {
            'version': self.VERSION,
            'total_analyses': self.stats['total_analyses'],
            'signals_generated': self.stats['signals_generated'],
            'cache_hits': self.stats['cache_hits'],
            'cache_size': len(self.cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_analyses'], 1),
            'errors_count': len(self.stats['errors']),
            'recent_errors': self.stats['errors'][-5:] if self.stats['errors'] else [],
            'configuration': {
                'min_confidence': self.min_confidence,
                'use_rsi': self.use_rsi,
                'use_macd': self.use_macd,
                'use_ma': self.use_ma,
                'use_bollinger': self.use_bollinger,
                'use_volume': self.use_volume,
                'use_divergence': self.use_divergence,
                'use_patterns': self.use_patterns
            }
        }

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def validate_signal(signal: Signal, current_price: float, 
                    max_price_deviation: float = 0.05) -> bool:
    """
    Проверяет валидность сигнала.
    
    Args:
        signal: Торговый сигнал для проверки
        current_price: Текущая цена
        max_price_deviation: Максимальное отклонение цены сигнала от текущей
        
    Returns:
        bool: True если сигнал валиден
    """
    if not signal:
        return False
    
    # Проверка направления
    if signal.direction not in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.NEUTRAL]:
        return False
    
    # Проверка цены
    price_diff = abs(signal.price - current_price) / current_price
    if price_diff > max_price_deviation:
        return False
    
    # Проверка уверенности
    if signal.confidence < 0.3:
        return False
    
    # Проверка стоп-лосса и тейк-профита
    if signal.direction in [SignalDirection.BUY, SignalDirection.SELL]:
        if signal.stop_loss is None or signal.take_profit is None:
            return False
        
        # Проверка логики стоп-лосса и тейк-профита
        if signal.direction == SignalDirection.BUY:
            if signal.stop_loss >= signal.price or signal.take_profit <= signal.price:
                return False
        elif signal.direction == SignalDirection.SELL:
            if signal.stop_loss <= signal.price or signal.take_profit >= signal.price:
                return False
    
    return True

def merge_similar_signals(signals: List[Signal], price_tolerance: float = 0.01) -> List[Signal]:
    """
    Объединяет похожие сигналы.
    
    Args:
        signals: Список сигналов
        price_tolerance: Допустимое отклонение в цене для объединения
        
    Returns:
        List[Signal]: Объединенный список сигналов
    """
    if not signals or len(signals) < 2:
        return signals
    
    merged_signals = []
    processed_indices = set()
    
    for i, signal1 in enumerate(signals):
        if i in processed_indices:
            continue
        
        similar_signals = [signal1]
        
        for j, signal2 in enumerate(signals[i+1:], i+1):
            if j in processed_indices:
                continue
            
            # Проверяем схожесть
            if (signal1.direction == signal2.direction and
                signal1.signal_type == signal2.signal_type and
                abs(signal1.price - signal2.price) / signal1.price <= price_tolerance):
                
                similar_signals.append(signal2)
                processed_indices.add(j)
        
        if len(similar_signals) == 1:
            merged_signals.append(signal1)
        else:
            # Объединяем похожие сигналы
            avg_price = np.mean([s.price for s in similar_signals])
            avg_confidence = np.mean([s.confidence for s in similar_signals])
            max_strength = max(similar_signals, key=lambda x: x.confidence).strength
            
            merged_signal = Signal(
                symbol=signal1.symbol,
                signal_type=signal1.signal_type,
                direction=signal1.direction,
                strength=max_strength,
                price=avg_price,
                confidence=avg_confidence,
                timestamp=max(similar_signals, key=lambda x: x.timestamp).timestamp,
                timeframe=signal1.timeframe,
                description=f"Merged from {len(similar_signals)} similar signals",
                metadata={'merged_count': len(similar_signals)}
            )
            
            merged_signals.append(merged_signal)
        
        processed_indices.add(i)
    
    return merged_signals

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    # Тестирование модуля генерации сигналов
    print("🧪 Тестирование SignalGenerator...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')
    prices = 50000 + np.cumsum(np.random.randn(200) * 100)
    
    df = pd.DataFrame({
        'open': prices - np.random.rand(200) * 100,
        'high': prices + np.random.rand(200) * 150,
        'low': prices - np.random.rand(200) * 150,
        'close': prices,
        'volume': np.random.rand(200) * 1000 + 500
    }, index=dates)
    
    # Тестовые уровни
    test_levels = {
        'supports': [
            {'price': 49000, 'strength': 'strong'},
            {'price': 49500, 'strength': 'medium'}
        ],
        'resistances': [
            {'price': 51000, 'strength': 'strong'},
            {'price': 51500, 'strength': 'medium'}
        ]
    }
    
    # Тестовый конфлюэнс
    test_confluence = {
        'score': 0.7,
        'strength': 'medium',
        'factors': {
            'multi_timeframe': 0.8,
            'volume_confirmation': 0.6
        }
    }
    
    # Создаем и тестируем генератор
    generator = SignalGenerator(
        min_confidence=0.5,
        use_rsi=True,
        use_macd=True,
        use_ma=True,
        use_bollinger=True,
        use_volume=True,
        use_divergence=True,
        use_patterns=True
    )
    
    result = generator.analyze(df, test_levels, test_confluence, "BTC/USDT", "1h")
    
    print(f"\n📊 Результаты анализа:")
    print(f"   Всего сигналов: {len(result.signals)}")
    print(f"   Время обработки: {result.statistics['processing_time_seconds']:.3f}с")
    print(f"   Средняя уверенность: {result.statistics['average_confidence']:.1%}")
    
    if result.signals:
        print(f"\n🎯 Обнаруженные сигналы:")
        for i, signal in enumerate(result.signals, 1):
            print(f"   {i}. {signal.direction.value} {signal.signal_type.value} @ ${signal.price:.2f}")
            print(f"      Уверенность: {signal.confidence:.1%}, Сила: {signal.strength.value}")
            print(f"      Описание: {signal.description}")
            
            if signal.stop_loss and signal.take_profit:
                print(f"      SL: ${signal.stop_loss:.2f}, TP: ${signal.take_profit:.2f}, "
                      f"R/R: {signal.risk_reward_ratio:.2f}")
            print()
    else:
        print("\n📭 Сигналы не обнаружены")
    
    # Статистика генератора
    stats = generator.get_statistics()
    print(f"\n📈 Статистика генератора:")
    print(f"   Всего анализов: {stats['total_analyses']}")
    print(f"   Сгенерировано сигналов: {stats['signals_generated']}")
    print(f"   Попаданий в кеш: {stats['cache_hits']}")
    print(f"   Размер кеша: {stats['cache_size']}")
    
    print("\n✅ Тестирование завершено!")
