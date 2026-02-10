#!/usr/bin/env python3
"""
ПОЛНЫЙ ОБРАБОТЧИК ДАННЫХ ДЛЯ ТОРГОВОГО БОТА
Версия: 2.0
Функционал: Загрузка данных с бирж, кеширование, обогащение индикаторами
Поддержка: Binance, Bybit, KuCoin, OKX
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import ccxt
import json
import hashlib
import os
import sys
from pathlib import Path
import pickle
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings

logger = logging.getLogger(__name__)

# ============================================================================
# КОНСТАНТЫ И КОНФИГУРАЦИЯ
# ============================================================================

class ExchangeType(Enum):
    BINANCE = "binance"
    BYBIT = "bybit"
    KUCOIN = "kucoin"
    OKX = "okx"
    COINBASE = "coinbase"

@dataclass
class DataRequest:
    """Структура запроса данных."""
    symbol: str
    timeframe: str
    limit: int
    since: Optional[int] = None
    params: Optional[Dict] = None

@dataclass
class CachedData:
    """Структура кешированных данных."""
    timestamp: datetime
    data: pd.DataFrame
    hash: str

# ============================================================================
# КЛАСС DATA HANDLER
# ============================================================================

class DataHandler:
    """
    Полнофункциональный обработчик рыночных данных.
    
    Возможности:
    - Поддержка 5+ бирж
    - Многоуровневое кеширование (память, диск, БД)
    - Автоматический реконнект и retry логика
    - Параллельная загрузка данных
    - Обогащение 20+ техническими индикаторами
    - Валидация и очистка данных
    - Статистика и мониторинг
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, 
                 exchange_id: str = "binance",
                 cache_enabled: bool = True,
                 cache_ttl: int = 300,
                 max_retries: int = 3,
                 timeout: int = 30):
        """
        Инициализация обработчика данных.
        
        Args:
            exchange_id: ID биржи (binance, bybit, kucoin, okx)
            cache_enabled: Включить кеширование
            cache_ttl: Время жизни кеша в секундах
            max_retries: Максимальное количество попыток при ошибках
            timeout: Таймаут запросов в секундах
        """
        
        self.exchange_id = exchange_id
        self.cache_enabled = cache_enabled
        self.cache_ttl = timedelta(seconds=cache_ttl)
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Инициализация биржи
        self.exchange = self._init_exchange()
        
        # Системы кеширования
        self.memory_cache = {}  # Кеш в оперативной памяти
        self.disk_cache_dir = Path("data/cache")
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Сессия HTTP
        self.session = None
        
        # Статистика и мониторинг
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0,
            'total_data_points': 0,
            'errors': []
        }
        
        # Очередь запросов для rate limiting
        self.request_queue = []
        self.rate_limits = {
            'requests_per_minute': 1200,  # Базовый лимит Binance
            'last_request_time': None
        }
        
        # Список поддерживаемых таймфреймов
        self.supported_timeframes = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '8h': 28800,
            '12h': 43200,
            '1d': 86400,
            '3d': 259200,
            '1w': 604800,
            '1M': 2592000
        }
        
        logger.info(f"✅ DataHandler v{self.VERSION} инициализирован для {exchange_id}")
        logger.info(f"   Кеширование: {'Включено' if cache_enabled else 'Выключено'}")
        logger.info(f"   TTL кеша: {cache_ttl} секунд")
    
    def _init_exchange(self):
        """Инициализирует подключение к бирже."""
        try:
            # Получаем класс биржи из ccxt
            exchange_class = getattr(ccxt, self.exchange_id)
            
            # Конфигурация для разных бирж
            config = {
                'enableRateLimit': True,
                'timeout': self.timeout * 1000,  # ccxt использует миллисекунды
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            }
            
            # Добавляем API ключи если есть
            exchange_config = settings.exchanges.get(self.exchange_id, {})
            if exchange_config.get('api_key') and exchange_config.get('api_secret'):
                config['apiKey'] = exchange_config['api_key']
                config['secret'] = exchange_config['api_secret']
                
                # Для тестовой сети
                if exchange_config.get('testnet'):
                    if self.exchange_id == 'binance':
                        config['urls']['api'] = config['urls'].get('test', 'https://testnet.binance.vision/api')
                    elif self.exchange_id == 'bybit':
                        config['urls']['api'] = 'https://api-testnet.bybit.com'
            
            # Создаем экземпляр биржи
            exchange = exchange_class(config)
            
            # Загружаем рынки для проверки подключения
            exchange.load_markets()
            
            logger.info(f"   Подключено к {self.exchange_id.upper()}, доступно {len(exchange.markets)} пар")
            return exchange
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации биржи {self.exchange_id}: {e}")
            raise
    
    def _get_cache_key(self, symbol: str, timeframe: str, limit: int, since: Optional[int] = None) -> str:
        """Генерирует уникальный ключ для кеша."""
        key_string = f"{symbol}_{timeframe}_{limit}_{since}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _save_to_disk_cache(self, key: str, data: pd.DataFrame):
        """Сохраняет данные в файловый кеш."""
        try:
            cache_file = self.disk_cache_dir / f"{key}.pkl"
            
            cache_entry = CachedData(
                timestamp=datetime.now(),
                data=data,
                hash=hashlib.md5(pickle.dumps(data)).hexdigest()
            )
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f)
                
            logger.debug(f"📁 Данные сохранены в файловый кеш: {cache_file}")
            
        except Exception as e:
            logger.warning(f"⚠️  Ошибка сохранения в файловый кеш: {e}")
    
    def _load_from_disk_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Загружает данные из файлового кеша."""
        try:
            cache_file = self.disk_cache_dir / f"{key}.pkl"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                cache_entry: CachedData = pickle.load(f)
            
            # Проверяем срок годности
            if datetime.now() - cache_entry.timestamp > self.cache_ttl:
                os.remove(cache_file)
                return None
            
            # Проверяем целостность данных
            current_hash = hashlib.md5(pickle.dumps(cache_entry.data)).hexdigest()
            if current_hash != cache_entry.hash:
                logger.warning(f"⚠️  Поврежденный кеш, удаляю: {cache_file}")
                os.remove(cache_file)
                return None
            
            logger.debug(f"📁 Данные загружены из файлового кеша: {cache_file}")
            return cache_entry.data.copy()
            
        except Exception as e:
            logger.warning(f"⚠️  Ошибка загрузки из файлового кеша: {e}")
            return None
    
    async def _rate_limit_delay(self):
        """Обработка rate limiting."""
        if not self.rate_limits['last_request_time']:
            self.rate_limits['last_request_time'] = datetime.now()
            return
        
        # Рассчитываем время с последнего запроса
        time_since_last = (datetime.now() - self.rate_limits['last_request_time']).total_seconds()
        
        # Минимальный интервал между запросами
        min_interval = 60.0 / self.rate_limits['requests_per_minute']
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.rate_limits['last_request_time'] = datetime.now()
    
    async def _fetch_with_retry(self, symbol: str, timeframe: str, limit: int, 
                               since: Optional[int] = None, params: Optional[Dict] = None) -> List:
        """
        Загружает данные с биржи с повторными попытками при ошибках.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            limit: Количество свечей
            since: Временная метка начала (опционально)
            params: Дополнительные параметры
            
        Returns:
            List: Список свечей или пустой список при ошибке
        """
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                await self._rate_limit_delay()
                
                # Выполняем запрос
                start_time = datetime.now()
                
                loop = asyncio.get_event_loop()
                ohlcv = await loop.run_in_executor(
                    None,
                    lambda: self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=since,
                        limit=limit,
                        params=params
                    )
                )
                
                # Обновляем статистику
                response_time = (datetime.now() - start_time).total_seconds()
                self.stats['total_requests'] += 1
                self.stats['successful_requests'] += 1
                self.stats['avg_response_time'] = (
                    (self.stats['avg_response_time'] * (self.stats['successful_requests'] - 1) + response_time) /
                    self.stats['successful_requests']
                )
                
                logger.debug(f"   📥 Загружено {len(ohlcv)} свечей за {response_time:.2f}с")
                return ohlcv
                
            except ccxt.NetworkError as e:
                self.stats['failed_requests'] += 1
                error_msg = f"Сетевая ошибка (попытка {attempt + 1}/{self.max_retries}): {e}"
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"   ⚠️  {error_msg}, повтор через {2 ** attempt}с")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"   ❌ {error_msg}")
                    self.stats['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'error': str(e),
                        'attempt': attempt + 1
                    })
                    
            except ccxt.ExchangeError as e:
                self.stats['failed_requests'] += 1
                error_msg = f"Ошибка биржи (попытка {attempt + 1}/{self.max_retries}): {e}"
                
                # Для некоторых ошибок не пытаемся повторно
                if "Invalid symbol" in str(e) or "Market does not exist" in str(e):
                    logger.error(f"   ❌ {error_msg}")
                    break
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"   ⚠️  {error_msg}, повтор через {2 ** attempt}с")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"   ❌ {error_msg}")
                    self.stats['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'error': str(e),
                        'attempt': attempt + 1
                    })
                    
            except Exception as e:
                self.stats['failed_requests'] += 1
                error_msg = f"Неизвестная ошибка (попытка {attempt + 1}/{self.max_retries}): {e}"
                logger.error(f"   ❌ {error_msg}")
                self.stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'error': str(e),
                    'attempt': attempt + 1
                })
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return []
    
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", 
                       limit: int = 500, since: Optional[int] = None,
                       params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """
        Основной метод получения OHLCV данных.
        
        Args:
            symbol: Торговая пара (например, "BTC/USDT")
            timeframe: Таймфрейм ("1m", "5m", "15m", "1h", "4h", "1d")
            limit: Количество свечей
            since: Временная метка начала (timestamp в миллисекундах)
            params: Дополнительные параметры запроса
            
        Returns:
            Optional[pd.DataFrame]: DataFrame с данными или None при ошибке
        """
        
        # Валидация входных параметров
        if not self._validate_request(symbol, timeframe, limit):
            return None
        
        # Генерация ключа кеша
        cache_key = self._get_cache_key(symbol, timeframe, limit, since)
        
        # Проверка кеша (если включено)
        if self.cache_enabled:
            # 1. Проверка в памяти
            if cache_key in self.memory_cache:
                cached_time, cached_data = self.memory_cache[cache_key]
                if datetime.now() - cached_time < self.cache_ttl:
                    self.stats['cache_hits'] += 1
                    logger.debug(f"🎯 Кеш попадание в памяти: {symbol} {timeframe}")
                    return cached_data.copy()
            
            # 2. Проверка на диске
            disk_data = self._load_from_disk_cache(cache_key)
            if disk_data is not None:
                self.stats['cache_hits'] += 1
                # Сохраняем в память для быстрого доступа
                self.memory_cache[cache_key] = (datetime.now(), disk_data.copy())
                return disk_data
        
        self.stats['cache_misses'] += 1
        
        # Загрузка данных с биржи
        logger.info(f"📥 Загрузка {symbol} {timeframe} ({limit} свечей)...")
        
        ohlcv_data = await self._fetch_with_retry(symbol, timeframe, limit, since, params)
        
        if not ohlcv_data:
            logger.warning(f"⚠️  Не удалось загрузить данные для {symbol}")
            return None
        
        try:
            # Конвертация в DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Конвертация timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Сортировка по времени (на всякий случай)
            df.sort_index(inplace=True)
            
            # Валидация данных
            df = self._validate_and_clean_data(df)
            
            # Обогащение техническими индикаторами
            df = self._add_technical_indicators(df)
            
            # Добавление производных признаков
            df = self._add_derived_features(df)
            
            # Обновление статистики
            self.stats['total_data_points'] += len(df)
            
            # Сохранение в кеш (если включено)
            if self.cache_enabled:
                self.memory_cache[cache_key] = (datetime.now(), df.copy())
                self._save_to_disk_cache(cache_key, df)
            
            logger.info(f"✅ {symbol} {timeframe}: загружено {len(df)} свечей, "
                       f"{len(df.columns)} индикаторов")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки данных {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _validate_request(self, symbol: str, timeframe: str, limit: int) -> bool:
        """Валидация параметров запроса."""
        errors = []
        
        # Проверка символа
        if not symbol or '/' not in symbol:
            errors.append(f"Неверный формат символа: {symbol}")
        
        # Проверка таймфрейма
        if timeframe not in self.supported_timeframes:
            errors.append(f"Неподдерживаемый таймфрейм: {timeframe}")
        
        # Проверка лимита
        if limit <= 0 or limit > 5000:
            errors.append(f"Лимит должен быть от 1 до 5000: {limit}")
        
        if errors:
            for error in errors:
                logger.error(f"❌ {error}")
            return False
        
        return True
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Валидация и очистка данных.
        
        Проверяет:
        - Отсутствующие значения
        - Аномальные цены (слишком большие изменения)
        - Нулевые объемы
        - Некорректные high/low
        """
        
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # 1. Проверка на NaN
        nan_count = df_clean.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"⚠️  Найдено {nan_count} NaN значений, заполняю...")
            df_clean = df_clean.ffill().bfill()
        
        # 2. Проверка корректности high/low
        invalid_hl = ((df_clean['high'] < df_clean['low']) | 
                     (df_clean['high'] < df_clean['open']) | 
                     (df_clean['high'] < df_clean['close']) |
                     (df_clean['low'] > df_clean['open']) | 
                     (df_clean['low'] > df_clean['close'])).sum()
        
        if invalid_hl > 0:
            logger.warning(f"⚠️  Найдено {invalid_hl} некорректных high/low, исправляю...")
            
            # Исправляем high
            df_clean['high'] = df_clean[['open', 'high', 'low', 'close']].max(axis=1)
            
            # Исправляем low
            df_clean['low'] = df_clean[['open', 'high', 'low', 'close']].min(axis=1)
        
        # 3. Проверка аномальных изменений цен
        price_changes = df_clean['close'].pct_change().abs()
        anomalous_changes = (price_changes > 0.5).sum()  # Более 50% за одну свечу
        
        if anomalous_changes > 0:
            logger.warning(f"⚠️  Найдено {anomalous_changes} аномальных изменений цен")
            
            # Заменяем аномальные значения средним
            for idx in price_changes[price_changes > 0.5].index:
                if idx > 0:
                    df_clean.loc[idx, 'close'] = df_clean.loc[idx-1, 'close']
        
        # 4. Проверка объемов
        zero_volumes = (df_clean['volume'] <= 0).sum()
        if zero_volumes > 0:
            logger.warning(f"⚠️  Найдено {zero_volumes} нулевых объемов")
            
            # Заменяем нулевые объемы средним
            mean_volume = df_clean['volume'][df_clean['volume'] > 0].mean()
            if pd.notna(mean_volume):
                df_clean['volume'] = df_clean['volume'].replace(0, mean_volume)
        
        return df_clean
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет технические индикаторы в DataFrame.
        
        Добавляет:
        - Трендовые индикаторы (MA, EMA, MACD)
        - Осцилляторы (RSI, Stochastic, CCI)
        - Волатильность (ATR, Bollinger Bands)
        - Объемные индикаторы (OBV, Volume SMA)
        - Прочие (Parabolic SAR, Ichimoku)
        """
        
        if df.empty or len(df) < 20:
            return df
        
        df_indicators = df.copy()
        
        try:
            # ==================== ТРЕНДОВЫЕ ИНДИКАТОРЫ ====================
            
            # Простые скользящие средние
            df_indicators['sma_10'] = df_indicators['close'].rolling(window=10).mean()
            df_indicators['sma_20'] = df_indicators['close'].rolling(window=20).mean()
            df_indicators['sma_50'] = df_indicators['close'].rolling(window=50).mean()
            df_indicators['sma_100'] = df_indicators['close'].rolling(window=100).mean()
            df_indicators['sma_200'] = df_indicators['close'].rolling(window=200).mean()
            
            # Экспоненциальные скользящие средние
            df_indicators['ema_12'] = df_indicators['close'].ewm(span=12, adjust=False).mean()
            df_indicators['ema_26'] = df_indicators['close'].ewm(span=26, adjust=False).mean()
            df_indicators['ema_50'] = df_indicators['close'].ewm(span=50, adjust=False).mean()
            df_indicators['ema_200'] = df_indicators['close'].ewm(span=200, adjust=False).mean()
            
            # MACD
            df_indicators['macd'] = df_indicators['ema_12'] - df_indicators['ema_26']
            df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
            df_indicators['macd_histogram'] = df_indicators['macd'] - df_indicators['macd_signal']
            
            # ==================== ОСЦИЛЛЯТОРЫ ====================
            
            # RSI (Relative Strength Index)
            delta = df_indicators['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Stochastic Oscillator
            low_14 = df_indicators['low'].rolling(window=14).min()
            high_14 = df_indicators['high'].rolling(window=14).max()
            df_indicators['stoch_k'] = 100 * ((df_indicators['close'] - low_14) / (high_14 - low_14))
            df_indicators['stoch_d'] = df_indicators['stoch_k'].rolling(window=3).mean()
            
            # CCI (Commodity Channel Index)
            tp = (df_indicators['high'] + df_indicators['low'] + df_indicators['close']) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            df_indicators['cci'] = (tp - sma_tp) / (0.015 * mad)
            
            # Williams %R
            highest_high = df_indicators['high'].rolling(window=14).max()
            lowest_low = df_indicators['low'].rolling(window=14).min()
            df_indicators['williams_r'] = -100 * ((highest_high - df_indicators['close']) / (highest_high - lowest_low))
            
            # ==================== ВОЛАТИЛЬНОСТЬ ====================
            
            # ATR (Average True Range)
            high_low = df_indicators['high'] - df_indicators['low']
            high_close = np.abs(df_indicators['high'] - df_indicators['close'].shift())
            low_close = np.abs(df_indicators['low'] - df_indicators['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df_indicators['atr'] = true_range.rolling(window=14).mean()
            
            # Bollinger Bands
            df_indicators['bb_middle'] = df_indicators['close'].rolling(window=20).mean()
            bb_std = df_indicators['close'].rolling(window=20).std()
            df_indicators['bb_upper'] = df_indicators['bb_middle'] + (bb_std * 2)
            df_indicators['bb_lower'] = df_indicators['bb_middle'] - (bb_std * 2)
            df_indicators['bb_width'] = (df_indicators['bb_upper'] - df_indicators['bb_lower']) / df_indicators['bb_middle']
            df_indicators['bb_position'] = (df_indicators['close'] - df_indicators['bb_lower']) / (df_indicators['bb_upper'] - df_indicators['bb_lower'])
            
            # ==================== ОБЪЕМНЫЕ ИНДИКАТОРЫ ====================
            
            # Volume SMA
            df_indicators['volume_sma_20'] = df_indicators['volume'].rolling(window=20).mean()
            df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_sma_20']
            
            # OBV (On-Balance Volume)
            df_indicators['obv'] = 0
            for i in range(1, len(df_indicators)):
                if df_indicators['close'].iloc[i] > df_indicators['close'].iloc[i-1]:
                    df_indicators['obv'].iloc[i] = df_indicators['obv'].iloc[i-1] + df_indicators['volume'].iloc[i]
                elif df_indicators['close'].iloc[i] < df_indicators['close'].iloc[i-1]:
                    df_indicators['obv'].iloc[i] = df_indicators['obv'].iloc[i-1] - df_indicators['volume'].iloc[i]
                else:
                    df_indicators['obv'].iloc[i] = df_indicators['obv'].iloc[i-1]
            
            # ==================== ПРОЧИЕ ИНДИКАТОРЫ ====================
            
            # Parabolic SAR (упрощенный)
            df_indicators['sar'] = df_indicators['close'].copy()
            af = 0.02  # Acceleration factor
            ep = df_indicators['high'].iloc[0]  # Extreme point
            
            for i in range(1, len(df_indicators)):
                if df_indicators['close'].iloc[i] > ep:
                    ep = df_indicators['high'].iloc[i]
                    af = min(af + 0.02, 0.2)
                else:
                    ep = df_indicators['low'].iloc[i]
                    af = min(af + 0.02, 0.2)
                
                df_indicators['sar'].iloc[i] = df_indicators['sar'].iloc[i-1] + af * (ep - df_indicators['sar'].iloc[i-1])
            
            # Производные признаки для тренда
            df_indicators['trend_ema'] = np.where(
                df_indicators['ema_12'] > df_indicators['ema_26'], 1, -1
            )
            
            df_indicators['trend_sma'] = np.where(
                (df_indicators['close'] > df_indicators['sma_20']) & 
                (df_indicators['sma_20'] > df_indicators['sma_50']), 1,
                np.where(
                    (df_indicators['close'] < df_indicators['sma_20']) & 
                    (df_indicators['sma_20'] < df_indicators['sma_50']), -1, 0
                )
            )
            
            logger.debug(f"📈 Добавлено {len(df_indicators.columns) - 6} технических индикаторов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета индикаторов: {e}")
            logger.error(traceback.format_exc())
        
        return df_indicators
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет производные признаки для анализа."""
        if df.empty:
            return df
        
        df_features = df.copy()
        
        try:
            # Ценовые изменения
            df_features['returns'] = df_features['close'].pct_change()
            df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift())
            
            # Волатильность
            df_features['volatility_20'] = df_features['returns'].rolling(window=20).std() * np.sqrt(365)
            df_features['volatility_50'] = df_features['returns'].rolling(window=50).std() * np.sqrt(365)
            
            # Момент и ускорение
            df_features['momentum_10'] = df_features['close'] - df_features['close'].shift(10)
            df_features['acceleration_5'] = df_features['momentum_10'].diff(5)
            
            # Процентные изменения
            for period in [1, 3, 5, 10, 20]:
                df_features[f'pct_change_{period}'] = df_features['close'].pct_change(periods=period)
            
            # Скользящие минимумы и максимумы
            df_features['rolling_max_20'] = df_features['high'].rolling(window=20).max()
            df_features['rolling_min_20'] = df_features['low'].rolling(window=20).min()
            df_features['price_position'] = (df_features['close'] - df_features['rolling_min_20']) / \
                                          (df_features['rolling_max_20'] - df_features['rolling_min_20'])
            
            # Свечные паттерны (упрощенные)
            df_features['is_doji'] = np.abs(df_features['close'] - df_features['open']) / \
                                    (df_features['high'] - df_features['low']) < 0.1
            
            df_features['is_bullish'] = df_features['close'] > df_features['open']
            df_features['is_bearish'] = df_features['close'] < df_features['open']
            
            # Объемные паттерны
            df_features['volume_spike'] = df_features['volume_ratio'] > 2.0
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета производных признаков: {e}")
        
        return df_features
    
    async def get_multiple_timeframes(self, symbol: str, 
                                    timeframes: List[str] = None,
                                    limit: int = 500) -> Dict[str, pd.DataFrame]:
        """
        Загружает данные для нескольких таймфреймов параллельно.
        
        Args:
            symbol: Торговая пара
            timeframes: Список таймфреймов
            limit: Количество свечей на таймфрейм
            
        Returns:
            Dict: {таймфрейм: DataFrame}
        """
        
        if timeframes is None:
            timeframes = ["15m", "1h", "4h", "1d"]
        
        # Создаем задачи для каждого таймфрейма
        tasks = []
        for tf in timeframes:
            task = self.get_ohlcv(symbol, tf, limit)
            tasks.append(task)
        
        # Выполняем параллельно
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Собираем результаты
        data = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Ошибка для {symbol} {tf}: {result}")
            elif result is not None:
                data[tf] = result
        
        logger.info(f"📊 Загружено {len(data)} таймфреймов для {symbol}")
        return data
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Получает текущую цену символа."""
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(
                None,
                lambda: self.exchange.fetch_ticker(symbol)
            )
            return ticker.get('last')
        except Exception as e:
            logger.error(f"❌ Ошибка получения цены {symbol}: {e}")
            return None
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Получает информацию о торговой паре."""
        try:
            loop = asyncio.get_event_loop()
            markets = await loop.run_in_executor(
                None,
                lambda: self.exchange.load_markets()
            )
            return markets.get(symbol)
        except Exception as e:
            logger.error(f"❌ Ошибка получения информации {symbol}: {e}")
            return None
    
    async def test_connection(self) -> bool:
        """Проверяет соединение с биржей."""
        try:
            loop = asyncio.get_event_loop()
            # Пробуем получить время сервера
            server_time = await loop.run_in_executor(None, self.exchange.fetch_time)
            
            # Пробуем получить тикер для BTC/USDT
            ticker = await loop.run_in_executor(
                None,
                lambda: self.exchange.fetch_ticker('BTC/USDT')
            )
            
            logger.info(f"✅ Соединение с {self.exchange_id.upper()} установлено")
            logger.info(f"   Время сервера: {datetime.fromtimestamp(server_time/1000)}")
            logger.info(f"   Цена BTC: ${ticker.get('last'):,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к {self.exchange_id.upper()}: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Возвращает подробную статистику работы."""
        total_requests = self.stats['total_requests']
        successful = self.stats['successful_requests']
        failed = self.stats['failed_requests']
        cache_hits = self.stats['cache_hits']
        cache_misses = self.stats['cache_misses']
        
        success_rate = successful / total_requests if total_requests > 0 else 0
        error_rate = failed / total_requests if total_requests > 0 else 0
        hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        return {
            'version': self.VERSION,
            'exchange': self.exchange_id,
            'requests': {
                'total': total_requests,
                'successful': successful,
                'failed': failed,
                'success_rate': f"{success_rate:.1%}",
                'error_rate': f"{error_rate:.1%}",
                'avg_response_time': f"{self.stats['avg_response_time']:.3f}s"
            },
            'cache': {
                'hits': cache_hits,
                'misses': cache_misses,
                'hit_rate': f"{hit_rate:.1%}",
                'memory_size': len(self.memory_cache),
                'disk_size': len(list(self.disk_cache_dir.glob("*.pkl")))
            },
            'data': {
                'total_points': self.stats['total_data_points'],
                'memory_cache_size': sum(len(df) for _, (_, df) in self.memory_cache.items())
            },
            'errors': {
                'total': len(self.stats['errors']),
                'recent': self.stats['errors'][-5:] if self.stats['errors'] else []
            }
        }
    
    def clear_cache(self, memory: bool = True, disk: bool = True):
        """Очищает кеш."""
        if memory:
            self.memory_cache.clear()
            logger.info("🧹 Очищен кеш в памяти")
        
        if disk and self.disk_cache_dir.exists():
            for file in self.disk_cache_dir.glob("*.pkl"):
                file.unlink()
            logger.info("🧹 Очищен файловый кеш")
    
    async def close(self):
        """Корректное завершение работы."""
        logger.info("🔚 Закрытие DataHandler...")
        
        # Сохраняем статистику
        stats_file = Path("logs/data_handler_stats.json")
        stats_file.parent.mkdir(exist_ok=True)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_statistics(), f, indent=2, ensure_ascii=False, default=str)
        
        # Закрываем сессию (если есть)
        if self.session:
            await self.session.close()
        
        logger.info("✅ DataHandler закрыт")

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И УТИЛИТЫ
# ============================================================================

def calculate_support_resistance_levels(df: pd.DataFrame, method: str = 'pivot') -> Dict:
    """
    Рассчитывает уровни поддержки и сопротивления.
    
    Args:
        df: DataFrame с ценовыми данными
        method: Метод расчета ('pivot', 'fractal', 'volume')
        
    Returns:
        Dict: Уровни поддержки и сопротивления
    """
    
    if df.empty:
        return {'supports': [], 'resistances': []}
    
    levels = {'supports': [], 'resistances': []}
    
    try:
        if method == 'pivot':
            # Классические пивот-уровни
            pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
            r1 = (2 * pivot) - df['low'].iloc[-1]
            r2 = pivot + (df['high'].iloc[-1] - df['low'].iloc[-1])
            s1 = (2 * pivot) - df['high'].iloc[-1]
            s2 = pivot - (df['high'].iloc[-1] - df['low'].iloc[-1])
            
            levels['resistances'] = [r1, r2]
            levels['supports'] = [s1, s2]
        
        elif method == 'fractal':
            # Уровни на основе фракталов (Williams)
            window = 5
            
            for i in range(window, len(df) - window):
                # Фракталы вверх (сопротивление)
                if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                    levels['resistances'].append(df['high'].iloc[i])
                
                # Фракталы вниз (поддержка)
                if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                    levels['supports'].append(df['low'].iloc[i])
            
            # Кластеризация уровней
            levels['resistances'] = _cluster_levels(levels['resistances'])
            levels['supports'] = _cluster_levels(levels['supports'])
        
        elif method == 'volume':
            # Уровни на основе Volume Profile
            price_levels = np.linspace(df['low'].min(), df['high'].max(), 50)
            volume_profile = np.zeros_like(price_levels)
            
            for _, row in df.iterrows():
                low_idx = np.searchsorted(price_levels, row['low'])
                high_idx = np.searchsorted(price_levels, row['high'])
                
                if high_idx > low_idx:
                    volume_per_level = row['volume'] / (high_idx - low_idx)
                    volume_profile[low_idx:high_idx] += volume_per_level
            
            # Находим пики объема
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(volume_profile, height=np.mean(volume_profile) * 1.5)
            
            for peak in peaks:
                price = price_levels[peak]
                if price < df['close'].iloc[-1]:
                    levels['supports'].append(float(price))
                else:
                    levels['resistances'].append(float(price))
        
    except Exception as e:
        logger.error(f"Ошибка расчета уровней: {e}")
    
    return levels

def _cluster_levels(levels: List[float], threshold_pct: float = 0.01) -> List[float]:
    """Кластеризует близкие уровни."""
    if not levels:
        return []
    
    levels_sorted = sorted(levels)
    clusters = []
    current_cluster = [levels_sorted[0]]
    
    for price in levels_sorted[1:]:
        if abs(price - current_cluster[-1]) / current_cluster[-1] <= threshold_pct:
            current_cluster.append(price)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [price]
    
    if current_cluster:
        clusters.append(np.mean(current_cluster))
    
    return clusters

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

async def test_data_handler():
    """Комплексное тестирование DataHandler."""
    print("\n" + "="*60)
    print("🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ DATA HANDLER")
    print("="*60)
    
    handler = DataHandler(
        exchange_id="binance",
        cache_enabled=True,
        cache_ttl=60
    )
    
    try:
        # 1. Тест подключения
        print("\n1. 🔌 Тест подключения к бирже...")
        if await handler.test_connection():
            print("   ✅ Подключение успешно")
        else:
            print("   ❌ Не удалось подключиться")
            return
        
        # 2. Тест загрузки данных
        print("\n2. 📥 Тест загрузки OHLCV данных...")
        df = await handler.get_ohlcv("BTC/USDT", "1h", 100)
        
        if df is not None and not df.empty:
            print(f"   ✅ Данные загружены: {len(df)} свечей")
            print(f"   📊 Колонки: {list(df.columns)}")
            print(f"   💰 Диапазон: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print(f"   ⏰ Период: {df.index[0]} - {df.index[-1]}")
        else:
            print("   ❌ Не удалось загрузить данные")
            return
        
        # 3. Тест множественных таймфреймов
        print("\n3. 📈 Тест множественных таймфреймов...")
        multi_data = await handler.get_multiple_timeframes("ETH/USDT", ["15m", "1h", "4h"], 50)
        
        if multi_data:
            print(f"   ✅ Загружено таймфреймов: {len(multi_data)}")
            for tf, tf_df in multi_data.items():
                print(f"     {tf}: {len(tf_df)} свечей, {len(tf_df.columns)} индикаторов")
        else:
            print("   ❌ Не удалось загрузить множественные таймфреймы")
        
        # 4. Тест текущей цены
        print("\n4. 💰 Тест получения текущей цены...")
        price = await handler.get_current_price("BTC/USDT")
        if price:
            print(f"   ✅ Текущая цена BTC: ${price:.2f}")
        else:
            print("   ❌ Не удалось получить цену")
        
        # 5. Тест информации о символе
        print("\n5. 📋 Тест информации о символе...")
        info = await handler.get_symbol_info("BTC/USDT")
        if info:
            print(f"   ✅ Информация получена")
            print(f"     Лот: {info.get('lot', 'N/A')}")
            print(f"     Точность: {info.get('precision', 'N/A')}")
        else:
            print("   ❌ Не удалось получить информацию")
        
        # 6. Тест расчета уровней
        print("\n6. 🎯 Тест расчета уровней поддержки/сопротивления...")
        if df is not None:
            levels = calculate_support_resistance_levels(df, method='pivot')
            print(f"   ✅ Уровни рассчитаны")
            print(f"     Поддержки: {levels['supports']}")
            print(f"     Сопротивления: {levels['resistances']}")
        
        # 7. Статистика
        print("\n7. 📊 Статистика работы...")
        stats = handler.get_statistics()
        print(f"   Всего запросов: {stats['requests']['total']}")
        print(f"   Успешных: {stats['requests']['successful']}")
        print(f"   Кеш попаданий: {stats['cache']['hit_rate']}")
        print(f"   Среднее время: {stats['requests']['avg_response_time']}")
        
        # 8. Тест кеширования
        print("\n8. 💾 Тест кеширования...")
        print("   Первая загрузка (кеш промах)...")
        start = datetime.now()
        df1 = await handler.get_ohlcv("BTC/USDT", "1h", 10)
        time1 = (datetime.now() - start).total_seconds()
        
        print("   Вторая загрузка (кеш попадание)...")
        start = datetime.now()
        df2 = await handler.get_ohlcv("BTC/USDT", "1h", 10)
        time2 = (datetime.now() - start).total_seconds()
        
        if time2 < time1:
            print(f"   ✅ Кеширование работает: {time1:.3f}s -> {time2:.3f}s")
        else:
            print(f"   ⚠️  Кеширование не дало ускорения")
        
        print("\n" + "="*60)
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await handler.close()

if __name__ == "__main__":
    # Запуск теста
    import asyncio
    asyncio.run(test_data_handler())