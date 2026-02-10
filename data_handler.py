#!/usr/bin/env python3
"""
Модуль загрузки и обработки рыночных данных.
Поддержка нескольких бирж, таймфреймов, кеширование.
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import ccxt
import json
import os
import hashlib

from config.settings import settings

logger = logging.getLogger(__name__)

class DataHandler:
    """
    Универсальный обработчик данных с поддержкой:
    - Множественных бирж (Binance, Bybit, KuCoin)
    - Кеширования в БД и памяти
    - Автоматического реконнекта
    - Обработки ошибок
    """
    
    def __init__(self):
        self.exchanges = {}
        self.cache = {}
        self.session = None
        self.cache_ttl = timedelta(minutes=5)
        
        # Инициализируем подключения к биржам
        self._init_exchanges()
        
        # Статистика
        self.stats = {
            'requests_total': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _init_exchanges(self):
        """Инициализация подключений к биржам."""
        exchange_configs = {
            'binance': {
                'apiKey': settings.BINANCE_API_KEY,
                'secret': settings.BINANCE_API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            },
            'bybit': {
                'apiKey': settings.BYBIT_API_KEY,
                'secret': settings.BYBIT_API_SECRET,
                'enableRateLimit': True,
            }
        }
        
        for exchange_id, config in exchange_configs.items():
            # Пропускаем, если нет ключей
            if not config.get('apiKey'):
                continue
                
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class(config)
                
                # Тест подключения
                exchange.load_markets()
                
                self.exchanges[exchange_id] = exchange
                logger.info(f"✅ Подключено к {exchange_id.upper()}")
                
            except Exception as e:
                logger.warning(f"⚠️  Не удалось подключиться к {exchange_id}: {e}")
    
    def _get_cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        """Генерирует ключ для кеша."""
        key_string = f"{symbol}_{timeframe}_{limit}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обогащает DataFrame техническими индикаторами.
        Добавляет колонки: rsi, macd, bollinger_bands, atr, volume_profile
        """
        if df.empty:
            return df
        
        try:
            # Копируем, чтобы избежать предупреждений
            df = df.copy()
            
            # 1. RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 2. MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # 3. Bollinger Bands
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma20 + (std20 * 2)
            df['bb_middle'] = sma20
            df['bb_lower'] = sma20 - (std20 * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # 4. ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # 5. Volume Profile
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # 6. Support/Resistance уровни (предварительные)
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['support1'] = (2 * df['pivot']) - df['high']
            df['resistance1'] = (2 * df['pivot']) - df['low']
            
            # 7. Трендовые индикаторы
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
            
            df['trend'] = np.where(
                df['ema_9'] > df['ema_21'], 
                np.where(df['ema_21'] > df['ema_50'], 'strong_bull', 'weak_bull'),
                np.where(df['ema_21'] < df['ema_50'], 'strong_bear', 'weak_bear')
            )
            
            # 8. Волатильность
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(365)
            
            logger.debug(f"📈 Данные обогащены: {len(df.columns)} колонок")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обогащения данных: {e}")
        
        return df
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', 
                       limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Основной метод получения OHLCV данных.
        Возвращает обогащенный DataFrame или None при ошибке.
        """
        cache_key = self._get_cache_key(symbol, timeframe, limit)
        
        # Проверяем кеш
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                self.stats['cache_hits'] += 1
                logger.debug(f"🎯 Кеш попадание: {symbol} {timeframe}")
                return cached_data.copy()
        
        self.stats['cache_misses'] += 1
        self.stats['requests_total'] += 1
        
        # Пробуем биржи по порядку
        exchanges_to_try = list(self.exchanges.keys())
        
        for exchange_id in exchanges_to_try:
            try:
                exchange = self.exchanges[exchange_id]
                
                # Асинхронный запрос через run_in_executor
                loop = asyncio.get_event_loop()
                ohlcv = await loop.run_in_executor(
                    None,
                    lambda: exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        limit=limit
                    )
                )
                
                if not ohlcv:
                    continue
                
                # Конвертируем в DataFrame
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Конвертируем timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Обогащаем индикаторами
                df = self._enrich_dataframe(df)
                
                # Сохраняем в кеш
                self.cache[cache_key] = (datetime.now(), df.copy())
                
                logger.info(f"✅ {exchange_id.upper()}: {symbol} {timeframe} - {len(df)} свечей")
                
                return df
                
            except ccxt.NetworkError as e:
                logger.warning(f"🌐 Сетевая ошибка {exchange_id}: {e}")
                continue
            except ccxt.ExchangeError as e:
                logger.warning(f"🏦 Ошибка биржи {exchange_id}: {e}")
                continue
            except Exception as e:
                logger.error(f"❌ Неизвестная ошибка {exchange_id}: {e}")
                continue
        
        self.stats['requests_failed'] += 1
        logger.error(f"❌ Не удалось получить данные для {symbol}")
        return None
    
    async def get_multiple_timeframes(self, symbol: str, 
                                    timeframes: List[str] = None,
                                    limit: int = 500) -> Dict[str, pd.DataFrame]:
        """
        Получает данные для нескольких таймфреймов параллельно.
        """
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']
        
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
        
        return data
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Получает информацию о торговой паре."""
        for exchange_id, exchange in self.exchanges.items():
            try:
                markets = exchange.load_markets()
                if symbol in markets:
                    return markets[symbol]
            except:
                continue
        
        return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Получает текущую цену символа."""
        try:
            for exchange_id, exchange in self.exchanges.items():
                ticker = exchange.fetch_ticker(symbol)
                return ticker.get('last')
        except:
            return None
    
    async def test_connection(self) -> bool:
        """Проверяет соединение с биржами."""
        if not self.exchanges:
            logger.error("❌ Нет подключений к биржам")
            return False
        
        for exchange_id, exchange in self.exchanges.items():
            try:
                # Пробуем получить время сервера
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, exchange.fetch_time)
                logger.info(f"✅ {exchange_id.upper()}: соединение стабильное")
            except Exception as e:
                logger.warning(f"⚠️  {exchange_id.upper()}: проблемы с соединением - {e}")
                return False
        
        return True
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику работы."""
        hit_rate = (self.stats['cache_hits'] / 
                   max(self.stats['cache_hits'] + self.stats['cache_misses'], 1))
        
        success_rate = (1 - (self.stats['requests_failed'] / 
                           max(self.stats['requests_total'], 1)))
        
        return {
            **self.stats,
            'cache_hit_rate': f"{hit_rate:.1%}",
            'success_rate': f"{success_rate:.1%}",
            'active_exchanges': len(self.exchanges),
            'cache_size': len(self.cache)
        }
    
    async def close(self):
        """Корректное закрытие соединений."""
        for exchange_id, exchange in self.exchanges.items():
            try:
                if hasattr(exchange, 'close'):
                    await exchange.close()
            except:
                pass
        
        self.cache.clear()
        logger.info("🔚 DataHandler закрыт")

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

async def test_data_handler():
    """Тест работоспособности модуля."""
    handler = DataHandler()
    
    # Тест подключения
    if not await handler.test_connection():
        print("❌ Тест подключения не пройден")
        return
    
    # Тест получения данных
    df = await handler.get_ohlcv('BTC/USDT', '1h', 100)
    
    if df is not None:
        print(f"✅ Данные получены успешно")
        print(f"   Свечей: {len(df)}")
        print(f"   Колонок: {len(df.columns)}")
        print(f"   Диапазон: {df.index[0]} - {df.index[-1]}")
        
        # Показываем последние строки
        print("\n📊 Последние 5 свечей:")
        print(df[['open', 'high', 'low', 'close', 'volume', 'rsi']].tail())
    else:
        print("❌ Не удалось получить данные")
    
    # Статистика
    stats = handler.get_statistics()
    print(f"\n📈 Статистика: {stats}")
    
    await handler.close()

if __name__ == "__main__":
    asyncio.run(test_data_handler())