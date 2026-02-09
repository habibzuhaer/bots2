"""
Загрузчик исторических данных для бэктестинга
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class HistoricalDataLoader:
    """Загрузчик исторических данных с Bybit"""
    
    def __init__(self):
        self.cache = {}
    
    async def load_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Загружает исторические данные для бэктеста
        
        Параметры:
        - symbol: тикер (например, "BTCUSDT")
        - timeframe: таймфрейм ("5m", "15m", "1h", "4h")
        - start_date: дата начала в формате "YYYY-MM-DD"
        - end_date: дата окончания в формате "YYYY-MM-DD"
        - use_cache: использовать кэш для ускорения
        """
        
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        # Проверяем кэш
        if use_cache and cache_key in self.cache:
            logger.debug(f"Используем кэшированные данные для {cache_key}")
            return self.cache[cache_key]
        
        try:
            logger.info(f"Загрузка данных {symbol} {timeframe} с {start_date} по {end_date}")
            
            # Преобразуем даты в timestamp
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Рассчитываем количество свечей
            tf_minutes = self._timeframe_to_minutes(timeframe)
            total_minutes = int((end_dt - start_dt).total_seconds() / 60)
            expected_candles = total_minutes // tf_minutes
            
            logger.info(f"Ожидаемое количество свечей: {expected_candles}")
            
            # Импортируем здесь, чтобы избежать циклических импортов
            try:
                from futures_bybit import fetch_kline
            except ImportError:
                # Альтернативный метод, если fetch_kline недоступен
                candles = await self._load_from_file_or_api(symbol, timeframe, start_dt, end_dt)
            else:
                # Используем асинхронную сессию для загрузки
                import aiohttp
                
                candles = []
                current_start = start_dt
                
                # Разбиваем на чанки по 1000 свечей (лимит Bybit)
                while current_start < end_dt:
                    # Загружаем чанк
                    chunk_end = min(current_start + timedelta(minutes=tf_minutes * 1000), end_dt)
                    
                    async with aiohttp.ClientSession() as session:
                        try:
                            # Преобразуем в формат запроса Bybit
                            chunk_candles = await fetch_kline(
                                session=session,
                                symbol=symbol,
                                interval=timeframe,
                                limit=1000,
                                start_time=int(current_start.timestamp() * 1000),
                                end_time=int(chunk_end.timestamp() * 1000)
                            )
                            
                            if chunk_candles:
                                candles.extend(chunk_candles)
                            
                        except Exception as e:
                            logger.warning(f"Ошибка загрузки чанка {symbol} {timeframe}: {e}")
                    
                    current_start = chunk_end
                    await asyncio.sleep(0.1)  # Небольшая пауза между запросами
            
            # Преобразуем в стандартный формат
            formatted_candles = self._format_candles(candles)
            
            # Кэшируем результат
            self.cache[cache_key] = formatted_candles
            
            logger.info(f"Загружено {len(formatted_candles)} свечей для {symbol} {timeframe}")
            return formatted_candles
            
        except Exception as e:
            logger.error(f"Критическая ошибка загрузки данных {symbol} {timeframe}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Преобразует таймфрейм в минуты"""
        tf_map = {
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        return tf_map.get(timeframe, 60)
    
    def _format_candles(self, candles: List[Dict]) -> List[Dict]:
        """Форматирует свечи в стандартный формат"""
        formatted = []
        
        for candle in candles:
            # Стандартизируем формат
            formatted_candle = {
                "ts": candle.get("ts") or candle.get("timestamp") or candle.get("time"),
                "open": float(candle.get("open", 0)),
                "high": float(candle.get("high", 0)),
                "low": float(candle.get("low", 0)),
                "close": float(candle.get("close", 0)),
                "volume": float(candle.get("volume", 0)),
                "open_time": candle.get("open_time"),
                "close_time": candle.get("close_time")
            }
            
            # Убедимся, что ts есть
            if not formatted_candle["ts"] and formatted_candle.get("open_time"):
                formatted_candle["ts"] = formatted_candle["open_time"]
            
            formatted.append(formatted_candle)
        
        # Сортируем по времени
        formatted.sort(key=lambda x: x["ts"])
        
        return formatted
    
    async def _load_from_file_or_api(self, symbol: str, timeframe: str, 
                                     start_dt: datetime, end_dt: datetime) -> List[Dict]:
        """Альтернативный метод загрузки данных"""
        # Здесь можно реализовать загрузку из локальных файлов
        # или других источников данных
        logger.warning(f"Используется заглушка для {symbol} {timeframe}")
        return []