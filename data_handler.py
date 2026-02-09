# bots2/data_handler.py
import ccxt
import pandas as pd
import asyncio
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataHandler:
    """
    Загружает рыночные данные с биржи (Binance).
    В будущем можно добавить кеширование, другие биржи, реконнект.
    """
    def __init__(self, exchange_id: str = 'binance'):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'enableRateLimit': True,  # Важно для соблюдения лимитов
            'options': {'defaultType': 'spot'}
        })
        logger.info(f"DataHandler подключен к {exchange_id}")

    async def get_ohlcv(self, 
                        symbol: str, 
                        timeframe: str = '1h', 
                        limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Асинхронно загружает свечи.
        Возвращает DataFrame с колонками: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        try:
            # CCxt работает синхронно, поэтому оборачиваем в потоки
            loop = asyncio.get_event_loop()
            ohlcv = await loop.run_in_executor(
                None, 
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Простой расчет индикатора для примера (можно добавить больше)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            logger.debug(f"Загружено {len(df)} свечей для {symbol} ({timeframe})")
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"Сетевая ошибка при загрузке {symbol}: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Ошибка биржи при загрузке {symbol}: {e}")
        except Exception as e:
            logger.error(f"Неизвестная ошибка: {e}")
        
        return None

    async def get_multiple_tf(self, symbol: str, timeframes: list, limit=100) -> Dict[str, pd.DataFrame]:
        """Загружает несколько таймфреймов одновременно."""
        tasks = [self.get_ohlcv(symbol, tf, limit) for tf in timeframes]
        results = await asyncio.gather(*tasks)
        return {tf: df for tf, df in zip(timeframes, results) if df is not None}

# Простой тест работы
if __name__ == "__main__":
    async def test():
        handler = DataHandler()
        df = await handler.get_ohlcv('BTC/USDT', '1h', 10)
        if df is not None:
            print(df[['close', 'sma_20']].tail())
        else:
            print("Не удалось загрузить данные")
    
    asyncio.run(test())