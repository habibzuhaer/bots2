"""
Унифицированный обработчик данных с поддержкой MTF/STF
"""
import pandas as pd
import ccxt
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config.settings import EXCHANGE_CONFIG

class DataHandler:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': EXCHANGE_CONFIG['api_key'],
            'secret': EXCHANGE_CONFIG['api_secret'],
            'enableRateLimit': True
        })
        
    async def fetch_ohlcv_multi_timeframe(
        self, 
        symbol: str, 
        timeframes: List[str], 
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """Загрузка данных для нескольких таймфреймов"""
        data = {}
        
        for tf in timeframes:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, 
                timeframe=tf, 
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Добавляем расчетные поля из crypto_lite_new
            df = self._calculate_indicators(df)
            data[tf] = df
            
        return data
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет индикаторов (можно добавить из trading_tab.py)"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        return df
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Получение информации о паре (из exchange_info.py)"""
        markets = self.exchange.load_markets()
        return markets.get(symbol, {})
