"""
Оркестратор всех расчетов с поддержкой новой логики
"""
import asyncio
from typing import Dict, List
from datetime import datetime
from core.data_handler import DataHandler
from engine.levels import LevelCalculator
from engine.margin import MarginCalculator
from engine.confluence import ConfluenceCalculator
from engine.cme import CMECalculator
from engine.signals import SignalGenerator
from storage.database import DatabaseManager
from notify.telegram import TelegramNotifier

class EngineRunner:
    def __init__(self):
        self.data_handler = DataHandler()
        self.level_calc = LevelCalculator()
        self.margin_calc = MarginCalculator()
        self.confluence_calc = ConfluenceCalculator()
        self.cme_calc = CMECalculator()
        self.signal_gen = SignalGenerator()
        self.db = DatabaseManager()
        self.notifier = TelegramNotifier()
        
        self.symbols = ['BTC/USDT', 'ETH/USDT']
        self.timeframes = ['1h', '4h', '1d']
        
    async def run_cycle(self):
        """Выполнение одного цикла расчетов"""
        for symbol in self.symbols:
            # 1. Загрузка данных
            data = await self.data_handler.fetch_ohlcv_multi_timeframe(
                symbol, self.timeframes
            )
            
            # 2. Расчет уровней (MTF)
            levels = {}
            for tf, df in data.items():
                levels[tf] = self.level_calc.calculate(df)
            
            # 3. Расчет маржинальных уровней
            margin_levels = self.margin_calc.calculate(data['1h'])
            
            # 4. Расчет CME
            cme_data = self.cme_calc.analyze(data['1h'])
            
            # 5. Расчет конфлюэнса
            confluence = self.confluence_calc.evaluate(
                levels, margin_levels, cme_data
            )
            
            # 6. Генерация сигналов (новая логика из trading_tab.py)
            signals = self.signal_gen.generate(
                data['1h'], levels, confluence
            )
            
            # 7. Сохранение в БД
            await self.db.save_analysis(
                symbol=symbol,
                levels=levels,
                confluence=confluence,
                signals=signals,
                timestamp=datetime.now()
            )
            
            # 8. Отправка уведомлений при наличии сигналов
            if signals:
                await self.notifier.send_signals(symbol, signals, confluence)
    
    async def start(self):
        """Запуск бесконечного цикла"""
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(60)  # Пауза 60 секунд
            except Exception as e:
                print(f"Error in engine cycle: {e}")
                await asyncio.sleep(300)  # При ошибке ждем 5 минут
