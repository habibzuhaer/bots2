"""
Интеграция бэктеста из crypto_lite_new
"""
import pandas as pd
from typing import Dict, List
from backtest.strategy_tester import StrategyTester
from storage.history_manager import HistoryManager

class BacktestRunner:
    def __init__(self, config_path=None):
        self.tester = StrategyTester(config_path)
        self.history_manager = HistoryManager()
        
    def run(self, 
            symbol: str, 
            start_date: str, 
            end_date: str, 
            timeframe: str = '1h'):
        """Запуск бэктеста"""
        
        # 1. Загрузка исторических данных
        data = self._load_historical_data(symbol, start_date, end_date, timeframe)
        
        # 2. Запуск тестирования стратегии
        results = self.tester.test_strategy(data)
        
        # 3. Анализ результатов
        analysis = self._analyze_results(results)
        
        # 4. Сохранение в историю
        test_id = self.history_manager.save_test(
            symbol=symbol,
            strategy=self.tester.strategy_name,
            results=results,
            analysis=analysis
        )
        
        # 5. Вывод отчета
        self._print_report(results, analysis)
        
        return test_id
    
    def _load_historical_data(self, symbol, start_date, end_date, timeframe):
        """Загрузка исторических данных"""
        # Реализация из data_handler.py crypto_lite_new
        pass