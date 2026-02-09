"""
Движок бэктестинга стратегии уровней и маржинальных зон
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import logging

from backtest.data_loader import HistoricalDataLoader
from backtest.strategy_tester import StrategyTester
from backtest.metrics import calculate_metrics
from backtest.visualizer import create_backtest_report

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Основной движок бэктестинга"""
    
    def __init__(self, config_path: str = "configs/backtest_tickers.json"):
        self.config = self._load_config(config_path)
        self.data_loader = HistoricalDataLoader()
        self.strategy_tester = StrategyTester()
        self.db_conn = None
        self.results = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации бэктеста"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфига {config_path}: {e}")
            raise
    
    async def run_backtest(
        self, 
        test_name: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Запускает бэктест для одного символа и таймфрейма
        """
        logger.info(f"Запуск бэктеста: {symbol} {timeframe} ({start_date} - {end_date})")
        
        try:
            # 1. Загружаем исторические данные
            logger.info(f"Загрузка данных для {symbol} {timeframe}...")
            candles = await self.data_loader.load_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(candles) < 100:
                logger.warning(f"Недостаточно данных для {symbol} {timeframe}: {len(candles)} свечей")
                return None
            
            # 2. Запускаем тестирование стратегии
            logger.info(f"Тестирование стратегии для {symbol} {timeframe}...")
            test_results = await self.strategy_tester.test_strategy(
                candles=candles,
                symbol=symbol,
                timeframe=timeframe,
                collision_threshold=self.config["parameters"]["collision_threshold"]
            )
            
            # 3. Рассчитываем метрики
            logger.info(f"Расчет метрик для {symbol} {timeframe}...")
            metrics = calculate_metrics(test_results)
            
            # 4. Формируем результат
            result = {
                "test_name": test_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "period_days": (datetime.strptime(end_date, "%Y-%m-%d") - 
                               datetime.strptime(start_date, "%Y-%m-%d")).days,
                "total_candles": len(candles),
                "metrics": metrics,
                "detailed_results": test_results,
                "created_at": datetime.now().isoformat()
            }
            
            # 5. Сохраняем в базу данных
            self._save_to_db(result)
            
            logger.info(f"Бэктест завершен: {symbol} {timeframe}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка бэктеста {symbol} {timeframe}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    async def run_comprehensive_backtest(self, test_period_index: int = 0) -> Dict[str, Any]:
        """
        Запускает комплексный бэктест по всем символам и таймфреймам
        """
        test_period = self.config["test_periods"][test_period_index]
        
        logger.info(f"Запуск комплексного бэктеста: {test_period['name']}")
        logger.info(f"Период: {test_period['start_date']} - {test_period['end_date']}")
        
        all_results = []
        tasks = []
        
        # Для каждого символа и таймфрейма создаем задачу бэктеста
        for symbol in self.config["symbols"]:
            for tf_group, timeframes in self.config["timeframes"].items():
                for timeframe in timeframes:
                    task = self.run_backtest(
                        test_name=test_period["name"],
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=test_period["start_date"],
                        end_date=test_period["end_date"]
                    )
                    tasks.append(task)
        
        # Запускаем все бэктесты параллельно
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Фильтруем успешные результаты
            for result in results:
                if isinstance(result, dict):
                    all_results.append(result)
        
        # Агрегируем результаты
        aggregated = self._aggregate_results(all_results)
        
        # Создаем отчет
        report_path = create_backtest_report(aggregated)
        
        logger.info(f"Комплексный бэктест завершен. Отчет: {report_path}")
        return aggregated
    
    def _save_to_db(self, result: Dict[str, Any]) -> None:
        """Сохраняет результат бэктеста в базу данных"""
        try:
            if not self.db_conn:
                self.db_conn = sqlite3.connect("db/backtest_results.db")
            
            cursor = self.db_conn.cursor()
            
            # Создаем таблицу если не существует
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    period_days INTEGER,
                    total_candles INTEGER,
                    metrics_json TEXT,
                    created_at TEXT
                )
            """)
            
            # Вставляем данные
            cursor.execute("""
                INSERT INTO backtest_results 
                (test_name, symbol, timeframe, start_date, end_date, 
                 period_days, total_candles, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result["test_name"],
                result["symbol"],
                result["timeframe"],
                result["start_date"],
                result["end_date"],
                result["period_days"],
                result["total_candles"],
                json.dumps(result["metrics"], ensure_ascii=False),
                result["created_at"]
            ))
            
            self.db_conn.commit()
            logger.info(f"Результат сохранен в БД: {result['symbol']} {result['timeframe']}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения в БД: {e}")
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Агрегирует результаты всех бэктестов"""
        if not results:
            return {}
        
        # Группируем по таймфреймам
        tf_groups = {"MTF": [], "STF": []}
        
        for result in results:
            tf = result["timeframe"]
            if tf in ["5m", "15m"]:
                tf_groups["MTF"].append(result)
            else:
                tf_groups["STF"].append(result)
        
        # Рассчитываем средние метрики
        aggregated = {
            "total_tests": len(results),
            "successful_tests": len([r for r in results if r["metrics"]["success_rate"] > 0]),
            "by_timeframe": {},
            "overall_metrics": {},
            "best_performer": None,
            "worst_performer": None
        }
        
        # Метрики по группам ТФ
        for tf_group, group_results in tf_groups.items():
            if group_results:
                aggregated["by_timeframe"][tf_group] = {
                    "count": len(group_results),
                    "avg_success_rate": np.mean([r["metrics"]["success_rate"] for r in group_results]),
                    "avg_accuracy": np.mean([r["metrics"]["accuracy"] for r in group_results]),
                    "avg_profit_factor": np.mean([r["metrics"]["profit_factor"] for r in group_results])
                }
        
        # Лучший и худший по успешности
        if results:
            sorted_results = sorted(results, key=lambda x: x["metrics"]["success_rate"], reverse=True)
            aggregated["best_performer"] = {
                "symbol": sorted_results[0]["symbol"],
                "timeframe": sorted_results[0]["timeframe"],
                "success_rate": sorted_results[0]["metrics"]["success_rate"]
            }
            aggregated["worst_performer"] = {
                "symbol": sorted_results[-1]["symbol"],
                "timeframe": sorted_results[-1]["timeframe"],
                "success_rate": sorted_results[-1]["metrics"]["success_rate"]
            }
        
        return aggregated
    
    def get_historical_results(self, limit: int = 10) -> List[Dict]:
        """Получает исторические результаты из БД"""
        try:
            if not self.db_conn:
                self.db_conn = sqlite3.connect("db/backtest_results.db")
            
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT * FROM backtest_results 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            columns = [column[0] for column in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                result["metrics"] = json.loads(result["metrics_json"])
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка чтения из БД: {e}")
            return []
    
    def compare_strategy_versions(self, version1: str, version2: str) -> Dict:
        """Сравнивает две версии стратегии"""
        # Реализация сравнения разных версий
        pass