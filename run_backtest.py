#!/usr/bin/env python3
"""
Скрипт для запуска бэктеста из командной строки
"""

import asyncio
import argparse
import logging
from datetime import datetime

from backtest.engine import BacktestEngine

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Запуск бэктеста стратегии")
    parser.add_argument("--period", type=str, choices=["quick", "medium", "full"], 
                       default="quick", help="Период тестирования")
    parser.add_argument("--symbol", type=str, help="Конкретный символ для теста")
    parser.add_argument("--timeframe", type=str, help="Конкретный таймфрейм для теста")
    parser.add_argument("--config", type=str, default="configs/backtest_tickers.json",
                       help="Путь к конфигурационному файлу")
    
    args = parser.parse_args()
    
    # Маппинг периодов
    period_map = {"quick": 0, "medium": 1, "full": 2}
    
    logger.info(f"Запуск бэктеста с параметрами:")
    logger.info(f"  Период: {args.period}")
    logger.info(f"  Конфиг: {args.config}")
    if args.symbol:
        logger.info(f"  Символ: {args.symbol}")
    if args.timeframe:
        logger.info(f"  Таймфрейм: {args.timeframe}")
    
    try:
        # Создаем движок бэктеста
        engine = BacktestEngine(config_path=args.config)
        
        if args.symbol and args.timeframe:
            # Тестируем конкретный символ и таймфрейм
            test_period = engine.config["test_periods"][period_map[args.period]]
            
            result = await engine.run_backtest(
                test_name=f"Custom Test - {args.symbol} {args.timeframe}",
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=test_period["start_date"],
                end_date=test_period["end_date"]
            )
            
            if result:
                logger.info(f"Бэктест завершен:")
                logger.info(f"  Символ: {result['symbol']}")
                logger.info(f"  Таймфрейм: {result['timeframe']}")
                logger.info(f"  Свечей: {result['total_candles']}")
                logger.info(f"  Успешность: {result['metrics'].get('success_rate', 0):.1f}%")
            else:
                logger.error("Бэктест не дал результатов")
        
        else:
            # Комплексный бэктест
            results = await engine.run_comprehensive_backtest(
                test_period_index=period_map[args.period]
            )
            
            logger.info(f"Комплексный бэктест завершен:")
            logger.info(f"  Всего тестов: {results.get('total_tests', 0)}")
            logger.info(f"  Успешных: {results.get('successful_tests', 0)}")
            
            # Показываем лучший результат
            best = results.get('best_performer')
            if best:
                logger.info(f"  Лучший: {best['symbol']} {best['timeframe']} "
                           f"({best['success_rate']:.1f}%)")
    
    except Exception as e:
        logger.error(f"Ошибка при запуске бэктеста: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
