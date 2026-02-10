import asyncio
import logging
from datetime import datetime
from typing import Dict, List
import traceback
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_handler import DataHandler
from engine.levels import LevelCalculator
from engine.confluence import ConfluenceCalculator
from engine.signals import SignalGenerator
from storage.database import DatabaseManager
from notify.telegram import TelegramNotifier
from config.settings import settings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EngineRunner:
    """
    Главный оркестратор торгового бота.
    Координирует работу всех компонентов системы.
    """
    
    def __init__(self, symbols: List[str] = None, timeframes: List[str] = None):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT"]
        self.timeframes = timeframes or ["1h", "4h"]
        
        # Инициализация компонентов
        self.data_handler = DataHandler()
        self.level_calc = LevelCalculator()
        self.confluence_calc = ConfluenceCalculator()
        self.signal_gen = SignalGenerator()
        self.db = DatabaseManager()
        self.telegram = TelegramNotifier()
        
        # Состояние
        self.is_running = False
        self.cycle_count = 0
        
        logger.info(f"EngineRunner инициализирован для {self.symbols}")
    
    async def initialize(self):
        """Инициализация всех компонентов."""
        try:
            logger.info("Инициализация компонентов...")
            
            # Инициализация БД
            self.db.initialize()
            
            # Проверка подключений
            if not await self.data_handler.test_connection():
                logger.error("Не удалось подключиться к источнику данных")
                return False
            
            # Проверка Telegram
            if settings.TELEGRAM_ENABLED:
                await self.telegram.test_connection()
            
            logger.info("Все компоненты успешно инициализированы")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            return False
    
    async def process_symbol(self, symbol: str) -> Dict:
        """Обрабатывает один символ за цикл."""
        result = {
            "symbol": symbol,
            "success": False,
            "signals": [],
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            logger.info(f"Обработка {symbol}...")
            
            # 1. Загрузка данных
            data = await self.data_handler.get_ohlcv(symbol, "1h", limit=100)
            if data is None or data.empty:
                result["error"] = "Не удалось загрузить данные"
                return result
            
            # 2. Расчет уровней
            levels = self.level_calc.calculate_support_resistance(data)
            
            # 3. Расчет конфлюэнса
            confluence = self.confluence_calc.evaluate({"1h": levels})
            
            # 4. Генерация сигналов
            current_price = data["close"].iloc[-1]
            signal = self.signal_gen.generate(
                symbol=symbol,
                price=current_price,
                levels={"1h": levels},
                confluence=confluence
            )
            
            if signal and signal.get("direction") != "NEUTRAL":
                result["signals"].append(signal)
                
                # 5. Сохранение в БД
                await self.db.save_signal(signal)
                
                # 6. Отправка уведомления
                if signal.get("strength") in ["STRONG", "MEDIUM"]:
                    await self.telegram.send_signal(signal)
            
            result["success"] = True
            logger.info(f"{symbol} обработан успешно")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Ошибка обработки {symbol}: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    async def run_cycle(self):
        """Выполняет один цикл анализа для всех символов."""
        self.cycle_count += 1
        logger.info(f"\n{'='*50}")
        logger.info(f"ЦИКЛ #{self.cycle_count} | {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*50}")
        
        tasks = [self.process_symbol(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks)
        
        # Статистика цикла
        successful = sum(1 for r in results if r["success"])
        total_signals = sum(len(r["signals"]) for r in results)
        
        logger.info(f"Итоги цикла #{self.cycle_count}:")
        logger.info(f"  Успешно обработано: {successful}/{len(self.symbols)}")
        logger.info(f"  Сигналов сгенерировано: {total_signals}")
        
        return results
    
    async def run_continuous(self, interval_seconds: int = 300):
        """Запускает бесконечный цикл анализа."""
        self.is_running = True
        
        # Инициализация
        if not await self.initialize():
            logger.error("Не удалось инициализировать систему")
            return
        
        logger.info("Запуск основного цикла...")
        
        # Отправка уведомления о старте
        if settings.TELEGRAM_ENABLED:
            await self.telegram.send_message(
                f"🚀 Trading Bot запущен!\n"
                f"Символы: {', '.join(self.symbols)}\n"
                f"Интервал: {interval_seconds} сек"
            )
        
        # Основной цикл
        while self.is_running:
            try:
                await self.run_cycle()
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Получен сигнал остановки...")
                self.is_running = False
                break
                
            except Exception as e:
                logger.error(f"Критическая ошибка в цикле: {e}")
                await asyncio.sleep(60)  # Пауза при ошибке
        
        # Завершение работы
        await self.shutdown()
    
    async def shutdown(self):
        """Корректное завершение работы."""
        logger.info("Завершение работы...")
        
        if settings.TELEGRAM_ENABLED:
            await self.telegram.send_message("🛑 Trading Bot остановлен")
        
        logger.info("Работа завершена")

# Точка входа
async def main():
    """Точка входа для запуска бота."""
    runner = EngineRunner(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1h", "4h"]
    )
    
    try:
        await runner.run_continuous(interval_seconds=300)
    except KeyboardInterrupt:
        logger.info("Завершение по запросу пользователя")
    except Exception as e:
        logger.error(f"Фатальная ошибка: {e}")
        raise

if __name__ == "__main__":
    # Создание необходимых директорий
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Запуск
    asyncio.run(main())