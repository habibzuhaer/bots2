#!/usr/bin/env python3
"""
Главный оркестратор торгового бота.
Координирует все компоненты: данные → анализ → сигналы → уведомления → хранение.
"""
import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import traceback

# Добавляем путь проекта
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
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EngineRunner:
    """
    Основной класс, управляющий жизненным циклом торгового бота.
    """
    
    def __init__(self, symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None):
        """Инициализация всех компонентов системы."""
        
        # Конфигурация
        self.symbols = symbols or settings.DEFAULT_SYMBOLS
        self.timeframes = timeframes or settings.TIMEFRAMES
        self.update_interval = settings.UPDATE_INTERVAL
        
        # Инициализация компонентов
        self.data_handler = DataHandler()
        self.level_calculator = LevelCalculator()
        self.confluence_calculator = ConfluenceCalculator()
        self.signal_generator = SignalGenerator()
        self.db = DatabaseManager()
        self.telegram_notifier = TelegramNotifier()
        
        # Состояние системы
        self.is_running = False
        self.cycle_count = 0
        self.last_signals = {}
        
        logger.info(f"""
╔══════════════════════════════════════════════════╗
║          TRADING BOT v2.0 ИНИЦИАЛИЗИРОВАН        ║
╠══════════════════════════════════════════════════╣
║ Символы:   {', '.join(self.symbols):<30} ║
║ Таймфреймы: {', '.join(self.timeframes):<30} ║
║ Интервал:  {self.update_interval} секунд{'':<19} ║
╚══════════════════════════════════════════════════╝
        """)
    
    async def initialize(self) -> bool:
        """Инициализация всех компонентов системы."""
        try:
            logger.info("🔄 Инициализация компонентов...")
            
            # Проверяем подключение к данным
            if not await self.data_handler.test_connection():
                logger.error("❌ Ошибка подключения к источнику данных")
                return False
            
            # Инициализируем базу данных
            await self.db.initialize()
            
            # Проверяем Telegram (если включен)
            if settings.TELEGRAM_ENABLED:
                await self.telegram_notifier.test_connection()
            
            logger.info("✅ Все компоненты успешно инициализированы")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    async def process_symbol(self, symbol: str) -> Dict:
        """
        Обрабатывает один символ за цикл.
        Возвращает результат анализа.
        """
        result = {
            'symbol': symbol,
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'signals': [],
            'levels': {},
            'error': None
        }
        
        try:
            logger.debug(f"🔍 Анализ {symbol}...")
            
            # 1. ЗАГРУЗКА ДАННЫХ
            data_frames = {}
            for tf in self.timeframes:
                df = await self.data_handler.get_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    limit=settings.DATA_LIMIT
                )
                
                if df is not None and not df.empty:
                    data_frames[tf] = df
                    logger.debug(f"   📊 {tf}: загружено {len(df)} свечей")
                else:
                    logger.warning(f"   ⚠️  {tf}: данные не получены")
            
            if not data_frames:
                result['error'] = "Нет данных для анализа"
                return result
            
            # 2. РАСЧЕТ УРОВНЕЙ (Multi-TimeFrame)
            all_levels = {}
            for tf, df in data_frames.items():
                levels = self.level_calculator.calculate(df)
                if levels:
                    all_levels[tf] = levels
            
            if not all_levels:
                result['error'] = "Не удалось рассчитать уровни"
                return result
            
            result['levels'] = all_levels
            
            # 3. ОЦЕНКА КОНФЛЮЭНСА
            confluence = self.confluence_calculator.analyze(all_levels)
            
            # 4. ГЕНЕРАЦИЯ СИГНАЛОВ
            main_df = data_frames.get('1h') or list(data_frames.values())[0]
            signals = self.signal_generator.analyze(
                symbol=symbol,
                df=main_df,
                levels=all_levels,
                confluence=confluence
            )
            
            if signals:
                result['signals'] = signals
                
                # 5. ОБРАБОТКА СИГНАЛОВ
                for signal in signals:
                    # Сохраняем в БД
                    await self.db.save_signal(signal)
                    
                    # Отправляем уведомление (если сигнал достаточно сильный)
                    if signal.get('strength', 0) >= settings.MIN_SIGNAL_STRENGTH:
                        await self.telegram_notifier.send_signal(signal)
                        
                        # Сохраняем для истории
                        self.last_signals[symbol] = {
                            'signal': signal,
                            'time': datetime.now()
                        }
            
            result['success'] = True
            logger.info(f"✅ {symbol}: обработан, сигналов: {len(signals)}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Ошибка обработки {symbol}: {e}")
            logger.debug(traceback.format_exc())
        
        return result
    
    async def run_cycle(self) -> Dict:
        """Выполняет один полный цикл анализа для всех символов."""
        cycle_start = datetime.now()
        self.cycle_count += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 ЦИКЛ #{self.cycle_count} | {cycle_start.strftime('%H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        results = {}
        
        # Обрабатываем все символы параллельно
        tasks = [self.process_symbol(symbol) for symbol in self.symbols]
        symbol_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Собираем результаты
        for i, symbol in enumerate(self.symbols):
            result = symbol_results[i]
            if isinstance(result, Exception):
                results[symbol] = {'error': str(result), 'success': False}
            else:
                results[symbol] = result
        
        # Статистика цикла
        successful = sum(1 for r in results.values() if r.get('success'))
        total_signals = sum(len(r.get('signals', [])) for r in results.values())
        
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        
        logger.info(f"\n📊 ИТОГИ ЦИКЛА #{self.cycle_count}:")
        logger.info(f"   Успешно: {successful}/{len(self.symbols)} символов")
        logger.info(f"   Сигналов: {total_signals}")
        logger.info(f"   Время: {cycle_time:.2f} секунд")
        
        return {
            'cycle': self.cycle_count,
            'timestamp': cycle_start.isoformat(),
            'duration': cycle_time,
            'results': results,
            'statistics': {
                'successful_symbols': successful,
                'total_signals': total_signals
            }
        }
    
    async def run_continuous(self):
        """Запускает бесконечный цикл анализа."""
        self.is_running = True
        
        # Инициализация
        if not await self.initialize():
            logger.error("❌ Не удалось инициализировать систему")
            return
        
        # Приветственное сообщение
        if settings.TELEGRAM_ENABLED:
            await self.telegram_notifier.send_message(
                f"🚀 Trading Bot v2.0 запущен!\n"
                f"📊 Символы: {', '.join(self.symbols)}\n"
                f"⏱️  Интервал: {self.update_interval} секунд"
            )
        
        logger.info("🚀 Запуск основного цикла...")
        
        # Основной цикл
        while self.is_running:
            try:
                cycle_result = await self.run_cycle()
                
                # Ждем до следующего цикла
                await asyncio.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Получен сигнал остановки...")
                self.is_running = False
                break
                
            except Exception as e:
                logger.error(f"❌ Критическая ошибка в основном цикле: {e}")
                logger.debug(traceback.format_exc())
                
                # Ждем перед повторной попыткой
                await asyncio.sleep(60)
        
        # Завершение работы
        await self.shutdown()
    
    async def shutdown(self):
        """Корректное завершение работы."""
        logger.info("🔚 Завершение работы...")
        
        if settings.TELEGRAM_ENABLED:
            await self.telegram_notifier.send_message(
                "🛑 Trading Bot остановлен."
            )
        
        # Закрываем соединения
        await self.data_handler.close()
        await self.db.close()
        
        logger.info("✅ Работа завершена корректно.")

# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

async def main():
    """Точка входа для запуска бота."""
    runner = EngineRunner()
    
    try:
        await runner.run_continuous()
    except KeyboardInterrupt:
        logger.info("👋 Завершение по запросу пользователя")
    except Exception as e:
        logger.error(f"💀 Фатальная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Запускаем бота
    asyncio.run(main())