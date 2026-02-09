# bots2/engine_runner.py
import asyncio
import pandas as pd
from datetime import datetime
import time
import sys
import os

# Добавляем путь для импорта модулей проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_handler import DataHandler
from engine.levels import LevelCalculator
from engine.confluence import ConfluenceCalculator
# Импортируем будущий модуль сигналов
from engine.signals import SignalGenerator

class EngineRunner:
    """
    Главный оркестратор. Цикл:
    1. Забирает данные -> 2. Считает уровни -> 3. Оценивает конфлюэнс -> 4. Генерирует сигнал.
    """
    def __init__(self, symbols=None, timeframes=None):
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.timeframes = timeframes or ['1h', '4h']
        self.data_handler = DataHandler()
        self.level_calc = LevelCalculator()
        self.confluence_calc = ConfluenceCalculator()
        self.signal_gen = SignalGenerator()
        
        print(f"[Engine] Инициализирован для {self.symbols} | Таймфреймы: {self.timeframes}")

    async def run_single_cycle(self, symbol: str):
        """Выполняет полный цикл анализа для одной пары."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Анализ {symbol}...")
        
        try:
            # 1. ПОЛУЧЕНИЕ ДАННЫХ
            data_frames = {}
            for tf in self.timeframes:
                df = await self.data_handler.get_ohlcv(symbol, tf, limit=100)
                if df is not None:
                    data_frames[tf] = df
                    print(f"   | Загружен {tf}, свечей: {len(df)}")
                else:
                    print(f"   | Ошибка загрузки {tf}")
            
            if not data_frames:
                print(f"   | Пропуск {symbol}: нет данных")
                return
            
            # 2. РАСЧЕТ УРОВНЕЙ (Multi-TimeFrame)
            all_levels = {}
            for tf, df in data_frames.items():
                levels = self.level_calc.calculate_support_resistance(df)
                all_levels[tf] = levels
                # (Для отладки) Вывод ключевых уровней
                if levels.get('resistances'):
                    key_res = levels['resistances'][0] if levels['resistances'] else None
                    key_sup = levels['supports'][0] if levels['supports'] else None
                    print(f"   | {tf}: R ~{key_res:.2f}, S ~{key_sup:.2f}")
            
            # 3. ОЦЕНКА КОНФЛЮЭНСА (Совпадение уровней на разных TF)
            confluence_report = self.confluence_calc.evaluate(all_levels)
            
            # 4. ГЕНЕРАЦИЯ СИГНАЛА НА ОСНОВЕ ЛОГИКИ
            current_price = data_frames['1h']['close'].iloc[-1]
            signal = self.signal_gen.generate(
                symbol=symbol,
                price=current_price,
                levels=all_levels,
                confluence=confluence_report
            )
            
            # 5. ВЫВОД ИТОГА
            if signal and signal.get('direction'):
                print(f"   --> СИГНАЛ: {signal['direction']} | Сила: {signal.get('strength', 'N/A')}")
                # Здесь позже добавим вызов модуля уведомлений (Telegram)
                # await self.notify(symbol, signal, current_price)
            else:
                print(f"   --> Сигналов нет")
                
        except Exception as e:
            print(f"   | Ошибка в цикле для {symbol}: {e}")

    async def run_continuous(self, interval_seconds=300):
        """Бесконечный цикл анализа с заданным интервалом."""
        print(f"\n[Engine] Запуск в непрерывном режиме. Интервал: {interval_seconds}с")
        while True:
            start_time = time.time()
            tasks = [self.run_single_cycle(symbol) for symbol in self.symbols]
            await asyncio.gather(*tasks)
            
            elapsed = time.time() - start_time
            sleep_for = max(0, interval_seconds - elapsed)
            print(f"\n[Engine] Цикл завершен за {elapsed:.1f}с. Следующий через {sleep_for:.0f}с...")
            await asyncio.sleep(sleep_for)

# ===== ТОЧКА ЗАПУСКА ДЛЯ ТЕСТА =====
if __name__ == "__main__":
    runner = EngineRunner(symbols=['BTC/USDT'], timeframes=['1h', '4h'])
    
    # Запускаем один полный цикл для проверки
    print("\n" + "="*50)
    print("ТЕСТОВЫЙ ЗАПУСК ENGINE_RUNNER")
    print("="*50)
    asyncio.run(runner.run_single_cycle('BTC/USDT'))
    
    # Раскомментируйте для запуска бесконечного цикла:
    # asyncio.run(runner.run_continuous())