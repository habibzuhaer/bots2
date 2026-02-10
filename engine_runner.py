#!/usr/bin/env python3
"""
ГЛАВНЫЙ ОРКЕСТРАТОР ТОРГОВОЙ СИСТЕМЫ
Версия: 2.0
Автор: Trading Bot Team
Описание: Координирует все компоненты системы: сбор данных → анализ → сигналы → уведомления
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
import os
import traceback
import json
import time

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты компонентов системы
try:
    from data_handler import DataHandler
    from engine.levels import LevelCalculator
    from engine.confluence import ConfluenceCalculator
    from engine.signals import SignalGenerator
    from engine.cme import CMECalculator
    from storage.database import DatabaseManager
    from notify.telegram import TelegramNotifier
    from config.settings import settings
except ImportError as e:
    print(f"❌ Ошибка импорта модулей: {e}")
    print("Убедитесь, что все файлы созданы и структура проекта правильная")
    sys.exit(1)

# ============================================================================
# КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ
# ============================================================================

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/engine.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# КЛАСС ENGINE RUNNER
# ============================================================================

class EngineRunner:
    """
    Главный класс-оркестратор торговой системы.
    Управляет жизненным циклом анализа, координирует работу всех компонентов.
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, 
                 symbols: Optional[List[str]] = None,
                 timeframes: Optional[List[str]] = None,
                 config_path: Optional[str] = None):
        """
        Инициализация EngineRunner.
        
        Args:
            symbols: Список торговых пар для анализа
            timeframes: Список таймфреймов для анализа
            config_path: Путь к конфигурационному файлу (опционально)
        """
        
        # Конфигурация
        self.symbols = symbols or settings.DEFAULT_SYMBOLS
        self.timeframes = timeframes or settings.TIMEFRAMES
        self.interval_seconds = settings.UPDATE_INTERVAL
        self.config_path = config_path
        
        # Состояние системы
        self.is_running = False
        self.cycle_count = 0
        self.start_time = None
        self.last_execution_time = {}
        self.errors = []
        self.performance_stats = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'failed_cycles': 0,
            'average_cycle_time': 0,
            'total_signals_generated': 0
        }
        
        # Инициализация компонентов
        self._initialize_components()
        
        # История сигналов
        self.signal_history = []
        self.alert_history = []
        
        logger.info(f"✅ EngineRunner v{self.VERSION} инициализирован")
        logger.info(f"   Символы: {', '.join(self.symbols)}")
        logger.info(f"   Таймфреймы: {', '.join(self.timeframes)}")
        logger.info(f"   Интервал: {self.interval_seconds} секунд")
    
    def _initialize_components(self):
        """Инициализирует все компоненты системы."""
        try:
            logger.info("🔄 Инициализация компонентов системы...")
            
            # 1. Обработчик данных
            self.data_handler = DataHandler(
                exchange_id=settings.DEFAULT_EXCHANGE,
                cache_enabled=True,
                cache_ttl=300
            )
            
            # 2. Калькулятор уровней
            self.level_calculator = LevelCalculator(
                cluster_threshold=0.005,
                min_touches=2,
                use_volume_profile=True,
                use_fibonacci=True
            )
            
            # 3. Калькулятор конфлюэнса
            self.confluence_calculator = ConfluenceCalculator(
                min_timeframes=2,
                weight_mapping={
                    '1h': 1.0,
                    '4h': 1.5,
                    '1d': 2.0
                }
            )
            
            # 4. Генератор сигналов
            self.signal_generator = SignalGenerator(
                rsi_overbought=70,
                rsi_oversold=30,
                macd_threshold=0,
                min_confidence=0.6
            )
            
            # 5. Калькулятор CME (если используется)
            self.cme_calculator = CMECalculator() if hasattr(settings, 'USE_CME') and settings.USE_CME else None
            
            # 6. Менеджер базы данных
            self.database = DatabaseManager(
                db_path=settings.DB_PATH,
                backup_enabled=True,
                backup_interval_hours=6
            )
            
            # 7. Telegram уведомитель
            self.telegram_notifier = TelegramNotifier(
                bot_token=settings.TELEGRAM_BOT_TOKEN,
                chat_id=settings.TELEGRAM_CHAT_ID,
                parse_mode="HTML"
            )
            
            logger.info("✅ Все компоненты успешно инициализированы")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации компонентов: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def initialize(self) -> bool:
        """
        Асинхронная инициализация системы.
        
        Returns:
            bool: True если инициализация успешна, False в противном случае
        """
        try:
            logger.info("🔧 Запуск асинхронной инициализации...")
            
            # 1. Проверка подключения к данным
            if not await self.data_handler.test_connection():
                logger.error("❌ Не удалось подключиться к источнику данных")
                return False
            
            # 2. Инициализация базы данных
            await self.database.initialize()
            
            # 3. Проверка Telegram (если включен)
            if settings.TELEGRAM_ENABLED:
                if not await self.telegram_notifier.test_connection():
                    logger.warning("⚠️  Не удалось подключиться к Telegram")
                else:
                    await self.telegram_notifier.send_message(
                        f"🚀 Trading Bot v{self.VERSION} запущен!\n"
                        f"📊 Начинаю анализ {len(self.symbols)} пар\n"
                        f"⏱️  Интервал: {self.interval_seconds} сек"
                    )
            
            # 4. Загрузка начальных данных
            await self._preload_initial_data()
            
            # 5. Запись в лог успешной инициализации
            logger.info("🎯 Система готова к работе")
            
            # 6. Стартовая статистика
            self.start_time = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации системы: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _preload_initial_data(self):
        """Предзагрузка начальных данных для всех символов."""
        logger.info("📥 Предзагрузка начальных данных...")
        
        tasks = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                task = self.data_handler.get_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=settings.DATA_LIMIT
                )
                tasks.append(task)
        
        # Параллельная загрузка
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        loaded_count = sum(1 for r in results if not isinstance(r, Exception) and r is not None)
        logger.info(f"✅ Предзагружено данных: {loaded_count}/{len(tasks)}")
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Полный цикл анализа для одного символа.
        
        Args:
            symbol: Торговая пара для анализа
            
        Returns:
            Dict: Результаты анализа
        """
        analysis_result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'stages': {},
            'signals': [],
            'errors': [],
            'processing_time': 0,
            'market_data': {}
        }
        
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Начинаю анализ {symbol}")
            
            # ЭТАП 1: ЗАГРУЗКА ДАННЫХ
            stage_start = time.time()
            data_frames = {}
            
            for timeframe in self.timeframes:
                try:
                    df = await self.data_handler.get_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=settings.DATA_LIMIT
                    )
                    
                    if df is not None and not df.empty:
                        data_frames[timeframe] = df
                        logger.debug(f"   📊 {timeframe}: {len(df)} свечей")
                    else:
                        logger.warning(f"   ⚠️  {timeframe}: данные не получены")
                        analysis_result['errors'].append(f"No data for {timeframe}")
                        
                except Exception as e:
                    error_msg = f"Ошибка загрузки {timeframe}: {str(e)}"
                    logger.error(f"   ❌ {error_msg}")
                    analysis_result['errors'].append(error_msg)
            
            if not data_frames:
                analysis_result['errors'].append("Нет данных для анализа")
                return analysis_result
            
            analysis_result['stages']['data_loading'] = {
                'time': time.time() - stage_start,
                'timeframes_loaded': len(data_frames),
                'total_candles': sum(len(df) for df in data_frames.values())
            }
            
            # ЭТАП 2: РАСЧЕТ УРОВНЕЙ (Multi-TimeFrame)
            stage_start = time.time()
            all_levels = {}
            
            for timeframe, df in data_frames.items():
                try:
                    levels = self.level_calculator.calculate(df, timeframe)
                    if levels:
                        all_levels[timeframe] = levels
                        
                        # Сохранение уровней в БД
                        await self.database.save_levels(symbol, timeframe, levels)
                        
                except Exception as e:
                    error_msg = f"Ошибка расчета уровней {timeframe}: {str(e)}"
                    logger.error(f"   ❌ {error_msg}")
                    analysis_result['errors'].append(error_msg)
            
            if not all_levels:
                analysis_result['errors'].append("Не удалось рассчитать уровни")
                return analysis_result
            
            analysis_result['stages']['levels_calculation'] = {
                'time': time.time() - stage_start,
                'timeframes_processed': len(all_levels),
                'total_levels': sum(len(lvls.get('supports', [])) + len(lvls.get('resistances', [])) 
                                  for lvls in all_levels.values())
            }
            
            # ЭТАП 3: ОЦЕНКА КОНФЛЮЭНСА
            stage_start = time.time()
            confluence = None
            
            try:
                confluence = self.confluence_calculator.evaluate(all_levels)
                analysis_result['confluence'] = confluence
            except Exception as e:
                error_msg = f"Ошибка оценки конфлюэнса: {str(e)}"
                logger.error(f"   ❌ {error_msg}")
                analysis_result['errors'].append(error_msg)
            
            analysis_result['stages']['confluence_evaluation'] = {
                'time': time.time() - stage_start
            }
            
            # ЭТАП 4: РАСЧЕТ CME (если включен)
            if self.cme_calculator:
                stage_start = time.time()
                try:
                    cme_data = self.cme_calculator.analyze(data_frames.get('1h'))
                    analysis_result['cme'] = cme_data
                except Exception as e:
                    error_msg = f"Ошибка расчета CME: {str(e)}"
                    logger.warning(f"   ⚠️  {error_msg}")
                    # CME не критично, продолжаем работу
                analysis_result['stages']['cme_analysis'] = {
                    'time': time.time() - stage_start
                }
            
            # ЭТАП 5: ГЕНЕРАЦИЯ СИГНАЛОВ
            stage_start = time.time()
            main_df = data_frames.get('1h') or list(data_frames.values())[0]
            current_price = main_df['close'].iloc[-1]
            
            signals = []
            try:
                signal = self.signal_generator.generate(
                    symbol=symbol,
                    price=current_price,
                    levels=all_levels,
                    confluence=confluence or {}
                )
                
                if signal and signal.get('direction') != 'NEUTRAL':
                    signals.append(signal)
                    
                    # Сохранение сигнала
                    signal_id = await self.database.save_signal(signal)
                    signal['db_id'] = signal_id
                    
                    # Отправка уведомления
                    if signal.get('confidence', 0) >= settings.MIN_CONFIDENCE:
                        await self._handle_signal_notification(signal)
                        
                        # Обновление статистики
                        self.performance_stats['total_signals_generated'] += 1
                    
                    analysis_result['signals'] = signals
                    
            except Exception as e:
                error_msg = f"Ошибка генерации сигналов: {str(e)}"
                logger.error(f"   ❌ {error_msg}")
                analysis_result['errors'].append(error_msg)
            
            analysis_result['stages']['signal_generation'] = {
                'time': time.time() - stage_start,
                'signals_generated': len(signals)
            }
            
            # ЭТАП 6: СОБИРАЕМ РЕЗУЛЬТАТЫ
            analysis_result['success'] = True
            analysis_result['market_data'] = {
                'current_price': current_price,
                'price_change_24h': self._calculate_price_change(main_df),
                'volume_24h': main_df['volume'].sum() if len(main_df) >= 24 else 0,
                'volatility': self._calculate_volatility(main_df)
            }
            
            # Запись в историю
            self._record_analysis_result(analysis_result)
            
            logger.info(f"✅ {symbol}: анализ завершен за {time.time() - start_time:.2f}с, "
                       f"сигналов: {len(signals)}")
            
        except Exception as e:
            error_msg = f"Критическая ошибка анализа {symbol}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            analysis_result['errors'].append(error_msg)
        
        finally:
            analysis_result['processing_time'] = time.time() - start_time
        
        return analysis_result
    
    async def _handle_signal_notification(self, signal: Dict[str, Any]):
        """Обработка уведомлений о сигналах."""
        try:
            if not settings.TELEGRAM_ENABLED:
                return
            
            # Формируем сообщение
            message = self._format_signal_message(signal)
            
            # Отправляем в Telegram
            await self.telegram_notifier.send_signal(message)
            
            # Логируем
            logger.info(f"📤 Отправлено уведомление о сигнале: {signal['symbol']} {signal['direction']}")
            
            # Сохраняем в историю
            self.alert_history.append({
                'timestamp': datetime.now().isoformat(),
                'signal': signal,
                'type': 'telegram'
            })
            
        except Exception as e:
            logger.error(f"❌ Ошибка отправки уведомления: {e}")
    
    def _format_signal_message(self, signal: Dict[str, Any]) -> str:
        """Форматирует сообщение о сигнале."""
        symbol = signal.get('symbol', 'N/A')
        direction = signal.get('direction', 'UNKNOWN')
        price = signal.get('price', 0)
        confidence = signal.get('confidence', 0) * 100
        strength = signal.get('strength', 'MEDIUM')
        
        emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
        
        return f"""
{emoji} <b>ТОРГОВЫЙ СИГНАЛ</b> {emoji}

<b>Пара:</b> {symbol}
<b>Направление:</b> <code>{direction}</code>
<b>Сила:</b> {strength}
<b>Уверенность:</b> {confidence:.1f}%
<b>Цена:</b> ${price:,.2f}

<b>Время:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<i>Сигнал сгенерирован Trading Bot v{self.VERSION}</i>
"""
    
    def _calculate_price_change(self, df: pd.DataFrame) -> float:
        """Рассчитывает изменение цены за 24 часа."""
        if len(df) < 24:
            return 0
        
        old_price = df['close'].iloc[-24]
        current_price = df['close'].iloc[-1]
        
        return ((current_price - old_price) / old_price) * 100
    
    def _calculate_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """Рассчитывает волатильность."""
        if len(df) < period:
            return 0
        
        returns = df['close'].pct_change().dropna()
        if len(returns) < period:
            return 0
        
        return returns.tail(period).std() * 100  # В процентах
    
    def _record_analysis_result(self, result: Dict[str, Any]):
        """Записывает результат анализа в историю."""
        self.signal_history.append(result)
        
        # Ограничиваем размер истории
        max_history = settings.MAX_HISTORY_SIZE
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]
    
    async def run_cycle(self) -> Dict[str, Any]:
        """
        Выполняет один полный цикл анализа для всех символов.
        
        Returns:
            Dict: Результаты цикла
        """
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 ЦИКЛ #{self.cycle_count} | {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        cycle_result = {
            'cycle_number': self.cycle_count,
            'start_time': cycle_start.isoformat(),
            'symbols_processed': [],
            'total_signals': 0,
            'total_errors': 0,
            'performance_metrics': {}
        }
        
        # Анализ всех символов параллельно
        tasks = []
        for symbol in self.symbols:
            task = self.analyze_symbol(symbol)
            tasks.append((symbol, task))
        
        # Запускаем задачи и собираем результаты
        symbol_results = {}
        for symbol, task in tasks:
            try:
                result = await task
                symbol_results[symbol] = result
                
                if result['success']:
                    cycle_result['symbols_processed'].append({
                        'symbol': symbol,
                        'success': True,
                        'signals': len(result.get('signals', [])),
                  