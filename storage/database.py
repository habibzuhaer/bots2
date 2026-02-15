#!/usr/bin/env python3
"""
ПОЛНЫЙ МОДУЛЬ РАБОТЫ С БАЗОЙ ДАННЫХ
Версия: 2.0
Использует SQLite для хранения сигналов, уровней, рыночных данных и настроек
Поддержка асинхронных операций через asyncio.to_thread
"""

import sqlite3
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import traceback
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# ============================================================================
# КЛАСС DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """
    Менеджер базы данных SQLite.
    
    Хранит:
    - Торговые сигналы (signals)
    - Уровни поддержки/сопротивления (levels)
    - Рыночные данные (market_data)
    - Настройки (settings)
    - Результаты циклов (cycle_results)
    - Ошибки (errors)
    
    Все операции выполняются асинхронно с использованием asyncio.to_thread.
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, db_path: str = "data/trading_bot.db", 
                 backup_enabled: bool = True,
                 backup_interval_hours: int = 6):
        """
        Инициализация менеджера БД.
        
        Args:
            db_path: Путь к файлу базы данных
            backup_enabled: Включить автоматическое резервное копирование
            backup_interval_hours: Интервал резервного копирования в часах
        """
        self.db_path = db_path
        self.backup_enabled = backup_enabled
        self.backup_interval_hours = backup_interval_hours
        
        # Создаем директорию для БД, если не существует
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Статистика
        self.stats = {
            'signals_saved': 0,
            'levels_saved': 0,
            'market_data_points': 0,
            'queries': 0,
            'errors': 0
        }
        
        # Кеш для настроек
        self.settings_cache = {}
        self.cache_ttl = timedelta(seconds=60)
        self.cache_timestamp = None
        
        logger.info(f"✅ DatabaseManager v{self.VERSION} инициализирован: {db_path}")
    
    async def initialize(self):
        """Асинхронная инициализация базы данных (создание таблиц)."""
        await self._init_db()
        logger.info("✅ База данных инициализирована")
    
    async def _init_db(self):
        """Создает необходимые таблицы, если они не существуют."""
        create_tables_sql = """
        -- Таблица сигналов
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            direction TEXT NOT NULL,
            strength TEXT NOT NULL,
            price REAL NOT NULL,
            confidence REAL,
            stop_loss REAL,
            take_profit REAL,
            risk_reward_ratio REAL,
            timeframe TEXT,
            indicators_json TEXT,
            levels_json TEXT,
            confluence_json TEXT,
            description TEXT,
            metadata_json TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            executed BOOLEAN DEFAULT FALSE,
            profit_loss REAL,
            notes TEXT
        );

        -- Таблица уровней
        CREATE TABLE IF NOT EXISTS levels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            level_type TEXT NOT NULL,
            price REAL NOT NULL,
            strength TEXT,
            confidence REAL,
            touches INTEGER DEFAULT 0,
            volume_profile REAL,
            calculation_method TEXT,
            cluster_size INTEGER DEFAULT 1,
            first_touch_time DATETIME,
            last_touch_time DATETIME,
            broken BOOLEAN DEFAULT FALSE,
            broken_time DATETIME,
            retests INTEGER DEFAULT 0,
            metadata_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME,
            UNIQUE(symbol, timeframe, level_type, price, created_at)
        );

        -- Таблица рыночных данных (кеш)
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            candle_timestamp DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            UNIQUE(symbol, timeframe, candle_timestamp)
        );

        -- Таблица результатов циклов
        CREATE TABLE IF NOT EXISTS cycle_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_number INTEGER,
            start_time DATETIME,
            end_time DATETIME,
            symbols_processed INTEGER,
            total_signals INTEGER,
            total_errors INTEGER,
            performance_metrics_json TEXT,
            details_json TEXT
        );

        -- Таблица ошибок
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            component TEXT,
            symbol TEXT,
            error_type TEXT,
            error_message TEXT,
            traceback TEXT
        );

        -- Таблица настроек
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- Индексы для производительности
        CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals(symbol, timestamp);
        CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals(direction);
        CREATE INDEX IF NOT EXISTS idx_levels_symbol_timeframe ON levels(symbol, timeframe);
        CREATE INDEX IF NOT EXISTS idx_levels_active ON levels(expires_at) WHERE expires_at IS NULL;
        CREATE INDEX IF NOT EXISTS idx_market_data_lookup ON market_data(symbol, timeframe, candle_timestamp);
        CREATE INDEX IF NOT EXISTS idx_errors_time ON errors(timestamp);
        """
        
        await self._execute_script(create_tables_sql)
    
    @asynccontextmanager
    async def get_connection(self):
        """Асинхронный контекстный менеджер для соединения с БД."""
        loop = asyncio.get_running_loop()
        conn = await loop.run_in_executor(None, sqlite3.connect, self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            await loop.run_in_executor(None, conn.commit)
        except Exception as e:
            await loop.run_in_executor(None, conn.rollback)
            raise e
        finally:
            await loop.run_in_executor(None, conn.close)
    
    async def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Выполняет SQL запрос и возвращает курсор."""
        self.stats['queries'] += 1
        async with self.get_connection() as conn:
            loop = asyncio.get_running_loop()
            cursor = await loop.run_in_executor(None, conn.execute, sql, params)
            return cursor
    
    async def _execute_script(self, sql: str):
        """Выполняет многострочный SQL скрипт."""
        async with self.get_connection() as conn:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, conn.executescript, sql)
    
    async def _fetch_all(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Выполняет запрос и возвращает все строки."""
        cursor = await self._execute(sql, params)
        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(None, cursor.fetchall)
        return rows
    
    async def _fetch_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Выполняет запрос и возвращает одну строку."""
        cursor = await self._execute(sql, params)
        loop = asyncio.get_running_loop()
        row = await loop.run_in_executor(None, cursor.fetchone)
        return row
    
    async def save_signal(self, signal: Union[Dict, Any]) -> int:
        """
        Сохраняет торговый сигнал в базу данных.
        
        Args:
            signal: Словарь с данными сигнала или объект Signal
            
        Returns:
            ID сохраненного сигнала или -1 при ошибке
        """
        try:
            # Преобразуем объект в словарь, если необходимо
            if hasattr(signal, 'to_dict'):
                signal_dict = signal.to_dict()
            else:
                signal_dict = signal
            
            # Подготовка JSON полей
            indicators_json = json.dumps(signal_dict.get('indicators', {}), default=str)
            levels_json = json.dumps(signal_dict.get('levels', {}), default=str)
            confluence_json = json.dumps(signal_dict.get('confluence', {}), default=str)
            metadata_json = json.dumps(signal_dict.get('metadata', {}), default=str)
            
            # Извлечение полей
            sql = """
                INSERT INTO signals (
                    symbol, signal_type, direction, strength, price, confidence,
                    stop_loss, take_profit, risk_reward_ratio, timeframe,
                    indicators_json, levels_json, confluence_json, description,
                    metadata_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                signal_dict.get('symbol'),
                signal_dict.get('type') or signal_dict.get('signal_type'),
                signal_dict.get('direction'),
                signal_dict.get('strength'),
                signal_dict.get('price'),
                signal_dict.get('confidence'),
                signal_dict.get('stop_loss'),
                signal_dict.get('take_profit'),
                signal_dict.get('risk_reward_ratio'),
                signal_dict.get('timeframe', '1h'),
                indicators_json,
                levels_json,
                confluence_json,
                signal_dict.get('description', ''),
                metadata_json,
                signal_dict.get('timestamp', datetime.now().isoformat())
            )
            
            cursor = await self._execute(sql, params)
            signal_id = cursor.lastrowid
            self.stats['signals_saved'] += 1
            logger.debug(f"✅ Сигнал сохранен с ID: {signal_id}")
            return signal_id
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Ошибка сохранения сигнала: {e}")
            logger.error(traceback.format_exc())
            await self.log_error('database', 'save_signal', str(e), traceback.format_exc())
            return -1
    
    async def save_levels(self, symbol: str, timeframe: str, 
                         levels: Dict[str, List]) -> bool:
        """
        Сохраняет уровни поддержки/сопротивления.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            levels: Словарь с поддержками и сопротивлениями (каждый элемент - объект Level)
            
        Returns:
            True если успешно, False при ошибке
        """
        try:
            # Отмечаем старые уровни как истекшие
            expire_sql = """
                UPDATE levels 
                SET expires_at = datetime('now')
                WHERE symbol = ? AND timeframe = ? AND expires_at IS NULL
            """
            await self._execute(expire_sql, (symbol, timeframe))
            
            # Сохраняем новые уровни
            count = 0
            for level_type, level_list in levels.items():
                for level in level_list:
                    # Преобразуем объект Level в словарь, если необходимо
                    if hasattr(level, 'to_dict'):
                        level_dict = level.to_dict()
                    else:
                        level_dict = level
                    
                    metadata_json = json.dumps(level_dict.get('metadata', {}), default=str)
                    
                    sql = """
                        INSERT INTO levels (
                            symbol, timeframe, level_type, price, strength,
                            confidence, touches, volume_profile, calculation_method,
                            cluster_size, first_touch_time, last_touch_time,
                            broken, broken_time, retests, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    params = (
                        symbol,
                        timeframe,
                        level_dict.get('type', level_type),
                        level_dict.get('price'),
                        level_dict.get('strength'),
                        level_dict.get('confidence', 0.5),
                        level_dict.get('touches', 0),
                        level_dict.get('volume_profile', 0.0),
                        level_dict.get('method', level_dict.get('calculation_method', 'unknown')),
                        level_dict.get('cluster_size', 1),
                        level_dict.get('first_touch_time'),
                        level_dict.get('last_touch_time'),
                        level_dict.get('broken', False),
                        level_dict.get('broken_time'),
                        level_dict.get('retests', 0),
                        metadata_json
                    )
                    await self._execute(sql, params)
                    count += 1
            
            self.stats['levels_saved'] += count
            logger.debug(f"✅ Сохранено {count} уровней для {symbol} {timeframe}")
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Ошибка сохранения уровней: {e}")
            await self.log_error('database', 'save_levels', str(e), traceback.format_exc())
            return False
    
    async def get_recent_signals(self, symbol: Optional[str] = None, 
                                 limit: int = 50,
                                 hours: Optional[int] = 24,
                                 min_confidence: float = 0.0) -> List[Dict]:
        """
        Получает последние сигналы.
        
        Args:
            symbol: Фильтр по символу
            limit: Максимальное количество сигналов
            hours: За последние N часов (если None, то без ограничения)
            min_confidence: Минимальная уверенность
            
        Returns:
            Список сигналов в виде словарей
        """
        try:
            sql = "SELECT * FROM signals WHERE 1=1"
            params = []
            
            if symbol:
                sql += " AND symbol = ?"
                params.append(symbol)
            
            if hours:
                sql += " AND timestamp > datetime('now', ?)"
                params.append(f'-{hours} hours')
            
            if min_confidence > 0:
                sql += " AND confidence >= ?"
                params.append(min_confidence)
            
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            rows = await self._fetch_all(sql, tuple(params))
            
            signals = []
            for row in rows:
                signal = dict(row)
                # Парсим JSON поля
                for json_field in ['indicators_json', 'levels_json', 'confluence_json', 'metadata_json']:
                    if signal.get(json_field):
                        signal[json_field.replace('_json', '')] = json.loads(signal[json_field])
                    signal.pop(json_field, None)
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения сигналов: {e}")
            return []
    
    async def get_active_levels(self, symbol: str, timeframe: str) -> Dict[str, List[Dict]]:
        """
        Получает активные уровни для символа и таймфрейма.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            
        Returns:
            Словарь с ключами 'supports' и 'resistances', каждый список словарей
        """
        try:
            sql = """
                SELECT * FROM levels 
                WHERE symbol = ? AND timeframe = ? 
                  AND (expires_at IS NULL OR expires_at > datetime('now'))
                ORDER BY level_type, price
            """
            rows = await self._fetch_all(sql, (symbol, timeframe))
            
            levels = {'supports': [], 'resistances': []}
            for row in rows:
                level = dict(row)
                if level.get('metadata_json'):
                    level['metadata'] = json.loads(level['metadata_json'])
                level.pop('metadata_json', None)
                
                if level['level_type'] == 'support':
                    levels['supports'].append(level)
                else:
                    levels['resistances'].append(level)
            
            return levels
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения уровней: {e}")
            return {'supports': [], 'resistances': []}
    
    async def cache_market_data(self, symbol: str, timeframe: str, 
                               df: pd.DataFrame) -> bool:
        """
        Кеширует рыночные данные.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            df: DataFrame с колонками open, high, low, close, volume и индексом datetime
            
        Returns:
            True если успешно
        """
        try:
            # Очищаем старые данные для этого символа/таймфрейма (оставляем последние 30 дней)
            cutoff = datetime.now() - timedelta(days=30)
            delete_sql = """
                DELETE FROM market_data 
                WHERE symbol = ? AND timeframe = ? AND candle_timestamp < ?
            """
            await self._execute(delete_sql, (symbol, timeframe, cutoff))
            
            # Вставляем новые данные
            records = []
            for idx, row in df.iterrows():
                # idx может быть Timestamp, конвертируем в строку для SQLite
                ts = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
                records.append((
                    symbol,
                    timeframe,
                    ts,
                    row.get('open'),
                    row.get('high'),
                    row.get('low'),
                    row.get('close'),
                    row.get('volume')
                ))
            
            insert_sql = """
                INSERT OR REPLACE INTO market_data 
                (symbol, timeframe, candle_timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # Выполняем вставку батчами
            batch_size = 500
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                async with self.get_connection() as conn:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, conn.executemany, insert_sql, batch)
            
            self.stats['market_data_points'] += len(records)
            logger.debug(f"✅ Закешировано {len(records)} свечей для {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка кеширования данных: {e}")
            return False
    
    async def get_cached_market_data(self, symbol: str, timeframe: str,
                                     limit: int = 1000,
                                     from_date: Optional[datetime] = None,
                                     to_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Получает кешированные рыночные данные.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            limit: Максимальное количество свечей
            from_date: Начальная дата (опционально)
            to_date: Конечная дата (опционально)
            
        Returns:
            DataFrame с данными
        """
        try:
            sql = """
                SELECT candle_timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if from_date:
                sql += " AND candle_timestamp >= ?"
                params.append(from_date.isoformat())
            if to_date:
                sql += " AND candle_timestamp <= ?"
                params.append(to_date.isoformat())
            
            sql += " ORDER BY candle_timestamp DESC LIMIT ?"
            params.append(limit)
            
            rows = await self._fetch_all(sql, tuple(params))
            
            if not rows:
                return pd.DataFrame()
            
            data = []
            for row in rows:
                data.append({
                    'timestamp': row['candle_timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения кешированных данных: {e}")
            return pd.DataFrame()
    
    async def save_cycle_result(self, result: Dict) -> bool:
        """
        Сохраняет результат цикла.
        
        Args:
            result: Словарь с результатами цикла
            
        Returns:
            True если успешно
        """
        try:
            sql = """
                INSERT INTO cycle_results 
                (cycle_number, start_time, end_time, symbols_processed, total_signals, total_errors, performance_metrics_json, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                result.get('cycle_number'),
                result.get('start_time'),
                result.get('end_time'),
                result.get('symbols_processed', 0),
                result.get('total_signals', 0),
                result.get('total_errors', 0),
                json.dumps(result.get('performance_metrics', {}), default=str),
                json.dumps(result.get('details', {}), default=str)
            )
            await self._execute(sql, params)
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результата цикла: {e}")
            return False
    
    async def log_error(self, component: str, symbol: str, error_type: str, 
                       error_message: str, traceback_str: str = ''):
        """
        Логирует ошибку в таблицу errors.
        
        Args:
            component: Компонент (например, 'data_handler', 'levels')
            symbol: Символ или 'N/A'
            error_type: Тип ошибки
            error_message: Сообщение об ошибке
            traceback_str: Трассировка
        """
        try:
            sql = """
                INSERT INTO errors (component, symbol, error_type, error_message, traceback)
                VALUES (?, ?, ?, ?, ?)
            """
            await self._execute(sql, (component, symbol, error_type, error_message, traceback_str))
        except Exception as e:
            logger.error(f"❌ Ошибка при логировании ошибки: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику работы БД.
        
        Returns:
            Словарь со статистикой
        """
        try:
            # Количество сигналов за последние 24 часа
            signals_24h = await self._fetch_one("""
                SELECT COUNT(*) as count FROM signals 
                WHERE timestamp > datetime('now', '-1 day')
            """)
            signals_24h = signals_24h['count'] if signals_24h else 0
            
            # Количество уровней активных
            levels_active = await self._fetch_one("""
                SELECT COUNT(*) as count FROM levels 
                WHERE expires_at IS NULL OR expires_at > datetime('now')
            """)
            levels_active = levels_active['count'] if levels_active else 0
            
            # Количество ошибок за последние 24 часа
            errors_24h = await self._fetch_one("""
                SELECT COUNT(*) as count FROM errors 
                WHERE timestamp > datetime('now', '-1 day')
            """)
            errors_24h = errors_24h['count'] if errors_24h else 0
            
            # Размер базы данных
            db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
            
            return {
                'version': self.VERSION,
                'signals_saved': self.stats['signals_saved'],
                'levels_saved': self.stats['levels_saved'],
                'market_data_points': self.stats['market_data_points'],
                'queries': self.stats['queries'],
                'errors_logged': self.stats['errors'],
                'signals_last_24h': signals_24h,
                'active_levels': levels_active,
                'errors_last_24h': errors_24h,
                'db_size_mb': db_size / (1024 * 1024),
                'db_path': self.db_path
            }
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    async def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Получает значение настройки.
        
        Args:
            key: Ключ настройки
            default: Значение по умолчанию, если ключ не найден
            
        Returns:
            Значение настройки (десериализованное из JSON)
        """
        try:
            # Проверка кеша
            if self.cache_timestamp and datetime.now() - self.cache_timestamp < self.cache_ttl:
                if key in self.settings_cache:
                    return self.settings_cache[key]
            
            sql = "SELECT value FROM settings WHERE key = ?"
            row = await self._fetch_one(sql, (key,))
            
            if row:
                value = json.loads(row['value'])
                # Обновляем кеш
                self.settings_cache[key] = value
                self.cache_timestamp = datetime.now()
                return value
            else:
                return default
        except Exception as e:
            logger.error(f"❌ Ошибка получения настройки {key}: {e}")
            return default
    
    async def set_setting(self, key: str, value: Any) -> bool:
        """
        Устанавливает значение настройки.
        
        Args:
            key: Ключ настройки
            value: Значение (будет сериализовано в JSON)
            
        Returns:
            True если успешно
        """
        try:
            value_json = json.dumps(value, default=str)
            sql = """
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
            """
            await self._execute(sql, (key, value_json))
            
            # Обновляем кеш
            self.settings_cache[key] = value
            self.cache_timestamp = datetime.now()
            
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка установки настройки {key}: {e}")
            return False
    
    async def vacuum(self):
        """Выполняет VACUUM для оптимизации базы данных."""
        try:
            await self._execute("VACUUM")
            logger.info("✅ База данных оптимизирована (VACUUM)")
        except Exception as e:
            logger.error(f"❌ Ошибка VACUUM: {e}")
    
    async def backup(self, backup_path: Optional[str] = None) -> bool:
        """
        Создает резервную копию базы данных.
        
        Args:
            backup_path: Путь для сохранения копии (если None, генерируется автоматически)
            
        Returns:
            True если успешно
        """
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"data/backups/trading_bot_{timestamp}.db"
            
            # Создаем директорию для бэкапов
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Копируем файл
            import shutil
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, shutil.copy2, self.db_path, backup_path)
            
            logger.info(f"✅ Резервная копия создана: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка создания резервной копии: {e}")
            return False
    
    async def close(self):
        """Закрывает соединения (ничего не делает, т.к. соединения закрываются в get_connection)."""
        logger.info("🔚 DatabaseManager закрыт")

# ============================================================================
# СИНГЛТОН ДЛЯ ГЛОБАЛЬНОГО ДОСТУПА
# ============================================================================

_db_instance = None

async def get_database() -> DatabaseManager:
    """Возвращает глобальный экземпляр DatabaseManager."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
        await _db_instance.initialize()
    return _db_instance

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("🧪 Тестирование DatabaseManager...")
        
        # Используем временную БД в памяти
        db = DatabaseManager(":memory:")
        await db.initialize()
        
        # 1. Сохранение сигнала
        test_signal = {
            'symbol': 'BTC/USDT',
            'type': 'breakout',
            'direction': 'BUY',
            'strength': 'strong',
            'price': 50000.0,
            'confidence': 0.85,
            'stop_loss': 49500.0,
            'take_profit': 51000.0,
            'risk_reward_ratio': 2.0,
            'timeframe': '1h',
            'indicators': {'rsi': 45, 'macd': 10},
            'levels': {'supports': [49000], 'resistances': [51000]},
            'confluence': {'score': 0.8},
            'description': 'Test signal',
            'timestamp': datetime.now().isoformat()
        }
        
        signal_id = await db.save_signal(test_signal)
        print(f"✅ Сигнал сохранен с ID: {signal_id}")
        
        # 2. Получение сигналов
        signals = await db.get_recent_signals('BTC/USDT', limit=5)
        print(f"✅ Получено сигналов: {len(signals)}")
        if signals:
            print(f"   Первый сигнал: {signals[0]}")
        
        # 3. Сохранение уровней
        from engine.levels import Level, LevelType, LevelStrength
        test_levels = {
            'supports': [
                Level(price=49000, level_type=LevelType.SUPPORT, strength=LevelStrength.STRONG, confidence=0.9, touches=3),
                Level(price=49500, level_type=LevelType.SUPPORT, strength=LevelStrength.MEDIUM, confidence=0.7, touches=1)
            ],
            'resistances': [
                Level(price=51000, level_type=LevelType.RESISTANCE, strength=LevelStrength.STRONG, confidence=0.9, touches=4),
                Level(price=51500, level_type=LevelType.RESISTANCE, strength=LevelStrength.MEDIUM, confidence=0.6, touches=2)
            ]
        }
        success = await db.save_levels('BTC/USDT', '1h', test_levels)
        print(f"✅ Уровни сохранены: {success}")
        
        # 4. Получение уровней
        levels = await db.get_active_levels('BTC/USDT', '1h')
        print(f"✅ Получено уровней: {len(levels['supports'])} поддержек, {len(levels['resistances'])} сопротивлений")
        
        # 5. Настройки
        await db.set_setting('test_key', {'value': 123})
        value = await db.get_setting('test_key')
        print(f"✅ Настройка получена: {value}")
        
        # 6. Статистика
        stats = await db.get_statistics()
        print(f"✅ Статистика: {stats}")
        
        print("\n🎉 Тестирование завершено успешно!")
    
    asyncio.run(test())
