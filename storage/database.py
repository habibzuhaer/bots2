# bots2/storage/database.py
import sqlite3
from sqlite3 import Connection
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Generator
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Менеджер базы данных SQLite для хранения всей торговой истории.
    Использует connection pooling и автоматическое создание схемы.
    """
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Инициализирует схему базы данных при первом запуске."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Таблица сигналов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strength TEXT,
                    price REAL NOT NULL,
                    confidence REAL,
                    levels_json TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    executed BOOLEAN DEFAULT FALSE,
                    profit_loss REAL,
                    notes TEXT
                )
            """)
            
            # Таблица уровней поддержки/сопротивления
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS levels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    level_type TEXT NOT NULL,  -- 'support' или 'resistance'
                    price REAL NOT NULL,
                    strength INTEGER,
                    touched_count INTEGER DEFAULT 0,
                    broken BOOLEAN DEFAULT FALSE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME,
                    UNIQUE(symbol, timeframe, level_type, price)
                )
            """)
            
            # Таблица для хранения сырых рыночных данных (кеш)
            cursor.execute("""
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
                )
            """)
            
            # Таблица настроек системы
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Индексы для быстрого поиска
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_levels_active ON levels(symbol, broken, expires_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_lookup ON market_data(symbol, timeframe, candle_timestamp)")
            
            conn.commit()
            logger.info(f"База данных инициализирована: {self.db_path}")
    
    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """
        Context manager для работы с подключением к БД.
        Автоматически закрывает соединение и откатывает при ошибке.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row  # Возвращает словари вместо кортежей
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Ошибка базы данных: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    # ===== ОСНОВНЫЕ МЕТОДЫ ДЛЯ СИГНАЛОВ =====
    
    async def save_signal(self, signal: Dict[str, Any]) -> int:
        """Сохраняет торговый сигнал в базу данных."""
        levels_json = json.dumps(signal.get('levels', {})) if signal.get('levels') else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signals 
                (symbol, direction, strength, price, confidence, levels_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.get('symbol'),
                signal.get('direction'),
                signal.get('strength'),
                signal.get('price'),
                signal.get('confidence'),
                levels_json,
                signal.get('timestamp', datetime.now().isoformat())
            ))
            return cursor.lastrowid
    
    async def get_recent_signals(self, 
                                symbol: Optional[str] = None, 
                                limit: int = 50,
                                hours: Optional[int] = 24) -> List[Dict]:
        """Получает последние сигналы с возможностью фильтрации."""
        query = "SELECT * FROM signals WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            query += " AND timestamp > ?"
            params.append(cutoff.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Преобразуем Row в словарь
            result = []
            for row in rows:
                row_dict = dict(row)
                # Парсим JSON поля
                if row_dict.get('levels_json'):
                    row_dict['levels'] = json.loads(row_dict['levels_json'])
                    del row_dict['levels_json']
                result.append(row_dict)
            
            return result
    
    async def mark_signal_executed(self, signal_id: int, profit_loss: Optional[float] = None):
        """Отмечает сигнал как исполненный и записывает P/L."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE signals 
                SET executed = TRUE, profit_loss = ?
                WHERE id = ?
            """, (profit_loss, signal_id))
    
    # ===== МЕТОДЫ ДЛЯ РАБОТЫ С УРОВНЯМИ =====
    
    async def save_levels(self, 
                         symbol: str, 
                         timeframe: str, 
                         supports: List[float], 
                         resistances: List[float]):
        """Сохраняет уровни поддержки и сопротивления."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Отмечаем старые уровни как истекшие
            cursor.execute("""
                UPDATE levels 
                SET expires_at = datetime('now', '+7 days')
                WHERE symbol = ? AND timeframe = ? AND expires_at IS NULL
            """, (symbol, timeframe))
            
            # Добавляем новые уровни поддержки
            for price in supports[:10]:  # Сохраняем только 10 ближайших
                cursor.execute("""
                    INSERT OR REPLACE INTO levels 
                    (symbol, timeframe, level_type, price, strength, touched_count)
                    VALUES (?, ?, 'support', ?, 1, 0)
                """, (symbol, timeframe, price))
            
            # Добавляем новые уровни сопротивления
            for price in resistances[:10]:
                cursor.execute("""
                    INSERT OR REPLACE INTO levels 
                    (symbol, timeframe, level_type, price, strength, touched_count)
                    VALUES (?, ?, 'resistance', ?, 1, 0)
                """, (symbol, timeframe, price))
            
            conn.commit()
    
    async def get_active_levels(self, 
                               symbol: str, 
                               timeframe: str) -> Dict[str, List[float]]:
        """Получает активные уровни для символа и таймфрейма."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT level_type, price, strength 
                FROM levels 
                WHERE symbol = ? 
                  AND timeframe = ? 
                  AND (expires_at IS NULL OR expires_at > datetime('now'))
                  AND broken = FALSE
                ORDER BY 
                    CASE level_type 
                        WHEN 'support' THEN 1 
                        WHEN 'resistance' THEN 2 
                    END,
                    price
            """, (symbol, timeframe))
            
            rows = cursor.fetchall()
            
            result = {'supports': [], 'resistances': []}
            for row in rows:
                if row['level_type'] == 'support':
                    result['supports'].append(row['price'])
                else:
                    result['resistances'].append(row['price'])
            
            return result
    
    # ===== МЕТОДЫ ДЛЯ РАБОТЫ С РЫНОЧНЫМИ ДАННЫМИ =====
    
    async def cache_market_data(self, 
                               symbol: str, 
                               timeframe: str, 
                               df: pd.DataFrame):
        """
        Кеширует рыночные данные в локальную БД.
        Полезно для бэктестов и оффлайн-работы.
        """
        if df.empty:
            return
        
        records = []
        for idx, row in df.iterrows():
            records.append((
                symbol,
                timeframe,
                idx.to_pydatetime(),
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row.get('volume', 0)
            ))
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO market_data 
                (symbol, timeframe, candle_timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            # Очищаем старые данные (старше 30 дней)
            cutoff = datetime.now() - timedelta(days=30)
            cursor.execute("""
                DELETE FROM market_data 
                WHERE candle_timestamp < ?
            """, (cutoff,))
            
            conn.commit()
    
    async def get_cached_data(self, 
                             symbol: str, 
                             timeframe: str, 
                             limit: int = 1000) -> pd.DataFrame:
        """Получает кешированные данные из БД."""
        with self.get_connection() as conn:
            query = """
                SELECT candle_timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY candle_timestamp DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, 
                                  params=(symbol, timeframe, limit),
                                  parse_dates=['candle_timestamp'])
            
            if not df.empty:
                df.set_index('candle_timestamp', inplace=True)
                df.sort_index(inplace=True)  # Сортируем по возрастанию времени
            
            return df
    
    # ===== СТАТИСТИКА И АНАЛИТИКА =====
    
    async def get_signal_statistics(self, 
                                   days: int = 30, 
                                   symbol: Optional[str] = None) -> Dict:
        """Возвращает статистику по сигналам за период."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    direction,
                    COUNT(*) as total,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN executed THEN 1 ELSE 0 END) as executed_count,
                    AVG(profit_loss) as avg_profit_loss
                FROM signals
                WHERE timestamp > datetime('now', ?)
            """
            
            params = [f'-{days} days']
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " GROUP BY direction"
            cursor.execute(query, params)
            
            stats = {}
            for row in cursor.fetchall():
                direction = row['direction']
                stats[direction] = {
                    'total': row['total'],
                    'executed': row['executed_count'],
                    'execution_rate': row['executed_count'] / row['total'] if row['total'] > 0 else 0,
                    'avg_confidence': row['avg_confidence'],
                    'avg_profit_loss': row['avg_profit_loss']
                }
            
            return stats

# Глобальный экземпляр для удобного доступа
_db_instance: Optional[DatabaseManager] = None

def get_database() -> DatabaseManager:
    """Возвращает глобальный экземпляр DatabaseManager."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance