#!/usr/bin/env python3
"""
Конфигурация всего проекта.
Все настройки в одном месте с валидацией.
"""
import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class LogLevel(str, Enum):
    """Уровни логирования."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ExchangeType(str, Enum):
    """Типы бирж."""
    BINANCE = "binance"
    BYBIT = "bybit"
    KUCOIN = "kucoin"
    OKX = "okx"

@dataclass
class ExchangeConfig:
    """Конфигурация для одной биржи."""
    name: ExchangeType
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    enabled: bool = True
    
    def is_configured(self) -> bool:
        """Проверяет, сконфигурирована ли биржа."""
        return bool(self.api_key and self.api_secret)

@dataclass
class TelegramConfig:
    """Конфигурация Telegram бота."""
    enabled: bool = True
    bot_token: str = ""
    chat_id: str = ""
    parse_mode: str = "HTML"
    send_signals: bool = True
    send_errors: bool = True
    send_summary: bool = True
    
    def is_configured(self) -> bool:
        """Проверяет, сконфигурирован ли Telegram."""
        return bool(self.bot_token and self.chat_id)

@dataclass
class TradingConfig:
    """Торговые настройки."""
    # Символы для анализа
    default_symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT", 
        "ETH/USDT", 
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT"
    ])
    
    # Таймфреймы
    timeframes: List[str] = field(default_factory=lambda: [
        "15m",
        "1h", 
        "4h",
        "1d"
    ])
    
    # Параметры анализа
    update_interval: int = 300  # секунды
    data_limit: int = 500  # свечей на таймфрейм
    min_signal_strength: float = 0.6  # минимальная сила сигнала
    
    # Управление рисками
    risk_per_trade: float = 0.02  # 2% на сделку
    max_open_positions: int = 3
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    trailing_stop_pct: float = 0.01  # 1%
    
    # Уровни
    level_cluster_threshold: float = 0.005  # 0.5%
    level_min_touches: int = 2
    level_volume_weight: float = 0.3
    level_time_weight: float = 0.7

@dataclass
class DatabaseConfig:
    """Конфигурация базы данных."""
    path: str = "data/trading_bot.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    backup_retention_days: int = 7
    cache_size: int = 1000  # записей в кеше

@dataclass
class WebConfig:
    """Конфигурация веб-интерфейса."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    @property
    def url(self) -> str:
        """Возвращает URL веб-интерфейса."""
        return f"http://{self.host}:{self.port}"

@dataclass
class BacktestConfig:
    """Конфигурация бэктеста."""
    initial_balance: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    default_period_days: int = 90

@dataclass
class Settings:
    """
    Главный класс настроек.
    Объединяет все конфигурации.
    """
    
    # Основные настройки
    project_name: str = "Trading Bot v2.0"
    version: str = "2.0.0"
    environment: str = "development"  # development, testing, production
    log_level: LogLevel = LogLevel.INFO
    debug: bool = False
    
    # Пути
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(project_root, "data")
    logs_dir: str = os.path.join(project_root, "logs")
    config_dir: str = os.path.join(project_root, "config")
    
    # Компоненты
    exchanges: Dict[ExchangeType, ExchangeConfig] = field(default_factory=dict)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    web: WebConfig = field(default_factory=WebConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    def __post_init__(self):
        """Инициализация после создания объекта."""
        # Загружаем настройки из переменных окружения
        self._load_from_env()
        
        # Инициализируем биржи
        self._init_exchanges()
        
        # Создаем директории
        self._create_directories()
        
        # Валидация
        self._validate()
    
    def _load_from_env(self):
        """Загружает настройки из переменных окружения."""
        
        # Основные настройки
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        self.debug = os.getenv("DEBUG", str(self.debug)).lower() == "true"
        
        log_level_str = os.getenv("LOG_LEVEL", self.log_level.value)
        self.log_level = LogLevel(log_level_str.upper())
        
        # Telegram
        self.telegram.enabled = os.getenv("TELEGRAM_ENABLED", str(self.telegram.enabled)).lower() == "true"
        self.telegram.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", self.telegram.bot_token)
        self.telegram.chat_id = os.getenv("TELEGRAM_CHAT_ID", self.telegram.chat_id)
        
        # Binance
        binance_config = ExchangeConfig(
            name=ExchangeType.BINANCE,
            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
            enabled=os.getenv("BINANCE_ENABLED", "true").lower() == "true"
        )
        self.exchanges[ExchangeType.BINANCE] = binance_config
        
        # Bybit
        bybit_config = ExchangeConfig(
            name=ExchangeType.BYBIT,
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
            testnet=os.getenv("BYBIT_TESTNET", "true").lower() == "true",
            enabled=os.getenv("BYBIT_ENABLED", "false").lower() == "true"
        )
        self.exchanges[ExchangeType.BYBIT] = bybit_config
        
        # База данных
        self.database.path = os.getenv("DB_PATH", self.database.path)
        
        # Веб-интерфейс
        self.web.host = os.getenv("WEB_HOST", self.web.host)
        self.web.port = int(os.getenv("WEB_PORT", self.web.port))
    
    def _init_exchanges(self):
        """Инициализирует конфигурации бирж."""
        # Убедимся, что все биржи присутствуют
        for exchange_type in ExchangeType:
            if exchange_type not in self.exchanges:
                self.exchanges[exchange_type] = ExchangeConfig(name=exchange_type)
    
    def _create_directories(self):
        """Создает необходимые директории."""
        directories = [
            self.data_dir,
            self.logs_dir,
            os.path.join(self.data_dir, "backups"),
            os.path.join(self.data_dir, "backtests"),
            os.path.join(self.data_dir, "cache"),
            os.path.join(self.logs_dir, "signals"),
            os.path.join(self.logs_dir, "errors"),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _validate(self):
        """Валидация настроек."""
        errors = []
        
        # Проверяем, что хотя бы одна биржа настроена
        configured_exchanges = [e for e in self.exchanges.values() 
                              if e.enabled and e.is_configured()]
        
        if not configured_exchanges:
            errors.append("Не настроена ни одна биржа. Установите API ключи.")
        
        # Проверяем Telegram
        if self.telegram.enabled and not self.telegram.is_configured():
            errors.append("Telegram включен, но не настроен.")
        
        # Проверяем пути
        if not os.path.exists(self.data_dir):
            errors.append(f"Директория данных не существует: {self.data_dir}")
        
        if errors:
            error_msg = "\n".join([f"  • {error}" for error in errors])
            raise ValueError(f"Ошибки конфигурации:\n{error_msg}")
    
    def get_configured_exchanges(self) -> List[ExchangeConfig]:
        """Возвращает список сконфигурированных бирж."""
        return [exchange for exchange in self.exchanges.values() 
                if exchange.enabled and exchange.is_configured()]
    
    def get_active_symbols(self) -> List[str]:
        """Возвращает активные символы для торговли."""
        return self.trading.default_symbols
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует настройки в словарь (без секретов)."""
        data = asdict(self)
        
        # Очищаем секретные данные
        for exchange_config in data['exchanges'].values():
            if exchange_config['api_key']:
                exchange_config['api_key'] = '***'
            if exchange_config['api_secret']:
                exchange_config['api_secret'] = '***'
        
        if data['telegram']['bot_token']:
            data['telegram']['bot_token'] = '***'
        
        return data
    
    def save(self, filename: str = "config/current_settings.json"):
        """Сохраняет текущие настройки в файл (без секретов)."""
        filepath = os.path.join(self.project_root, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logging.info(f"⚙️  Настройки сохранены: {filepath}")
    
    def print_summary(self):
        """Выводит сводку настроек."""
        print("\n" + "="*60)
        print(f"⚙️  КОНФИГУРАЦИЯ: {self.project_name} v{self.version}")
        print("="*60)
        
        # Основные настройки
        print(f"\n📋 ОСНОВНЫЕ:")
        print(f"   Окружение: {self.environment}")
        print(f"   Логирование: {self.log_level.value}")
        print(f"   Отладка: {self.debug}")
        
        # Биржи
        print(f"\n🏦 БИРЖИ:")
        for exchange in self.get_configured_exchanges():
            status = "✅" if exchange.testnet else "⚠️ "
            print(f"   {status} {exchange.name.value}: {'Testnet' if exchange.testnet else 'Mainnet'}")
        
        # Telegram
        print(f"\n📱 TELEGRAM:")
        status = "✅ Настроен" if self.telegram.is_configured() else "❌ Не настроен"
        print(f"   Статус: {status}")
        
        # Торговля
        print(f"\n📊 ТОРГОВЛЯ:")
        print(f"   Символы: {', '.join(self.trading.default_symbols[:3])}...")
        print(f"   Таймфреймы: {', '.join(self.trading.timeframes)}")
        print(f"   Интервал: {self.trading.update_interval} сек")
        
        # Пути
        print(f"\n📁 ПУТИ:")
        print(f"   Данные: {self.data_dir}")
        print(f"   Логи: {self.logs_dir}")
        print(f"   БД: {self.database.path}")
        
        print("="*60)
    
    @property
    def is_production(self) -> bool:
        """Проверяет, работает ли в production режиме."""
        return self.environment.lower() == "production"

# Глобальный экземпляр настроек
settings = Settings()

def get_settings() -> Settings:
    """Возвращает глобальный экземпляр настроек."""
    return settings

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    # Тестируем загрузку настроек
    print("🧪 Тестирование загрузки настроек...")
    
    # Создаем временные переменные окружения для теста
    os.environ["TELEGRAM_BOT_TOKEN"] = "test_token"
    os.environ["TELEGRAM_CHAT_ID"] = "test_chat"
    os.environ["BINANCE_API_KEY"] = "test_key"
    os.environ["BINANCE_API_SECRET"] = "test_secret"
    
    # Перезагружаем настройки
    settings = Settings()
    
    # Выводим сводку
    settings.print_summary()
    
    # Проверяем методы
    print(f"\n✅ Сконфигурированные биржи: {len(settings.get_configured_exchanges())}")
    print(f"✅ Активные символы: {len(settings.get_active_symbols())}")
    print(f"✅ Production режим: {settings.is_production}")
    
    # Сохраняем настройки
    settings.save()
    
    print("\n✅ Тест пройден успешно!")