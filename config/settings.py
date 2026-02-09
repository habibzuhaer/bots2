# bots2/config/settings.py
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import json

# Загружаем переменные окружения из .env файла
load_dotenv()

@dataclass
class ExchangeConfig:
    """Конфигурация биржи."""
    name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    
    @classmethod
    def from_env(cls, prefix: str = "BINANCE"):
        """Создает конфигурацию из переменных окружения."""
        return cls(
            name=os.getenv(f"{prefix}_EXCHANGE", "binance"),
            api_key=os.getenv(f"{prefix}_API_KEY", ""),
            api_secret=os.getenv(f"{prefix}_API_SECRET", ""),
            testnet=os.getenv(f"{prefix}_TESTNET", "true").lower() == "true"
        )

@dataclass
class TelegramConfig:
    """Конфигурация Telegram бота."""
    enabled: bool = True
    bot_token: str = ""
    chat_id: str = ""
    parse_mode: str = "HTML"
    alerts_enabled: bool = True
    signals_enabled: bool = True
    
    @classmethod
    def from_env(cls):
        """Создает конфигурацию из переменных окружения."""
        return cls(
            enabled=os.getenv("TELEGRAM_ENABLED", "true").lower() == "true",
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            parse_mode=os.getenv("TELEGRAM_PARSE_MODE", "HTML"),
            alerts_enabled=os.getenv("TELEGRAM_ALERTS_ENABLED", "true").lower() == "true",
            signals_enabled=os.getenv("TELEGRAM_SIGNALS_ENABLED", "true").lower() == "true"
        )

@dataclass
class TradingConfig:
    """Торговые настройки."""
    # Общие настройки
    default_symbols: list = None
    timeframes: list = None
    update_interval: int = 300  # секунды
    
    # Управление рисками
    risk_per_trade: float = 0.02  # 2% на сделку
    max_open_positions: int = 3
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    
    # Настройки уровней
    level_sensitivity: float = 0.002  # 0.2%
    volume_weight: float = 0.3
    time_weight: float = 0.7
    
    def __post_init__(self):
        if self.default_symbols is None:
            self.default_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        if self.timeframes is None:
            self.timeframes = ["15m", "1h", "4h", "1d"]

@dataclass
class DatabaseConfig:
    """Конфигурация базы данных."""
    path: str = "data/trading_bot.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 7
    
    @classmethod
    def from_env(cls):
        """Создает конфигурацию из переменных окружения."""
        return cls(
            path=os.getenv("DB_PATH", "data/trading_bot.db"),
            backup_enabled=os.getenv("DB_BACKUP_ENABLED", "true").lower() == "true",
            backup_interval_hours=int(os.getenv("DB_BACKUP_INTERVAL", "24")),
            backup_retention_days=int(os.getenv("DB_BACKUP_RETENTION", "7"))
        )

@dataclass
class BacktestConfig:
    """Конфигурация бэктеста."""
    initial_balance: float = 10000.0
    commission: float = 0.001  # 0.1%
    default_period_days: int = 30
    data_source: str = "database"  # database, csv, api

class Settings:
    """
    Главный класс настроек приложения.
    Объединяет все конфигурации в одном месте.
    """
    
    def __init__(self):
        self.exchange = ExchangeConfig.from_env()
        self.telegram = TelegramConfig.from_env()
        self.trading = TradingConfig()
        self.database = DatabaseConfig.from_env()
        self.backtest = BacktestConfig()
        
        # Динамические настройки
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Пути
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "data")
        self.logs_dir = os.path.join(self.project_root, "logs")
        
        # Создаем необходимые директории
        self._create_directories()
    
    def _create_directories(self):
        """Создает необходимые директории для работы приложения."""
        directories = [
            self.data_dir,
            self.logs_dir,
            os.path.join(self.data_dir, "backtests"),
            os.path.join(self.data_dir, "cache"),
            os.path.join(self.logs_dir, "signals"),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует настройки в словарь."""
        return {
            'exchange': asdict(self.exchange),
            'telegram': asdict(self.telegram),
            'trading': asdict(self.trading),
            'database': asdict(self.database),
            'backtest': asdict(self.backtest),
            'environment': self.environment,
            'debug': self.debug,
            'log_level': self.log_level,
            'paths': {
                'project_root': self.project_root,
                'data_dir': self.data_dir,
                'logs_dir': self.logs_dir
            }
        }
    
    def save(self, filename: str = "config/current_settings.json"):
        """Сохраняет текущие настройки в файл."""
        filepath = os.path.join(self.project_root, filename)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Настройки сохранены в {filepath}")
    
    def print_summary(self):
        """Выводит краткую информацию о настройках."""
        print("\n" + "="*60)
        print("КОНФИГУРАЦИЯ ПРИЛОЖЕНИЯ")
        print("="*60)
        
        # Безопасный вывод (без секретов)
        safe_config = self.to_dict()
        safe_config['exchange']['api_key'] = '***' if safe_config['exchange']['api_key'] else 'not set'
        safe_config['exchange']['api_secret'] = '***' if safe_config['exchange']['api_secret'] else 'not set'
        safe_config['telegram']['bot_token'] = '***' if safe_config['telegram']['bot_token'] else 'not set'
        
        for section, values in safe_config.items():
            if isinstance(values, dict):
                print(f"\n[{section.upper()}]")
                for key, value in values.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for subkey, subvalue in value.items():
                            print(f"    {subkey}: {subvalue}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"{section}: {values}")
        
        print("="*60)

# Глобальный экземпляр настроек
settings = Settings()

# Функция для быстрого доступа к настройкам
def get_settings() -> Settings:
    """Возвращает глобальный экземпляр настроек."""
    return settings

if __name__ == "__main__":
    # При запуске этого файла выводим сводку настроек
    settings.print_summary()
    
    # Проверяем обязательные настройки
    if not settings.exchange.api_key:
        print("\n⚠️  Внимание: API ключ биржи не установлен")
        print("   Установите переменную окружения BINANCE_API_KEY")
    
    if not settings.telegram.bot_token:
        print("\n⚠️  Внимание: Telegram бот не настроен")
        print("   Установите переменные окружения TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID")
    
    # Сохраняем настройки в файл
    settings.save()