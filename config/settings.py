import os
from typing import Dict, List

# Настройки биржи
EXCHANGE_CONFIG = {
    'default_exchange': 'binance',
    'exchanges': {
        'binance': {
            'api_key': os.getenv('BINANCE_API_KEY', ''),
            'api_secret': os.getenv('BINANCE_API_SECRET', ''),
            'testnet': False
        },
        'bybit': {
            'api_key': os.getenv('BYBIT_API_KEY', ''),
            'api_secret': os.getenv('BYBIT_API_SECRET', ''),
            'testnet': True
        }
    }
}

# Торговые настройки
TRADING_SETTINGS = {
    'default_symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    'timeframes': ['15m', '1h', '4h', '1d'],
    'risk_per_trade': 0.02,  # 2% на сделку
    'max_open_positions': 3,
    'stop_loss_percent': 0.02,  # 2% стоп-лосс
    'take_profit_ratios': [1.5, 2.0, 3.0],
    
    # Настройки уровней
    'level_sensitivity': 0.002,  # 0.2% для определения уровня
    'volume_weight': 0.3,
    'time_weight': 0.7,
}

# Настройки уведомлений
NOTIFICATION_SETTINGS = {
    'telegram': {
        'enabled': True,
        'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
        'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
    },
    'webhook': {
        'enabled': False,
        'url': os.getenv('WEBHOOK_URL', '')
    }
}

# Настройки базы данных
DATABASE_CONFIG = {
    'path': 'data/bot_database.db',
    'backup_interval_hours': 24
}