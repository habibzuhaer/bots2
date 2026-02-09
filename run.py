#!/usr/bin/env python3
# bots2/run.py
"""
Универсальный скрипт запуска торгового бота.
Поддерживает все режимы работы.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

def setup_environment():
    """Настройка окружения."""
    # Добавляем корень проекта в PYTHONPATH
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Проверяем наличие .env файла
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  Файл .env не найден. Создаю шаблон...")
        create_env_template(env_file)
    
    # Проверяем зависимости
    check_dependencies()

def create_env_template(env_file: Path):
    """Создает шаблон .env файла."""
    template = """# ===== ОКРУЖЕНИЕ =====
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# ===== БИРЖА (Binance) =====
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true

# ===== TELEGRAM =====
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# ===== ТОРГОВЫЕ НАСТРОЙКИ =====
RISK_PER_TRADE=0.02
UPDATE_INTERVAL=300
"""
    
    env_file.write_text(template)
    print(f"✅ Шаблон .env создан: {env_file}")
    print("⚠️  Заполните его своими данными перед запуском!")

def check_dependencies():
    """Проверяет установленные зависимости."""
    try:
        import ccxt
        import pandas
        import fastapi
        print("✅ Основные зависимости установлены")
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        sys.exit(1)

def run_bot(symbols=None, timeframes=None, interval=300):
    """Запускает торгового бота."""
    from engine_runner import EngineRunner
    
    symbols = symbols or ["BTC/USDT"]
    timeframes = timeframes or ["1h", "4h"]
    
    print(f"🤖 Запуск торгового бота")
    print(f"   Пары: {', '.join(symbols)}")
    print(f"   Таймфреймы: {', '.join(timeframes)}")
    print(f"   Интервал анализа: {interval} секунд")
    print()
    
    runner = EngineRunner(symbols=symbols, timeframes=timeframes)
    
    import asyncio
    asyncio.run(runner.run_continuous(interval_seconds=interval))

def run_web(host="0.0.0.0", port=8000, reload=False):
    """Запускает веб-интерфейс."""
    cmd = ["uvicorn", "web.app:app", f"--host={host}", f"--port={port}"]
    
    if reload:
        cmd.append("--reload")
    
    print(f"🌐 Запуск веб-интерфейса")
    print(f"   URL: http://{host}:{port}")
    print(f"   Документация: http://{host}:{port}/api/docs")
    print()
    
    subprocess.run(cmd)

def run_backtest_cli():
    """Запускает CLI бэктеста."""
    from backtest.runner import run_backtest_cli as run_cli
    
    import asyncio
    asyncio.run(run_cli())

def run_database_setup():
    """Настраивает базу данных."""
    from storage.database import DatabaseManager
    
    print("🗄️  Настройка базы данных...")
    db = DatabaseManager()
    print("✅ База данных готова к работе")

def main():
    parser = argparse.ArgumentParser(description='Trading Bot v2 - Универсальный запуск')
    
    # Основные команды
    subparsers = parser.add_subparsers(dest='command', help='Команда')
    
    # Команда bot
    bot_parser = subparsers.add_parser('bot', help='Запустить торгового бота')
    bot_parser.add_argument('--symbols', nargs='+', default=['BTC/USDT'], 
                           help='Торговые пары')
    bot_parser.add_argument('--timeframes', nargs='+', default=['1h', '4h'],
                           help='Таймфреймы')
    bot_parser.add_argument('--interval', type=int, default=300,
                           help='Интервал анализа (секунды)')
    
    # Команда web
    web_parser = subparsers.add_parser('web', help='Запустить веб-интерфейс')
    web_parser.add_argument('--host', default='0.0.0.0', help='Хост')
    web_parser.add_argument('--port', type=int, default=8000, help='Порт')
    web_parser.add_argument('--reload', action='store_true', help='Автоперезагрузка')
    
    # Команда backtest
    backtest_parser = subparsers.add_parser('backtest', help='Запустить бэктест')
    backtest_parser.add_argument('symbol', help='Торговая пара (например, BTC/USDT)')
    backtest_parser.add_argument('--start', required=True, help='Дата начала (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', help='Дата окончания (YYYY-MM-DD)')
    backtest_parser.add_argument('--timeframe', default='1h', help='Таймфрейм')
    backtest_parser.add_argument('--balance', type=float, default=10000.0,
                                help='Начальный баланс')
    
    # Команда setup
    subparsers.add_parser('setup', help='Настроить окружение и базу данных')
    
    # Команда status
    subparsers.add_parser('status', help='Показать статус системы')
    
    args = parser.parse_args()
    
    # Настройка окружения
    setup_environment()
    
    # Выполнение команды
    if args.command == 'bot':
        run_bot(args.symbols, args.timeframes, args.interval)
    elif args.command == 'web':
        run_web(args.host, args.port, args.reload)
    elif args.command == 'backtest':
        # Сохраняем аргументы для бэктеста
        import os
        os.environ['BACKTEST_ARGS'] = f"{args.symbol} {args.start} {args.end or ''} {args.timeframe} {args.balance}"
        run_backtest_cli()
    elif args.command == 'setup':
        run_database_setup()
    elif args.command == 'status':
        from config.settings import settings
        settings.print_summary()
    else:
        # Если команда не указана, показываем справку
        parser.print_help()
        
        # Показываем примеры использования
        print("\n📚 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:")
        print("  python run.py bot --symbols BTC/USDT ETH/USDT --timeframes 1h 4h")
        print("  python run.py web --host localhost --port 8080 --reload")
        print("  python run.py backtest BTC/USDT --start 2024-01-01 --end 2024-01-31")
        print("  python run.py setup")
        print("  python run.py status")

if __name__ == "__main__":
    main()