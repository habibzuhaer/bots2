# bots2/main.py
#!/usr/bin/env python3
"""
Главная точка входа в приложение bots2.
Запуск: python main.py [mode] [options]
"""
import argparse
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_bot():
    """Запуск торгового движка."""
    from engine_runner import EngineRunner
    print("Запуск торгового движка...")
    runner = EngineRunner(symbols=['BTC/USDT', 'ETH/USDT'])
    asyncio.run(runner.run_continuous(interval_seconds=300))

def run_backtest():
    """Запуск бэктеста."""
    print("Режим бэктеста. (В разработке)")
    # Здесь будет вызов backtest/runner.py
    from backtest.runner import BacktestRunner
    runner = BacktestRunner()
    runner.run()

def run_web():
    """Запуск веб-интерфейса."""
    import uvicorn
    print("Запуск веб-сервера на http://localhost:8000")
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)

def main():
    parser = argparse.ArgumentParser(description='Trading Bot Bots2')
    parser.add_argument('mode', choices=['bot', 'backtest', 'web'], 
                       nargs='?', default='web',
                       help='Режим работы: bot, backtest или web')
    
    args = parser.parse_args()
    
    if args.mode == 'bot':
        run_bot()
    elif args.mode == 'backtest':
        run_backtest()
    elif args.mode == 'web':
        run_web()
    else:
        print("Неизвестный режим. Используйте: bot, backtest или web")

if __name__ == '__main__':
    main()