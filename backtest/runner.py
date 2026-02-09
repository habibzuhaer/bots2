# bots2/backtest/runner.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio

from storage.database import get_database
from engine.levels import LevelCalculator
from engine.confluence import ConfluenceCalculator
from engine.signals import SignalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Trade:
    """Класс для представления одной сделки."""
    id: int
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None

@dataclass
class BacktestResult:
    """Результаты бэктеста."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    
    # Статистика
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Метрики
    win_rate: float
    profit_factor: float
    total_pnl: float
    total_return_percent: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Детали
    trades: List[Trade]
    equity_curve: List[Dict[str, float]]
    daily_returns: List[float]

class BacktestRunner:
    """
    Основной движок бэктестирования.
    Тестирует стратегию на исторических данных.
    """
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,  # 0.1%
                 stop_loss_pct: float = 0.02,  # 2%
                 take_profit_pct: float = 0.04  # 4%
                 ):
        self.initial_balance = initial_balance
        self.commission = commission
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.level_calc = LevelCalculator()
        self.confluence_calc = ConfluenceCalculator()
        self.signal_gen = SignalGenerator()
        
        self.current_balance = initial_balance
        self.position = None  # Текущая открытая позиция
        self.trades = []
        self.equity_curve = []
        
    def run(self, 
            df: pd.DataFrame,
            symbol: str,
            timeframe: str) -> BacktestResult:
        """
        Запускает бэктест на переданных данных.
        
        :param df: DataFrame с историческими данными
        :param symbol: Торговая пара
        :param timeframe: Таймфрейм данных
        :return: BacktestResult
        """
        logger.info(f"Запуск бэктеста для {symbol} ({timeframe})")
        logger.info(f"Период: {df.index[0]} - {df.index[-1]}")
        logger.info(f"Баланс: ${self.initial_balance:,.2f}")
        
        self.current_balance = self.initial_balance
        self.trades = []
        self.equity_curve = []
        
        # Скользящее окно для анализа (имитация реального времени)
        window_size = 100
        trade_id = 1
        
        for i in range(window_size, len(df)):
            current_data = df.iloc[:i].copy()
            current_price = df['close'].iloc[i]
            current_time = df.index[i]
            
            # Обновляем кривую капитала
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': self.current_balance,
                'price': current_price
            })
            
            # Закрываем позиции по стоп-лоссу или тейк-профиту
            if self.position:
                self._check_position_exit(current_price, current_time)
            
            # Генерируем сигналы только если нет открытой позиции
            if not self.position:
                # 1. Рассчитываем уровни на текущем окне
                levels = self.level_calc.calculate_support_resistance(
                    current_data.tail(200)  # Используем последние 200 свечей
                )
                
                # 2. Оцениваем конфлюэнс (для простоты используем один ТФ)
                confluence = self.confluence_calc.evaluate({'current': levels})
                
                # 3. Генерируем сигнал
                signal = self.signal_gen.generate(
                    symbol=symbol,
                    price=current_price,
                    levels={'current': levels},
                    confluence=confluence
                )
                
                # 4. Если есть сигнал - открываем позицию
                if signal and signal['direction'] in ['BUY', 'SELL']:
                    self._open_position(
                        trade_id=trade_id,
                        direction=signal['direction'],
                        entry_price=current_price,
                        entry_time=current_time,
                        stop_loss_pct=self.stop_loss_pct,
                        take_profit_pct=self.take_profit_pct
                    )
                    trade_id += 1
        
        # Закрываем все открытые позиции по последней цене
        if self.position:
            self._close_position(
                exit_price=df['close'].iloc[-1],
                exit_time=df.index[-1]
            )
        
        # Формируем финальные результаты
        return self._compile_results(
            symbol=symbol,
            timeframe=timeframe,
            start_date=df.index[0],
            end_date=df.index[-1]
        )
    
    def _open_position(self, 
                      trade_id: int,
                      direction: str,
                      entry_price: float,
                      entry_time: datetime,
                      stop_loss_pct: float,
                      take_profit_pct: float):
        """Открывает новую позицию."""
        
        # Рассчитываем стоп-лосс и тейк-профит
        if direction == 'BUY':
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # SELL
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
        
        # Расчет размера позиции (рискуем 2% от баланса)
        risk_amount = self.current_balance * 0.02
        position_size = risk_amount / (abs(entry_price - stop_loss) / entry_price)
        
        trade = Trade(
            id=trade_id,
            direction=TradeDirection(direction),
            entry_price=entry_price,
            entry_time=entry_time,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.position = {
            'trade': trade,
            'size': position_size,
            'entry_balance': self.current_balance
        }
        
        logger.debug(f"Открыта позиция #{trade_id}: {direction} по ${entry_price:,.2f}")
    
    def _check_position_exit(self, current_price: float, current_time: datetime):
        """Проверяет условия выхода из позиции."""
        if not self.position:
            return
        
        trade = self.position['trade']
        
        # Проверка стоп-лосса
        if (trade.direction == TradeDirection.BUY and current_price <= trade.stop_loss) or \
           (trade.direction == TradeDirection.SELL and current_price >= trade.stop_loss):
            self._close_position(current_price, current_time, reason="SL")
        
        # Проверка тейк-профита
        elif (trade.direction == TradeDirection.BUY and current_price >= trade.take_profit) or \
             (trade.direction == TradeDirection.SELL and current_price <= trade.take_profit):
            self._close_position(current_price, current_time, reason="TP")
    
    def _close_position(self, 
                       exit_price: float, 
                       exit_time: datetime, 
                       reason: str = "MANUAL"):
        """Закрывает текущую позицию."""
        if not self.position:
            return
        
        trade = self.position['trade']
        position_size = self.position['size']
        
        # Расчет PnL
        if trade.direction == TradeDirection.BUY:
            pnl = (exit_price - trade.entry_price) * position_size
        else:  # SELL
            pnl = (trade.entry_price - exit_price) * position_size
        
        # Учитываем комиссию
        pnl -= (trade.entry_price * position_size * self.commission)
        pnl -= (exit_price * position_size * self.commission)
        
        pnl_percent = (pnl / self.position['entry_balance']) * 100
        
        # Обновляем баланс
        self.current_balance += pnl
        
        # Заполняем данные сделки
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        
        self.trades.append(trade)
        
        logger.debug(
            f"Закрыта позиция #{trade.id}: {trade.direction.value} "
            f"PnL: ${pnl:,.2f} ({pnl_percent:+.2f}%) "
            f"Причина: {reason}"
        )
        
        # Сбрасываем позицию
        self.position = None
    
    def _compile_results(self,
                        symbol: str,
                        timeframe: str,
                        start_date: datetime,
                        end_date: datetime) -> BacktestResult:
        """Компилирует все результаты бэктеста."""
        
        # Базовые метрики
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl and t.pnl > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Общий PnL
        total_pnl = sum(t.pnl or 0 for t in self.trades)
        total_return_percent = ((self.current_balance / self.initial_balance) - 1) * 100
        
        # Profit Factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl and t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Максимальная просадка
        max_drawdown = self._calculate_max_drawdown()
        
        # Коэффициент Шарпа (упрощенный)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=self.current_balance,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            total_return_percent=total_return_percent,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades,
            equity_curve=self.equity_curve,
            daily_returns=self._calculate_daily_returns()
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Рассчитывает максимальную просадку по кривой капитала."""
        if not self.equity_curve:
            return 0
        
        equities = [point['equity'] for point in self.equity_curve]
        peak = equities[0]
        max_dd = 0
        
        for equity in equities:
            if equity > peak:
                peak = equity
            
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Рассчитывает упрощенный коэффициент Шарпа."""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i]['equity'] / self.equity_curve[i-1]['equity']) - 1
            returns.append(ret)
        
        if not returns:
            return 0
        
        avg_return = np.mean(returns) * 252  # Годовой доход
        std_return = np.std(returns) * np.sqrt(252)  # Годовая волатильность
        
        if std_return == 0:
            return 0
        
        return (avg_return - risk_free_rate) / std_return
    
    def _calculate_daily_returns(self) -> List[float]:
        """Рассчитывает дневные доходности."""
        # Группируем по дням
        daily_equity = {}
        for point in self.equity_curve:
            date = point['timestamp'].date()
            daily_equity[date] = point['equity']
        
        # Сортируем по дате
        sorted_dates = sorted(daily_equity.keys())
        returns = []
        
        for i in range(1, len(sorted_dates)):
            prev = daily_equity[sorted_dates[i-1]]
            curr = daily_equity[sorted_dates[i]]
            returns.append((curr - prev) / prev)
        
        return returns

def save_backtest_result(result: BacktestResult, filename: str = None):
    """Сохраняет результаты бэктеста в JSON файл."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{result.symbol.replace('/', '_')}_{timestamp}.json"
    
    # Преобразуем объект в словарь
    result_dict = {
        'symbol': result.symbol,
        'timeframe': result.timeframe,
        'start_date': result.start_date.isoformat(),
        'end_date': result.end_date.isoformat(),
        'metrics': {
            'initial_balance': result.initial_balance,
            'final_balance': result.final_balance,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_return_percent': result.total_return_percent,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio
        },
        'trades': [
            {
                'id': t.id,
                'direction': t.direction.value,
                'entry_price': t.entry_price,
                'entry_time': t.entry_time.isoformat(),
                'exit_price': t.exit_price,
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent
            }
            for t in result.trades
        ]
    }
    
    with open(f"data/backtests/{filename}", 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    logger.info(f"Результаты сохранены в {filename}")
    return filename

# CLI интерфейс для запуска бэктеста
async def run_backtest_cli():
    """Запускает бэктест из командной строки."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Запуск бэктеста торговой стратегии')
    parser.add_argument('symbol', help='Торговая пара (например, BTC/USDT)')
    parser.add_argument('--start', help='Дата начала (YYYY-MM-DD)', required=True)
    parser.add_argument('--end', help='Дата окончания (YYYY-MM-DD)', default=None)
    parser.add_argument('--timeframe', help='Таймфрейм', default='1h')
    parser.add_argument('--balance', help='Начальный баланс', type=float, default=10000.0)
    parser.add_argument('--output', help='Файл для сохранения результатов', default=None)
    
    args = parser.parse_args()
    
    if args.end is None:
        args.end = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Запуск бэктеста для {args.symbol}")
    logger.info(f"Период: {args.start} - {args.end}")
    logger.info(f"Таймфрейм: {args.timeframe}")
    logger.info(f"Баланс: ${args.balance:,.2f}")
    
    # Загрузка исторических данных
    from data_handler import DataHandler
    handler = DataHandler()
    
    # Здесь нужно реализовать загрузку исторических данных
    # Временная заглушка
    logger.warning("Загрузка исторических данных не реализована")
    
    # Создаем тестовые данные
    dates = pd.date_range(start=args.start, end=args.end, freq='1h')
    df = pd.DataFrame({
        'open': np.random.normal(50000, 1000, len(dates)),
        'high': np.random.normal(50200, 1000, len(dates)),
        'low': np.random.normal(49800, 1000, len(dates)),
        'close': np.random.normal(50000, 1000, len(dates)),
        'volume': np.random.normal(100, 20, len(dates))
    }, index=dates)
    
    # Запускаем бэктест
    runner = BacktestRunner(initial_balance=args.balance)
    result = runner.run(df, args.symbol, args.timeframe)
    
    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print("="*60)
    print(f"Пара: {result.symbol}")
    print(f"Период: {result.start_date.date()} - {result.end_date.date()}")
    print(f"Таймфрейм: {result.timeframe}")
    print()
    print(f"Начальный баланс: ${result.initial_balance:,.2f}")
    print(f"Конечный баланс:  ${result.final_balance:,.2f}")
    print(f"Общая доходность: {result.total_return_percent:+.2f}%")
    print()
    print(f"Сделок: {result.total_trades}")
    print(f"Прибыльных: {result.winning_trades}")
    print(f"Убыточных: {result.losing_trades}")
    print(f"Win Rate: {result.win_rate*100:.1f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Макс. просадка: {result.max_drawdown*100:.1f}%")
    print(f"Коэф. Шарпа: {result.sharpe_ratio:.2f}")
    print("="*60)
    
    # Сохраняем результаты
    if args.output:
        save_backtest_result(result, args.output)
    
    return result

if __name__ == "__main__":
    # Запуск из командной строки
    asyncio.run(run_backtest_cli())