# backtest/engine.py

from typing import Iterable

class BacktestEngine:
    def __init__(self, executor):
        """
        executor: функция исполнения сделки
        (одна и та же для всех сценариев)
        """
        self.executor = executor
        self.trades = []

    def run(self, bars: Iterable, decision_fn):
        """
        bars: реальные свечи из БД
        decision_fn: функция принятия решения
        """
        for bar in bars:
            decision = decision_fn(bar)
            if decision is not None:
                trade = self.executor(bar, decision)
                if trade:
                    self.trades.append(trade)

        return self.trades
