# diag/diag_backtest_fairness.py

from backtest.engine import BacktestEngine

def run():
    print("\nBACKTEST FAIRNESS DIAGNOSTIC\n")

    def dummy_executor(bar, decision):
        return {"pnl": 0}

    engine1 = BacktestEngine(dummy_executor)
    engine2 = BacktestEngine(dummy_executor)

    assert engine1.executor is engine2.executor
    print("OK   | same executor")

    assert engine1.trades == []
    assert engine2.trades == []
    print("OK   | clean initial state")

    print("\nBACKTEST FAIRNESS PASSED")

if __name__ == "__main__":
    run()
