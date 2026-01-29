# backtest/runner.py

from backtest.guard import ensure_contract_passed
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics


def run_backtest(
    *,
    bars,
    executor,
    decision_no_cme,
    decision_with_cme,
):
    ensure_contract_passed()

    engine = BacktestEngine(executor)

    trades_plain = engine.run(bars, decision_no_cme)
    metrics_plain = compute_metrics(trades_plain)

    engine = BacktestEngine(executor)
    trades_cme = engine.run(bars, decision_with_cme)
    metrics_cme = compute_metrics(trades_cme)

    return {
        "no_cme": metrics_plain,
        "with_cme": metrics_cme,
    }
