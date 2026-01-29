# backtest/phase_runner.py

from backtest.runner import run_backtest
from backtest.market_phase import split_by_phase


def run_phase_backtests(
    *,
    bars,
    executor,
    decision_no_cme,
    decision_with_cme,
):
    results = {}

    by_phase = split_by_phase(bars)

    for phase, phase_bars in by_phase.items():
        metrics = run_backtest(
            bars=phase_bars,
            executor=executor,
            decision_no_cme=decision_no_cme,
            decision_with_cme=decision_with_cme,
        )

        results[phase] = metrics

    return results
