# backtest/walkforward.py

from backtest.runner import run_backtest


def split_windows(bars, window_size, step):
    i = 0
    while i + window_size <= len(bars):
        yield bars[i : i + window_size]
        i += step


def run_walk_forward(
    *,
    bars,
    window_size,
    step,
    executor,
    decision_no_cme,
    decision_with_cme,
):
    results = []

    for idx, window in enumerate(
        split_windows(bars, window_size, step)
    ):
        metrics = run_backtest(
            bars=window,
            executor=executor,
            decision_no_cme=decision_no_cme,
            decision_with_cme=decision_with_cme,
        )

        results.append(
            {
                "window": idx,
                "no_cme": metrics["no_cme"],
                "with_cme": metrics["with_cme"],
            }
        )

    return results
