# backtest/metrics.py

def compute_metrics(trades):
    if not trades:
        return {
            "count": 0,
            "winrate": 0.0,
            "pnl": 0.0,
            "max_dd": 0.0,
        }

    pnl = [t["pnl"] for t in trades]
    equity = []
    cur = 0.0
    peak = 0.0
    max_dd = 0.0

    for p in pnl:
        cur += p
        peak = max(peak, cur)
        dd = peak - cur
        max_dd = max(max_dd, dd)
        equity.append(cur)

    wins = sum(1 for p in pnl if p > 0)

    return {
        "count": len(trades),
        "winrate": wins / len(trades),
        "pnl": sum(pnl),
        "max_dd": max_dd,
    }
