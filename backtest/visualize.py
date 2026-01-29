# backtest/visualize.py

import matplotlib.pyplot as plt


def build_equity_curve(trades):
    equity = []
    cur = 0.0
    for t in trades:
        cur += t["pnl"]
        equity.append(cur)
    return equity


def plot_equity(trades_no_cme, trades_with_cme):
    eq_plain = build_equity_curve(trades_no_cme)
    eq_cme = build_equity_curve(trades_with_cme)

    plt.figure(figsize=(12, 6))
    plt.plot(eq_plain, label="NO CME", alpha=0.8)
    plt.plot(eq_cme, label="WITH CME", alpha=0.8)

    plt.title("Equity Curve Comparison")
    plt.xlabel("Trades")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
