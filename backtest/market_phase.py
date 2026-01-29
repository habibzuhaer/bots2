# backtest/market_phase.py

def classify_phase(bar):
    """
    Минимальный контракт.
    Реальную логику фазы (trend / range / volatile)
    ты подставляешь сам.
    """
    return bar["phase"]


def split_by_phase(bars):
    phases = {}

    for bar in bars:
        phase = classify_phase(bar)
        phases.setdefault(phase, []).append(bar)

    return phases
