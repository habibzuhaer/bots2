# technical/acdf/adapter.py

from technical.acdf.calculator import calculate_acdf
from technical.acdf.model import ACDFLevels


def build_acdf_levels(market_state: dict) -> ACDFLevels:
    """
    market_state — абстрактное состояние рынка.
    CME, scoring, decision сюда НЕ ПЕРЕДАЮТСЯ.
    """

    required = {"impulse_high", "impulse_low"}
    if not required.issubset(market_state):
        raise ValueError("Incomplete market_state for A–C–D–F")

    return calculate_acdf(
        impulse_high=market_state["impulse_high"],
        impulse_low=market_state["impulse_low"],
    )
