# context/scoring/scorer.py

from context.scoring.model import ScoreResult
from context.scoring.components import (
    range_position_score,
    range_width_score,
    expiration_proximity_score,
)

# Веса — часть модели, не стратегии
WEIGHTS = {
    "range_position": 0.5,
    "range_width": 0.3,
    "expiration": 0.2,
}


def compute_cme_score(
    *,
    price: float,
    low: float,
    high: float,
    days_to_expiration: int | None = None,
) -> ScoreResult:
    components = {}

    components["range_position"] = range_position_score(
        price, low, high
    )

    components["range_width"] = range_width_score(
        low, high
    )

    if days_to_expiration is not None:
        components["expiration"] = expiration_proximity_score(
            days_to_expiration
        )
    else:
        components["expiration"] = 0.0

    # взвешенная сумма
    score = 0.0
    for k, v in components.items():
        score += WEIGHTS.get(k, 0.0) * v

    # жёсткое ограничение
    score = max(0.0, min(1.0, score))

    return ScoreResult(
        score=score,
        components=components,
    )
