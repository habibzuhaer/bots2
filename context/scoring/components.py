# context/scoring/components.py

def range_position_score(price: float, low: float, high: float) -> float:
    """
    Чем ближе к центру диапазона — тем выше score.
    1.0 в центре, 0.0 за пределами диапазона.
    """
    if low >= high:
        return 0.0

    if price < low or price > high:
        return 0.0

    mid = (low + high) / 2
    half = (high - low) / 2

    return 1.0 - abs(price - mid) / half


def range_width_score(low: float, high: float) -> float:
    """
    Узкий диапазон = выше значимость контекста.
    Не нормализуем под рынок — только относительная мера.
    """
    width = high - low
    if width <= 0:
        return 0.0

    # инверсия: чем шире диапазон, тем ниже score
    return 1.0 / (1.0 + width)


def expiration_proximity_score(days_to_exp: int) -> float:
    """
    Чем ближе экспирация — тем выше влияние CME.
    """
    if days_to_exp < 0:
        return 0.0

    if days_to_exp == 0:
        return 1.0

    return 1.0 / (1.0 + days_to_exp)
