# technical/acdf/calculator.py

from technical.acdf.model import ACDFLevels


def calculate_acdf(
    *,
    impulse_high: float,
    impulse_low: float,
) -> ACDFLevels:
    """
    ЧИСТАЯ техника.

    impulse_high / impulse_low
    получаются из твоей логики импульса.
    """

    A = impulse_high
    C = impulse_low

    # симметрия (пример, не стратегия)
    D = A + (A - C)
    F = C - (A - C)

    return ACDFLevels(A=A, C=C, D=D, F=F)
