# diag/diag_core_integrity.py

from context.range_model import PriceContext
from context.bias import detect_bias
from filters.cme_filter import cme_allows_trade
from cme.expirations import days_to_expiration, is_expiration_near
from config.settings import CME_RANGE_TOLERANCE_PCT

def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_range_math():
    header("RANGE MODEL — MATHEMATICAL CHECK")

    r = PriceContext(expected_low=42000, expected_high=44000)

    cases = [
        (41000, False, 1000),
        (42000, True, 0),
        (43000, True, 0),
        (44000, True, 0),
        (45000, False, 1000),
    ]

    for price, exp_in, exp_dist in cases:
        in_range = r.contains(price)
        dist = r.distance_to_range(price)

        ok = (in_range == exp_in) and (dist == exp_dist)
        status = "OK" if ok else "FAIL"

        print(
            f"{status:4} | price={price} | "
            f"in_range={in_range} | distance={dist}"
        )


def test_bias_detection():
    header("BIAS DETECTION")

    low, high = 42000, 44000
    mid = (low + high) / 2

    cases = [
        (mid + 100, "BULLISH"),
        (mid - 100, "BEARISH"),
        (mid, "NEUTRAL"),
    ]

    for price, expected in cases:
        bias = detect_bias(price, low, high)
        status = "OK" if bias == expected else "FAIL"

        print(
            f"{status:4} | price={price} | bias={bias}"
        )


def test_cme_tolerance():
    header("CME RANGE TOLERANCE")

    low, high = 42000, 44000
    range_size = high - low
    tol = range_size * CME_RANGE_TOLERANCE_PCT

    cases = [
        (high + tol * 0.9, True),
        (high + tol * 1.1, False),
        (low - tol * 0.9, True),
        (low - tol * 1.1, False),
    ]

    for price, expected in cases:
        allowed = cme_allows_trade(price, low, high)
        status = "OK" if allowed == expected else "FAIL"

        print(
            f"{status:4} | price={price:.2f} | can_trade={allowed}"
        )


def test_expiration_logic():
    header("EXPIRATION LOGIC")

    now = "2026-01-10"
    window = 3

    cases = [
        ("2026-01-11", True),
        ("2026-01-13", True),
        ("2026-01-14", False),
        ("2026-01-20", False),
    ]

    for exp_date, expected in cases:
        near = is_expiration_near(exp_date, now, window)
        days = days_to_expiration(exp_date, now)

        status = "OK" if near == expected else "FAIL"

        print(
            f"{status:4} | exp={exp_date} | days_left={days} | near={near}"
        )


def run():
    test_range_math()
    test_bias_detection()
    test_cme_tolerance()
    test_expiration_logic()


if __name__ == "__main__":
    run()
