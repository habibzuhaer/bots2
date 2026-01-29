# diag/diag_range_math.py

from context.range_model import PriceContext

def run():
    r = PriceContext(expected_low=42000, expected_high=44000)

    prices = [
        41000,
        42000,
        43000,
        44000,
        45000
    ]

    print("=== RANGE MODEL DIAGNOSTIC ===")

    for p in prices:
        print(
            f"price={p} | "
            f"in_range={r.contains(p)} | "
            f"distance={r.distance_to_range(p)}"
        )


if __name__ == "__main__":
    run()
