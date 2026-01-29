# diag/diag_cme_context.py

from engine.context_builder import build_context
from engine.decision import can_trade

def run():
    tests = [
        {
            "name": "Цена внутри диапазона",
            "price": 43000,
            "range": {"low": 42000, "high": 44000},
            "expect": True
        },
        {
            "name": "Цена за пределами диапазона (в допуске)",
            "price": 44100,
            "range": {"low": 42000, "high": 44000},
            "expect": True
        },
        {
            "name": "Цена сильно вне диапазона",
            "price": 46000,
            "range": {"low": 42000, "high": 44000},
            "expect": False
        },
        {
            "name": "Цена ниже диапазона",
            "price": 40000,
            "range": {"low": 42000, "high": 44000},
            "expect": False
        }
    ]

    print("=== CME CONTEXT DIAGNOSTIC ===")

    for t in tests:
        ctx = build_context(t["price"], t["range"])
        result = can_trade(t["price"], ctx)

        status = "OK" if result == t["expect"] else "FAIL"

        print(
            f"{status:4} | {t['name']:<35} | "
            f"bias={ctx['bias']:<8} | can_trade={result}"
        )


if __name__ == "__main__":
    run()
