# diag/diag_full_pipeline.py

from engine.context_builder import build_context
from engine.decision import can_trade

def run():
    scenarios = [
        {
            "name": "Нормальный рынок",
            "price": 43200,
            "range": {"low": 42000, "high": 44500}
        },
        {
            "name": "Цена у верхней границы",
            "price": 44480,
            "range": {"low": 42000, "high": 44500}
        },
        {
            "name": "Вне контекста CME",
            "price": 47000,
            "range": {"low": 42000, "high": 44500}
        }
    ]

    print("=== FULL PIPELINE DIAGNOSTIC ===")

    for s in scenarios:
        ctx = build_context(s["price"], s["range"])
        allowed = can_trade(s["price"], ctx)

        print(
            f"{s['name']:<25} | "
            f"price={s['price']} | "
            f"bias={ctx['bias']:<8} | "
            f"can_trade={allowed}"
        )


if __name__ == "__main__":
    run()
