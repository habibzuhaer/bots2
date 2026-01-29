# diag/diag_system_pipeline.py

from engine.context_builder import build_context
from engine.decision import can_trade
from cme.expirations import is_expiration_near

def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_pipeline_case(name, price, cme_range, exp_date=None, now=None):
    ctx = build_context(price, cme_range)
    allowed = can_trade(price, ctx)

    exp_flag = None
    if exp_date and now:
        exp_flag = is_expiration_near(exp_date, now, window=3)

    print(
        f"{name:<30} | "
        f"price={price:<7} | "
        f"range=({cme_range['low']},{cme_range['high']}) | "
        f"bias={ctx['bias']:<8} | "
        f"can_trade={allowed} | "
        f"exp_near={exp_flag}"
    )


def run():
    header("FULL SYSTEM PIPELINE DIAGNOSTIC")

    scenarios = [
        {
            "name": "Нормальный рынок",
            "price": 43200,
            "range": {"low": 42000, "high": 44500},
        },
        {
            "name": "Верх диапазона",
            "price": 44490,
            "range": {"low": 42000, "high": 44500},
        },
        {
            "name": "Ниже диапазона (допуск)",
            "price": 41850,
            "range": {"low": 42000, "high": 44500},
        },
        {
            "name": "Полностью вне контекста",
            "price": 47000,
            "range": {"low": 42000, "high": 44500},
        },
        {
            "name": "Рынок + экспирация рядом",
            "price": 43100,
            "range": {"low": 42000, "high": 44500},
            "exp_date": "2026-01-12",
            "now": "2026-01-10"
        },
    ]

    for s in scenarios:
        run_pipeline_case(
            name=s["name"],
            price=s["price"],
            cme_range=s["range"],
            exp_date=s.get("exp_date"),
            now=s.get("now")
        )


if __name__ == "__main__":
    run()
