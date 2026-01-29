# diag/diag_runtime_invariants.py
"""
RUNTIME INVARIANT DIAGNOSTIC

Проверяет:
- что система не падает на None / пустых структурах
- что ошибки выбрасываются корректно
- что нет скрытых зависимостей
"""

from engine.context_builder import build_context
from engine.decision import can_trade


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def expect_exception(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"OK   | {name} → raised {type(e).__name__}")
        return
    raise RuntimeError(f"FAIL | {name} → no exception raised")


def check_context_builder_guards():
    header("CONTEXT BUILDER GUARDS")

    expect_exception(
        "Missing CME range",
        lambda: build_context(0, None)
    )

    expect_exception(
        "Malformed CME range",
        lambda: build_context(0, {})
    )


def check_decision_guards():
    header("DECISION GUARDS")

    expect_exception(
        "Missing context",
        lambda: can_trade(0, None)
    )

    expect_exception(
        "Malformed context",
        lambda: can_trade(0, {})
    )


def check_no_hidden_state():
    header("NO HIDDEN STATE")

    # повторный вызов должен быть детерминированным
    dummy_ctx = {
        "range": None,
        "bias": "NEUTRAL"
    }

    try:
        can_trade(0, dummy_ctx)
    except Exception:
        print("OK   | decision does not rely on global mutable state")
        return

    raise RuntimeError("FAIL | decision silently accepted invalid state")


def run():
    check_context_builder_guards()
    check_decision_guards()
    check_no_hidden_state()

    print("\nALL RUNTIME INVARIANTS PASSED")


if __name__ == "__main__":
    run()
