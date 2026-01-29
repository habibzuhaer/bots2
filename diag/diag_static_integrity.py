# diag/diag_static_integrity.py
"""
STATIC / STRUCTURAL DIAGNOSTIC

Цель:
- проверить, что все ключевые модули существуют
- проверить сигнатуры функций
- проверить логические инварианты
- проверить, что система не содержит скрытых сигналов
"""

import inspect
import sys

# --- core imports ---
from config import settings
from context.range_model import PriceContext
from context.bias import detect_bias
from filters.cme_filter import cme_allows_trade
from engine.context_builder import build_context
from engine.decision import can_trade
from cme.expirations import days_to_expiration, is_expiration_near


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_settings():
    header("CONFIG / SETTINGS CHECK")

    required = [
        "CME_RANGE_TOLERANCE_PCT",
        "EXPIRATION_WARNING_DAYS",
        "MIN_CONTEXT_SCORE",
    ]

    for name in required:
        exists = hasattr(settings, name)
        value = getattr(settings, name, None)

        ok = exists and isinstance(value, (int, float))
        status = "OK" if ok else "FAIL"

        print(f"{status:4} | {name} = {value}")

        if not ok:
            raise RuntimeError(f"Invalid or missing setting: {name}")


def check_price_context_contract():
    header("PriceContext CONTRACT")

    sig = inspect.signature(PriceContext)

    required_fields = {"expected_low", "expected_high"}
    actual_fields = set(sig.parameters.keys())

    ok = required_fields == actual_fields
    status = "OK" if ok else "FAIL"

    print(f"{status:4} | fields={actual_fields}")

    if not ok:
        raise RuntimeError("PriceContext fields mismatch")

    # methods existence
    for method in ["contains", "distance_to_range"]:
        exists = hasattr(PriceContext, method)
        status = "OK" if exists else "FAIL"
        print(f"{status:4} | method={method}")

        if not exists:
            raise RuntimeError(f"Missing method: {method}")


def check_bias_is_not_signal():
    header("BIAS FUNCTION SAFETY")

    sig = inspect.signature(detect_bias)
    params = list(sig.parameters.keys())

    ok = params == ["price", "low", "high"]
    status = "OK" if ok else "FAIL"

    print(f"{status:4} | detect_bias signature={params}")

    # sanity: bias must be str
    ret_ann = sig.return_annotation
    print(f"INFO | return_annotation={ret_ann}")


def check_cme_filter_purity():
    header("CME FILTER PURITY")

    src = inspect.getsource(cme_allows_trade)

    forbidden = ["buy", "sell", "long", "short", "entry", "signal"]

    violated = [w for w in forbidden if w in src.lower()]

    ok = not violated
    status = "OK" if ok else "FAIL"

    print(f"{status:4} | forbidden_words_found={violated}")

    if not ok:
        raise RuntimeError("CME filter contains signal logic")


def check_context_builder_contract():
    header("CONTEXT BUILDER CONTRACT")

    sig = inspect.signature(build_context)
    params = list(sig.parameters.keys())

    ok = params == ["price", "cme_range"]
    status = "OK" if ok else "FAIL"

    print(f"{status:4} | build_context(price, cme_range)")

    if not ok:
        raise RuntimeError("Invalid build_context signature")


def check_decision_is_gate_only():
    header("DECISION MODULE SAFETY")

    src = inspect.getsource(can_trade)

    forbidden = ["open", "close", "order", "position", "tp", "sl"]

    violated = [w for w in forbidden if w in src.lower()]

    ok = not violated
    status = "OK" if ok else "FAIL"

    print(f"{status:4} | forbidden_words_found={violated}")

    if not ok:
        raise RuntimeError("Decision module contains execution logic")


def check_expiration_logic_purity():
    header("EXPIRATION LOGIC")

    for fn in [days_to_expiration, is_expiration_near]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())

        ok = "price" not in params
        status = "OK" if ok else "FAIL"

        print(f"{status:4} | {fn.__name__} params={params}")

        if not ok:
            raise RuntimeError("Expiration logic depends on price")


def run():
    check_settings()
    check_price_context_contract()
    check_bias_is_not_signal()
    check_cme_filter_purity()
    check_context_builder_contract()
    check_decision_is_gate_only()
    check_expiration_logic_purity()

    print("\nALL STATIC CHECKS PASSED")


if __name__ == "__main__":
    run()
