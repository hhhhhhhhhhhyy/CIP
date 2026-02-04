from __future__ import annotations

from typing import Any, Literal


FinalResult = Literal["Correct", "Incorrect", "Spam", "NeedDiscussion"]


def normalize_final_result(value: Any) -> FinalResult | None:
    """Normalize various judgement values to the canonical set.

    Canonical values:
      - Correct
      - Incorrect
      - Spam
      - NeedDiscussion

    Returns None when the value cannot be confidently mapped.
    """

    if value is None:
        return None

    # Handle numeric-like values
    if isinstance(value, (int, float)):
        if value == 1:
            return "Correct"
        if value == 0:
            return "Incorrect"
        return None

    s = str(value).strip()
    if not s:
        return None

    low = s.lower().strip()
    low = " ".join(low.split())

    # Direct canonical matches
    canon_map = {
        "correct": "Correct",
        "incorrect": "Incorrect",
        "spam": "Spam",
        "needdiscussion": "NeedDiscussion",
        "need discussion": "NeedDiscussion",
        "need_discussion": "NeedDiscussion",
        "need review": "NeedDiscussion",
        "manual review": "NeedDiscussion",
    }
    if low in canon_map:
        return canon_map[low]

    # Common yes/no variants
    if low in {"y", "yes", "true", "t", "pass", "passed", "1"}:
        return "Correct"
    if low in {"n", "no", "false", "f", "fail", "failed", "0"}:
        return "Incorrect"

    # Chinese common variants (best-effort)
    if low in {"正确", "对", "是", "通过"}:
        return "Correct"
    if low in {"错误", "不对", "否", "不通过"}:
        return "Incorrect"
    if "spam" in low or "垃圾" in low or "灌水" in low:
        return "Spam"
    if "need" in low and "discussion" in low:
        return "NeedDiscussion"
    if "复核" in low or "人工" in low or "讨论" in low:
        return "NeedDiscussion"

    return None


def judge_ai_match(
    ai_final_result: Any,
    vendor_judgement: Any,
    *,
    ignore: set[FinalResult] | None = None,
) -> bool | None:


    if ignore is None:
        ignore = {"NeedDiscussion", "Spam"}

    ai = normalize_final_result(ai_final_result)
    vendor = normalize_final_result(vendor_judgement)

    if ai is None or vendor is None:
        return None

    if ai in ignore or vendor in ignore:
        return None

    return ai == vendor
