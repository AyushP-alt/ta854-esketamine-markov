"""Presentation-layer economics helpers for the Streamlit dashboard.

These functions interpret outputs from :mod:`model`; they do not reproduce the
Markov calculations.
"""

from __future__ import annotations

import math
from collections.abc import Callable


TOLERANCE = 1e-12


def format_icer(inc_cost: float, inc_qaly: float, icer: float | None = None) -> str:
    """Format an ICER without obscuring dominance or near-zero effects."""
    if abs(inc_qaly) <= TOLERANCE:
        return "Not meaningful (incremental QALYs are approximately zero)"
    if inc_qaly > 0 and inc_cost < 0:
        return "Dominant (lower cost, more QALYs)"
    if inc_qaly < 0 and inc_cost > 0:
        return "Dominated (higher cost, fewer QALYs)"
    if inc_qaly < 0 and inc_cost <= 0:
        return "Less effective at lower or equal cost; use NMB, not the ICER ratio"
    if icer is None:
        return "ICER not supplied by the model"
    if not math.isfinite(icer):
        return "Not meaningful for this result"
    return f"£{icer:,.0f} per QALY"


def decision_interpretation(inc_cost: float, inc_qaly: float, nmb: float) -> str:
    """Return a plain-language interpretation for the selected threshold."""
    if abs(inc_qaly) <= TOLERANCE:
        if inc_cost < -TOLERANCE:
            effect_text = "Costs are lower with approximately no incremental QALY difference, so an ICER is not meaningful."
        elif inc_cost > TOLERANCE:
            effect_text = "Costs are higher with approximately no incremental QALY difference, so an ICER is not meaningful."
        else:
            effect_text = "Incremental costs and QALYs are approximately zero, so an ICER is not meaningful."
    elif inc_qaly > 0 and inc_cost < 0:
        effect_text = "Esketamine is dominant in this model run: lower cost and more QALYs."
    elif inc_qaly < 0 and inc_cost > 0:
        effect_text = "Esketamine is dominated in this model run: higher cost and fewer QALYs."
    elif inc_qaly < 0 and inc_cost < 0:
        effect_text = "Esketamine costs less but also produces fewer QALYs; the ICER alone is not a useful threshold summary."
    elif inc_qaly > 0 and inc_cost > 0:
        effect_text = "Esketamine costs more and produces more QALYs; compare the ICER and net benefit at the selected threshold."
    elif inc_qaly < 0:
        effect_text = "Esketamine produces fewer QALYs at the same cost; the ICER is not a useful threshold summary."
    elif inc_qaly > 0:
        effect_text = "Esketamine produces more QALYs at the same cost."
    elif inc_cost < 0:
        effect_text = "Costs are lower with approximately no incremental QALY difference."
    elif inc_cost > 0:
        effect_text = "Costs are higher with approximately no incremental QALY difference."
    else:
        effect_text = "The strategies have no material incremental cost or QALY difference."

    if nmb > TOLERANCE:
        threshold_text = "Incremental NMB is positive at the selected threshold, favouring esketamine on net benefit under these assumptions."
    elif nmb < -TOLERANCE:
        threshold_text = "Incremental NMB is negative at the selected threshold, favouring the comparator on net benefit under these assumptions."
    else:
        threshold_text = "Incremental NMB is approximately zero at the selected threshold."
    return f"{effect_text} {threshold_text} This is a model-based value assessment, not a coverage or reimbursement decision."


def economic_summary(result: dict, wtp_threshold: float) -> dict:
    """Summarise an existing model result and calculate threshold NMB."""
    inc_cost = float(result["inc_cost"])
    inc_qaly = float(result["inc_qaly"])
    icer = float(result["icer"])
    nmb = float(wtp_threshold) * inc_qaly - inc_cost
    return {
        "inc_cost": inc_cost,
        "inc_qaly": inc_qaly,
        "icer": icer,
        "icer_display": format_icer(inc_cost, inc_qaly, icer),
        "nmb": nmb,
        "interpretation": decision_interpretation(inc_cost, inc_qaly, nmb),
    }


def break_even_price(
    result_at_zero_price: dict,
    result_at_one_pound: dict,
    wtp_threshold: float,
    search_min: float = 0.0,
    search_max: float = 489.0,
) -> dict:
    """Solve for the per-session acquisition price using engine-run outputs.

    The model is run externally at £0 and £1/session. This helper interpolates
    between those outputs; it does not calculate treatment costs.
    """
    cost_zero = float(result_at_zero_price["inc_cost"])
    cost_one = float(result_at_one_pound["inc_cost"])
    qaly_zero = float(result_at_zero_price["inc_qaly"])
    qaly_one = float(result_at_one_pound["inc_qaly"])
    slope = cost_one - cost_zero

    if abs(qaly_one - qaly_zero) > 1e-10 or slope <= TOLERANCE:
        return {"price": None, "in_range": False, "reason": "No supported linear price solution."}

    price = (float(wtp_threshold) * qaly_zero - cost_zero) / slope
    return {
        "price": price,
        "in_range": search_min - 1e-9 <= price <= search_max + 1e-9,
        "reason": None,
    }


def find_nmb_zero_by_search(
    evaluate_nmb: Callable[[float], float],
    search_min: float,
    search_max: float,
    *,
    scan_intervals: int = 40,
    price_tolerance: float = 0.01,
) -> dict:
    """Find the first NMB zero using evaluated points and bracketed bisection.

    The evaluator should run the actual model at the supplied price. Scanning
    first avoids assuming a linear price-to-NMB relationship and identifies a
    sign-changing interval before bisection refines the threshold.
    """
    if not math.isfinite(search_min) or not math.isfinite(search_max) or search_max <= search_min:
        raise ValueError("search_max must be greater than search_min")
    if scan_intervals < 1:
        raise ValueError("scan_intervals must be at least 1")
    if not math.isfinite(price_tolerance) or price_tolerance <= 0:
        raise ValueError("price_tolerance must be positive")

    prices = [search_min + (search_max - search_min) * i / scan_intervals for i in range(scan_intervals + 1)]
    values = [float(evaluate_nmb(price)) for price in prices]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("evaluate_nmb must return finite values over the search range")
    tolerance = 1e-9
    for price, value in zip(prices, values):
        if abs(value) <= tolerance:
            return {"price": price, "in_range": True, "nmb": value, "reason": None}

    bracket = next(
        (
            (prices[i], prices[i + 1], values[i], values[i + 1])
            for i in range(len(prices) - 1)
            if (values[i] < 0 < values[i + 1]) or (values[i] > 0 > values[i + 1])
        ),
        None,
    )
    if bracket is None:
        return {
            "price": None,
            "in_range": False,
            "nmb": None,
            "reason": "No NMB zero crossing was found within the searched price range.",
        }

    low, high, nmb_low, _ = bracket
    while high - low > price_tolerance:
        mid = (low + high) / 2.0
        nmb_mid = float(evaluate_nmb(mid))
        if not math.isfinite(nmb_mid):
            raise ValueError("evaluate_nmb must return finite values over the search range")
        if abs(nmb_mid) <= tolerance:
            return {"price": mid, "in_range": True, "nmb": nmb_mid, "reason": None}
        if (nmb_low < 0 < nmb_mid) or (nmb_low > 0 > nmb_mid):
            high = mid
        else:
            low, nmb_low = mid, nmb_mid

    price = (low + high) / 2.0
    return {"price": price, "in_range": True, "nmb": float(evaluate_nmb(price)), "reason": None}


def validate_probability(value: float, name: str) -> float:
    """Check probability controls at the UI/helper boundary."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def split_diagnostics(messages: list[str]) -> tuple[list[str], list[str]]:
    """Separate failed checks from informational model diagnostics."""
    failures = [message for message in messages if message.startswith("FAIL:")]
    informational = [message for message in messages if not message.startswith("FAIL:")]
    return failures, informational
