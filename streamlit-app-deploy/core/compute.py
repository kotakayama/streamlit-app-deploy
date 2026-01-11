from typing import Optional, Tuple
import pandas as pd
from core.normalize import match_item


def extract_metric(
    df: pd.DataFrame,
    metric_key: str,
    year: str
) -> Tuple[Optional[float], Optional[str]]:
    """Return (value, source_detail)"""
    patterns = None
    try:
        from core.normalize import ALIASES
        patterns = ALIASES[metric_key]
    except Exception:
        patterns = []
    hits = df[df["Item"].apply(lambda x: match_item(x, patterns))]
    if hits.empty:
        return None, None
    row_idx = hits.index[0]
    label = df.loc[row_idx, "Item"]
    val = df.loc[row_idx, year]
    if pd.isna(val):
        return None, None
    source_detail = f"{label} @ {year}"
    return float(val), source_detail


def safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def calc_valuation(
    market_cap: float,
    cash: float,
    debt: float,
    revenue: float,
    ebitda: Optional[float],
    net_income: Optional[float],
    equity: Optional[float],
    fcf: Optional[float],
) -> dict:
    net_debt = debt - cash
    ev = market_cap + net_debt
    out = {
        "Market Cap": market_cap,
        "Cash": cash,
        "Debt": debt,
        "Net Debt": net_debt,
        "EV": ev,
        "EV/Sales": (ev / revenue) if revenue else None,
        "EV/EBITDA": (ev / ebitda) if (ebitda not in [None, 0]) else None,
        "PER": (market_cap / net_income) if (net_income not in [None, 0]) else None,
        "PBR": (market_cap / equity) if (equity not in [None, 0]) else None,
        "FCF Yield": (fcf / market_cap) if (fcf not in [None, 0]) else None,
    }
    return out


# --- 追加: EV / マルチプル計算ユーティリティ ---
def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def compute_metrics(
    market_cap: float | None,
    cash: float | None,
    debt_short: float | None,
    debt_long: float | None,
    lease_liabilities: float | None,
    revenue: float | None,
    ebitda: float | None,
    net_income: float | None,
):
    debt_total = (debt_short or 0) + (debt_long or 0)
    net_debt = debt_total - (cash or 0)

    # v1: EV = Market Cap + Net Debt + Lease(任意)
    # まずは Lease を含める/含めないをUIで切替可能にする
    return {
        "market_cap": market_cap,
        "debt_total": debt_total,
        "cash": cash,
        "net_debt": net_debt,
        "lease_liabilities": lease_liabilities,
        "revenue": revenue,
        "ebitda": ebitda,
        "net_income": net_income,
    }


def compute_valuation_table(company: str, base: dict, include_lease: bool):
    market_cap = base["market_cap"]
    net_debt = base["net_debt"]
    lease = base.get("lease_liabilities") or 0
    ev = None if market_cap is None else market_cap + net_debt + (lease if include_lease else 0)

    revenue = base.get("revenue")
    ebitda = base.get("ebitda")
    net_income = base.get("net_income")

    rows = [
        ("Company", company),
        ("Market Cap", market_cap),
        ("Net Debt", net_debt),
        ("Lease (included)" if include_lease else "Lease (excluded)", lease),
        ("EV", ev),
        ("EV/Sales", safe_div(ev, revenue)),
        ("EV/EBITDA", safe_div(ev, ebitda)),
        ("PER", safe_div(market_cap, net_income)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])