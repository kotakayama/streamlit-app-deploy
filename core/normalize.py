import re
from typing import List
import pandas as pd

# 科目名の揺れ（v1はルールベース。v2でLLMマッピングに置換）
ALIASES = {
    # PL
    "revenue": [r"売上", r"revenue", r"net sales", r"sales"],
    "ebitda": [r"ebitda"],
    "operating_income": [r"営業利益", r"operating income", r"ebit\b"],
    "net_income": [r"当期純利益", r"net income", r"profit attributable", r"純利益"],
    # BS
    "cash": [r"現金", r"cash and cash equivalents", r"cash equivalents"],
    "debt": [r"有利子負債", r"interest[- ]?bearing", r"debt"],
    "equity": [r"純資産", r"equity", r"net assets"],
    "shares": [r"発行済株式", r"shares outstanding", r"issued shares"],
    # CF
    "cfo": [r"営業cf", r"営業キャッシュフロー", r"cash flow from operating", r"\bcfo\b"],
    "cfi": [r"投資cf", r"投資キャッシュフロー", r"cash flow from investing", r"\bcfi\b"],
    "fcf": [r"フリーキャッシュフロー", r"\bfcf\b", r"free cash flow"],
}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def match_item(label: str, patterns: List[str]) -> bool:
    t = normalize_text(label)
    return any(re.search(p, t) for p in patterns)


def read_sheet_table(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """Read a sheet and normalize columns: expect first column to be Item and FYXXXX columns."""
    df = pd.read_excel(xls, sheet_name=sheet)
    if df.empty:
        raise ValueError(f"{sheet} is empty.")
    df = df.rename(columns={df.columns[0]: "Item"})
    year_cols = [c for c in df.columns if re.match(r"^FY\d{4}", str(c))]
    if not year_cols:
        raise ValueError(f"{sheet}: No FY columns found (e.g., FY2023).")
    df = df[["Item"] + year_cols].copy()
    return df


# --- Optional LLM-assisted normalization (useful for absorbing naming variance) ---
import os
import json
import logging
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

STANDARD_SCHEMA_HINT = """
{
  "revenue": number|null,
  "operating_income": number|null,
  "ebitda": number|null,
  "net_income": number|null,
  "cash": number|null,
  "debt_short": number|null,
  "debt_long": number|null,
  "lease_liabilities": number|null,
  "total_liabilities": number|null,
  "total_equity": number|null,
  "shares_total": number|null,
  "notes": string|null
}
"""


def _rule_based_normalize(raw: dict) -> dict:
    """Fallback deterministic normalizer that flattens typical `pl`/`bs`/`shares` structures into the standard schema."""
    keys = [
        "revenue",
        "operating_income",
        "ebitda",
        "net_income",
        "cash",
        "debt_short",
        "debt_long",
        "lease_liabilities",
        "total_liabilities",
        "total_equity",
        "shares_total",
        "notes",
    ]
    out = {k: None for k in keys}

    # direct numeric keys
    for k in keys:
        if k in raw and isinstance(raw[k], (int, float)):
            out[k] = raw[k]

    # pl/bs nested
    pl = raw.get("pl") or {}
    bs = raw.get("bs") or {}
    shares = raw.get("shares") or {}

    if isinstance(pl, dict):
        out["revenue"] = out.get("revenue") or pl.get("revenue")
        out["operating_income"] = out.get("operating_income") or pl.get("operating_income")
        out["ebitda"] = out.get("ebitda") or pl.get("ebitda")
        out["net_income"] = out.get("net_income") or pl.get("net_income")
    if isinstance(bs, dict):
        out["cash"] = out.get("cash") or bs.get("cash_and_deposits") or bs.get("cash")
        out["debt_long"] = out.get("debt_long") or bs.get("long_term_debt")
        out["debt_short"] = out.get("debt_short") or bs.get("short_term_debt")
        out["lease_liabilities"] = out.get("lease_liabilities") or bs.get("lease_liabilities")
        out["total_liabilities"] = out.get("total_liabilities") or bs.get("total_liabilities")
        out["total_equity"] = out.get("total_equity") or bs.get("total_equity") or bs.get("net_assets")
    if isinstance(shares, dict):
        out["shares_total"] = out.get("shares_total") or shares.get("shares_total")

    return out


def normalize_with_llm(raw: dict, model: str = "gpt-4.1-mini") -> dict:
    """Normalize extracted numeric fields into the STANDARD_SCHEMA, optionally using an LLM to absorb naming variance.

    Behavior:
      - If `OPENAI_API_KEY` is not set or OpenAI client not available, a deterministic rule-based fallback is used.
      - If the LLM call fails for any reason, fall back to the rule-based normalizer and include an explanatory note in the returned dict under `notes`.

    Input `raw` should be a minimal numeric dictionary (e.g. flattened or containing `pl`/`bs`/`shares` dicts).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        logging.info("OpenAI API key not set or OpenAI client not available; using rule-based normalization.")
        out = _rule_based_normalize(raw)
        out["notes"] = out.get("notes") or "rule_based_fallback"
        return out

    client = OpenAI(api_key=api_key)

    prompt = f"""
Convert the following extracted Japanese financial data into the standard schema below. Return ONLY valid JSON that matches the schema. If a value is missing, set it to null. Use JPY amounts as numbers.

Schema:
{STANDARD_SCHEMA_HINT}

Input JSON:
{json.dumps(raw, ensure_ascii=False)}
"""

    try:
        resp = client.responses.create(
            model=model,
            instructions="Return ONLY valid JSON. No markdown. No extra keys.",
            input=prompt,
        )
        # `resp.output_text` is a convenience on some SDK versions
        txt = getattr(resp, "output_text", None) or json.dumps(resp.output, ensure_ascii=False)
        parsed = json.loads(txt)
        # Ensure missing keys are present and set to None
        for k in ["revenue","operating_income","ebitda","net_income","cash","debt_short","debt_long","lease_liabilities","total_liabilities","total_equity","shares_total","notes"]:
            if k not in parsed:
                parsed[k] = None
        return parsed
    except Exception as e:
        logging.exception("LLM normalization failed; falling back to rule-based normalizer")
        out = _rule_based_normalize(raw)
        out["notes"] = (out.get("notes") or "llm_failure_fallback") + f" (err: {e})"
        return out