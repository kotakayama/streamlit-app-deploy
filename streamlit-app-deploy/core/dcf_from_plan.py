import pandas as pd
import numpy as np
import re
import os
import json
import difflib
from datetime import datetime
from pathlib import Path

OP_CF_LABELS = ["営業活動によるキャッシュフロー"]
INV_CF_LABELS = ["投資活動によるキャッシュフロー"]

# default mapping filename (in same folder)
FCF_MAPPINGS_FILENAME = os.path.join(os.path.dirname(__file__), "fcf_mappings.json")

def _load_mappings(path: str | None = None) -> dict:
    """Load synonyms mappings for metrics. If file doesn't exist, returns built-in defaults."""
    default = {
        "fcf": ["fcf", "free cash flow", "フリーキャッシュフロー", "フリーCF", "FCF"],
        "operating_cf": ["営業活動によるキャッシュフロー", "営業CF", "営業キャッシュフロー", "operating cash flow", "operating cf"],
        "investing_cf": ["投資活動によるキャッシュフロー", "投資CF", "投資キャッシュフロー", "investing cash flow", "investing cf"]
    }
    p = path or FCF_MAPPINGS_FILENAME
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            # merge with defaults (allow overriding or extending)
            for k, v in default.items():
                if k in data:
                    if isinstance(data[k], list):
                        data[k] = list(dict.fromkeys(data[k] + v))
                else:
                    data[k] = v
            return data
    except FileNotFoundError:
        return default
    except Exception:
        return default


def _pick_series(cf_long: pd.DataFrame, label_candidates: list[str]) -> pd.Series:
    """
    cf_long columns: sheet, metric, period, value, unit
    returns Series indexed by period
    """
    df = cf_long[cf_long["metric"].isin(label_candidates)].copy()
    if df.empty:
        return pd.Series(dtype=float)
    s = df.groupby("period")["value"].sum()
    return s.sort_index()


def build_fcf_from_cf(plan_tidy: pd.DataFrame, periods: list[str]) -> pd.DataFrame:
    """
    FCF = Operating CF + Investing CF
    returns DataFrame with columns: period, op_cf, inv_cf, fcf
    """
    cf = plan_tidy[plan_tidy["sheet"].eq("CF")].copy()
    if cf.empty:
        raise ValueError("CFシートが取り込まれていません（sheet='CF'が空です）")

    op = _pick_series(cf, OP_CF_LABELS)
    inv = _pick_series(cf, INV_CF_LABELS)

    # 期間を揃える（指定periodsだけ）
    idx = pd.Index(periods)
    op = op.reindex(idx)
    inv = inv.reindex(idx)

    out = pd.DataFrame({
        "period": idx,
        "op_cf": op.values,
        "inv_cf": inv.values,
    })
    # 欠損は0扱い（該当期に項目がない場合）
    out["op_cf"] = out["op_cf"].fillna(0.0)
    out["inv_cf"] = out["inv_cf"].fillna(0.0)
    out["fcf"] = out["op_cf"] + out["inv_cf"]
    return out


def run_dcf(fcf_df: pd.DataFrame, wacc: float, g: float, mid_year: bool = True) -> dict:
    """
    fcf_df: period, fcf
    wacc/g: decimals (e.g., 0.09, 0.015)
    """
    if wacc <= g:
        raise ValueError("WACC <= g です（永続成長率法が成立しません）")

    fcf = fcf_df["fcf"].astype(float).to_numpy()
    n = len(fcf)

    # t=1..n
    t = np.arange(1, n + 1, dtype=float)
    if mid_year:
        t = t - 0.5

    disc = 1 / np.power(1 + wacc, t)
    pv_fcf = fcf * disc
    pv_sum = float(np.nansum(pv_fcf))

    # Terminal Value (perpetuity growth) using last year FCF
    fcf_last = fcf[-1]
    tv = (fcf_last * (1 + g)) / (wacc - g)
    pv_tv = float(tv / np.power(1 + wacc, (n - 0.5) if mid_year else n))

    ev = pv_sum + pv_tv

    detail = fcf_df.copy()
    detail["t"] = (np.arange(1, n + 1) - (0.5 if mid_year else 0.0))
    detail["discount_factor"] = disc
    detail["pv_fcf"] = pv_fcf

    return {
        "detail": detail,
        "pv_fcf_sum": pv_sum,
        "tv": float(tv),
        "pv_tv": pv_tv,
        "enterprise_value": float(ev),
    }


def sensitivity_wacc_g(fcf_df: pd.DataFrame, wacc_grid: list[float], g_grid: list[float], mid_year: bool = True) -> pd.DataFrame:
    """
    returns DataFrame indexed by wacc, columns by g, values EV
    """
    res = {}
    for w in wacc_grid:
        row = {}
        for gg in g_grid:
            if w <= gg:
                row[gg] = np.nan
            else:
                row[gg] = run_dcf(fcf_df, w, gg, mid_year=mid_year)["enterprise_value"]
        res[w] = row
    return pd.DataFrame(res).T.sort_index()


# -------------------- Excel extraction helpers --------------------

def _is_date_like(val: object) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip()
    if not s:
        return False
    # common date-like patterns
    if re.search(r"\d{4}[-/年]", s):
        return True
    if re.match(r"FY\d{2,4}", s, flags=re.IGNORECASE):
        return True
    if re.match(r"\d{4}$", s):
        return True
    # month names or 'yyyy/mm/dd'
    try:
        datetime.fromisoformat(s)
        return True
    except Exception:
        pass
    return False


def _find_header_row(df: pd.DataFrame, max_rows: int = 20) -> int | None:
    """Find a header row index where many cells look like periods/dates or contain FY/yyyy."""
    nrows = min(max_rows, df.shape[0])
    for r in range(nrows):
        row = df.iloc[r].astype(str).fillna("")
        date_like_count = sum(_is_date_like(c) for c in row)
        contains_fy = any(str(c).strip().lower().startswith("fy") for c in row)
        if date_like_count >= 3 or contains_fy:
            return r
    return None


def _normalize_header_multirow(df: pd.DataFrame, header_row_idx: int) -> list[str]:
    """Flatten multi-row header by concatenating non-empty cells above header_row_idx if present."""
    # start with header row
    cols = df.columns.tolist()
    header = [str(x).strip() for x in df.iloc[header_row_idx].fillna("").astype(str).tolist()]
    # look up to 3 rows above
    for r in range(header_row_idx - 1, max(-1, header_row_idx - 4), -1):
        upper = df.iloc[r].fillna("").astype(str).tolist()
        # prepend non-empty pieces
        header = [("" if str(u).strip() == "" else str(u).strip() + " ") + h for u, h in zip(upper, header)]
    header = [re.sub(r"\s+", " ", h).strip() for h in header]
    return header


def _detect_unit_nearby(df: pd.DataFrame, header_row_idx: int) -> str | None:
    hay = "\n".join(df.iloc[max(0, header_row_idx - 3): header_row_idx + 3].astype(str).fillna("").apply(lambda r: ",".join(r), axis=1).tolist())
    for token in ["円", "千", "百万円", "¥", "USD", "$", "EUR", "JPY", "MUSD", "Millions"]:
        if token in hay:
            return token
    return None


def _tidy_table_from_sheet(path: str, sheet: str) -> pd.DataFrame:
    """Return tidy table rows with columns: sheet, metric, period, value, unit"""
    df = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    if df.empty:
        return pd.DataFrame(columns=["sheet", "metric", "period", "value", "unit"])
    header_idx = _find_header_row(df)
    if header_idx is None:
        return pd.DataFrame(columns=["sheet", "metric", "period", "value", "unit"])
    header = _normalize_header_multirow(df, header_idx)
    # determine period columns (indices)
    period_cols = [i for i, h in enumerate(header) if _is_date_like(h) or re.search(r"FY|yyyy|年", str(h), re.IGNORECASE)]
    if not period_cols:
        # try numeric-like year headers
        period_cols = [i for i, h in enumerate(header) if re.match(r"\d{4}$", str(h).strip())]
    if not period_cols:
        return pd.DataFrame(columns=["sheet", "metric", "period", "value", "unit"])
    first_period_col = period_cols[0]
    periods = [str(header[i]).strip() for i in period_cols]

    unit = _detect_unit_nearby(df, header_idx)

    # collect metric rows below header
    rows = []
    for r in range(header_idx + 1, df.shape[0]):
        row = df.iloc[r].astype(object).tolist()
        metric = None
        # metric likely in left area before first_period_col
        for c in range(0, first_period_col):
            if row[c] is not None and str(row[c]).strip() != "":
                metric = str(row[c]).strip()
                break
        if metric is None:
            # might be row of totals or stray, stop when many empty
            if all((str(x).strip() == "" or pd.isna(x)) for x in row):
                continue
            else:
                metric = ""
        for idx_col, period in zip(period_cols, periods):
            raw = row[idx_col] if idx_col < len(row) else None
            try:
                val = float(raw)
            except Exception:
                # try to strip commas, percent, etc
                s = str(raw).replace(',', '').replace('%', '').strip()
                try:
                    val = float(s)
                except Exception:
                    val = np.nan
            rows.append({"sheet": sheet, "metric": metric, "period": period, "value": val, "unit": unit})
    tidy = pd.DataFrame(rows)
    return tidy


def extract_cf_tidy_from_workbook(path: str) -> pd.DataFrame:
    """Scan workbook and build a tidy CF-like DataFrame of metrics x periods."""
    x = []
    wb = pd.ExcelFile(path)
    for s in wb.sheet_names:
        try:
            t = _tidy_table_from_sheet(path, s)
            if not t.empty:
                x.append(t)
        except Exception:
            continue
    if not x:
        return pd.DataFrame(columns=["sheet", "metric", "period", "value", "unit"])
    return pd.concat(x, ignore_index=True)


def _find_series_metric(tidy: pd.DataFrame, target_synonyms: list[str]) -> pd.Series:
    # case-insensitive substring and similarity matching
    if tidy.empty:
        return pd.Series(dtype=float)
    candidates = tidy[~tidy["metric"].astype(str).str.strip().eq("")].copy()
    candidates["metric_low"] = candidates["metric"].astype(str).str.lower()
    syn_low = [s.lower() for s in target_synonyms]
    # direct substring
    mask = candidates["metric_low"].apply(lambda m: any(s in m for s in syn_low))
    if mask.any():
        df = candidates[mask].copy()
        s = df.groupby("period")["value"].sum()
        return s.sort_index()
    # fuzzy matching using difflib
    unique_metrics = candidates["metric_low"].unique().tolist()
    matches = set()
    for syn in syn_low:
        close = difflib.get_close_matches(syn, unique_metrics, n=5, cutoff=0.7)
        for c in close:
            matches.add(c)
    if matches:
        df = candidates[candidates["metric_low"].isin(matches)].copy()
        s = df.groupby("period")["value"].sum()
        return s.sort_index()
    return pd.Series(dtype=float)


def extract_fcf_from_workbook(path: str) -> pd.DataFrame:
    """Return DataFrame with columns ['period','fcf'] extracted from workbook using mappings and fallbacks."""
    mappings = _load_mappings()
    tidy = extract_cf_tidy_from_workbook(path)
    # try direct FCF metric
    fcf_s = _find_series_metric(tidy, mappings.get("fcf", []))
    if not fcf_s.empty:
        df = fcf_s.rename("fcf").reset_index()
        df.columns = ["period", "fcf"]
        return df
    # fallbacks: operating - investing
    op_s = _find_series_metric(tidy, mappings.get("operating_cf", []))
    inv_s = _find_series_metric(tidy, mappings.get("investing_cf", []))
    if not op_s.empty and not inv_s.empty:
        # align indices
        idx = op_s.index.union(inv_s.index)
        op = op_s.reindex(idx).fillna(0.0)
        inv = inv_s.reindex(idx).fillna(0.0)
        fcf = (op + inv)  # note: in some datasets investing is negative; this assumes sign convention
        df = fcf.rename("fcf").reset_index()
        df.columns = ["period", "fcf"]
        return df
    # as last resort, try heuristic: last numeric column in a sheet that looks like a cashflow
    if not tidy.empty:
        # pick metric with highest non-nan values sum
        agg = tidy.groupby("metric")["value"].apply(lambda s: s.dropna().abs().sum()).sort_values(ascending=False)
        if not agg.empty:
            metric = agg.index[0]
            s = tidy[tidy["metric"] == metric].groupby("period")["value"].sum()
            df = s.rename("fcf").reset_index()
            df.columns = ["period", "fcf"]
            return df
    # nothing found
    return pd.DataFrame(columns=["period", "fcf"])


def extract_future_fcf_plan(xlsx_file: str, forecast_years: int | None = None) -> pd.DataFrame:
    """
    Extract future FCF plan from uploaded Excel (FS_年次 sheet, CF section).
    Returns DataFrame with columns: [period, 営業CF, 投資CF, FCF]
    suitable for DCF analysis and display in business plan section.
    
    Args:
        xlsx_file: Path to Excel file
        forecast_years: Number of forecast years to include (if None, includes all)
    
    Returns:
        DataFrame with period and FCF data
    """
    try:
        # Try to import ingest_plan_excel to extract data by section
        import sys
        from pathlib import Path
        
        # Add parent directory to path to import ingest_plan_excel
        core_dir = Path(__file__).parent
        sys.path.insert(0, str(core_dir))
        from ingest_plan_excel import extract_yearly_table_by_section
        
        # Extract CF section from FS_年次 sheet
        section_results = extract_yearly_table_by_section(xlsx_file, "FS_年次")
        cf_section = section_results.get("キャッシュフロー計算書")
        
        if not cf_section:
            return pd.DataFrame(columns=["period", "営業CF", "投資CF", "FCF"])
        
        cf_long = cf_section["long"]
        
        # Extract operating CF
        op_cf_keywords = ["営業", "operating"]
        op_rows = cf_long[cf_long["metric"].str.contains("|".join(op_cf_keywords), case=False, na=False)]
        # Look for the main operating cash flow line (usually "～キャッシュフロー")
        main_op = op_rows[op_rows["metric"].str.contains("キャッシュフロー$", regex=True, case=False, na=False)]
        if main_op.empty:
            main_op = op_rows[op_rows["metric"].str.contains("営業", case=False, na=False)]
        
        op_by_period = {}
        if not main_op.empty:
            # Get the first matching metric (usually operating CF)
            metric_name = main_op["metric"].iloc[0]
            op_data = cf_long[cf_long["metric"] == metric_name].groupby("period")["value"].sum()
            op_by_period = op_data.to_dict()
        
        # Extract investing CF
        inv_cf_keywords = ["投資", "investing"]
        inv_rows = cf_long[cf_long["metric"].str.contains("|".join(inv_cf_keywords), case=False, na=False)]
        # Look for the main investing cash flow line
        main_inv = inv_rows[inv_rows["metric"].str.contains("キャッシュフロー$", regex=True, case=False, na=False)]
        if main_inv.empty:
            main_inv = inv_rows[inv_rows["metric"].str.contains("投資", case=False, na=False)]
        
        inv_by_period = {}
        if not main_inv.empty:
            metric_name = main_inv["metric"].iloc[0]
            inv_data = cf_long[cf_long["metric"] == metric_name].groupby("period")["value"].sum()
            inv_by_period = inv_data.to_dict()
        
        # Extract FCF (フリーキャッシュフロー)
        fcf_keywords = ["フリー", "フリーキャッシュ", "FCF", "free"]
        fcf_rows = cf_long[cf_long["metric"].str.contains("|".join(fcf_keywords), case=False, na=False)]
        # If FCF row exists in the data, use it; otherwise calculate
        if not fcf_rows.empty:
            # Use the first FCF metric found
            metric_name = fcf_rows["metric"].iloc[0]
            fcf_data = cf_long[cf_long["metric"] == metric_name].groupby("period")["value"].sum()
            fcf_by_period = fcf_data.to_dict()
        else:
            # Calculate: FCF = Operating CF + Investing CF
            all_periods = set(op_by_period.keys()) | set(inv_by_period.keys())
            fcf_by_period = {}
            for period in all_periods:
                op_val = op_by_period.get(period, 0)
                inv_val = inv_by_period.get(period, 0)
                if pd.notna(op_val) and pd.notna(inv_val):
                    fcf_by_period[period] = op_val + inv_val
        
        # Combine into result DataFrame
        all_periods = sorted(set(op_by_period.keys()) | set(inv_by_period.keys()) | set(fcf_by_period.keys()))
        
        result_data = {
            "period": all_periods,
            "営業CF": [op_by_period.get(p, np.nan) for p in all_periods],
            "投資CF": [inv_by_period.get(p, np.nan) for p in all_periods],
            "FCF": [fcf_by_period.get(p, np.nan) for p in all_periods],
        }
        
        result_df = pd.DataFrame(result_data)
        
        # Convert period to datetime-like string for sorting
        result_df = result_df.sort_values("period").reset_index(drop=True)
        
        # Filter to forecast years if specified
        if forecast_years and forecast_years > 0:
            # Keep the last N years (forecast period)
            result_df = result_df.tail(forecast_years).reset_index(drop=True)
        
        return result_df
        
    except Exception as e:
        # If extraction fails, return empty DataFrame
        print(f"Warning: extract_future_fcf_plan failed: {e}")
        return pd.DataFrame(columns=["period", "営業CF", "投資CF", "FCF"])
