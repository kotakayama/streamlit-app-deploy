import pandas as pd
import numpy as np

OP_CF_LABELS = ["営業活動によるキャッシュフロー"]
INV_CF_LABELS = ["投資活動によるキャッシュフロー"]


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
