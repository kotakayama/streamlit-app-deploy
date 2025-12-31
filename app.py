import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st

# ----------------------------
# Config
# ----------------------------
REQUIRED_SHEETS = ["PL", "BS", "CF"]
TZ = timezone.utc  # ログはUTCで統一（社内方針でJSTにしてもOK）

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

@dataclass
class Provenance:
    item: str
    value: float
    unit: str
    source_type: str  # excel/manual
    source_detail: str  # e.g. "PL!A12:FY2024" or "manual:market"
    captured_at: str
    formula: Optional[str] = None
    note: Optional[str] = None

def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def match_item(label: str, patterns: List[str]) -> bool:
    t = normalize_text(label)
    return any(re.search(p, t) for p in patterns)

def read_sheet_table(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xls, sheet_name=sheet)
    if df.empty:
        raise ValueError(f"{sheet} is empty.")
    # 先頭列を科目名として扱う
    df = df.rename(columns={df.columns[0]: "Item"})
    # 年度列のフィルタ（FY2023等を想定）
    year_cols = [c for c in df.columns if re.match(r"^FY\d{4}", str(c))]
    if not year_cols:
        raise ValueError(f"{sheet}: No FY columns found (e.g., FY2023).")
    df = df[["Item"] + year_cols].copy()
    return df

def extract_metric(
    df: pd.DataFrame,
    metric_key: str,
    year: str
) -> Tuple[Optional[float], Optional[str]]:
    """Return (value, source_detail)"""
    patterns = ALIASES[metric_key]
    hits = df[df["Item"].apply(lambda x: match_item(x, patterns))]
    if hits.empty:
        return None, None
    # 最初の一致を採用（v1）
    row_idx = hits.index[0]
    label = df.loc[row_idx, "Item"]
    val = df.loc[row_idx, year]
    if pd.isna(val):
        return None, None
    # excel cell referenceっぽく残す（厳密なセル番地はv2で改善）
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
) -> Dict[str, Optional[float]]:
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

def export_excel(
    valuation_df: pd.DataFrame,
    peers_df: pd.DataFrame,
    prov_df: pd.DataFrame,
    memo_text: str
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        valuation_df.to_excel(writer, sheet_name="Valuation", index=False)
        peers_df.to_excel(writer, sheet_name="Peers", index=False)
        prov_df.to_excel(writer, sheet_name="Provenance", index=False)

        # Memo sheet
        workbook  = writer.book
        worksheet = workbook.add_worksheet("MTG Memo")
        writer.sheets["MTG Memo"] = worksheet
        for i, line in enumerate(memo_text.splitlines() or [""]):
            worksheet.write(i, 0, line)

    return output.getvalue()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Valuation Automation (v1)", layout="wide")
st.title("Valuation Automation (v1) — テンプレExcelから指標抽出→同業比較→根拠ログ")

st.caption("v1: Web取得なし（手入力/CSVで代替）。v2で科目マッピングをLLM化、v3でWeb抽出を追加。")

colA, colB = st.columns([1, 2])

with colA:
    company = st.text_input("対象企業", value="キッチハイク")
    target_year = st.text_input("対象年度（FYxxxx）", value="FY2024")
    currency = st.selectbox("通貨", ["JPY", "USD", "EUR"], index=0)
    unit = st.selectbox("数値単位（Excelと揃える）", ["円", "千円", "百万円", "億円"], index=2)

with colB:
    st.subheader("① 財務パッケージ（Excel）アップロード")
    f = st.file_uploader("PL/BS/CFシートを含む .xlsx", type=["xlsx"])

st.divider()

prov_logs: List[Provenance] = []
financials = {}

if f is not None:
    xls = pd.ExcelFile(f)
    missing = [s for s in REQUIRED_SHEETS if s not in xls.sheet_names]
    if missing:
        st.error(f"シートが足りません: {missing}（必要: {REQUIRED_SHEETS}）")
        st.stop()

    pl = read_sheet_table(xls, "PL")
    bs = read_sheet_table(xls, "BS")
    cf = read_sheet_table(xls, "CF")

    # Extract metrics
    rev, rev_src = extract_metric(pl, "revenue", target_year)
    ebitda, ebitda_src = extract_metric(pl, "ebitda", target_year)
    op, op_src = extract_metric(pl, "operating_income", target_year)
    ni, ni_src = extract_metric(pl, "net_income", target_year)

    cash, cash_src = extract_metric(bs, "cash", target_year)
    debt, debt_src = extract_metric(bs, "debt", target_year)
    eq, eq_src = extract_metric(bs, "equity", target_year)
    shares, shares_src = extract_metric(bs, "shares", target_year)

    cfo, cfo_src = extract_metric(cf, "cfo", target_year)
    cfi, cfi_src = extract_metric(cf, "cfi", target_year)
    fcf, fcf_src = extract_metric(cf, "fcf", target_year)

    # if FCF not present, estimate CFO+CFI
    if fcf is None and (cfo is not None and cfi is not None):
        fcf = cfo + cfi
        fcf_src = "estimated from CFO + CFI"

    extracted = {
        "Revenue": rev,
        "EBITDA": ebitda,
        "Operating Income": op,
        "Net Income": ni,
        "Cash": cash,
        "Debt": debt,
        "Equity": eq,
        "Shares": shares,
        "CFO": cfo,
        "CFI": cfi,
        "FCF": fcf,
    }

    # Provenance logs (excel)
    ts = now_iso()
    def add_prov(item, value, src):
        if value is None:
            return
        prov_logs.append(Provenance(
            item=item, value=float(value), unit=unit,
            source_type="excel", source_detail=src or "",
            captured_at=ts
        ))

    add_prov("Revenue", rev, f"PL: {rev_src}")
    add_prov("EBITDA", ebitda, f"PL: {ebitda_src}")
    add_prov("Operating Income", op, f"PL: {op_src}")
    add_prov("Net Income", ni, f"PL: {ni_src}")
    add_prov("Cash", cash, f"BS: {cash_src}")
    add_prov("Debt", debt, f"BS: {debt_src}")
    add_prov("Equity", eq, f"BS: {eq_src}")
    add_prov("Shares", shares, f"BS: {shares_src}")
    add_prov("CFO", cfo, f"CF: {cfo_src}")
    add_prov("CFI", cfi, f"CF: {cfi_src}")
    if fcf is not None:
        prov_logs.append(Provenance(
            item="FCF", value=float(fcf), unit=unit,
            source_type="excel" if fcf_src != "estimated from CFO + CFI" else "derived",
            source_detail=f"CF: {fcf_src}",
            captured_at=ts,
            formula=None if fcf_src != "estimated from CFO + CFI" else "FCF = CFO + CFI",
            note=None if fcf_src != "estimated from CFO + CFI" else "定義は案件により異なるため要確認"
        ))

    st.subheader("② 抽出結果（プレビュー）")
    st.dataframe(pd.DataFrame([extracted]).T.rename(columns={0: target_year}))

    st.subheader("③ 市場データ（v1は手入力）")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        price = st.number_input("株価（1株）", min_value=0.0, value=0.0, step=1.0)
    with c2:
        shares_in = st.number_input("発行株式数（株）", min_value=0.0, value=float(shares or 0.0), step=1.0)
    with c3:
        market_cap_override = st.number_input("時価総額（入力する場合）", min_value=0.0, value=0.0, step=1.0)
    with c4:
        st.markdown("**根拠URL（市場データ）**")
        market_url = st.text_input("例: 証券会社/取引所/IR", value="")

    if market_cap_override > 0:
        market_cap = market_cap_override
        mc_formula = "manual"
    else:
        market_cap = price * shares_in
        mc_formula = "Market Cap = Price × Shares"

    # Market provenance
    ts2 = now_iso()
    prov_logs.append(Provenance(
        item="Price", value=float(price), unit=currency,
        source_type="manual", source_detail=f"url:{market_url}",
        captured_at=ts2
    ))
    prov_logs.append(Provenance(
        item="Shares (input)", value=float(shares_in), unit="shares",
        source_type="manual" if shares is None else "excel+manual",
        source_detail=f"url:{market_url}",
        captured_at=ts2
    ))
    prov_logs.append(Provenance(
        item="Market Cap", value=float(market_cap), unit=unit,
        source_type="derived" if mc_formula != "manual" else "manual",
        source_detail=f"url:{market_url}",
        captured_at=ts2,
        formula=None if mc_formula == "manual" else mc_formula
    ))

    # If cash/debt missing, allow overrides
    st.subheader("④ バリュエーション計算（不足値は補完）")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        cash_in = st.number_input("現金（上書き可）", min_value=0.0, value=float(cash or 0.0), step=1.0)
    with cc2:
        debt_in = st.number_input("有利子負債（上書き可）", min_value=0.0, value=float(debt or 0.0), step=1.0)
    with cc3:
        st.markdown("**根拠URL（BS補完）**")
        bs_url = st.text_input("例: 決算短信/有報/説明資料", value="")

    if cash is None and cash_in > 0:
        prov_logs.append(Provenance("Cash", float(cash_in), unit, "manual", f"url:{bs_url}", now_iso()))
    if debt is None and debt_in > 0:
        prov_logs.append(Provenance("Debt", float(debt_in), unit, "manual", f"url:{bs_url}", now_iso()))

    # Calculate
    if rev is None or rev == 0:
        st.warning("売上が抽出できていないため、EV/Salesが計算できません。PLの科目名をテンプレに寄せてください。")

    metrics = calc_valuation(
        market_cap=market_cap,
        cash=float(cash_in),
        debt=float(debt_in),
        revenue=float(rev or 0.0),
        ebitda=safe_float(ebitda),
        net_income=safe_float(ni),
        equity=safe_float(eq),
        fcf=safe_float(fcf),
    )
    valuation_df = pd.DataFrame([{
        "Company": company,
        "Year": target_year,
        "Currency": currency,
        **metrics
    }])
    st.dataframe(valuation_df)

    st.subheader("⑤ Peer比較（v1: CSV貼り付け/手入力）")
    st.caption("最低限: Company, EV/Sales, EV/EBITDA, PER, PBR, FCF Yield の列があるCSVを貼り付けると集計できます。")
    peers_csv = st.text_area("Peers CSV（ヘッダあり）", height=140, value="Company,EV/Sales,EV/EBITDA,PER,PBR,FCF Yield\n")
    peers_df = pd.DataFrame()
    if peers_csv.strip():
        try:
            peers_df = pd.read_csv(io.StringIO(peers_csv))
            # Append target company
            peers_df = pd.concat([
                peers_df,
                valuation_df[["Company","EV/Sales","EV/EBITDA","PER","PBR","FCF Yield"]].rename(columns={"Company":"Company"})
            ], ignore_index=True)
            st.dataframe(peers_df)

            # Summary stats
            numeric_cols = [c for c in peers_df.columns if c != "Company"]
            stats = peers_df[numeric_cols].apply(pd.to_numeric, errors="coerce").describe(percentiles=[0.25, 0.5, 0.75]).T
            stats = stats[["25%","50%","75%","mean","std","count"]].rename(columns={"50%":"median"})
            st.subheader("Peer統計（四分位・平均など）")
            st.dataframe(stats)

        except Exception as e:
            st.error(f"Peers CSVの読み込みに失敗: {e}")

    st.subheader("⑥ MTGメモ（v1は手書き。v2でLLM自動生成）")
    memo = st.text_area("結論・論点・確認事項", height=180, value=f"- 対象: {company}（{target_year}）\n- 結論:\n- 論点:\n- 確認質問:\n")

    # Provenance DF
    prov_df = pd.DataFrame([p.__dict__ for p in prov_logs])

    st.subheader("⑦ 根拠ログ（Provenance）")
    st.dataframe(prov_df)

    st.subheader("⑧ エクスポート")
    xlsx_bytes = export_excel(
        valuation_df=valuation_df,
        peers_df=peers_df if not peers_df.empty else pd.DataFrame(),
        prov_df=prov_df,
        memo_text=memo
    )
    st.download_button(
        "Excelでダウンロード（Valuation / Peers / Provenance / MTG Memo）",
        data=xlsx_bytes,
        file_name=f"{company}_valuation_{target_year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("まずはPL/BS/CF入りのテンプレExcelをアップロードしてください。")
