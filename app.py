import os
import json
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

from core.ingest_pdf import ingest_financials_from_pdf
from core.evidence import EvidenceLog
from core.compute import compute_metrics, compute_valuation_table
from core.export import to_excel_bytes

# LLM正規化は任意（OFFでも動く）
USE_LLM = True
if USE_LLM:
    try:
        from core.normalize import normalize_with_llm
    except Exception:
        normalize_with_llm = None

load_dotenv()

st.set_page_config(page_title="Valuation Workbench (PoC)", layout="wide")
st.title("Valuation Workbench (PoC) - PDF Ingest v1")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY が未設定です（LLM正規化をONにするなら .env に設定してください）")

left, right = st.columns([1, 2])

with left:
    st.header("1) Upload")
    pdf_file = st.file_uploader("決算書PDF", type=["pdf"])

    st.header("2) Company & assumptions")
    company = st.text_input("対象企業名", "")
    include_lease = st.checkbox("EVにリース債務を含める", value=True)

    st.header("3) External (v1: 手動入力)")
    market_cap_manual = st.number_input("時価総額（円）", value=0.0, help="未入力なら price×shares でも可")
    price = st.number_input("株価（任意）", value=0.0)
    shares_override = st.number_input("発行株式数（上書き・任意）", value=0.0)

    st.header("4) Evidence URLs")
    urls = st.text_area("参照URL（1行1つ）", "")

    run = st.button("Run")

with right:
    st.header("Outputs")

    if run:
        if not pdf_file:
            st.error("PDFをアップロードしてください。")
            st.stop()

        evlog = EvidenceLog()
        url_list = [u.strip() for u in urls.splitlines() if u.strip()]
        for u in url_list:
            evlog.add_url(u)

        raw, meta = ingest_financials_from_pdf(pdf_file)

        # Evidence: PDF抽出値（ページ番号つき）
        bs_page = (meta.get("bs_page") or 0) + 1
        pl_page = (meta.get("pl_page") or 0) + 1
        sh_page = (meta.get("shares_page") or 0) + 1 if meta.get("shares_page") is not None else None

        for k, v in raw.get("pl", {}).items():
            evlog.add(field_name=f"pl.{k}", value=v, source_type="internal_pdf", page=pl_page, raw_label=k, unit="JPY")
        for k, v in raw.get("bs", {}).items():
            evlog.add(field_name=f"bs.{k}", value=v, source_type="internal_pdf", page=bs_page, raw_label=k, unit="JPY")
        for k, v in raw.get("shares", {}).items():
            evlog.add(field_name=f"shares.{k}", value=v, source_type="internal_pdf", page=sh_page, raw_label=k, unit="shares")

        st.subheader("A) Raw extracted (from PDF)")
        st.json({"raw": raw, "meta": meta})

        # 標準化（LLM or ルール）
        standardized = {
            "revenue": raw["pl"].get("revenue"),
            "operating_income": raw["pl"].get("operating_income"),
            "ebitda": None,  # v1: PDFに無ければnull
            "net_income": raw["pl"].get("net_income"),
            "cash": raw["bs"].get("cash_and_deposits"),
            "debt_short": raw["bs"].get("short_term_debt"),
            "debt_long": raw["bs"].get("long_term_debt"),
            "lease_liabilities": raw["bs"].get("lease_liabilities"),
            "total_liabilities": raw["bs"].get("total_liabilities"),
            "total_equity": raw["bs"].get("total_equity"),
            "shares_total": raw["shares"].get("shares_total"),
            "notes": None,
        }

        if USE_LLM and os.getenv("OPENAI_API_KEY") and normalize_with_llm is not None:
            try:
                standardized = normalize_with_llm(raw)
                evlog.add("standardize.status", "ok", source_type="llm", notes="LLM normalization applied")
            except Exception as e:
                evlog.add("standardize.status", "failed", source_type="llm", notes=str(e))

        st.subheader("B) Standardized (for valuation)")
        st.dataframe(pd.DataFrame(standardized.items(), columns=["field", "value"]), use_container_width=True)

        # 時価総額の決定：手入力優先 → 価格×株式数（株式数はPDF or 上書き）
        shares_for_mcap = shares_override if shares_override > 0 else (standardized.get("shares_total") or 0)
        mcap = None
        if market_cap_manual and market_cap_manual > 0:
            mcap = float(market_cap_manual)
            evlog.add("market_cap", mcap, source_type="manual", notes="manual input")
        elif price > 0 and shares_for_mcap > 0:
            mcap = float(price) * float(shares_for_mcap)
            evlog.add("price", float(price), source_type="manual")
            evlog.add("shares_for_mcap", float(shares_for_mcap), source_type="calc", calc_formula="override_or_pdf_shares")
            evlog.add("market_cap", mcap, source_type="calc", calc_formula="price * shares_for_mcap")
        else:
            evlog.add("market_cap", None, source_type="manual", notes="not provided")

        base = compute_metrics(
            market_cap=mcap,
            cash=standardized.get("cash"),
            debt_short=standardized.get("debt_short"),
            debt_long=standardized.get("debt_long"),
            lease_liabilities=standardized.get("lease_liabilities"),
            revenue=standardized.get("revenue"),
            ebitda=standardized.get("ebitda"),
            net_income=standardized.get("net_income"),
        )

        # Evidence: 計算項目
        evlog.add("debt_total", base["debt_total"], source_type="calc", calc_formula="debt_short + debt_long", unit="JPY")
        evlog.add("net_debt", base["net_debt"], source_type="calc", calc_formula="debt_total - cash", unit="JPY")

        table = compute_valuation_table(company or "Target", base, include_lease=include_lease)

        st.subheader("C) Valuation Table")
        st.dataframe(table, use_container_width=True)

        st.subheader("D) Evidence Log")
        ev_df = evlog.to_df()
        st.dataframe(ev_df, use_container_width=True)

        # Export
        xlsx = to_excel_bytes({
            "valuation": table,
            "standardized": pd.DataFrame(standardized.items(), columns=["field", "value"]),
            "evidence": ev_df,
        })
        st.download_button("Download Excel", data=xlsx, file_name="valuation_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download Evidence JSON", data=json.dumps(evlog.to_dict(), ensure_ascii=False, indent=2), file_name="evidence.json", mime="application/json")


