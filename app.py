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

        # --- 人間向けラベル（日本語） ---
        DISPLAY_LABELS = {
            "revenue": "売上高",
            "gross_profit": "売上総利益",
            "operating_income": "営業利益",
            "ordinary_income": "経常利益",
            "net_income": "当期純利益",
            "ebitda": "EBITDA",
            "cash": "現金及び預金",
            "cash_and_deposits": "現金及び預金",
            "current_assets": "流動資産",
            "fixed_assets": "固定資産",
            "debt_short": "短期借入金",
            "debt_long": "長期借入金",
            "short_term_debt": "短期借入金",
            "lease_liabilities": "リース債務",
            "current_liabilities": "流動負債",
            "fixed_liabilities": "固定負債",
            "total_liabilities": "負債合計",
            "total_equity": "純資産",
            "shares_total": "発行済株式数",
            "shares_common": "普通株式(株数)",
        }

        def humanize_keys(d: dict) -> dict:
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[DISPLAY_LABELS.get(k, k)] = humanize_keys(v)
                else:
                    out[DISPLAY_LABELS.get(k, k)] = v
            return out

        st.subheader("A) Raw extracted (from PDF)")
        # 表示用にキーを日本語化して出力
        try:
            human_raw = {
                "pl": humanize_keys(raw.get("pl", {})),
                "bs": humanize_keys(raw.get("bs", {})),
                "shares": humanize_keys(raw.get("shares", {})),
            }
        except Exception:
            human_raw = {"raw": raw}
        st.json({"raw": human_raw, "meta": meta})

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
        # 日本語ラベルを併記して表示
        std_df = pd.DataFrame([{"field": k, "label": DISPLAY_LABELS.get(k, k), "value": v} for k, v in standardized.items()])
        st.dataframe(std_df["label value"].tolist() if False else std_df[["label", "value"]].rename(columns={"label": "項目", "value": "値"}), use_container_width=True)

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

        # Valuation 表の項目を日本語化して表示
        VAL_LABELS = {
            "Company": "対象企業",
            "Market Cap": "時価総額",
            "Net Debt": "ネット・デット（純有利子負債）",
            "Lease (included)": "リース（含む）",
            "Lease (excluded)": "リース（除く）",
            "EV": "企業価値（EV）",
            "EV/Sales": "EV/売上高",
            "EV/EBITDA": "EV/EBITDA",
            "PER": "PER",
        }
        disp_table = table.copy()
        disp_table["Metric"] = disp_table["Metric"].map(VAL_LABELS).fillna(disp_table["Metric"])
        disp_table = disp_table.rename(columns={"Metric": "項目", "Value": "値"})
        st.subheader("C) Valuation Table")
        st.dataframe(disp_table, use_container_width=True)

        st.subheader("D) Evidence Log")
        ev_df = evlog.to_df()
        # 人間向けラベル列を追加
        def resolve_field_label(field_name: str) -> str:
            if not field_name:
                return ""
            key = field_name.split('.')[-1]
            return DISPLAY_LABELS.get(key, field_name)
        if not ev_df.empty:
            ev_df = ev_df.copy()
            ev_df["項目"] = ev_df["field_name"].apply(resolve_field_label)
            # 表示順を調整
            cols = ["項目"] + [c for c in ev_df.columns if c != "項目"]
            st.dataframe(ev_df[cols], use_container_width=True)
        else:
            st.dataframe(ev_df, use_container_width=True)

        # Export
        # 標準化シートは日本語ラベル列（項目, 値）を出力
        standardized_export = std_df[["label", "value"]].rename(columns={"label": "項目", "value": "値"}) if 'std_df' in locals() else pd.DataFrame(standardized.items(), columns=["field", "value"])
        xlsx = to_excel_bytes({
            "valuation": table,
            "standardized": standardized_export,
            "evidence": ev_df,
        })
        st.download_button("Download Excel", data=xlsx, file_name="valuation_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download Evidence JSON", data=json.dumps(evlog.to_dict(), ensure_ascii=False, indent=2), file_name="evidence.json", mime="application/json")


