import os
import json
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

from core.ingest_pdf import ingest_financials_from_pdf
from core.ingest_plan_excel import list_sheet_names, extract_yearly_table
from core.evidence import EvidenceLog
from core.compute import compute_metrics, compute_valuation_table
from core.export import to_excel_bytes
import numpy as np
from core.dcf_from_plan import (
    build_fcf_from_cf,
    run_dcf,
    sensitivity_wacc_g,
    extract_future_fcf_plan_nopat,
)

# LLM正規化は任意（OFFでも動く）
USE_LLM = True
if USE_LLM:
    try:
        from core.normalize import normalize_with_llm
    except Exception:
        normalize_with_llm = None

load_dotenv()

st.set_page_config(page_title="Valuation App", layout="wide")
st.title("Valuation App")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY が未設定です（LLM正規化をONにするなら .env に設定してください）")

left, right = st.columns([1, 2])

with left:
    st.markdown("<h2>Financial Inputs <span style='font-size: 0.6em;'>（決算・事業計画）</span></h2>", unsafe_allow_html=True)
    pdf_file = st.file_uploader("直近の決算書（PDF）をアップロードしてください", type=["pdf"])
    
    # Store PDF file in session for later use
    if pdf_file is not None:
        st.session_state['pdf_file'] = pdf_file

    plan_file = st.file_uploader("中期事業計画（Excel）をアップロードしてください", type=["xls", "xlsx"], help="※ 将来キャッシュフローの算出に使用します")
    if plan_file is not None:
        try:
            sheets = list_sheet_names(plan_file)
            sheet_options = ["シートが選択されていません"] + sheets
            sheet_choice = st.selectbox("将来の売上・費用計画が記載されたシートを選択してください", sheet_options)
            if sheet_choice != "シートが選択されていません" and st.button("▶️ 事業計画からキャッシュフローを生成", key="extract_plan", type="secondary"):
                try:
                    plan_results = extract_yearly_table(plan_file, sheet_choice)
                    st.session_state['plan_extract'] = plan_results
                    # plan_tidy は long format を保持（sheet, metric, period, value, unit）
                    st.session_state['plan_tidy'] = plan_results['long']
                    st.success(f"Sheet {sheet_choice} extracted: {len(plan_results['wide'])} rows, {len(plan_results['wide'].columns)} periods")
                    
                    # FCF計画（NOPATベース）も一緒に抽出（デフォルト税率30%）
                    try:
                        fcf_plan = extract_future_fcf_plan_nopat(plan_file, tax_rate=0.30)
                        st.session_state['fcf_plan'] = fcf_plan
                        st.info(f"FCF plan (NOPAT-based) extracted: {len(fcf_plan)} periods")
                    except Exception as fcf_err:
                        st.warning(f"FCF plan extraction failed: {str(fcf_err)}")
                    
                    # PDFが既にアップロードされている場合は、自動的に再処理してBS情報を取得
                    stored_pdf = st.session_state.get('pdf_file')
                    if stored_pdf is not None:
                        try:
                            raw, meta = ingest_financials_from_pdf(stored_pdf)
                            standardized = {
                                "revenue": raw["pl"].get("revenue"),
                                "operating_income": raw["pl"].get("operating_income"),
                                "ebitda": None,
                                "net_income": raw["pl"].get("net_income"),
                                "cash": raw["bs"].get("cash_and_deposits"),
                                "debt_short": raw["bs"].get("short_term_debt"),
                                "debt_long": raw["bs"].get("long_term_debt"),
                                "lease_liabilities": raw["bs"].get("lease_liabilities"),
                                "total_liabilities": raw["bs"].get("total_liabilities"),
                                "total_equity": raw["bs"].get("total_equity"),
                                "shares_outstanding": raw["shares"].get("shares_total"),
                                "notes": None,
                            }
                            st.session_state['standardized_bs'] = standardized
                            st.info("決算書のBS情報を自動読込しました")
                        except Exception as pdf_err:
                            st.warning(f"PDF再処理に失敗しました（WACCの資本構成は手動入力してください）: {str(pdf_err)}")
                except Exception as e:
                    st.error(f"Plan extraction failed: {e}")
                except Exception as e:
                    st.error(f"Plan extraction failed: {e}")
        except Exception as e:
            st.error(f"Excel読み込みエラー: {e}")

with right:
    st.markdown("<h2>Valuation Summary <span style='font-size: 0.6em;'>（株価算定結果）</span></h2>", unsafe_allow_html=True)

    if pdf_file:

        evlog = EvidenceLog()

        # もし事業計画がセッションにあれば、根拠ログに追加
        if 'plan_extract' in st.session_state:
            plan = st.session_state['plan_extract']
            evlog.add("plan.sheet", plan['sheet'], source_type="plan_excel", raw_label=plan['sheet'], notes=f"unit={plan.get('unit')}")

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

        # 時価総額の決定
        mcap = None
        evlog.add("market_cap", None, source_type="manual", notes="not provided")

        # Store standardized data for WACC calculation
        st.session_state['standardized_bs'] = standardized

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

        table = compute_valuation_table("Target", base, include_lease=True)

    # Plan preview (アップロード済みの事業計画を表示)
    if 'plan_extract' in st.session_state:
        plan = st.session_state['plan_extract']
        st.subheader("①将来FCFの算出")
        
        # FCF計画を最初に表示（NOPATベースがある場合）
        if 'fcf_plan' in st.session_state and not st.session_state['fcf_plan'].empty:
            st.write("**FCF計画（NOPAT + 減価償却 − CAPEX − Δ運転資本）※税率デフォルト30%**")
            fcf_plan = st.session_state['fcf_plan'].copy()
            # 数値列のフォーマット
            for col in fcf_plan.columns:
                if col != 'period':
                    fcf_plan[col] = fcf_plan[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            st.dataframe(fcf_plan, use_container_width=True)
            
            # WACC計算セクション（FCF表の後に表示）
            if 'standardized_bs' in st.session_state:
                st.write("---")
                st.subheader("②WACCの算出")
                st.write("WACC = (E/V)×Re + (D/V)×Rd×(1−Tc)")
                
                # Get BS data from standardized
                bs_data = st.session_state.get('standardized_bs', {})
                equity_default = float(bs_data.get('total_equity') or 0.0)
                debt_short_default = float(bs_data.get('debt_short') or 0.0)
                debt_long_default = float(bs_data.get('debt_long') or 0.0)
                total_debt_default = debt_short_default + debt_long_default
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write("**自己（株主）資本コスト**")
                    cost_of_equity = st.number_input("Re (%)", value=8.0, step=0.5, key="wacc_re")
                with col2:
                    st.write("**有利子負債コスト**")
                    cost_of_debt = st.number_input("Rd (%)", value=2.0, step=0.1, key="wacc_rd")
                with col3:
                    st.write("**資本構成（帳簿価額）**")
                    book_equity = st.number_input("自己資本 E (円)", value=equity_default, min_value=0.0, key="wacc_equity", help="最新の決算書（BS）の純資産")
                    book_debt = st.number_input("有利子負債 D (円)", value=total_debt_default, min_value=0.0, key="wacc_debt", help="短期借入金 + 長期借入金")
                with col4:
                    st.write("**法人税率**")
                    tax_rate_wacc = st.number_input("Tc (%)", value=30.0, min_value=0.0, max_value=100.0, step=0.5, key="wacc_tc")
                
                if st.button("WACC計算", key="wacc_calc_btn"):
                    # Calculate WACC
                    E = float(book_equity)
                    D = float(book_debt)
                    V = E + D
                    Re = float(cost_of_equity) / 100.0
                    Rd = float(cost_of_debt) / 100.0
                    Tc = float(tax_rate_wacc) / 100.0
                    
                    if V == 0:
                        st.error("自己資本または有利子負債が必要です")
                    else:
                        wacc = (E/V) * Re + (D/V) * Rd * (1 - Tc)
                        st.session_state['wacc_calculated'] = wacc
                        st.session_state['wacc_inputs'] = {
                            "E": E,
                            "D": D,
                            "V": V,
                            "Re": Re,
                            "Rd": Rd,
                            "Tc": Tc,
                        }
                
                if 'wacc_calculated' in st.session_state:
                    wacc = st.session_state['wacc_calculated']
                    wacc_inputs = st.session_state.get('wacc_inputs', {})
                    E = wacc_inputs.get('E', 0)
                    D = wacc_inputs.get('D', 0)
                    V = wacc_inputs.get('V', E + D if (E + D) > 0 else 1)
                    Re = wacc_inputs.get('Re', float(cost_of_equity) / 100.0)
                    Rd = wacc_inputs.get('Rd', float(cost_of_debt) / 100.0)
                    Tc = wacc_inputs.get('Tc', float(tax_rate_wacc) / 100.0)
                    
                    # Display results (persistent)
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("計算されたWACC", f"{wacc*100:.2f}%")
                    with col_res2:
                        st.write("")
                        st.write("")
                        st.write(f"E/V = {E/V*100:.1f}%, D/V = {D/V*100:.1f}%")
                    
                    # Show breakdown
                    st.write("**内訳:**")
                    st.write(f"- 株主資本部分：(E/V)×Re = {E/V:.3f} × {Re*100:.1f}% = {(E/V)*Re*100:.2f}%")
                    st.write(f"- 負債部分：(D/V)×Rd×(1−Tc) = {D/V:.3f} × {Rd*100:.1f}% × (1−{Tc*100:.0f}%) = {(D/V)*Rd*(1-Tc)*100:.2f}%")
                
                # Terminal Value計算セクション（WACC計算後に表示）
                if 'wacc_calculated' in st.session_state:
                    st.write("---")
                    st.subheader("F) Terminal Value計算")
                    st.write("TV = FCF_last × (1 + g) / (WACC − g)")
                    
                    wacc_tv = st.session_state['wacc_calculated']
                    fcf_plan_data = st.session_state.get('fcf_plan', pd.DataFrame())
                    
                    if not fcf_plan_data.empty:
                        # Get last year FCF
                        fcf_col = [c for c in fcf_plan_data.columns if c.upper() == 'FCF']
                        if fcf_col:
                            fcf_values = pd.to_numeric(fcf_plan_data[fcf_col[0]], errors='coerce')
                            fcf_last = fcf_values.dropna().iloc[-1] if len(fcf_values.dropna()) > 0 else 0.0
                            forecast_years = len(fcf_values.dropna())
                        else:
                            fcf_last = 0.0
                            forecast_years = 0
                    else:
                        fcf_last = 0.0
                        forecast_years = 0
                    
                    col_tv1, col_tv2 = st.columns(2)
                    with col_tv1:
                        st.write(f"**最終年FCF**: {fcf_last:,.0f} 円")
                        st.write(f"**予測期間**: {forecast_years} 年")
                        g = st.number_input("永続成長率 g (%)", value=2.0, step=0.1, key="tv_growth_rate") / 100.0
                    
                    with col_tv2:
                        st.write("")
                        st.write("")
                        if st.button("Terminal Value計算", key="tv_calc_btn"):
                            if wacc_tv <= g:
                                st.error("WACC > g である必要があります（現在のWACC≤g）")
                            elif fcf_last <= 0:
                                st.error("最終年FCFが正の値である必要があります")
                            else:
                                tv = (fcf_last * (1 + g)) / (wacc_tv - g)
                                pv_tv = tv / (1 + wacc_tv) ** forecast_years
                                
                                st.session_state['terminal_value'] = tv
                                st.session_state['pv_terminal_value'] = pv_tv
                
                if 'pv_terminal_value' in st.session_state and 'terminal_value' in st.session_state:
                    tv = st.session_state['terminal_value']
                    pv_tv = st.session_state['pv_terminal_value']
                    col_tv_res1, col_tv_res2 = st.columns(2)
                    with col_tv_res1:
                        st.metric("Terminal Value", f"{tv:,.0f} 円")
                    with col_tv_res2:
                        st.metric("Terminal Value (PV)", f"{pv_tv:,.0f} 円")
                    st.write("**計算式:**")
                    st.write(f"TV = {fcf_last:,.0f} × (1 + {g*100:.2f}%) / ({wacc_tv*100:.2f}% − {g*100:.2f}%) = {tv:,.0f} 円")
                    st.write(f"PV(TV) = {tv:,.0f} / (1 + {wacc_tv*100:.2f}%)^{forecast_years} = {pv_tv:,.0f} 円")
                
                # 事業価値・株式価値セクション（Terminal Value計算後に表示）
                if 'pv_terminal_value' in st.session_state and 'wacc_calculated' in st.session_state:
                    st.write("---")
                    st.subheader("G) 事業価値・株式価値")
                    
                    fcf_plan_data = st.session_state.get('fcf_plan', pd.DataFrame())
                    wacc_ev = st.session_state['wacc_calculated']
                    pv_tv = st.session_state['pv_terminal_value']
                    
                    # FCF予測期間のPV計算
                    if not fcf_plan_data.empty:
                        fcf_col = [c for c in fcf_plan_data.columns if c.upper() == 'FCF']
                        if fcf_col:
                            fcf_values = pd.to_numeric(fcf_plan_data[fcf_col[0]], errors='coerce').dropna()
                            
                            # Mid-year convention: 各年のPV計算
                            pv_fcf_list = []
                            for i, fcf in enumerate(fcf_values, start=1):
                                pv = fcf / ((1 + wacc_ev) ** (i - 0.5))  # Mid-year convention
                                pv_fcf_list.append(pv)
                            
                            pv_fcf_sum = sum(pv_fcf_list)
                        else:
                            pv_fcf_sum = 0.0
                            pv_fcf_list = []
                    else:
                        pv_fcf_sum = 0.0
                        pv_fcf_list = []
                    
                    # 事業価値 = PV(FCF予測) + PV(TV)
                    enterprise_value = pv_fcf_sum + pv_tv
                    
                    col_ev1, col_ev2, col_ev3 = st.columns(3)
                    with col_ev1:
                        st.metric("PV(FCF予測期間)", f"{pv_fcf_sum:,.0f} 円")
                    with col_ev2:
                        st.metric("PV(Terminal Value)", f"{pv_tv:,.0f} 円")
                    with col_ev3:
                        st.metric("事業価値 (EV)", f"{enterprise_value:,.0f} 円", 
                                  help="Enterprise Value = PV(FCF予測) + PV(TV)")
                    
                    st.write("**FCF予測期間の内訳（mid-year convention）:**")
                    if pv_fcf_list:
                        fcf_detail_data = []
                        fcf_values_display = pd.to_numeric(fcf_plan_data[fcf_col[0]], errors='coerce').dropna()
                        for i, (fcf, pv) in enumerate(zip(fcf_values_display, pv_fcf_list), start=1):
                            fcf_detail_data.append({
                                "年": f"Year {i}",
                                "FCF": f"{fcf:,.0f}",
                                "PV": f"{pv:,.0f}"
                            })
                        st.dataframe(pd.DataFrame(fcf_detail_data), use_container_width=True)
                    
                    # 株式価値計算（有利子負債を除く）
                    st.write("---")
                    st.write("**株式価値計算**")
                    st.write("株式価値 = 事業価値 − 純有利子負債")
                    st.write("純有利子負債 = 有利子負債 − 現金及び現金同等物")
                    
                    standardized_bs = st.session_state.get('standardized_bs', {})
                    
                    col_eq1, col_eq2, col_eq3 = st.columns(3)
                    with col_eq1:
                        debt_short = standardized_bs.get('debt_short', 0) or 0
                        debt_long = standardized_bs.get('debt_long', 0) or 0
                        total_debt = debt_short + debt_long
                        st.number_input("有利子負債 (円)", value=float(total_debt), 
                                        format="%.0f", key="debt_for_equity")
                    
                    with col_eq2:
                        cash = standardized_bs.get('cash', 0) or 0
                        st.number_input("現金及び現金同等物 (円)", value=float(cash), 
                                        format="%.0f", key="cash_for_equity")
                    
                    with col_eq3:
                        st.write("")
                        st.write("")
                        if st.button("株式価値計算", key="equity_value_calc_btn"):
                            debt_input = st.session_state.get('debt_for_equity', total_debt)
                            cash_input = st.session_state.get('cash_for_equity', cash)
                            net_debt = debt_input - cash_input
                            equity_value = enterprise_value - net_debt
                            
                            st.session_state['net_debt'] = net_debt
                            st.session_state['equity_value'] = equity_value
                
                if 'equity_value' in st.session_state and 'net_debt' in st.session_state:
                    net_debt = st.session_state['net_debt']
                    equity_value = st.session_state['equity_value']
                    debt_input = st.session_state.get('debt_for_equity', standardized_bs.get('debt_short', 0) + standardized_bs.get('debt_long', 0))
                    cash_input = st.session_state.get('cash_for_equity', standardized_bs.get('cash', 0))
                    col_eq_res1, col_eq_res2 = st.columns(2)
                    with col_eq_res1:
                        st.metric("純有利子負債", f"{net_debt:,.0f} 円")
                    with col_eq_res2:
                        st.metric("株式価値", f"{equity_value:,.0f} 円",
                                  help="Equity Value = EV − Net Debt")
                    st.write("**計算式:**")
                    st.write(f"純有利子負債 = {debt_input:,.0f} − {cash_input:,.0f} = {net_debt:,.0f} 円")
                    st.write(f"株式価値 = {enterprise_value:,.0f} − {net_debt:,.0f} = {equity_value:,.0f} 円")
                
                # 1株当たり価値計算セクション（株式価値計算後に表示）
                if 'equity_value' in st.session_state:
                    st.write("---")
                    st.subheader("H) 1株当たり価値")
                    st.write("1株当たり価値 = 株式価値 ÷ 発行済株式数")
                    
                    equity_value_for_share = st.session_state['equity_value']
                    standardized_bs = st.session_state.get('standardized_bs', {})
                    shares_default = standardized_bs.get('shares_outstanding', 0) or 0
                    
                    col_sh1, col_sh2 = st.columns(2)
                    with col_sh1:
                        st.write(f"**株式価値**: {equity_value_for_share:,.0f} 円")
                        shares_input = st.number_input("発行済株式数 (株)", 
                                                       value=float(shares_default), 
                                                       format="%.0f", 
                                                       key="shares_for_price",
                                                       help="完全希薄化後の発行済株式数を入力してください")
                    
                    with col_sh2:
                        st.write("")
                        st.write("")
                        if st.button("1株当たり価値計算", key="price_per_share_calc_btn"):
                            shares_calc = st.session_state.get('shares_for_price', shares_default)
                            
                            if shares_calc <= 0:
                                st.error("発行済株式数は正の値である必要があります")
                            else:
                                price_per_share = equity_value_for_share / shares_calc
                                st.session_state['price_per_share'] = price_per_share
                                st.session_state['shares_used'] = shares_calc
                                
                                st.metric("1株当たり価値", f"{price_per_share:,.2f} 円",
                                          help=f"株式価値 {equity_value_for_share:,.0f}円 ÷ {shares_calc:,.0f}株")
                                
                                st.write("**計算式:**")
                                st.write(f"1株当たり価値 = {equity_value_for_share:,.0f} ÷ {shares_calc:,.0f} = {price_per_share:,.2f} 円")
