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

# ファイルアップローダーのボタンテキストをカスタマイズ
st.markdown("""
<style>
[data-testid="stFileUploader"] section button[kind="secondary"] {
    font-size: 0;
}
[data-testid="stFileUploader"] section button[kind="secondary"]::after {
    content: "ファイルを選択する";
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

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
        # ファイルIDをチェックして、新しいファイルの場合はfcf_planをクリア
        file_id = f"{plan_file.name}_{plan_file.size}"
        if 'last_plan_file_id' not in st.session_state or st.session_state['last_plan_file_id'] != file_id:
            st.session_state['last_plan_file_id'] = file_id
            # 新しいファイルなので、古いfcf_planをクリア
            if 'fcf_plan' in st.session_state:
                del st.session_state['fcf_plan']
        
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
            
            # データをクリアして再計算ボタン（FCF生成後のみ表示）
            if 'fcf_plan' in st.session_state:
                if st.button("🔄 データをクリアして再計算", help="アップロードしたデータをクリアして、最初からやり直します"):
                    keys_to_clear = ['fcf_plan', 'wacc_calculated', 'wacc_inputs', 'terminal_value', 'pv_terminal_value', 
                                   'forecast_years', 'tv_g_used', 'tv_fcf_last', 'tv_forecast_years', 'tv_display_start', 
                                   'tv_display_end', 'net_debt', 'equity_value', 'price_per_share', 'shares_used']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.success("データをクリアしました。Excelファイルを再アップロードしてください。")
                    st.rerun()
                
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
        st.markdown("<h3>① Forecast FCF <span style='font-size: 0.7em;'>（将来FCFの算出）</span></h3>", unsafe_allow_html=True)
        
        # FCF計画を最初に表示（NOPATベースがある場合）
        if 'fcf_plan' in st.session_state and not st.session_state['fcf_plan'].empty:
            st.write("**FCF計画（NOPAT + 減価償却 − CAPEX − Δ運転資本）※税率デフォルト30%、単位：百万円**")
            fcf_plan = st.session_state['fcf_plan'].copy()
            # 数値列のフォーマット（データは既に百万円単位）
            for col in fcf_plan.columns:
                if col != 'period':
                    fcf_plan[col] = fcf_plan[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            st.dataframe(fcf_plan, use_container_width=True)
            
            # WACC計算セクション（FCF表の後に表示）
            if 'standardized_bs' in st.session_state:
                st.write("---")
                st.markdown("<h3>② Calculate WACC <span style='font-size: 0.7em;'>（WACCの算出）</span></h3>", unsafe_allow_html=True)
                st.write("WACC = (E/V)×Re + (D/V)×Rd×(1−Tc)")
                
                # Get BS data from standardized
                bs_data = st.session_state.get('standardized_bs', {})
                equity_default = float(bs_data.get('total_equity') or 0.0) / 1_000_000  # 百万円に変換
                debt_short_default = float(bs_data.get('debt_short') or 0.0) / 1_000_000
                debt_long_default = float(bs_data.get('debt_long') or 0.0) / 1_000_000
                total_debt_default = debt_short_default + debt_long_default
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write("**自己（株主）資本コスト**")
                    cost_of_equity = st.number_input("Re (%)", value=14.58, step=0.01, key="wacc_re")
                with col2:
                    st.write("**有利子負債コスト**")
                    cost_of_debt = st.number_input("Rd (%)", value=2.17, step=0.01, key="wacc_rd")
                with col3:
                    st.write("**資本構成（帳簿価額）**")
                    book_equity = st.number_input("自己資本 E (百万円)", value=equity_default, min_value=0.0, key="wacc_equity", help="最新の決算書（BS）の純資産")
                    book_debt = st.number_input("有利子負債 D (百万円)", value=total_debt_default, min_value=0.0, key="wacc_debt", help="短期借入金 + 長期借入金")
                with col4:
                    st.write("**法人税率**")
                    tax_rate_wacc = st.number_input("Tc (%)", value=30.0, min_value=0.0, max_value=100.0, step=0.5, key="wacc_tc")
                
                if st.button("▶️ WACC（割引率）を計算する", key="wacc_calc_btn", type="secondary"):
                    # Calculate WACC (入力は百万円単位なので円に変換)
                    E = float(book_equity) * 1_000_000
                    D = float(book_debt) * 1_000_000
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
                        st.markdown(f"<div style='font-size: 0.875rem; color: rgb(49, 51, 63);'>計算されたWACC</div><div style='font-size: 2.25rem; font-weight: 600;'>{wacc*100:.2f}<span style='font-size: 0.875rem; font-weight: 400;'>%</span></div>", unsafe_allow_html=True)
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
                    st.markdown("<h3>③ Calculate Terminal Value <span style='font-size: 0.7em;'>（残存価値の算出）</span></h3>", unsafe_allow_html=True)
                    st.write("TV = FCF_last × (1 + g) / (WACC − g)")
                    
                    wacc_tv = st.session_state['wacc_calculated']
                    fcf_plan_data = st.session_state.get('fcf_plan', pd.DataFrame())
                    
                    # 年度選択のための準備
                    available_periods = []
                    period_to_fcf = {}
                    
                    if not fcf_plan_data.empty:
                        period_col = [c for c in fcf_plan_data.columns if c.lower() == 'period']
                        fcf_col = [c for c in fcf_plan_data.columns if c.upper() == 'FCF']
                        
                        if period_col and fcf_col:
                            for idx, row in fcf_plan_data.iterrows():
                                period_val = row[period_col[0]]
                                fcf_val = pd.to_numeric(row[fcf_col[0]], errors='coerce')
                                if pd.notna(fcf_val):
                                    available_periods.append(str(period_val))
                                    period_to_fcf[str(period_val)] = fcf_val
                    
                    col_tv1, col_tv2 = st.columns(2)
                    with col_tv1:
                        if available_periods:
                            # フォームを使用して、値の変更時の自動再実行を防止
                            with st.form(key="tv_calculation_form"):
                                # 開始年度選択ドロップダウン
                                # 保存された開始年度を使用（なければ最初の年度）
                                saved_start = st.session_state.get('tv_display_start')
                                if saved_start and saved_start in available_periods:
                                    default_start_idx = available_periods.index(saved_start)
                                else:
                                    default_start_idx = 0
                                
                                start_period = st.selectbox(
                                    "開始年度を選択",
                                    options=available_periods,
                                    index=default_start_idx,
                                    help="予測期間の開始年度を選択してください"
                                )
                                
                                # 最終年度選択ドロップダウン（開始年度以降のみ）
                                start_index = available_periods.index(start_period)
                                end_period_options = available_periods[start_index:]
                                
                                # 保存された最終年度を使用（なければ最後の年度）
                                saved_end = st.session_state.get('tv_display_end')
                                
                                # 保存された最終年度が現在のオプションリストに存在する場合はそれを使用
                                if saved_end and saved_end in end_period_options:
                                    default_end_idx = end_period_options.index(saved_end)
                                else:
                                    default_end_idx = len(end_period_options) - 1
                                
                                end_period = st.selectbox(
                                    "最終年度を選択",
                                    options=end_period_options,
                                    index=default_end_idx,
                                    help="Terminal Value計算に使用する最終年度を選択してください"
                                )
                                
                                # デフォルト成長率を設定
                                default_g = st.session_state.get('tv_g_used', 0.01) * 100 if 'tv_g_used' in st.session_state else 1.0
                                g_input = st.number_input("永続成長率 g (%)", value=default_g, step=0.1)
                                
                                st.write("")
                                submit_button = st.form_submit_button("▶️ Terminal Valueを計算する", type="secondary")
                            
                            # 計算済みの値がある場合は表示（ボタン押下後のみ更新）
                            if 'tv_fcf_last' in st.session_state and 'tv_forecast_years' in st.session_state:
                                display_fcf = st.session_state['tv_fcf_last']
                                display_years = st.session_state['tv_forecast_years']
                                display_start = st.session_state.get('tv_display_start', start_period)
                                display_end = st.session_state.get('tv_display_end', end_period)
                            else:
                                # 初期表示用（ボタン押下前）
                                end_index = available_periods.index(end_period)
                                display_fcf = period_to_fcf[end_period]
                                display_years = end_index - start_index + 1
                                display_start = start_period
                                display_end = end_period
                            
                            st.markdown(f"**最終年FCF**: {display_fcf:,.0f} <span style='font-size: 0.8em;'>百万円</span>", unsafe_allow_html=True)
                            st.write(f"**予測期間**: {display_years} 年 ({display_start} 〜 {display_end})")
                            
                            # フォーム送信時の処理
                            if submit_button:
                                # まず最初に選択された値をsession_stateに保存
                                st.session_state['tv_display_start'] = start_period
                                st.session_state['tv_display_end'] = end_period
                                
                                g = g_input / 100.0
                                start_idx = available_periods.index(start_period)
                                end_idx = available_periods.index(end_period)
                                fcf_last = period_to_fcf[end_period]
                                forecast_years = end_idx - start_idx + 1
                                
                                # 計算用の値を保存
                                st.session_state['tv_fcf_last'] = fcf_last
                                st.session_state['tv_forecast_years'] = forecast_years
                                
                                if wacc_tv <= g:
                                    st.error("WACC > g である必要があります（現在のWACC≤g）")
                                elif fcf_last <= 0:
                                    st.error("最終年FCFが正の値である必要があります")
                                else:
                                    tv = (fcf_last * (1 + g)) / (wacc_tv - g)
                                    pv_tv = tv / (1 + wacc_tv) ** forecast_years
                                    
                                    st.session_state['terminal_value'] = tv
                                    st.session_state['pv_terminal_value'] = pv_tv
                                    st.session_state['forecast_years'] = forecast_years
                                    st.session_state['tv_g_used'] = g
                                    st.rerun()
                        else:
                            st.warning("FCFデータが見つかりません")
                    
                    with col_tv2:
                        pass
                
                if 'pv_terminal_value' in st.session_state and 'terminal_value' in st.session_state:
                    tv = st.session_state['terminal_value']
                    pv_tv = st.session_state['pv_terminal_value']
                    fcf_last_display = st.session_state.get('tv_fcf_last', 0)
                    forecast_years_display = st.session_state.get('tv_forecast_years', 0)
                    g_used = st.session_state.get('tv_g_used', 0)
                    
                    col_tv_res1, col_tv_res2 = st.columns(2)
                    with col_tv_res1:
                        st.markdown(f"<div style='font-size: 0.875rem; color: rgb(49, 51, 63);'>Terminal Value</div><div style='font-size: 2.25rem; font-weight: 600;'>{tv:,.0f} <span style='font-size: 0.875rem; font-weight: 400;'>百万円</span></div>", unsafe_allow_html=True)
                    with col_tv_res2:
                        st.markdown(f"<div style='font-size: 0.875rem; color: rgb(49, 51, 63);'>Terminal Value (PV)</div><div style='font-size: 2.25rem; font-weight: 600;'>{pv_tv:,.0f} <span style='font-size: 0.875rem; font-weight: 400;'>百万円</span></div>", unsafe_allow_html=True)
                    st.write("**計算式:**")
                    st.markdown(f"TV = {fcf_last_display:,.0f} × (1 + {g_used*100:.2f}%) / ({wacc_tv*100:.2f}% − {g_used*100:.2f}%) = {tv:,.0f} <span style='font-size: 0.8em;'>百万円</span>", unsafe_allow_html=True)
                    st.markdown(f"PV(TV) = {tv:,.0f} / (1 + {wacc_tv*100:.2f}%)^{forecast_years_display} = {pv_tv:,.0f} <span style='font-size: 0.8em;'>百万円</span>", unsafe_allow_html=True)
                
                # 事業価値・株式価値セクション（Terminal Value計算後に表示）
                if 'pv_terminal_value' in st.session_state and 'wacc_calculated' in st.session_state:
                    st.write("---")
                    st.markdown("<h3>④ Calculate Enterprise & Equity Value <span style='font-size: 0.7em;'>（事業価値・株主価値の算出）</span></h3>", unsafe_allow_html=True)
                    
                    fcf_plan_data = st.session_state.get('fcf_plan', pd.DataFrame())
                    wacc_ev = st.session_state['wacc_calculated']
                    pv_tv = st.session_state['pv_terminal_value']
                    
                    # Terminal Valueで選択された期間の情報を取得
                    tv_display_start = st.session_state.get('tv_display_start', None)
                    tv_display_end = st.session_state.get('tv_display_end', None)
                    
                    # FCF予測期間のPV計算
                    if not fcf_plan_data.empty:
                        fcf_col = [c for c in fcf_plan_data.columns if c.upper() == 'FCF']
                        period_col = [c for c in fcf_plan_data.columns if c.lower() == 'period']
                        
                        if fcf_col:
                            fcf_values = pd.to_numeric(fcf_plan_data[fcf_col[0]], errors='coerce').dropna()
                            
                            # Terminal Valueで選択された期間のインデックスを取得
                            start_idx = 0
                            end_idx = len(fcf_values) - 1
                            
                            if period_col and tv_display_start and tv_display_end:
                                periods = fcf_plan_data[period_col[0]].astype(str).tolist()
                                try:
                                    start_idx = periods.index(tv_display_start)
                                    end_idx = periods.index(tv_display_end)
                                except ValueError:
                                    # 期間が見つからない場合は全期間を使用
                                    pass
                            
                            # 選択された期間のみを抽出
                            fcf_values_selected = fcf_values.iloc[start_idx:end_idx+1]
                            
                            # Mid-year convention: 各年のPV計算（選択された期間のみ）
                            pv_fcf_list = []
                            for i, fcf in enumerate(fcf_values_selected, start=1):
                                pv = fcf / ((1 + wacc_ev) ** (i - 0.5))  # Mid-year convention
                                pv_fcf_list.append(pv)
                            
                            pv_fcf_sum = sum(pv_fcf_list)
                        else:
                            pv_fcf_sum = 0.0
                            pv_fcf_list = []
                            fcf_values_selected = pd.Series()
                            start_idx = 0
                    else:
                        pv_fcf_sum = 0.0
                        pv_fcf_list = []
                        fcf_values_selected = pd.Series()
                        start_idx = 0
                    
                    # 事業価値 = PV(FCF予測) + PV(TV)
                    enterprise_value = pv_fcf_sum + pv_tv
                    
                    col_ev1, col_ev2, col_ev3 = st.columns(3)
                    with col_ev1:
                        st.markdown(f"<div style='font-size: 0.875rem; color: rgb(49, 51, 63);'>PV(FCF予測期間)</div><div style='font-size: 2.25rem; font-weight: 600;'>{pv_fcf_sum:,.0f} <span style='font-size: 0.875rem; font-weight: 400;'>百万円</span></div>", unsafe_allow_html=True)
                    with col_ev2:
                        st.markdown(f"<div style='font-size: 0.875rem; color: rgb(49, 51, 63);'>PV(Terminal Value)</div><div style='font-size: 2.25rem; font-weight: 600;'>{pv_tv:,.0f} <span style='font-size: 0.875rem; font-weight: 400;'>百万円</span></div>", unsafe_allow_html=True)
                    with col_ev3:
                        st.markdown(f"<div style='font-size: 0.875rem; color: rgb(49, 51, 63);'>事業価値 (EV)</div><div style='font-size: 2.25rem; font-weight: 600;'>{enterprise_value:,.0f} <span style='font-size: 0.875rem; font-weight: 400;'>百万円</span></div>", unsafe_allow_html=True)
                    
                    st.write("**FCF予測期間の内訳（mid-year convention）:**")
                    if pv_fcf_list and len(fcf_values_selected) > 0:
                        fcf_detail_data = []
                        # 選択された期間のFCF値を使用
                        period_labels = []
                        if period_col and tv_display_start and tv_display_end:
                            periods = fcf_plan_data[period_col[0]].astype(str).tolist()
                            period_labels = periods[start_idx:start_idx+len(fcf_values_selected)]
                        
                        for i, (fcf, pv) in enumerate(zip(fcf_values_selected, pv_fcf_list), start=1):
                            year_label = period_labels[i-1] if i-1 < len(period_labels) else f"Year {i}"
                            fcf_detail_data.append({
                                "年": year_label,
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
                    
                    # 有利子負債と現金の値を先に取得（百万円単位に変換）
                    debt_short = standardized_bs.get('debt_short', 0) or 0
                    debt_long = standardized_bs.get('debt_long', 0) or 0
                    total_debt = (debt_short + debt_long) / 1_000_000
                    cash = (standardized_bs.get('cash', 0) or 0) / 1_000_000
                    
                    col_eq1, col_eq2, col_eq3 = st.columns(3)
                    with col_eq1:
                        st.number_input("有利子負債 (百万円)", value=float(total_debt), 
                                        format="%.2f", key="debt_for_equity")
                        
                        st.write("")
                        if st.button("▶️ 株式価値を計算する", key="equity_value_calc_btn", type="secondary"):
                            debt_input = st.session_state.get('debt_for_equity', total_debt)  # 百万円単位
                            cash_input = st.session_state.get('cash_for_equity', cash)  # 百万円単位
                            net_debt = debt_input - cash_input  # 百万円単位
                            equity_value = enterprise_value - net_debt  # 百万円単位
                            
                            st.session_state['net_debt'] = net_debt
                            st.session_state['equity_value'] = equity_value
                    
                    with col_eq2:
                        st.number_input("現金及び現金同等物 (百万円)", value=float(cash), 
                                        format="%.2f", key="cash_for_equity")
                    
                    with col_eq3:
                        pass
                
                if 'equity_value' in st.session_state and 'net_debt' in st.session_state:
                    net_debt = st.session_state['net_debt']
                    equity_value = st.session_state['equity_value']
                    debt_input = st.session_state.get('debt_for_equity', (standardized_bs.get('debt_short', 0) + standardized_bs.get('debt_long', 0)) / 1_000_000)
                    cash_input = st.session_state.get('cash_for_equity', standardized_bs.get('cash', 0) / 1_000_000)
                    col_eq_res1, col_eq_res2 = st.columns(2)
                    with col_eq_res1:
                        st.markdown(f"<div style='font-size: 0.875rem; color: rgb(49, 51, 63);'>純有利子負債</div><div style='font-size: 2.25rem; font-weight: 600;'>{net_debt:,.0f} <span style='font-size: 0.875rem; font-weight: 400;'>百万円</span></div>", unsafe_allow_html=True)
                    with col_eq_res2:
                        st.markdown(f"<div style='font-size: 0.875rem; color: rgb(49, 51, 63);'>株式価値</div><div style='font-size: 2.25rem; font-weight: 600;'>{equity_value:,.0f} <span style='font-size: 0.875rem; font-weight: 400;'>百万円</span></div>", unsafe_allow_html=True)
                    st.write("**計算式:**")
                    st.write(f"純有利子負債 = {debt_input:,.0f} − {cash_input:,.0f} = {net_debt:,.0f} 円")
                    st.write(f"株式価値 = {enterprise_value:,.0f} − {net_debt:,.0f} = {equity_value:,.0f} 円")
                
                # 1株当たり価値計算セクション（株式価値計算後に表示）
                if 'equity_value' in st.session_state:
                    st.write("---")
                    st.markdown("<h3>⑤ Implied Share Price <span style='font-size: 0.7em;'>（理論株価）</span></h3>", unsafe_allow_html=True)
                    st.write("1株当たり価値 = 株式価値 ÷ 発行済株式数")
                    
                    equity_value_for_share = st.session_state['equity_value']
                    standardized_bs = st.session_state.get('standardized_bs', {})
                    shares_default = standardized_bs.get('shares_outstanding', 0) or 0
                    
                    col_sh1, col_sh2 = st.columns(2)
                    with col_sh1:
                        st.markdown(f"**株式価値**: {equity_value_for_share:,.0f} <span style='font-size: 0.8em;'>百万円</span>", unsafe_allow_html=True)
                        
                        # フォームを使用して、値の変更時の自動再実行を防止
                        with st.form(key="share_price_form"):
                            # デフォルト値を設定（保存された値があればそれを使用）
                            default_shares = st.session_state.get('shares_used', shares_default) if 'shares_used' in st.session_state else shares_default
                            
                            shares_input = st.number_input("発行済株式数 (株)", 
                                                           value=float(default_shares), 
                                                           format="%.0f",
                                                           help="完全希薄化後の発行済株式数を入力してください")
                            
                            st.write("")
                            submit_button = st.form_submit_button("▶️ 理論株価を計算する", type="secondary")
                        
                        # フォーム送信時の処理
                        if submit_button:
                            if shares_input <= 0:
                                st.error("発行済株式数は正の値である必要があります")
                            else:
                                # equity_value_for_shareは百万円単位なので、円に変換してから計算
                                price_per_share = (equity_value_for_share * 1_000_000) / shares_input
                                st.session_state['price_per_share'] = price_per_share
                                st.session_state['shares_used'] = shares_input
                                st.rerun()
                        
                        # 計算済みの結果を表示
                        if 'price_per_share' in st.session_state:
                            price_per_share = st.session_state['price_per_share']
                            shares_used = st.session_state['shares_used']
                            
                            st.markdown(f"<div style='font-size: 0.875rem; color: rgb(49, 51, 63);'>1株当たり価値</div><div style='font-size: 2.25rem; font-weight: 600;'>{price_per_share:,.0f} <span style='font-size: 0.875rem; font-weight: 400;'>円</span></div>", unsafe_allow_html=True)
                            
                            st.write("**計算式:**")
                            st.markdown(f"1株当たり価値 = {equity_value_for_share:,.0f}<span style='font-size: 0.8em;'>百万円</span> × 1,000,000 ÷ {shares_used:,.0f}株 = {price_per_share:,.0f} <span style='font-size: 0.8em;'>円</span>", unsafe_allow_html=True)
                    
                    with col_sh2:
                        pass
