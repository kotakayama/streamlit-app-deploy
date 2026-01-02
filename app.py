import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List

import pandas as pd
import pdfplumber
# pdfminer exception class (may not be available in all pdfplumber versions/environments)
try:
    from pdfplumber.utils import PdfminerException
except Exception:
    PdfminerException = None
# pytesseract (optional runtime dependency) - check availability at import time
try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
except Exception:
    pytesseract = None
    TesseractNotFoundError = None
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

def parse_pdf_financials(uploaded_file) -> dict:
    """Extract tables from a PDF and attempt to map them to PL/BS/CF.
    Returns a dict with keys:
      - mapping: {"PL": df, "BS": df, "CF": df} for confident matches
      - candidates: list of {"id": int, "page": int, "table_index": int, "df": DataFrame, "preview": str}
    """
    from io import BytesIO
    data = uploaded_file.read()
    dfs = []
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            for p_idx, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for t_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    header = table[0]
                    rows = table[1:]
                    try:
                        df = pd.DataFrame(rows, columns=header)
                    except Exception:
                        df = pd.DataFrame(rows)
                    df = df.rename(columns={df.columns[0]: "Item"})
                    dfs.append((p_idx, t_idx, df))
    except Exception as e:
        import logging
        logging.exception("Error during PDF parsing")
        # If pdfminer-specific exception class is available, check instance
        if PdfminerException is not None and isinstance(e, PdfminerException):
            raise ValueError("PDFの解析に失敗しました（内部パーサーでエラー）。このPDFは暗号化、スキャン画像、または特殊フォーマットの可能性があります。PDFを別名で保存して再アップロードするか、Excelやテキストに変換して再度アップロードしてください。") from e
        raise ValueError("PDF解析中に予期しないエラーが発生しました。別のPDFをお試しください。") from e

    candidates = []
    for idx, (p_idx, t_idx, df) in enumerate(dfs):
        # Detect year-like columns or numeric columns that look like financial columns
        year_cols = [c for c in df.columns if re.search(r"FY\d{4}", str(c)) or re.search(r"\b\d{4}\b", str(c))]
        # If no explicit year columns, consider columns with many numeric-like cells
        if not year_cols:
            numeric_like = []
            for i, c in enumerate(list(df.columns)[1:], start=1):
                col_series = df.iloc[:, i].astype(str).fillna("")
                # Some pages yield DataFrame slices; ensure we have a Series
                if hasattr(col_series, 'str'):
                    num_count = col_series.str.contains(r"\d").sum()
                else:
                    num_count = sum(1 for v in col_series if re.search(r"\d", str(v)))
                if num_count >= max(1, int(0.4 * len(col_series))):
                    numeric_like.append(c)
            year_cols = numeric_like
        # Coerce detected numeric columns (use positional access to avoid DataFrame-like columns)
        for c in year_cols:
            try:
                col_idx = df.columns.get_loc(c)
            except Exception:
                # If get_loc fails (e.g., unexpected column label), skip coercion for this entry
                continue
            # Work with positional Series to avoid .str issues when column labels are tuple/multiindex
            col_ser = df.iloc[:, col_idx].astype(str).fillna("")
            if hasattr(col_ser, 'str'):
                cleaned = col_ser.str.replace(r"[^0-9.\-]", "", regex=True)
            else:
                cleaned = col_ser.apply(lambda x: re.sub(r"[^0-9.\-]", "", str(x)))
            df.iloc[:, col_idx] = pd.to_numeric(cleaned, errors="coerce")
        # First column may be duplicated or ambiguous; use positional index for preview
        preview = "\n".join(df.iloc[:, 0].astype(str).head(10).tolist())
        candidates.append({"id": idx, "page": p_idx, "table_index": t_idx, "df": df, "preview": preview})

    mapping = {}
    # Prioritize PL (損益計算書) and BS (貸借対照表) detection only
    for c in candidates:
        df = c['df']
        text_sample = " ".join(df.iloc[:, 0].astype(str).head(40).tolist()).lower()
        pl_score = sum(bool(re.search(p, text_sample)) for p in ALIASES["revenue"] + ALIASES["ebitda"] + ALIASES["operating_income"] + ALIASES["net_income"])
        bs_score = sum(bool(re.search(p, text_sample)) for p in ALIASES["cash"] + ALIASES["debt"] + ALIASES["equity"] + ALIASES["shares"])
        scores = {"PL": pl_score, "BS": bs_score}
        # Require at least one matching keyword to consider it a candidate
        if pl_score > 0 and "PL" not in mapping:
            mapping["PL"] = {"df": df, "score": pl_score, "page": c['page'], "table_index": c['table_index']}
        if bs_score > 0 and "BS" not in mapping:
            mapping["BS"] = {"df": df, "score": bs_score, "page": c['page'], "table_index": c['table_index']}

    return {"mapping": mapping, "candidates": candidates, "raw_bytes": data}

# ----------------------------
# OCR helpers (fallback for scanned PDFs)
# ----------------------------

def _parse_number_like(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # Handle parentheses as negatives
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    # Remove currency symbols and spaces
    s = re.sub(r"[^0-9\.,\-]", "", s)
    if not s:
        return None
    # If comma used as thousand separator, remove it
    if s.count(",") > 0 and s.count(".") == 0:
        s = s.replace(",", "")
    else:
        # remove commas, keep dot as decimal
        s = s.replace(",", "")
    try:
        v = float(s)
        if negative:
            v = -v
        return v
    except Exception:
        return None


def ocr_extract_metrics_from_bytes(data: bytes, target_year: str) -> dict:
    """Use OCR to extract metric-like lines from PDF pages and build synthetic PL/BS DataFrames.
    Returns dict with possible keys: 'pl_df', 'bs_df', 'metrics' where metrics is {metric_key: (value, detail)}.
    """
    if pytesseract is None:
        raise ValueError("pytesseract is not available. Install pytesseract and system Tesseract to enable OCR.")

    from io import BytesIO
    metrics = {}
    page_texts = []
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            for p_idx, page in enumerate(pdf.pages):
                try:
                    img = page.to_image(resolution=200).original
                except Exception:
                    # Fallback: extract a rasterized image of the whole page if available
                    try:
                        img = page.to_image().original
                    except Exception:
                        continue
                try:
                    txt = pytesseract.image_to_string(img, lang='jpn+eng') if pytesseract is not None else ''
                except Exception as e:
                    # Reraise Tesseract not found specifically to allow the caller to handle it
                    if TesseractNotFoundError is not None and isinstance(e, TesseractNotFoundError):
                        raise
                    continue
                if not txt:
                    continue
                page_texts.append((p_idx, txt))
                for line in txt.splitlines():
                    t = line.strip()
                    if not t:
                        continue
                    low = t.lower()
                    # search for metric labels
                    for metric_key, patterns in ALIASES.items():
                        if any(re.search(p, low) for p in patterns):
                            # find first numeric token in the line
                            m = re.search(r"[-\(]?[0-9][0-9\.,\)\(\-]+", t)
                            if m:
                                num = _parse_number_like(m.group(0))
                                if num is not None and metric_key not in metrics:
                                    metrics[metric_key] = (num, f"page {p_idx}: {t}")
    except Exception as e:
        import logging
        logging.exception("OCR extraction failed")
        raise

    # Build synthetic DataFrames
    pl_rows = []
    bs_rows = []
    for k in ['revenue', 'ebitda', 'operating_income', 'net_income']:
        if k in metrics:
            label_used = metrics[k][1].split(":", 1)[1].strip() if ':' in metrics[k][1] else k
            pl_rows.append({'Item': label_used, target_year: metrics[k][0]})
    for k in ['cash', 'debt', 'equity', 'shares']:
        if k in metrics:
            label_used = metrics[k][1].split(":", 1)[1].strip() if ':' in metrics[k][1] else k
            bs_rows.append({'Item': label_used, target_year: metrics[k][0]})

    res = {'metrics': metrics}
    if pl_rows:
        res['pl_df'] = pd.DataFrame(pl_rows)
    if bs_rows:
        res['bs_df'] = pd.DataFrame(bs_rows)
    res['raw_texts'] = page_texts
    return res

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Valuation App", layout="wide")
st.title("Valuation App")

st.caption("v1: Web取得なし（手入力/CSVで代替）。v2で科目マッピングをLLM化、v3でWeb抽出を追加。")

colA, colB = st.columns([1, 2])

with colA:
    company = st.text_input("対象企業", value="キッチハイク")
    target_year = st.text_input("対象年度（FYxxxx）", value="FY2024")
    currency = st.selectbox("通貨", ["JPY", "USD", "EUR"], index=0)
    unit = st.selectbox("数値単位（Excelと揃える）", ["円", "千円", "百万円", "億円"], index=2)

with colB:
    st.subheader("① 決算書アップロード（Excel または PDF）")
    f = st.file_uploader("PL/BS/CFシートを含む .xlsx または 決算書.pdf", type=["xlsx","pdf"])

st.divider()

prov_logs: List[Provenance] = []
financials = {}

if f is not None:
    is_pdf = f.name.lower().endswith('.pdf') if getattr(f, 'name', None) else False
    if is_pdf:
        st.info('PDF をアップロードしました。自動抽出を試みます。解析に失敗した場合は手動で入力してください。')
        try:
            pdf_sheets = parse_pdf_financials(f)
        except Exception as e:
            import logging
            logging.exception("PDF parsing failed in app flow")
            st.error(f"PDF解析に失敗しました: {e}")
            st.info("対処例: PDFを別名で保存して再アップロード、または Excel に変換してアップロードしてください。詳しいエラーはサーバーログに記録されています。")
            st.stop()

        pdf_result = parse_pdf_financials(f)
        mapping = pdf_result.get('mapping', {})
        candidates = pdf_result.get('candidates', [])
        used_ocr = False  # track whether OCR fallback provided data
        # For PDFs, we require only PL and BS for now
        missing = [s for s in ["PL", "BS"] if s not in mapping]

        # If mapping is incomplete, try OCR fallback when available
        if missing and pytesseract is not None and pdf_result.get('raw_bytes'):
            st.info('テキスト抽出に失敗しました。OCRで画像から数値を抽出します（Tesseractが必要）。')
            try:
                ocr_res = ocr_extract_metrics_from_bytes(pdf_result.get('raw_bytes'), target_year)
                if ocr_res.get('pl_df') is not None and 'PL' not in mapping:
                    mapping['PL'] = {'df': ocr_res['pl_df'], 'score': 0, 'page': None, 'table_index': None}
                    used_ocr = True
                if ocr_res.get('bs_df') is not None and 'BS' not in mapping:
                    mapping['BS'] = {'df': ocr_res['bs_df'], 'score': 0, 'page': None, 'table_index': None}
                    used_ocr = True
                # Recompute missing after OCR
                missing = [s for s in ["PL", "BS"] if s not in mapping]
                if not missing:
                    st.success('OCRでPL/BSを抽出しました。結果を確認してください。')
                    # Show OCR details
                    if ocr_res.get('metrics'):
                        st.subheader('OCR抽出（検出された項目）')
                        ocr_rows = []
                        for k, (v, d) in ocr_res.get('metrics', {}).items():
                            ocr_rows.append({'metric_key': k, 'value': v, 'detail': d})
                        st.dataframe(pd.DataFrame(ocr_rows))
                    if ocr_res.get('raw_texts'):
                        with st.expander('OCR元テキスト（ページ別）'):
                            for p_idx, txt in ocr_res.get('raw_texts'):
                                st.markdown(f'**Page {p_idx}**')
                                st.text(txt)
            except TesseractNotFoundError:
                st.warning('システムにTesseractが見つかりません。OCRを使うにはTesseractをインストールしてください: https://github.com/tesseract-ocr/tesseract')
            except Exception as e:
                st.error(f'OCR実行中にエラーが発生しました: {e}')

        if missing:
            st.warning(f"PDFからPL/BSが自動検出できませんでした: {missing}。候補テーブルを表示します。手動で確認・選択してください。")
            st.subheader("検出されたテーブル候補（PL/BS 優先）")
            for c in candidates:
                with st.expander(f"Table {c['id']} - page {c['page']}"):
                    st.dataframe(c['df'])
            manual_mapping = {}
            for s in missing:
                opts = ["(未選択)"] + [f"Table {c['id']}" for c in candidates]
                choice = st.selectbox(f"{s} に割り当てるテーブルを選択してください", opts, key=f"map_{s}")
                if choice.startswith("Table "):
                    sel_id = int(choice.split()[1])
                    sel_df = next((cd['df'] for cd in candidates if cd['id'] == sel_id), None)
                    manual_mapping[s] = sel_df
                else:
                    manual_mapping[s] = None

            pl = mapping.get('PL', {}).get('df') or manual_mapping.get('PL')
            bs = mapping.get('BS', {}).get('df') or manual_mapping.get('BS')

            if pl is None or bs is None:
                st.error("PL と BS の両方が割り当てられていません。続行するには両方を割り当ててください。")
                st.stop()
        else:
            pl = mapping.get('PL', {}).get('df')
            bs = mapping.get('BS', {}).get('df')
    else:
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

    # Provenance logs
    ts = now_iso()
    def add_prov(item, value, src, source_type="excel"):
        if value is None:
            return
        prov_logs.append(Provenance(
            item=item, value=float(value), unit=unit,
            source_type=source_type, source_detail=src or "",
            captured_at=ts
        ))

    source_type = "pdf+ocr" if (is_pdf and used_ocr) else ("pdf" if is_pdf else "excel")
    add_prov("Revenue", rev, f"PL: {rev_src}", source_type=source_type)
    add_prov("EBITDA", ebitda, f"PL: {ebitda_src}", source_type=source_type)
    add_prov("Operating Income", op, f"PL: {op_src}", source_type=source_type)
    add_prov("Net Income", ni, f"PL: {ni_src}", source_type=source_type)
    add_prov("Cash", cash, f"BS: {cash_src}", source_type=source_type)
    add_prov("Debt", debt, f"BS: {debt_src}", source_type=source_type)
    add_prov("Equity", eq, f"BS: {eq_src}", source_type=source_type)
    add_prov("Shares", shares, f"BS: {shares_src}", source_type=source_type)
    add_prov("CFO", cfo, f"CF: {cfo_src}", source_type=source_type)
    add_prov("CFI", cfi, f"CF: {cfi_src}", source_type=source_type)
    if fcf is not None:
        prov_logs.append(Provenance(
            item="FCF", value=float(fcf), unit=unit,
            source_type=source_type if fcf_src != "estimated from CFO + CFI" else "derived",
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

