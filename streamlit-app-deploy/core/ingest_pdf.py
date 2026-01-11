import re
from io import BytesIO
from datetime import datetime
from typing import Optional

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

from core.normalize import ALIASES


def _parse_number_like(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    s = re.sub(r"[^0-9\.,\-]", "", s)
    if not s:
        return None
    if s.count(",") > 0 and s.count(".") == 0:
        s = s.replace(",", "")
    else:
        s = s.replace(",", "")
    try:
        v = float(s)
        if negative:
            v = -v
        return v
    except Exception:
        return None


# --- テキストベースの堅牢な抽出ユーティリティ（日本語見出しに強い） ---
# △（三角）や全角カンマ、全角数字などに耐える
def parse_jpy_number(s: str) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    s = s.replace("，", ",").replace("−", "-").replace("▲", "-").replace("△", "-")
    # 全角数字を半角に変換
    full2half = str.maketrans("０１２３４５６７８９", "0123456789")
    s = s.translate(full2half)
    # 許容する文字以外を除去
    s = re.sub(r"[^\d\-,\.]", "", s)
    if s in ("", "-", ","):
        return None
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def extract_pdf_text_by_page(pdf_file) -> list[str]:
    """Return list of page-extracted text (preserves page ordering)."""
    texts = []
    with pdfplumber.open(pdf_file) as pdf:
        for p in pdf.pages:
            texts.append(p.extract_text() or "")
    return texts


def find_page_index(texts: list[str], keyword: str) -> int | None:
    for i, t in enumerate(texts):
        if keyword in t:
            return i
    return None


def extract_pl(text: str) -> dict:
    patterns = {
        "revenue": r"売上高\s+([0-9,△▲\-]+)",
        "gross_profit": r"売上総利益\s+([0-9,△▲\-]+)",
        "sga": r"販売費及び一般管理費\]?[\s　]*([0-9,△▲\-]+)",
        "operating_income": r"営業(利益|損失)\s+([0-9,△▲\-]+)",
        "ordinary_income": r"経常(利益|損失)\s+([0-9,△▲\-]+)",
        "pre_tax_income": r"税引前当期純(利益|損失)\s+([0-9,△▲\-]+)",
        "net_income": r"当期純(利益|損失)\s+([0-9,△▲\-]+)",
    }
    out = {}
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if not m:
            continue
        # 数値キャプチャが第2グループにある場合があるので検査
        val_str = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
        out[k] = parse_jpy_number(val_str)
    return out


def extract_bs(text: str) -> dict:
    patterns = {
        "current_assets": r"【流動資産】\s+([0-9,△▲\-]+)",
        "cash_and_deposits": r"現金及び預金\s+([0-9,△▲\-]+)",
        "fixed_assets": r"【固定資産】\s+([0-9,△▲\-]+)",
        "current_liabilities": r"【流動負債】\s+([0-9,△▲\-]+)",
        "short_term_debt": r"短期借入金\s+([0-9,△▲\-]+)",
        "fixed_liabilities": r"【固定負債】\s+([0-9,△▲\-]+)",
        "long_term_debt": r"長期借入金\s+([0-9,△▲\-]+)",
        "lease_liabilities": r"長期リース債務\s+([0-9,△▲\-]+)",
        "total_liabilities": r"負債の部合計\s+([0-9,△▲\-]+)",
        "total_equity": r"純資産の部合計\s+([0-9,△▲\-]+)",
        "total_assets": r"資産の部合計\s+([0-9,△▲\-]+)",
    }
    out = {}
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if not m:
            continue
        out[k] = parse_jpy_number(m.group(1))
    return out


def extract_shares(text: str) -> dict:
    def parse_shares(s):
        if not s:
            return None
        s = s.replace("，", ",").replace(",", "")
        # 全角→半角
        full2half = str.maketrans("０１２３４５６７８９", "0123456789")
        s = s.translate(full2half)
        s = re.sub(r"[^\d]", "", s)
        return int(s) if s else None

    patterns = {
        "shares_common": r"普通株式\s*[:：]\s*([0-9０-９,，]+)株",
        "shares_class_kou": r"甲種種類株式\s*[:：]\s*([0-9０-９,，]+)株",
        "shares_class_a": r"Ａ種種類株式\s*[:：]\s*([0-9０-９,，]+)株",
    }
    out = {}
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[k] = parse_shares(m.group(1))
    vals = [v for v in out.values() if isinstance(v, int)]
    if vals:
        out["shares_total"] = sum(vals)
    return out


def ingest_financials_from_pdf(pdf_file) -> tuple[dict, dict]:
    """
    returns: (raw_data, meta)
    raw_data: dict with 'pl','bs','shares'
    meta: page indices used
    """
    texts = extract_pdf_text_by_page(pdf_file)

    idx_bs = find_page_index(texts, "貸借対照表")
    idx_pl = find_page_index(texts, "損益計算書")

    # 株式数は注記ページにあることが多いので、後半ページから総当りで拾う
    shares = {}
    shares_page = None
    for i in reversed(range(len(texts))):
        s = extract_shares(texts[i])
        if s:
            shares = s
            shares_page = i
            break

    bs = extract_bs(texts[idx_bs]) if idx_bs is not None else {}
    pl = extract_pl(texts[idx_pl]) if idx_pl is not None else {}

    raw = {"pl": pl, "bs": bs, "shares": shares}
    meta = {"bs_page": idx_bs, "pl_page": idx_pl, "shares_page": shares_page}
    return raw, meta


def parse_pdf_financials(uploaded_file) -> dict:
    """Extract tables from a PDF and attempt to map them to PL/BS/CF.
    Returns a dict with keys:
      - mapping: {"PL": df, "BS": df, "CF": df} for confident matches
      - candidates: list of {"id": int, "page": int, "table_index": int, "df": DataFrame, "preview": str}
    """
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
        import logging, os, traceback
        logging.exception("Error during PDF parsing")
        if PdfminerException is not None and isinstance(e, PdfminerException):
            raise ValueError("PDFの解析に失敗しました（内部パーサーでエラー）。このPDFは暗号化、スキャン画像、または特殊フォーマットの可能性があります。PDFを別名で保存して再アップロードするか、Excelやテキストに変換して再度アップロードしてください。") from e
        try:
            log_dir = os.path.join(os.path.dirname(__file__), "..", "tmp_pdf_logs")
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            err_id = f"pdferr_{ts}"
            pdf_path = os.path.join(log_dir, f"{err_id}.pdf")
            log_path = os.path.join(log_dir, f"{err_id}.log")
            with open(pdf_path, "wb") as fh:
                fh.write(data)
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write("Traceback (most recent call last):\n")
                fh.write(traceback.format_exc())
                fh.write("\nException repr: ")
                fh.write(repr(e))
        except Exception:
            logging.exception("Failed to write PDF parsing diagnostics")
            err_id = "unknown"
        raise ValueError(f"PDF解析中に予期しないエラーが発生しました（error id: {err_id}）。詳細はサーバ側ログまたは tmp_pdf_logs/{err_id}.log を確認してください。") from e

    candidates = []
    for idx, (p_idx, t_idx, df) in enumerate(dfs):
        year_cols = [c for c in df.columns if re.search(r"FY\d{4}", str(c)) or re.search(r"\b\d{4}\b", str(c))]
        if not year_cols:
            numeric_like = []
            for i, c in enumerate(list(df.columns)[1:], start=1):
                col_series = df.iloc[:, i].astype(str).fillna("")
                if hasattr(col_series, 'str'):
                    num_count = col_series.str.contains(r"\d").sum()
                else:
                    num_count = sum(1 for v in col_series if re.search(r"\d", str(v)))
                if num_count >= max(1, int(0.4 * len(col_series))):
                    numeric_like.append(c)
            year_cols = numeric_like
        for c in year_cols:
            try:
                col_idx = df.columns.get_loc(c)
            except Exception:
                continue
            col_ser = df.iloc[:, col_idx].astype(str).fillna("")
            if hasattr(col_ser, 'str'):
                cleaned = col_ser.str.replace(r"[^0-9.\-]", "", regex=True)
            else:
                cleaned = col_ser.apply(lambda x: re.sub(r"[^0-9.\-]", "", str(x)))
            df.iloc[:, col_idx] = pd.to_numeric(cleaned, errors="coerce")
        preview = "\n".join(df.iloc[:, 0].astype(str).head(10).tolist())
        candidates.append({"id": idx, "page": p_idx, "table_index": t_idx, "df": df, "preview": preview})

    mapping = {}
    for c in candidates:
        df = c['df']
        text_sample = " ".join(df.iloc[:, 0].astype(str).head(40).tolist()).lower()
        pl_score = sum(bool(re.search(p, text_sample)) for p in ALIASES["revenue"] + ALIASES["ebitda"] + ALIASES["operating_income"] + ALIASES["net_income"])
        bs_score = sum(bool(re.search(p, text_sample)) for p in ALIASES["cash"] + ALIASES["debt"] + ALIASES["equity"] + ALIASES["shares"])
        if pl_score > 0 and "PL" not in mapping:
            mapping["PL"] = {"df": df, "score": pl_score, "page": c['page'], "table_index": c['table_index']}
        if bs_score > 0 and "BS" not in mapping:
            mapping["BS"] = {"df": df, "score": bs_score, "page": c['page'], "table_index": c['table_index']}

    return {"mapping": mapping, "candidates": candidates, "raw_bytes": data}


def ocr_extract_metrics_from_bytes(data: bytes, target_year: str) -> dict:
    if pytesseract is None:
        raise ValueError("pytesseract is not available. Install pytesseract and system Tesseract to enable OCR.")
    metrics = {}
    page_texts = []
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            for p_idx, page in enumerate(pdf.pages):
                try:
                    img = page.to_image(resolution=200).original
                except Exception:
                    try:
                        img = page.to_image().original
                    except Exception:
                        continue
                try:
                    txt = pytesseract.image_to_string(img, lang='jpn+eng') if pytesseract is not None else ''
                except Exception as e:
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
                    for metric_key, patterns in ALIASES.items():
                        if any(re.search(p, low) for p in patterns):
                            m = re.search(r"[-\(]?[0-9][0-9\.,\)\(\-]+", t)
                            if m:
                                num = _parse_number_like(m.group(0))
                                if num is not None and metric_key not in metrics:
                                    metrics[metric_key] = (num, f"page {p_idx}: {t}")
    except Exception as e:
        import logging
        logging.exception("OCR extraction failed")
        raise

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