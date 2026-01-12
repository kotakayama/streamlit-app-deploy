import re
import pandas as pd
from datetime import datetime
import numpy as np

FISCAL_RE = re.compile(r"^\s*\d{1,2}[/／]\d{1,2}期\s*$")  # 例: 24/11期, 25/5期（全角スラッシュ対応）


def _norm_cell(x) -> str:
    # セル内のよくある全角/不揃い文字を正規化してマッチングしやすくする
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("／", "/")
    s = s.replace("\uFF0F", "/")
    s = s.replace("\u00A0", " ")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _is_date_like(val: object) -> bool:
    """Check if a cell value looks like a date or period marker"""
    if pd.isna(val):
        return False
    if isinstance(val, datetime):
        return True
    s = str(val).strip()
    if not s:
        return False
    # common date-like patterns
    if re.search(r"\d{4}[-/年]", s):
        return True
    if re.match(r"FY\s*\d{2,4}", s, flags=re.IGNORECASE):
        return True
    if re.match(r"CY\s*\d{2,4}", s, flags=re.IGNORECASE):
        return True
    if re.match(r"^\d{4}$", s):
        return True
    if FISCAL_RE.match(s):  # ◯◯/◯◯期
        return True
    # check datetime
    try:
        datetime.fromisoformat(s)
        return True
    except Exception:
        pass
    return False


def _find_header_row(df: pd.DataFrame) -> int | None:
    # "Unit:" がある行をヘッダ行とみなす（このサンプルに強い）
    for r in range(min(30, len(df))):
        row = df.iloc[r].astype(object).apply(_norm_cell)
        if row.str.contains("Unit:", na=False).any():
            return r
    
    # Fallback: 日付パターンが3つ以上ある行をヘッダー行とする
    for r in range(min(30, len(df))):
        row = df.iloc[r]
        date_like_count = sum(_is_date_like(c) for c in row)
        if date_like_count >= 3:
            return r
    
    return None


def _find_label_col(df: pd.DataFrame, header_row: int) -> int:
    # ヘッダ行の次行以降で「文字列が多い列」を科目列として推定（0〜5列から）
    body = df.iloc[header_row + 1 : header_row + 40, :6]
    best_col, best_score = 2, -1
    for c in range(body.shape[1]):
        s = body.iloc[:, c]
        score = s.apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 0 and str(x) != "nan").sum()
        if score > best_score:
            best_col, best_score = c, score
    return best_col


def _detect_section_boundaries(df: pd.DataFrame) -> list[tuple[int, int, str]]:
    """
    Detect data sections (PL/BS/CF) in the sheet. Returns list of (start_row, end_row, section_name).
    Heuristic: Look for rows where column B contains 損益, 貸借, キャッシュ, CF, PL, BS keywords.
    """
    sections = []
    section_keywords = {
        "損益計算書": ["損益", "PL"],
        "貸借対照表": ["貸借", "BS", "バランスシート"],
        "キャッシュフロー計算書": ["キャッシュ", "CF", "現金"],
    }
    
    section_start_rows = {}
    for row_idx in range(len(df)):
        b_val = str(df.iloc[row_idx, 1]) if df.shape[1] > 1 else ""
        for section_name, keywords in section_keywords.items():
            if any(kw in b_val for kw in keywords):
                section_start_rows[section_name] = row_idx
                break
    
    # Sort by start row and create (start, end) pairs
    sorted_sections = sorted(section_start_rows.items(), key=lambda x: x[1])
    for i, (section_name, start_row) in enumerate(sorted_sections):
        # End row is the next section's start - 1, or the end of the dataframe
        if i + 1 < len(sorted_sections):
            end_row = sorted_sections[i + 1][1] - 1
        else:
            end_row = len(df) - 1
        sections.append((start_row, end_row, section_name))
    
    return sections


def _parse_numeric(val) -> float:
    """Robust numeric parser for Excel cells with commas, currency, and negative markers (△, parentheses)."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    neg = False
    # Japanese negative marker '△' or '▲'
    if s.startswith("△") or s.startswith("▲"):
        neg = True
        s = s[1:].strip()
    # Parentheses negative (e.g., (123))
    if re.match(r"^\(.*\)$", s):
        neg = True
        s = s.strip("()")
    # Remove common decorations
    s = s.replace(",", "").replace("円", "").replace("¥", "").replace("%", "").replace(" ", "")
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return np.nan


def extract_yearly_table(xlsx_file, sheet_name: str) -> dict:
    """
    returns:
      {
        "sheet": str,
        "unit": str|None,
        "header_row": int,
        "label_col": int,
        "period_cols": [{"col": int, "period": str}, ...],
        "wide": DataFrame (index=metric, columns=period),
        "long": DataFrame (sheet, metric, period, value),
        "sections": [section info] or None if single section
      }
    """
    df = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None)

    # Try to detect multiple sections
    sections = _detect_section_boundaries(df)
    
    # If multiple sections, process the entire dataframe as-is
    # but note that it contains mixed content
    if not sections:
        sections = [(0, len(df) - 1, "Default")]
    
    header_row = _find_header_row(df)
    if header_row is None:
        raise ValueError(f"Header row not found in sheet={sheet_name}")

    headers = df.iloc[header_row].tolist()

    # 単位
    unit = None
    for h in headers[:10]:
        if isinstance(h, str) and "Unit:" in h:
            unit = h.strip()
            break

    # 年次列（◯◯/◯◯期）だけ拾う（見つからなければゆるめの判定も試す）
    period_cols = []
    for c, h in enumerate(headers):
        h_norm = _norm_cell(h)
        if FISCAL_RE.match(h_norm):
            period_cols.append({"col": c, "period": h_norm})
    
    if not period_cols:
        # ゆるい判定: '期' と '/' を含むヘッダを年度列とみなす
        for c, h in enumerate(headers):
            h_norm = _norm_cell(h)
            if "期" in h_norm and "/" in h_norm:
                period_cols.append({"col": c, "period": h_norm})
    
    if not period_cols:
        # さらにフォールバック: _is_date_like で判定（datetime、FY、年数値など）
        for c, h in enumerate(headers):
            if _is_date_like(h):
                # period文字列を正規化
                if isinstance(h, datetime):
                    period_str = h.strftime("%Y-%m-%d")
                else:
                    period_str = str(h).strip()
                period_cols.append({"col": c, "period": period_str})

    if not period_cols:
        raise ValueError(f"No period columns found in sheet={sheet_name}")

    label_col = _find_label_col(df, header_row)

    body = df.iloc[header_row + 1 :, :].copy()
    # 指定科目列の文字列抽出（NaNはNoneに）
    metrics_raw = body.iloc[:, label_col]
    metrics = metrics_raw.where(metrics_raw.notna(), None).astype(object).apply(lambda x: _norm_cell(x) if x is not None else None)
    metrics = pd.Index([m if (m is not None and m != "") else None for m in metrics])

    wide = pd.DataFrame(index=metrics)
    for pc in period_cols:
        col_series = body.iloc[:, pc["col"]].apply(_parse_numeric)
        wide[pc["period"]] = col_series.values

    # インデックスがNoneの行を落とす
    wide = wide[wide.index.notna()]
    # 全部NaN行は落とす
    wide = wide.dropna(how="all")

    # long 形式に展開
    tmp = wide.reset_index()
    metric_col = tmp.columns[0]
    long = tmp.melt(id_vars=[metric_col], var_name="period", value_name="value")
    long = long.rename(columns={metric_col: "metric"}).dropna(subset=["value"])
    long.insert(0, "sheet", sheet_name)
    long["unit"] = unit

    # raw preview around header for diagnostics
    start = max(0, header_row - 2)
    raw_preview = df.iloc[start : header_row + 6, :10].copy()

    return {
        "sheet": sheet_name,
        "unit": unit,
        "header_row": header_row,
        "label_col": label_col,
        "period_cols": period_cols,
        "wide": wide,
        "long": long,
        "raw_preview": raw_preview,
        "sections": sections,  # Include detected sections
    }


def list_sheet_names(xlsx_file) -> list[str]:
    # pandas でExcelFileを使うのが手軽
    xl = pd.ExcelFile(xlsx_file)
    return xl.sheet_names

def extract_yearly_table_by_section(xlsx_file, sheet_name: str) -> dict[str, dict]:
    """
    Extract yearly table separately by section (PL, BS, CF, etc).
    Returns dict with section names as keys, each containing the same structure as extract_yearly_table.
    Each section may have a different label column position.
    """
    result = extract_yearly_table(xlsx_file, sheet_name)
    sections = result.get("sections", [])
    
    if not sections:
        # No sections detected, return as single section
        return {"all": result}
    
    df = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None)
    header_row = result["header_row"]
    period_cols = result["period_cols"]
    unit = result["unit"]
    
    section_results = {}
    
    for start_row, end_row, section_name in sections:
        # For each section, extract the data
        if start_row > end_row:
            continue
        
        # All sections share the same header row
        # Extract body for this section (skip the section header line itself)
        body_start = start_row + 1
        body = df.iloc[body_start : end_row + 1, :].copy()
        if body.empty:
            continue
        
        # Detect label column for this section (best column with non-empty strings)
        label_col = _find_label_col(body, 0)  # Start from first row of body
        
        if label_col >= body.shape[1]:
            continue
        
        # 指定科目列の文字列抽出
        metrics_raw = body.iloc[:, label_col]
        metrics = metrics_raw.where(metrics_raw.notna(), None).astype(object).apply(lambda x: _norm_cell(x) if x is not None else None)
        metrics = pd.Index([m if (m is not None and m != "") else None for m in metrics])
        
        # Create wide dataframe for this section
        wide = pd.DataFrame(index=metrics)
        for pc in period_cols:
            col_idx = pc["col"]
            if col_idx < body.shape[1]:
                col_series = body.iloc[:, col_idx].apply(_parse_numeric)
                wide[pc["period"]] = col_series.values
        
        # Clean up
        wide = wide[wide.index.notna()]
        wide = wide.dropna(how="all")
        
        if wide.empty:
            continue
        
        # Convert to long format
        tmp = wide.reset_index()
        metric_col = tmp.columns[0]
        long = tmp.melt(id_vars=[metric_col], var_name="period", value_name="value")
        long = long.rename(columns={metric_col: "metric"}).dropna(subset=["value"])
        long.insert(0, "sheet", sheet_name)
        long.insert(1, "section", section_name)
        long["unit"] = unit
        
        section_results[section_name] = {
            "sheet": sheet_name,
            "section": section_name,
            "unit": unit,
            "header_row": header_row,
            "label_col": label_col,
            "period_cols": period_cols,
            "wide": wide,
            "long": long,
        }
    
    return section_results if section_results else {"all": result}