import re
import pandas as pd

FISCAL_RE = re.compile(r"^\s*\d{1,2}[/／]\d{1,2}期\s*$")  # 例: 24/11期, 25/5期（全角スラッシュ対応）


def _norm_cell(x) -> str:
    # セル内のよくある全角/不揃い文字を正規化してマッチングしやすくする
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("／", "/")
    s = s.replace("\uFF0F", "/")
    s = s.replace("\u00A0", " ")
    return s.strip()


def _find_header_row(df: pd.DataFrame) -> int | None:
    # "Unit:" がある行をヘッダ行とみなす（このサンプルに強い）
    for r in range(min(30, len(df))):
        row = df.iloc[r].astype(object).apply(_norm_cell)
        if row.str.contains("Unit:", na=False).any():
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
        "long": DataFrame (sheet, metric, period, value)
      }
    """
    df = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None)

    header_row = _find_header_row(df)
    if header_row is None:
        raise ValueError(f"Header row not found in sheet={sheet_name} (no 'Unit:')")

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
        raise ValueError(f"No fiscal-year columns like '24/11期' found in sheet={sheet_name}")

    label_col = _find_label_col(df, header_row)

    body = df.iloc[header_row + 1 :, :].copy()
    # 指定科目列の文字列抽出（NaNはNoneに）
    metrics_raw = body.iloc[:, label_col]
    metrics = metrics_raw.where(metrics_raw.notna(), None).astype(object).apply(lambda x: _norm_cell(x) if x is not None else None)
    metrics = pd.Index([m if (m is not None and m != "") else None for m in metrics])

    wide = pd.DataFrame(index=metrics)
    for pc in period_cols:
        col_series = pd.to_numeric(body.iloc[:, pc["col"]], errors="coerce")
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
    }


def list_sheet_names(xlsx_file) -> list[str]:
    # pandas でExcelFileを使うのが手軽
    xl = pd.ExcelFile(xlsx_file)
    return xl.sheet_names
