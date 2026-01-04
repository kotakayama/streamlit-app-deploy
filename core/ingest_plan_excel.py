import re
import pandas as pd

FISCAL_RE = re.compile(r"^\s*\d{2}/\d{1,2}期\s*$")  # 例: 24/11期, 25/5期


def _find_header_row(df: pd.DataFrame) -> int | None:
    # "Unit:" がある行をヘッダ行とみなす（このサンプルに強い）
    for r in range(min(30, len(df))):
        row = df.iloc[r].astype(str)
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

    # 年次列（◯◯/◯◯期）だけ拾う
    period_cols = []
    for c, h in enumerate(headers):
        if isinstance(h, str) and FISCAL_RE.match(h.strip()):
            period_cols.append({"col": c, "period": h.strip()})

    if not period_cols:
        raise ValueError(f"No fiscal-year columns like '24/11期' found in sheet={sheet_name}")

    label_col = _find_label_col(df, header_row)

    body = df.iloc[header_row + 1 :, :]
    metrics = body.iloc[:, label_col].astype(str).str.strip()
    metrics = metrics.replace({"nan": None, "": None})

    wide = pd.DataFrame(index=metrics)
    for pc in period_cols:
        wide[pc["period"]] = pd.to_numeric(body.iloc[:, pc["col"]], errors="coerce")

    wide = wide[wide.index.notna()]
    wide = wide.dropna(how="all")  # 全部NaN行は落とす

    long = (
        wide.reset_index()
        .melt(id_vars=["index"], var_name="period", value_name="value")
        .rename(columns={"index": "metric"})
        .dropna(subset=["value"])
    )
    long.insert(0, "sheet", sheet_name)
    long["unit"] = unit

    return {
        "sheet": sheet_name,
        "unit": unit,
        "header_row": header_row,
        "label_col": label_col,
        "period_cols": period_cols,
        "wide": wide,
        "long": long,
    }


def list_sheet_names(xlsx_file) -> list[str]:
    # pandas でExcelFileを使うのが手軽
    xl = pd.ExcelFile(xlsx_file)
    return xl.sheet_names
