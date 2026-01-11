import io
import pandas as pd


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

        workbook  = writer.book
        worksheet = workbook.add_worksheet("MTG Memo")
        writer.sheets["MTG Memo"] = worksheet
        for i, line in enumerate(memo_text.splitlines() or [""]):
            worksheet.write(i, 0, line)

    return output.getvalue()


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Write multiple DataFrames to an in-memory Excel (sheet name -> DataFrame).

    Uses `openpyxl` engine for broader compatibility with downstream tools.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            # sheet names must be <= 31 chars for Excel
            sheet_name = str(name)[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()