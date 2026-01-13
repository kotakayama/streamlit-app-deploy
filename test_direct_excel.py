import pandas as pd
import numpy as np

# Excelファイル直接読み込み
xlsx_file = 'streamlit-app-deploy/Model_PS_資本政策詳細.xlsx'
df_raw = pd.read_excel(xlsx_file, sheet_name="FS_年次", header=None)

# 期間列を特定（行0に日付がある）
date_row = df_raw.iloc[0]
period_cols = {}
for col_idx in range(len(date_row)):
    val = date_row.iloc[col_idx]
    if pd.notna(val) and hasattr(val, 'strftime'):
        period_str = val.strftime('%Y-%m-%d')
        period_cols[col_idx] = period_str

print("=" * 100)
print("検出された期間列:")
print("=" * 100)
for col_idx, period in period_cols.items():
    print(f"列{col_idx}: {period}")

# 営業利益の行を探す
ebit_row_idx = None
for row_idx in range(len(df_raw)):
    for col_idx in range(min(5, len(df_raw.columns))):
        val = df_raw.iloc[row_idx, col_idx]
        if pd.notna(val) and str(val).strip() == '営業利益':
            ebit_row_idx = row_idx
            break
    if ebit_row_idx is not None:
        break

print(f"\n営業利益の行: {ebit_row_idx}")

if ebit_row_idx is not None and period_cols:
    # 営業利益の値を抽出
    ebit_row = df_raw.iloc[ebit_row_idx]
    print("\n" + "=" * 100)
    print("営業利益（EBIT）の値:")
    print("=" * 100)
    for col_idx, period_str in sorted(period_cols.items()):
        val = ebit_row.iloc[col_idx]
        if pd.notna(val) and isinstance(val, (int, float)):
            nopat = val * 0.7
            print(f"{period_str}: EBIT = {val:>12.2f} 百万円, NOPAT = {nopat:>12.2f} 百万円")
