import pandas as pd
import numpy as np
from datetime import datetime

# Excelファイル直接読み込み
xlsx_file = 'streamlit-app-deploy/Model_PS_資本政策詳細.xlsx'
df_raw = pd.read_excel(xlsx_file, sheet_name="FS_年次", header=None)

# 期間列を特定（行0に日付がある）
date_row = df_raw.iloc[0]
print("=" * 100)
print("行0の内容（列15-23）:")
print("=" * 100)
for col_idx in range(15, 23):
    val = date_row.iloc[col_idx]
    print(f"列{col_idx}: {val} (型: {type(val).__name__})")

# datetime型またはTimestamp型の列を探す
period_cols = {}
for col_idx in range(len(date_row)):
    val = date_row.iloc[col_idx]
    if pd.notna(val):
        # pandas.Timestamp または datetime.datetime をチェック
        if isinstance(val, (pd.Timestamp, datetime)):
            period_str = pd.to_datetime(val).strftime('%Y-%m-%d')
            period_cols[col_idx] = period_str

print("\n" + "=" * 100)
print("検出された期間列:")
print("=" * 100)
for col_idx, period in sorted(period_cols.items()):
    print(f"列{col_idx}: {period}")

# 営業利益の行（行75）から値を取得
ebit_row_idx = 75
ebit_row = df_raw.iloc[ebit_row_idx]

print("\n" + "=" * 100)
print(f"営業利益（行{ebit_row_idx}）の各期間の値:")
print("=" * 100)
tax_rate = 0.30
for col_idx, period_str in sorted(period_cols.items()):
    val = ebit_row.iloc[col_idx]
    if pd.notna(val) and isinstance(val, (int, float)):
        nopat = val * (1 - tax_rate)
        print(f"{period_str}: EBIT = {val:>12.2f} 百万円 → NOPAT = {nopat:>12.2f} 百万円")
