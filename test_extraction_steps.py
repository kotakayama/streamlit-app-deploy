import pandas as pd
import numpy as np
from datetime import datetime

xlsx_file = 'streamlit-app-deploy/Model_PS_資本政策詳細.xlsx'
df_raw = pd.read_excel(xlsx_file, sheet_name="FS_年次")

print("=" * 100)
print("ステップ1: 期間列の検出")
print("=" * 100)

date_row = df_raw.iloc[0]
period_cols = {}
for col_name in df_raw.columns:
    val = date_row[col_name]
    if pd.notna(val) and isinstance(val, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp, datetime)):
        period_str = pd.to_datetime(val).strftime('%Y-%m-%d')
        period_cols[col_name] = period_str
        print(f"  {col_name}: {period_str}")

print(f"\n検出された期間数: {len(period_cols)}")

print("\n" + "=" * 100)
print("ステップ2: 営業利益の行を検索")
print("=" * 100)

ebit_row_idx = None
for row_idx in range(len(df_raw)):
    row = df_raw.iloc[row_idx]
    for col_name in list(df_raw.columns)[:5]:
        val = row[col_name]
        if pd.notna(val) and str(val).strip() == '営業利益':
            ebit_row_idx = row_idx
            print(f"  営業利益を発見: 行{ebit_row_idx}, 列{col_name}")
            break
    if ebit_row_idx is not None:
        break

if ebit_row_idx is None:
    print("  営業利益が見つかりませんでした")
else:
    print("\n" + "=" * 100)
    print("ステップ3: 営業利益の値を抽出")
    print("=" * 100)
    
    ebit_row = df_raw.iloc[ebit_row_idx]
    ebit_dict = {}
    for col_name, period_str in period_cols.items():
        val = ebit_row[col_name]
        if pd.notna(val) and isinstance(val, (int, float, np.integer, np.floating)):
            ebit_dict[period_str] = float(val)
            print(f"  {period_str}: {val:.2f} 百万円")
    
    print("\n" + "=" * 100)
    print("ステップ4: NOPATを計算（税率30%）")
    print("=" * 100)
    
    tax_rate = 0.30
    for period_str, ebit in ebit_dict.items():
        nopat = ebit * (1 - tax_rate)
        print(f"  {period_str}: EBIT {ebit:>10.2f} × 0.70 = NOPAT {nopat:>10.2f} 百万円")
