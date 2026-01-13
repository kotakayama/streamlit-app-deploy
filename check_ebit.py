import pandas as pd

# Excelファイル直接読み込み
xlsx_path = 'streamlit-app-deploy/Model_PS_資本政策詳細.xlsx'
df = pd.read_excel(xlsx_path, sheet_name='FS_年次', header=None)

# 期間列を特定（行0に日付がある）
print("=== 期間列の確認 ===")
date_row = df.iloc[0]
for i, val in enumerate(date_row):
    if pd.notna(val) and '2023' in str(val) or '2024' in str(val) or '2025' in str(val):
        print(f"列{i}: {val}")

# 営業利益の行を探す
print("\n=== 営業利益の行を探す ===")
for idx, row in df.iterrows():
    if '営業利益' in str(row.iloc[2]):  # 3列目に項目名
        print(f"\n行{idx}: 営業利益を発見")
        print("項目列:", row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3])
        print("\n各期間の値:")
        # 列15-22あたりが2022-2029年度
        for col_idx in range(15, 23):
            period = date_row.iloc[col_idx]
            value = row.iloc[col_idx]
            if pd.notna(value):
                print(f"  列{col_idx} ({period}): {value}")
