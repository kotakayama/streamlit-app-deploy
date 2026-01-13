import pandas as pd

# Excelファイル直接読み込み
xlsx_path = 'streamlit-app-deploy/Model_PS_資本政策詳細.xlsx'
df = pd.read_excel(xlsx_path, sheet_name='FS_年次', header=None)

# 期間列を特定
date_row = df.iloc[0]  # 行0に期末日
print("=== 期間ヘッダー（日付） ===")
for i in range(15, 23):
    print(f"列{i}: {date_row.iloc[i]}")

# 営業利益の行（行75）
ebit_row = df.iloc[75]
print("\n=== 営業利益（EBIT）の各期間の値 ===")
for col_idx in range(16, 23):  # 列16から（2023年度から）
    period = date_row.iloc[col_idx]
    value = ebit_row.iloc[col_idx]
    if pd.notna(value):
        print(f"{period}: {value:.2f} 百万円")

print("\n=== NOPATの計算（営業利益 × (1 - 0.30)） ===")
tax_rate = 0.30
for col_idx in range(16, 23):
    period = date_row.iloc[col_idx]
    ebit = ebit_row.iloc[col_idx]
    if pd.notna(ebit):
        nopat = ebit * (1 - tax_rate)
        print(f"{period}: EBIT {ebit:.2f} × 0.70 = NOPAT {nopat:.2f} 百万円")
