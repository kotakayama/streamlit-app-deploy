import pandas as pd

# Excelファイル読み込み
xlsx_path = 'streamlit-app-deploy/Model_PS_資本政策詳細.xlsx'
df = pd.read_excel(xlsx_path, sheet_name='FS_年次')

# 行0に期末日、行74に営業利益
date_row = df.iloc[0]
ebit_row = df.iloc[74]

print("=" * 100)
print("FCF計算に使用されるNOPATの算出方法")
print("=" * 100)
print("\nファイル: Model_PS_資本政策詳細.xlsx")
print("シート: FS_年次")
print("営業利益の行: Excelの75行目（Pythonのインデックス74）")
print("\n計算式: NOPAT = 営業利益（EBIT）× (1 - 税率)")
print("税率: 30% (デフォルト値)")
print("=" * 100)

tax_rate = 0.30

# 列の対応
columns_to_check = [
    ('Unnamed: 16', '2023-09-30'),
    ('Unnamed: 17', '2024-09-30'),
    ('Unnamed: 18', '2025-09-30'),
    ('Unnamed: 19', '2026-09-30'),
    ('Unnamed: 20', '2027-09-30'),
]

print("\n期間別NOPAT計算:")
print("-" * 100)
for col_name, period_label in columns_to_check:
    ebit = ebit_row[col_name]
    
    if pd.notna(ebit):
        nopat = ebit * (1 - tax_rate)
        print(f"\n{period_label}:")
        print(f"  営業利益（EBIT）: {ebit:>15,.2f} 百万円  ← Excelの FS_年次シート 75行目")
        print(f"  計算: {ebit:>15,.2f} × (1 - 0.30)")
        print(f"  NOPAT:            {nopat:>15,.2f} 百万円")
    else:
        print(f"\n{period_label}: データなし")

print("\n" + "=" * 100)
print("備考:")
print("  - Excelに「税引後営業利益」または「NOPAT」という項目が直接存在する場合はそちらを使用")
print("  - 今回のExcelには該当項目がないため、営業利益から計算しています")
print("=" * 100)
