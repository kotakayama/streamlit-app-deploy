import pandas as pd

# Excelファイル読み込み
xlsx_path = 'streamlit-app-deploy/Model_PS_資本政策詳細.xlsx'
df = pd.read_excel(xlsx_path, sheet_name='FS_年次')

# 行0に期末日（日付）
# 行75に営業利益
date_row = df.iloc[0]
ebit_row = df.iloc[75]

print("=" * 80)
print("各期間のNOPAT計算詳細")
print("=" * 80)
print("\nExcelファイル: Model_PS_資本政策詳細.xlsx")
print("シート: FS_年次")
print("営業利益の行: 75行目（Excelでは76行目）")
print("計算式: NOPAT = 営業利益（EBIT）× (1 - 税率)")
print("税率: 30%")
print("=" * 80)

tax_rate = 0.30

# 列15から22が2022-2029年度
print("\n期間別NOPAT計算:")
print("-" * 80)
for col_name in ['Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'IPO', 'Unnamed: 22']:
    period = date_row[col_name]
    ebit = ebit_row[col_name]
    
    if pd.notna(ebit) and pd.notna(period):
        nopat = ebit * (1 - tax_rate)
        period_str = str(period).split()[0]  # 日付部分のみ
        print(f"{period_str}:")
        print(f"  営業利益（EBIT）= {ebit:>12.2f} 百万円 (FS_年次シート 76行目)")
        print(f"  NOPAT = {ebit:>12.2f} × (1 - 0.30) = {nopat:>12.2f} 百万円")
        print()
