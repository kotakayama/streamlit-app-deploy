import pandas as pd
import sys
import os
os.chdir('streamlit-app-deploy')
sys.path.insert(0, 'core')
from ingest_plan_excel import extract_yearly_table_by_section
from dcf_from_plan import extract_future_fcf_plan_nopat

# セクション抽出
sections = extract_yearly_table_by_section('Model_PS_資本政策詳細.xlsx', 'FS_年次')
pl = sections.get('損益計算書')

if pl and 'long' in pl:
    df = pl['long']
    print("=== 損益計算書のmetric一覧 ===")
    for metric in df['metric'].unique():
        print(f"  - {metric}")
    
    print("\n=== 各期間の営業利益の値 ===")
    ebit_df = df[df['metric'].str.contains('営業', na=False)]
    if not ebit_df.empty:
        for _, row in ebit_df.iterrows():
            print(f"  {row['period']}: {row['metric']} = {row['value']}")
    
    print("\n=== 税引後営業利益/NOPATの値 ===")
    nopat_df = df[df['metric'].str.contains('税引後|NOPAT', na=False)]
    if not nopat_df.empty:
        for _, row in nopat_df.iterrows():
            print(f"  {row['period']}: {row['metric']} = {row['value']}")
    else:
        print("  (税引後営業利益/NOPATの項目は見つかりませんでした)")

# FCF計算結果を表示
print("\n=== 実際に計算されたNOPAT（税率30%） ===")
fcf_df = extract_future_fcf_plan_nopat('Model_PS_資本政策詳細.xlsx', tax_rate=0.30)
for _, row in fcf_df.iterrows():
    print(f"  {row['period']}: NOPAT = {row['NOPAT']:.2f} 百万円")
