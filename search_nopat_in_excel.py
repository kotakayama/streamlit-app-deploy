import pandas as pd
import sys
import os

# Change to the app directory
os.chdir('streamlit-app-deploy')

# Read the Excel file directly and check what's in there
df = pd.read_excel('Model_PS_資本政策詳細.xlsx', sheet_name='FS_年次')

# Look for NOPAT in the Excel
print("=" * 100)
print("Excelファイル内の'NOPAT'または'税引後'を含む行を検索")
print("=" * 100)

for idx in range(len(df)):
    row = df.iloc[idx]
    # Check all columns for NOPAT or 税引後営業利益
    row_text = ' '.join([str(val) for val in row[:5] if pd.notna(val)])
    if 'NOPAT' in row_text or '税引後営業利益' in row_text or '税引後' in row_text:
        print(f"\n行{idx}: {row_text}")
        # Show the date headers
        date_row = df.iloc[0]
        print("期間別の値:")
        for col_idx in range(16, 23):
            period = date_row.iloc[col_idx]
            value = row.iloc[col_idx]
            if pd.notna(value):
                print(f"  列{col_idx} ({period}): {value}")
