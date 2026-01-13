import pandas as pd
import numpy as np

# Excelファイル直接読み込み
xlsx_path = 'streamlit-app-deploy/Model_PS_資本政策詳細.xlsx'
df = pd.read_excel(xlsx_path, sheet_name='FS_年次')

print("=== FS_年次シートの内容（先頭50行） ===")
print(df.head(50))

print("\n=== 損益計算書セクションを探す ===")
# '損益計算書'というテキストを含む行を探す
for idx, row in df.iterrows():
    if any('損益計算書' in str(val) for val in row if pd.notna(val)):
        print(f"行 {idx}: {row.tolist()}")
        # その後10行も表示
        print("\n次の15行:")
        print(df.iloc[idx:idx+15])
        break
