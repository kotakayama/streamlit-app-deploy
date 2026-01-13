import pandas as pd
import numpy as np

# Excelファイル直接読み込み
xlsx_path = 'streamlit-app-deploy/Model_PS_資本政策詳細.xlsx'
df = pd.read_excel(xlsx_path, sheet_name='FS_年次')

# 期間ヘッダーを探す（行0と行1）
print("=== 期間ヘッダー ===")
print("行0 (日付):", df.iloc[0].tolist())
print("行1 (年度):", df.iloc[1].tolist())

# 損益計算書の行番号を見つける
pl_start = None
for idx, row in df.iterrows():
    if any('損益計算書' in str(val) for val in row if pd.notna(val)):
        pl_start = idx
        break

if pl_start is not None:
    print(f"\n=== 損益計算書は行 {pl_start} から開始 ===")
    
    # 損益計算書セクションから主要な項目を抽出
    # 通常は売上高、売上原価、売上総利益、販管費、営業利益などがある
    print("\n=== 損益計算書の主要項目（行14-70あたり） ===")
    for idx in range(pl_start, min(pl_start + 70, len(df))):
        row = df.iloc[idx]
        # 2列目（Unnamed: 1）または3列目（Unnamed: 2）に項目名がある
        item1 = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
        item2 = str(row.iloc[2]) if pd.notna(row.iloc[2]) else ""
        
        # 主要な項目のみ表示
        keywords = ['売上高', '売上原価', '売上総利益', '販売費', '販管費', '営業利益', '経常利益', '税引前', '当期純利益', 'EBIT', 'NOPAT']
        if any(kw in item1 or kw in item2 for kw in keywords):
            # 値のサンプル（2023年度から数期間）
            values = []
            for col_idx in range(6, min(12, len(row))):  # 列6-11あたりに実績/計画値
                val = row.iloc[col_idx]
                if pd.notna(val) and str(val) != 'nan':
                    values.append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
                else:
                    values.append("-")
            
            print(f"行{idx}: [{item1}] [{item2}] = {' | '.join(values[:6])}")
    
    # 営業利益を探す
    print("\n=== '営業利益'を含む行の詳細 ===")
    for idx in range(pl_start, min(pl_start + 100, len(df))):
        row = df.iloc[idx]
        text = ' '.join([str(val) for val in row[:5] if pd.notna(val)])
        if '営業利益' in text:
            print(f"\n行{idx}: {text}")
            # 全ての値を表示
            print("値:", [f"{row.iloc[i]:.2f}" if pd.notna(row.iloc[i]) and isinstance(row.iloc[i], (int, float)) else str(row.iloc[i]) for i in range(6, min(15, len(row)))])
