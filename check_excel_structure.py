import pandas as pd

# Excelファイル直接読み込み
df = pd.read_excel('streamlit-app-deploy/Model_PS_資本政策詳細.xlsx', sheet_name='FS_年次', header=None)

# 損益計算書セクション（行13から）
pl_start = 13
pl_end = 200  # 適当に広めに

print("=" * 100)
print("損益計算書セクション（行13-90）の構造")
print("=" * 100)

# 各列に何が入っているかチェック
for col_idx in range(5):
    print(f"\n列{col_idx}の内容（行13-90）:")
    for row_idx in range(pl_start, min(pl_start + 80, len(df))):
        val = df.iloc[row_idx, col_idx]
        if pd.notna(val) and str(val).strip() != '' and str(val) != 'nan':
            # 主要な項目だけ表示
            val_str = str(val).strip()
            if any(kw in val_str for kw in ['売上', '営業', '総利益', '販管費', '原価', '損益計算書']):
                print(f"  行{row_idx}: {val_str[:50]}")

print("\n" + "=" * 100)
print("営業利益の行（行74-75付近）の詳細:")
print("=" * 100)
for row_idx in range(72, 78):
    print(f"\n行{row_idx}:")
    for col_idx in range(8):
        val = df.iloc[row_idx, col_idx]
        if pd.notna(val) and str(val).strip() != '':
            print(f"  列{col_idx}: {val}")
