import pandas as pd
import sys
sys.path.insert(0, 'streamlit-app-deploy/core')
from ingest_plan_excel import extract_yearly_table_by_section

# セクション別抽出
sections = extract_yearly_table_by_section('streamlit-app-deploy/Model_PS_資本政策詳細.xlsx', 'FS_年次')
pl = sections.get('損益計算書')

if pl and 'long' in pl:
    pl_long = pl['long']
    
    print("=" * 100)
    print("損益計算書（PL）のlong形式データ")
    print("=" * 100)
    print("\nmetricの一覧:")
    for metric in pl_long['metric'].unique()[:30]:
        print(f"  - {metric}")
    
    print("\n" + "=" * 100)
    print("各metricの合計値（絶対値）- EBITが見つからない場合に使用される")
    print("=" * 100)
    agg = pl_long.groupby("metric")["value"].agg(lambda x: x.dropna().abs().sum())
    top_10 = agg.nlargest(10)
    for idx, (metric, value) in enumerate(top_10.items(), 1):
        marker = " ← 最大値（フォールバックで使用される）" if idx == 1 else ""
        print(f"{idx}. {metric}: {value:.2f}{marker}")
    
    top_metric = agg.idxmax()
    print(f"\n" + "=" * 100)
    print(f"フォールバックで使用されるmetric: '{top_metric}'")
    print("=" * 100)
    
    # この metric の各期間の値
    top_data = pl_long[pl_long["metric"] == top_metric].copy()
    print(f"\n'{top_metric}' の期間別の値:")
    for _, row in top_data.iterrows():
        print(f"  {row['period']}: {row['value']:.2f} 百万円")
    
    print("\n" + "=" * 100)
    print("NOPATへの変換（税率30%適用）:")
    print("=" * 100)
    for _, row in top_data.iterrows():
        nopat = row['value'] * 0.7
        print(f"  {row['period']}: {row['value']:.2f} × 0.70 = {nopat:.2f} 百万円")
