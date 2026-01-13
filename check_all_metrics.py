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
    print("損益計算書（PL）の全metric一覧（最初の50個）")
    print("=" * 100)
    all_metrics = pl_long['metric'].unique()
    for i, metric in enumerate(all_metrics[:50], 1):
        print(f"{i:3d}. {metric}")
    
    print(f"\n全部で {len(all_metrics)} 個のmetricがあります")
    
    # '営業'を含むmetricを探す
    print("\n" + "=" * 100)
    print("'営業'を含むmetric:")
    print("=" * 100)
    for metric in all_metrics:
        if '営業' in str(metric):
            print(f"  - {metric}")
            # その値も表示
            data = pl_long[pl_long['metric'] == metric]
            print(f"    期間数: {len(data)}")
            if not data.empty:
                print(f"    サンプル: {data.iloc[0]['period']} = {data.iloc[0]['value']}")
    
    if not any('営業' in str(m) for m in all_metrics):
        print("  (見つかりませんでした)")
