import sys
import os
os.chdir('streamlit-app-deploy')
sys.path.insert(0, 'core')

from dcf_from_plan import extract_future_fcf_plan_nopat

# 実際にアプリが使用しているFCFデータを取得
fcf_df = extract_future_fcf_plan_nopat('Model_PS_資本政策詳細.xlsx', tax_rate=0.30)

print("=" * 100)
print("アプリに実際に表示されるFCF計画データ")
print("=" * 100)
print("\n", fcf_df.to_string(index=False))

print("\n" + "=" * 100)
print("各期間のNOPAT詳細:")
print("=" * 100)
for _, row in fcf_df.iterrows():
    print(f"\n{row['period']}: NOPAT = {row['NOPAT']:.2f} 百万円")
