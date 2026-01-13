import sys
import os
os.chdir('streamlit-app-deploy')
sys.path.insert(0, 'core')

# 強制的にモジュールをリロード
if 'dcf_from_plan' in sys.modules:
    del sys.modules['dcf_from_plan']
if 'ingest_plan_excel' in sys.modules:
    del sys.modules['ingest_plan_excel']

from dcf_from_plan import extract_future_fcf_plan_nopat

# 修正後のデータを取得
print("=" * 100)
print("修正後のFCF計算結果")
print("=" * 100)

fcf_df = extract_future_fcf_plan_nopat('Model_PS_資本政策詳細.xlsx', tax_rate=0.30)

print("\n", fcf_df.to_string(index=False))

print("\n" + "=" * 100)
print("各期間のNOPAT詳細:")
print("=" * 100)
for _, row in fcf_df.iterrows():
    period = row['period']
    nopat = row['NOPAT']
    fcf = row['FCF']
    print(f"{period}: NOPAT = {nopat:>10.2f} 百万円, FCF = {fcf:>10.2f} 百万円")

print("\n" + "=" * 100)
print("期待値との比較:")
print("=" * 100)
expected = {
    '2023-09-30': {'NOPAT': -61.15, 'FCF': -8},
    '2024-09-30': {'NOPAT': -130.09, 'FCF': -1},
    '2025-09-30': {'NOPAT': -170.28, 'FCF': -13},
    '2026-09-30': {'NOPAT': -69.09, 'FCF': 429},
    '2027-09-30': {'NOPAT': 2120.40, 'FCF': 1076},
}

for period, expected_vals in expected.items():
    if period in fcf_df['period'].values:
        row = fcf_df[fcf_df['period'] == period].iloc[0]
        nopat_match = abs(row['NOPAT'] - expected_vals['NOPAT']) < 1.0
        fcf_match = abs(row['FCF'] - expected_vals['FCF']) < 2.0
        nopat_status = "✓" if nopat_match else "✗"
        fcf_status = "✓" if fcf_match else "✗"
        print(f"{period}: NOPAT {nopat_status} (実際: {row['NOPAT']:>10.2f}, 期待: {expected_vals['NOPAT']:>10.2f}), FCF {fcf_status} (実際: {row['FCF']:>10.2f}, 期待: {expected_vals['FCF']:>10.2f})")
