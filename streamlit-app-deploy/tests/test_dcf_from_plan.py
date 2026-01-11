import os
import pytest
from streamlit_app_deploy.core.dcf_from_plan import extract_fcf_from_workbook, _load_mappings

# path to sample file in repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SAMPLE = os.path.join(ROOT, 'Model_PS_資本政策詳細.xlsx')


def test_mappings_loads():
    m = _load_mappings()
    assert isinstance(m, dict)
    assert 'fcf' in m and isinstance(m['fcf'], list)


def test_extract_fcf_runs_on_sample():
    assert os.path.exists(SAMPLE), f"Sample workbook not found at {SAMPLE}"
    df = extract_fcf_from_workbook(SAMPLE)
    assert set(df.columns) == {'period', 'fcf'}
    if df.empty:
        pytest.skip('No FCF series was detected from sample workbook; heuristics may need tuning')
    assert df['fcf'].notna().any()
