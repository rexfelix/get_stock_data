"""정확 amount (N,K) 재검증 단위 테스트 (TDD Red).

검증:
- T1: 모듈 임포트 + 함수 존재
- T2: convert_amount_to_won — 백만원 → 원 변환
- T3: NaN amount 처리
- T4: 빈 DataFrame 처리
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

def test_module_can_be_imported():
    import backtest_n_k_accurate_amount as m
    assert hasattr(m, "convert_amount_to_won")
    assert hasattr(m, "load_price_data_accurate")


# ---------------------------------------------------------------------------
# T2: 단위 변환
# ---------------------------------------------------------------------------

def test_convert_amount_to_won_basic():
    """amount 컬럼이 백만원 단위 → 원 단위로 변환 (× 1_000_000)."""
    import backtest_n_k_accurate_amount as m
    df = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "close": [100.0, 110.0],
        "amount": [5000, 6000],  # 백만원
    })
    out = m.convert_amount_to_won(df)
    assert out.iloc[0]["amount"] == 5_000_000_000  # 50억원 = 5000 * 1M
    assert out.iloc[1]["amount"] == 6_000_000_000


# ---------------------------------------------------------------------------
# T3: NaN 처리
# ---------------------------------------------------------------------------

def test_convert_amount_with_nan():
    import backtest_n_k_accurate_amount as m
    df = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "close": [100.0, 110.0],
        "amount": [5000, np.nan],
    })
    out = m.convert_amount_to_won(df)
    assert out.iloc[0]["amount"] == 5_000_000_000
    assert pd.isna(out.iloc[1]["amount"])


# ---------------------------------------------------------------------------
# T4: 빈 DataFrame
# ---------------------------------------------------------------------------

def test_convert_amount_empty_df():
    import backtest_n_k_accurate_amount as m
    df = pd.DataFrame(columns=["ticker", "date", "close", "amount"])
    out = m.convert_amount_to_won(df)
    assert "amount" in out.columns
    assert len(out) == 0
