"""1500억 (15,3) 2019~2023 백테스트 (amount 추정) 단위 테스트 (TDD Red).

검증:
- T1: 모듈 임포트
- T2: estimate_amount_column(df) 함수 정의 + close × volume 정확성
- T3: NaN 처리
- T4: 빈 DataFrame 처리
- T5: 기존 amount 컬럼 덮어쓰기 동작
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
    import backtest_n15_k3_2019_2023  # noqa: F401


# ---------------------------------------------------------------------------
# T2: amount = close × volume 정확성
# ---------------------------------------------------------------------------

def test_estimate_amount_basic():
    import backtest_n15_k3_2019_2023 as m

    df = pd.DataFrame({
        "ticker": ["A", "A", "A"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "close": [100.0, 200.0, 300.0],
        "volume": [1_000_000, 2_000_000, 500_000],
    })
    out = m.estimate_amount_column(df)
    assert "amount" in out.columns
    assert out.iloc[0]["amount"] == 100.0 * 1_000_000  # 100M
    assert out.iloc[1]["amount"] == 200.0 * 2_000_000  # 400M
    assert out.iloc[2]["amount"] == 300.0 * 500_000    # 150M


def test_estimate_amount_realistic_15_eok():
    """1500억 임계 식별 가능성 검증: close=70000원, volume=2.2M → 1540억."""
    import backtest_n15_k3_2019_2023 as m

    df = pd.DataFrame({
        "ticker": ["005930"],
        "date": pd.to_datetime(["2024-01-02"]),
        "close": [70000.0],
        "volume": [2_200_000],
    })
    out = m.estimate_amount_column(df)
    assert out.iloc[0]["amount"] == 70000.0 * 2_200_000  # 154,000,000,000원 = 1540억


# ---------------------------------------------------------------------------
# T3: NaN 처리
# ---------------------------------------------------------------------------

def test_estimate_amount_with_nan():
    import backtest_n15_k3_2019_2023 as m

    df = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "close": [100.0, np.nan],
        "volume": [1_000_000, 2_000_000],
    })
    out = m.estimate_amount_column(df)
    assert out.iloc[0]["amount"] == 100_000_000
    assert pd.isna(out.iloc[1]["amount"])  # NaN × 2M = NaN


# ---------------------------------------------------------------------------
# T4: 빈 DataFrame
# ---------------------------------------------------------------------------

def test_estimate_amount_empty_df():
    import backtest_n15_k3_2019_2023 as m

    df = pd.DataFrame(columns=["ticker", "date", "close", "volume"])
    out = m.estimate_amount_column(df)
    assert "amount" in out.columns
    assert len(out) == 0


# ---------------------------------------------------------------------------
# T5: 기존 amount 컬럼 덮어쓰기
# ---------------------------------------------------------------------------

def test_estimate_amount_overwrites_existing():
    """기존 amount 컬럼이 있어도 close × volume 으로 덮어씀 (추정값 일관성)."""
    import backtest_n15_k3_2019_2023 as m

    df = pd.DataFrame({
        "ticker": ["A"],
        "date": pd.to_datetime(["2024-01-02"]),
        "close": [100.0],
        "volume": [1_000_000],
        "amount": [999_999_999],  # 기존 잘못된 값
    })
    out = m.estimate_amount_column(df)
    assert out.iloc[0]["amount"] == 100_000_000  # 100M (추정), 기존 999_999_999 덮어씀
