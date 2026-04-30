"""1500억 (N,K) + 월봉 MA10 위 필터 단위 테스트 (TDD Red).

검증:
- T1: 모듈 임포트
- T2: compute_monthly_ma10 — 12개월 인공 데이터에서 11번째 월 MA10 정확성
- T3: apply_monthly_ma10_filter — close > MA10 통과/불통
- T4: NaN MA10 (초기 구간) 시 불통
- T5: 빈 입력
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


def _make_monthly_close_df(closes_per_month: list[float]) -> pd.DataFrame:
    """월별 close 시계열을 일봉 형태로 만든다 (각 월 첫 영업일만 사용).

    closes_per_month: [(month_idx 1)=close1, ..., (month_idx N)=closeN]
    각 월의 마지막 거래일에 close 가 해당 값.
    """
    rows = []
    base_year = 2023
    for i, close in enumerate(closes_per_month):
        # 월의 마지막 영업일 (간단히 매월 28일)
        month = ((i) % 12) + 1
        year = base_year + (i // 12)
        last_day = pd.Timestamp(f"{year}-{month:02d}-28")
        rows.append({
            "date": last_day,
            "close": close,
            "open": close, "high": close, "low": close,
            "amount": 1_000_000_000, "volume": 1000,
            "ma5": close, "ma20": close, "shares": 1000,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

def test_module_can_be_imported():
    import backtest_n_k_monthly_ma10  # noqa: F401


# ---------------------------------------------------------------------------
# T2: compute_monthly_ma10 정확성
# ---------------------------------------------------------------------------

def test_monthly_ma10_eleventh_month():
    """1~10월 close 평균 = 11월 시점 month_ma10."""
    import backtest_n_k_monthly_ma10 as m

    closes = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
              200.0, 300.0]  # 12개월
    df = _make_monthly_close_df(closes)
    daily_data = {"A": df}

    out = m.compute_monthly_ma10(daily_data)
    df_out = out["A"]
    # 11번째 월(2023-11-28) 시점에서 MA10 = mean(1~10월) = (10+20+...+100)/10 = 55
    nov_row = df_out[df_out["date"] == pd.Timestamp("2023-11-28")].iloc[0]
    assert abs(nov_row["monthly_ma10"] - 55.0) < 1e-6


def test_monthly_ma10_initial_period_is_nan():
    """초기 9개월은 MA10 미계산 (NaN)."""
    import backtest_n_k_monthly_ma10 as m

    closes = [float(i) for i in range(1, 11)]
    df = _make_monthly_close_df(closes)
    daily_data = {"A": df}
    out = m.compute_monthly_ma10(daily_data)
    df_out = out["A"]
    # 1~9번째 월 NaN
    for i in range(9):
        row = df_out.iloc[i]
        assert pd.isna(row["monthly_ma10"]), f"month {i+1} should be NaN"
    # 10번째 월부터 계산 (1~10월 평균이 11번째 월부터 적용 — 직전 완료 월 기준)
    # 즉 10번째 월(2023-10) 시점에서 MA10 = NaN (아직 9개월만 완료)
    # 11번째 월부터 MA10 가용
    # 단, 정확한 정의에 따라 다름 — 본 테스트는 NaN 구간 확인까지만


# ---------------------------------------------------------------------------
# T3: apply_monthly_ma10_filter — close > MA10 분리
# ---------------------------------------------------------------------------

def test_close_above_monthly_ma10_passes():
    import backtest_n_k_monthly_ma10 as m

    d = pd.Timestamp("2024-06-15")
    df_a = pd.DataFrame({
        "date": [d], "close": [110.0],
        "monthly_ma10": [100.0],  # 110 > 100 통과
        "open": [110.0], "high": [111.0], "low": [109.0],
        "amount": [1_000_000_000], "volume": [1000],
        "ma5": [109.0], "ma20": [108.0], "shares": [1000],
    })
    df_b = pd.DataFrame({
        "date": [d], "close": [90.0],
        "monthly_ma10": [100.0],  # 90 < 100 불통
        "open": [90.0], "high": [91.0], "low": [89.0],
        "amount": [1_000_000_000], "volume": [1000],
        "ma5": [89.0], "ma20": [88.0], "shares": [1000],
    })
    daily_data = {"A": df_a, "B": df_b}
    signals = {d: ["A", "B"]}
    filtered = m.apply_monthly_ma10_filter(signals, daily_data)
    assert filtered.get(d, []) == ["A"]


# ---------------------------------------------------------------------------
# T4: NaN MA10 시 불통
# ---------------------------------------------------------------------------

def test_nan_monthly_ma10_excluded():
    import backtest_n_k_monthly_ma10 as m

    d = pd.Timestamp("2019-03-15")
    df_a = pd.DataFrame({
        "date": [d], "close": [100.0],
        "monthly_ma10": [np.nan],
        "open": [100.0], "high": [101.0], "low": [99.0],
        "amount": [1_000_000_000], "volume": [1000],
        "ma5": [99.0], "ma20": [98.0], "shares": [1000],
    })
    daily_data = {"A": df_a}
    signals = {d: ["A"]}
    filtered = m.apply_monthly_ma10_filter(signals, daily_data)
    assert filtered.get(d, []) == []


# ---------------------------------------------------------------------------
# T5: 빈 입력
# ---------------------------------------------------------------------------

def test_empty_signals_returns_empty():
    import backtest_n_k_monthly_ma10 as m
    assert m.apply_monthly_ma10_filter({}, {}) == {}
