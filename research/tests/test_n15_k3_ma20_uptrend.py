"""1500억 N=15 + MA20 5일 연속 상승 필터 + K=3 단위 테스트 (TDD Red).

검증:
- T1: 모듈 임포트 가능
- T2: is_ma20_uptrend_5d — 단조 증가 → True
- T3: 단조 증가 아님 (평탄/등락) → False
- T4: NaN 포함 시 False
- T5: apply_ma20_uptrend_filter 통합 동작
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


def _make_daily_with_ma20(ticker: str, dates: list[pd.Timestamp],
                          ma20_values: list[float]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "open": [100.0] * n,
        "close": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "amount": [1_000_000_000] * n,
        "shares": [1000] * n,
        "ma5": [99.0] * n,
        "ma20": ma20_values,
    })


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

def test_module_can_be_imported():
    import backtest_n15_k3_ma20_uptrend  # noqa: F401


# ---------------------------------------------------------------------------
# T2: 단조 증가
# ---------------------------------------------------------------------------

def test_strictly_increasing_returns_true():
    import backtest_n15_k3_ma20_uptrend as m
    ma20 = [10.0, 11.0, 12.0, 13.0, 14.0]
    assert m.is_ma20_uptrend_5d(ma20) is True


def test_strictly_increasing_with_small_steps():
    import backtest_n15_k3_ma20_uptrend as m
    ma20 = [100.0, 100.01, 100.02, 100.03, 100.04]
    assert m.is_ma20_uptrend_5d(ma20) is True


# ---------------------------------------------------------------------------
# T3: 단조 증가 아님
# ---------------------------------------------------------------------------

def test_flat_segment_returns_false():
    """평탄 구간이 있으면 strictly increasing 아님."""
    import backtest_n15_k3_ma20_uptrend as m
    ma20 = [10.0, 11.0, 12.0, 12.0, 14.0]  # 12 → 12 평탄
    assert m.is_ma20_uptrend_5d(ma20) is False


def test_decreasing_segment_returns_false():
    import backtest_n15_k3_ma20_uptrend as m
    ma20 = [10.0, 11.0, 10.0, 13.0, 14.0]  # 11 → 10 하락
    assert m.is_ma20_uptrend_5d(ma20) is False


def test_all_decreasing_returns_false():
    import backtest_n15_k3_ma20_uptrend as m
    ma20 = [14.0, 13.0, 12.0, 11.0, 10.0]
    assert m.is_ma20_uptrend_5d(ma20) is False


# ---------------------------------------------------------------------------
# T4: NaN 처리
# ---------------------------------------------------------------------------

def test_nan_in_values_returns_false():
    import backtest_n15_k3_ma20_uptrend as m
    ma20 = [np.nan, 11.0, 12.0, 13.0, 14.0]
    assert m.is_ma20_uptrend_5d(ma20) is False


def test_wrong_length_returns_false():
    """5개 아닌 길이 → False."""
    import backtest_n15_k3_ma20_uptrend as m
    assert m.is_ma20_uptrend_5d([10.0, 11.0, 12.0]) is False
    assert m.is_ma20_uptrend_5d([]) is False


# ---------------------------------------------------------------------------
# T5: apply_ma20_uptrend_filter 통합
# ---------------------------------------------------------------------------

def test_apply_filter_passes_only_uptrend_tickers():
    import backtest_n15_k3_ma20_uptrend as m

    dates = pd.to_datetime([f"2024-01-{d:02d}" for d in [2, 3, 4, 5, 8, 9]])
    # 종목 A: dates[5]=2024-01-09 기준 5일 단조 증가
    df_a = _make_daily_with_ma20("A", list(dates),
                                 ma20_values=[8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    # 종목 B: dates[5] 기준 평탄 포함 → 불통
    df_b = _make_daily_with_ma20("B", list(dates),
                                 ma20_values=[8.0, 9.0, 10.0, 11.0, 11.0, 12.0])
    daily_data = {"A": df_a, "B": df_b}

    target = dates[5]  # 2024-01-09
    signals = {target: ["A", "B"]}
    filtered = m.apply_ma20_uptrend_filter(signals, daily_data)
    assert filtered.get(target, []) == ["A"]


def test_apply_filter_empty_signals():
    import backtest_n15_k3_ma20_uptrend as m
    assert m.apply_ma20_uptrend_filter({}, {}) == {}
