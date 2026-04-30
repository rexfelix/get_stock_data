"""(N=15, K=3) + 시총 캡 사전 제외 단위 테스트 (TDD Red).

검증:
- T1: 모듈 임포트
- T2: cap 미만/초과/같음 분리 (cap=30조)
- T3: NaN mcap 시 불통
- T4: 입력 순서 보존
- T5: 빈 입력 시 빈 dict
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


def _make_daily_with_mcap(ticker: str, date: pd.Timestamp,
                          mcap: float) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [date],
        "open": [100.0], "close": [100.0],
        "high": [101.0], "low": [99.0],
        "amount": [1_000_000_000],
        "shares": [1000], "ma5": [99.0], "ma20": [98.0],
        "mcap": [mcap],
    })


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

def test_module_can_be_imported():
    import backtest_n15_k3_mcap_cap  # noqa: F401


# ---------------------------------------------------------------------------
# T2: cap 비교
# ---------------------------------------------------------------------------

def test_below_cap_passes():
    """mcap < cap → 통과."""
    import backtest_n15_k3_mcap_cap as m

    d = pd.Timestamp("2024-01-10")
    cap = 30_000_000_000_000  # 30조
    daily_data = {
        "A": _make_daily_with_mcap("A", d, mcap=20_000_000_000_000),  # 20조 통과
        "B": _make_daily_with_mcap("B", d, mcap=50_000_000_000_000),  # 50조 불통
    }
    signals = {d: ["A", "B"]}
    filtered = m.apply_mcap_cap_filter(signals, daily_data, cap)
    assert filtered.get(d, []) == ["A"]


def test_equal_cap_passes():
    """mcap == cap → 통과 (≤)."""
    import backtest_n15_k3_mcap_cap as m

    d = pd.Timestamp("2024-01-10")
    cap = 30_000_000_000_000
    daily_data = {
        "X": _make_daily_with_mcap("X", d, mcap=30_000_000_000_000),  # 정확히 30조
    }
    signals = {d: ["X"]}
    filtered = m.apply_mcap_cap_filter(signals, daily_data, cap)
    assert filtered.get(d, []) == ["X"]


def test_above_cap_excluded():
    """mcap > cap → 불통."""
    import backtest_n15_k3_mcap_cap as m

    d = pd.Timestamp("2024-01-10")
    cap = 30_000_000_000_000
    daily_data = {
        "BIG": _make_daily_with_mcap("BIG", d, mcap=350_000_000_000_000),  # 350조 (삼성전자 수준)
    }
    signals = {d: ["BIG"]}
    filtered = m.apply_mcap_cap_filter(signals, daily_data, cap)
    assert filtered.get(d, []) == []


# ---------------------------------------------------------------------------
# T3: NaN mcap
# ---------------------------------------------------------------------------

def test_nan_mcap_excluded():
    import backtest_n15_k3_mcap_cap as m

    d = pd.Timestamp("2024-01-10")
    daily_data = {
        "A": _make_daily_with_mcap("A", d, mcap=20_000_000_000_000),
        "N": _make_daily_with_mcap("N", d, mcap=np.nan),
    }
    signals = {d: ["A", "N"]}
    filtered = m.apply_mcap_cap_filter(signals, daily_data, 30_000_000_000_000)
    assert filtered.get(d, []) == ["A"]


# ---------------------------------------------------------------------------
# T4: 순서 보존
# ---------------------------------------------------------------------------

def test_preserves_order():
    import backtest_n15_k3_mcap_cap as m

    d = pd.Timestamp("2024-01-10")
    cap = 30_000_000_000_000
    daily_data = {
        "Z": _make_daily_with_mcap("Z", d, mcap=15_000_000_000_000),
        "Y": _make_daily_with_mcap("Y", d, mcap=20_000_000_000_000),
        "X": _make_daily_with_mcap("X", d, mcap=50_000_000_000_000),  # 제외
        "W": _make_daily_with_mcap("W", d, mcap=10_000_000_000_000),
    }
    signals = {d: ["Z", "Y", "X", "W"]}  # amount 내림차순
    filtered = m.apply_mcap_cap_filter(signals, daily_data, cap)
    assert filtered.get(d, []) == ["Z", "Y", "W"]


# ---------------------------------------------------------------------------
# T5: 빈 입력
# ---------------------------------------------------------------------------

def test_empty_signals_returns_empty():
    import backtest_n15_k3_mcap_cap as m
    assert m.apply_mcap_cap_filter({}, {}, 30_000_000_000_000) == {}
