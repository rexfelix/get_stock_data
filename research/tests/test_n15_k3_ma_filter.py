"""1500억 N=15 + MA 정배열 필터 + K=3 백테스트 단위 테스트 (TDD Red).

검증:
- T1: backtest_n15_k3_ma_filter 모듈 임포트 가능
- T2: close>MA5 만족 / 불만족 분리
- T3: MA5>MA20 만족 / 불만족 분리
- T4: 두 조건 모두 만족하는 종목만 통과
- T5: 빈 입력 시 빈 dict 반환

이 테스트들은 backtest_n15_k3_ma_filter 모듈 구현 후 모두 통과해야 한다.
"""
import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def _make_daily_row(ticker: str, date: pd.Timestamp,
                    close: float, ma5: float, ma20: float) -> pd.DataFrame:
    """단일 일자 daily_data 행 생성."""
    return pd.DataFrame({
        "date": [date],
        "open": [close],
        "close": [close],
        "high": [close + 1],
        "low": [close - 1],
        "amount": [1_000_000_000],
        "shares": [1000],
        "ma5": [ma5],
        "ma20": [ma20],
    })


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

def test_module_can_be_imported():
    import backtest_n15_k3_ma_filter  # noqa: F401


# ---------------------------------------------------------------------------
# T2: close > MA5 필터
# ---------------------------------------------------------------------------

def test_close_above_ma5_passes():
    """close > MA5 만족 시 통과."""
    import backtest_n15_k3_ma_filter as m

    d = pd.Timestamp("2024-01-10")
    daily_data = {
        "A": _make_daily_row("A", d, close=100.0, ma5=95.0, ma20=90.0),  # 통과 (모두 만족)
        "B": _make_daily_row("B", d, close=90.0, ma5=95.0, ma20=90.0),   # 불통 (close<MA5)
    }
    signals = {d: ["A", "B"]}
    filtered = m.apply_ma_filter(signals, daily_data)
    assert filtered.get(d, []) == ["A"]


# ---------------------------------------------------------------------------
# T3: MA5 > MA20 필터
# ---------------------------------------------------------------------------

def test_ma5_above_ma20_passes():
    """MA5 > MA20 만족 시 통과."""
    import backtest_n15_k3_ma_filter as m

    d = pd.Timestamp("2024-01-10")
    daily_data = {
        "A": _make_daily_row("A", d, close=100.0, ma5=95.0, ma20=90.0),  # 통과
        "C": _make_daily_row("C", d, close=100.0, ma5=85.0, ma20=90.0),  # 불통 (MA5<MA20). 단 close>MA5도 거짓
    }
    # C는 close>MA5(100>85) 만족하지만 MA5<MA20 (85<90) 으로 불통
    signals = {d: ["A", "C"]}
    filtered = m.apply_ma_filter(signals, daily_data)
    assert filtered.get(d, []) == ["A"]


# ---------------------------------------------------------------------------
# T4: 두 조건 동시 만족
# ---------------------------------------------------------------------------

def test_both_conditions_required():
    """close>MA5 AND MA5>MA20 모두 만족해야 통과."""
    import backtest_n15_k3_ma_filter as m

    d = pd.Timestamp("2024-01-10")
    daily_data = {
        "A": _make_daily_row("A", d, close=100.0, ma5=95.0, ma20=90.0),  # 통과
        "B": _make_daily_row("B", d, close=90.0, ma5=95.0, ma20=85.0),   # close<MA5 불통
        "C": _make_daily_row("C", d, close=100.0, ma5=85.0, ma20=90.0),  # MA5<MA20 불통
        "D": _make_daily_row("D", d, close=80.0, ma5=85.0, ma20=90.0),   # 둘 다 불통
    }
    signals = {d: ["A", "B", "C", "D"]}
    filtered = m.apply_ma_filter(signals, daily_data)
    assert filtered.get(d, []) == ["A"]


def test_preserves_order_when_multiple_pass():
    """여러 종목이 통과하면 입력 순서 보존 (amount 정렬 유지)."""
    import backtest_n15_k3_ma_filter as m

    d = pd.Timestamp("2024-01-10")
    daily_data = {
        "X": _make_daily_row("X", d, close=110.0, ma5=100.0, ma20=95.0),  # 통과
        "Y": _make_daily_row("Y", d, close=100.0, ma5=95.0, ma20=90.0),   # 통과
        "Z": _make_daily_row("Z", d, close=90.0, ma5=95.0, ma20=90.0),    # 불통
    }
    signals = {d: ["X", "Y", "Z"]}
    filtered = m.apply_ma_filter(signals, daily_data)
    assert filtered.get(d, []) == ["X", "Y"]


# ---------------------------------------------------------------------------
# T5: 빈 입력
# ---------------------------------------------------------------------------

def test_empty_signals_returns_empty():
    import backtest_n15_k3_ma_filter as m
    filtered = m.apply_ma_filter({}, {})
    assert filtered == {}
