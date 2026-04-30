"""(N=15, K=3) 실전 운영성 검증 단위 테스트 (TDD Red 단계).

검증:
- T1: backtest_n15_k3_validation 모듈 임포트 가능
- T2: compute_no_trade_gap(trades) — 인접 buy_date 사이 최대 gap (일)
- T3: stress_test_one_trade_loss(trades, idx, loss_pct) — 한 거래 net_ret 강제 손실 후 합계 변화
- T4: unique_tickers / repeated_ticker_count 정확성

이 테스트들은 backtest_n15_k3_validation 모듈 구현 후 모두 통과해야 한다.
"""
import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


# ---------------------------------------------------------------------------
# T1: 모듈 임포트
# ---------------------------------------------------------------------------

def test_module_can_be_imported():
    import backtest_n15_k3_validation  # noqa: F401


# ---------------------------------------------------------------------------
# T2: 무거래 구간 계산
# ---------------------------------------------------------------------------

def test_compute_no_trade_gap_basic():
    """buy_date 정렬 후 인접 차이의 max (일 단위)."""
    import backtest_n15_k3_validation as m

    trades = [
        {"buy_date": pd.Timestamp("2023-01-10")},
        {"buy_date": pd.Timestamp("2023-01-15")},  # gap 5
        {"buy_date": pd.Timestamp("2023-03-01")},  # gap 45
        {"buy_date": pd.Timestamp("2023-03-05")},  # gap 4
    ]
    gap_days, gap_start, gap_end = m.compute_no_trade_gap(trades)
    assert gap_days == 45
    assert gap_start == pd.Timestamp("2023-01-15")
    assert gap_end == pd.Timestamp("2023-03-01")


def test_compute_no_trade_gap_empty():
    """trades 비어있으면 (0, None, None)."""
    import backtest_n15_k3_validation as m
    gap_days, gap_start, gap_end = m.compute_no_trade_gap([])
    assert gap_days == 0
    assert gap_start is None
    assert gap_end is None


def test_compute_no_trade_gap_single():
    """trades 1개면 gap=0."""
    import backtest_n15_k3_validation as m
    trades = [{"buy_date": pd.Timestamp("2023-01-10")}]
    gap_days, _, _ = m.compute_no_trade_gap(trades)
    assert gap_days == 0


# ---------------------------------------------------------------------------
# T3: Stress Test — 한 거래 강제 손실
# ---------------------------------------------------------------------------

def test_stress_test_one_trade_loss():
    """한 거래의 net_ret을 -50%로 강제했을 때 net_ret 변화 계산."""
    import backtest_n15_k3_validation as m

    trades = [
        {"net_ret": 0.10, "ticker": "A"},
        {"net_ret": 0.20, "ticker": "B"},
        {"net_ret": -0.05, "ticker": "C"},
    ]
    # idx=1 (B) net_ret을 -0.50으로 강제
    new_trades = m.stress_test_one_trade_loss(trades, idx=1, forced_net_ret=-0.50)
    assert len(new_trades) == 3
    assert new_trades[0]["net_ret"] == 0.10  # A 그대로
    assert new_trades[1]["net_ret"] == -0.50  # B 강제
    assert new_trades[2]["net_ret"] == -0.05  # C 그대로
    # 원본 불변
    assert trades[1]["net_ret"] == 0.20


# ---------------------------------------------------------------------------
# T4: 종목 다양성 계산
# ---------------------------------------------------------------------------

def test_unique_tickers_and_repeats():
    import backtest_n15_k3_validation as m

    trades = [
        {"ticker": "A"}, {"ticker": "B"}, {"ticker": "A"},
        {"ticker": "C"}, {"ticker": "A"}, {"ticker": "B"},
    ]
    counts = m.count_ticker_trades(trades)
    assert counts["A"] == 3
    assert counts["B"] == 2
    assert counts["C"] == 1
    assert m.unique_ticker_count(trades) == 3
    assert m.most_repeated_ticker(trades) == ("A", 3)
