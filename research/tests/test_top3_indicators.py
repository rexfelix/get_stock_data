"""Top 3 지표 비교 백테스트 단위 테스트 (TDD Red 단계).

핵심 로직 검증:
- T1: compute_ma 단순 평균 정확성
- T2: compute_top3 지표별 상위 3종목 추출
- T3: simulate_one 매도 규칙 LIST_EXIT
- T4: simulate_one 매도 규칙 MA5
- T5: simulate_one 매도 규칙 HOLD_N
- T6: 수수료/세금 적용 정확성

이 테스트들은 backtest_top3_indicators 모듈 구현 후 모두 통과해야 한다.
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

import backtest_top3_indicators as bt  # noqa: E402


# ---------------------------------------------------------------------------
# T1: MA 계산
# ---------------------------------------------------------------------------

def test_compute_ma_matches_rolling_mean():
    prices = pd.Series(np.arange(1, 11, dtype=float))  # 1..10
    ma5 = bt.compute_ma(prices, 5)
    # MA5[4] = mean(1..5) = 3.0
    assert ma5.iloc[4] == 3.0
    # MA5[9] = mean(6..10) = 8.0
    assert ma5.iloc[9] == 8.0
    # MA5[3] = NaN (window 미달)
    assert pd.isna(ma5.iloc[3])


# ---------------------------------------------------------------------------
# T2: Top3 추출
# ---------------------------------------------------------------------------

def test_compute_top3_by_amount():
    """date 1일치, 5종목 amount → 상위 3 추출."""
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E"],
        "amount": [100, 500, 300, 700, 200],
    })
    top3 = bt.compute_top3_by_column(df, "amount", n=3)
    assert top3 == ["D", "B", "C"]  # 700, 500, 300 순


def test_compute_top3_handles_ties():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "amount": [100, 100, 100],
    })
    top3 = bt.compute_top3_by_column(df, "amount", n=3)
    assert set(top3) == {"A", "B", "C"}
    assert len(top3) == 3


# ---------------------------------------------------------------------------
# T3: LIST_EXIT 매도 규칙
# ---------------------------------------------------------------------------

def test_list_exit_sells_when_dropped_from_top3():
    """T-1 Top3 중 하나가 T 마감 Top3에서 빠지면 T+1 시가 매도."""
    # 3일치, 1종목 시뮬레이션
    # day0: top3 = [A, B, C], A 신호
    # day1: A 시가 매수
    # day1: top3 = [B, C, D] → A 이탈 → 매도 신호
    # day2: A 시가 매도
    dates = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-06"])
    daily_data = {
        "A": pd.DataFrame({
            "date": dates,
            "open": [100.0, 110.0, 120.0],
            "close": [105.0, 115.0, 125.0],
            "high": [108.0, 118.0, 128.0],
            "low": [99.0, 109.0, 119.0],
            "amount": [1000, 500, 200],  # day0 큼, day1 작음
            "shares": [1000, 1000, 1000],
        })
    }
    top3_per_day = {
        dates[0]: ["A", "B", "C"],
        dates[1]: ["B", "C", "D"],   # A 이탈
        dates[2]: ["B", "C", "D"],
    }
    trades = bt.simulate_strategy(
        daily_data=daily_data,
        top3_per_day=top3_per_day,
        dates=list(dates),
        rule="LIST_EXIT",
        hold_n=None,
    )
    assert len(trades) == 1
    t = trades[0]
    assert t["ticker"] == "A"
    assert t["buy_price"] == 110.0      # day1 시가
    assert t["sell_price"] == 120.0     # day2 시가
    # gross = (120-110)/110 = 9.09%, net = 9.09 - 0.21 (수수료+세금) = ~8.88%
    assert abs(t["gross_ret"] - (120/110 - 1)) < 1e-6


# ---------------------------------------------------------------------------
# T4: MA5 매도 규칙
# ---------------------------------------------------------------------------

def test_ma5_exit_sells_when_close_below_ma5():
    """종가 < MA5 첫 발생 시 다음날 시가 매도."""
    # 7일치, MA5는 day4부터 유효
    dates = pd.to_datetime([f"2026-01-{d:02d}" for d in [2, 3, 6, 7, 8, 9, 10]])
    closes = [100.0, 102.0, 104.0, 106.0, 108.0, 90.0, 95.0]  # day5(idx5) 급락
    opens = [99.0, 101.0, 103.0, 105.0, 107.0, 91.0, 96.0]
    df_a = pd.DataFrame({
        "date": dates,
        "open": opens,
        "close": closes,
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "amount": [1000] * 7,
        "shares": [1000] * 7,
    })
    df_a["ma5"] = bt.compute_ma(df_a["close"], 5)
    df_a["ma20"] = bt.compute_ma(df_a["close"], 20)
    daily_data = {"A": df_a}
    # day0에 매수 신호 있음, day1에 매수
    top3_per_day = {dates[0]: ["A", "B", "C"]}
    for d in dates[1:]:
        top3_per_day[d] = ["A", "B", "C"]  # 계속 Top3 (LIST_EXIT 영향 안받음)
    trades = bt.simulate_strategy(
        daily_data=daily_data,
        top3_per_day=top3_per_day,
        dates=list(dates),
        rule="MA5",
        hold_n=None,
    )
    # MA5(day4) = mean(100,102,104,106,108) = 104. close[4]=108 > 104 → 보유
    # MA5(day5) = mean(102,104,106,108,90) = 102. close[5]=90 < 102 → 신호
    # day6 시가에 매도 = 96
    assert len(trades) == 1
    t = trades[0]
    assert t["buy_price"] == opens[1]  # day1 시가 = 101
    assert t["sell_price"] == opens[6]  # day6 시가 = 96


# ---------------------------------------------------------------------------
# T5: HOLD_N 매도 규칙
# ---------------------------------------------------------------------------

def test_hold_n_sells_after_n_days():
    """N영업일 보유 후 시가 매도. (매수 당일=1일차)"""
    dates = pd.to_datetime([f"2026-01-{d:02d}" for d in [2, 3, 6, 7, 8, 9]])
    df_a = pd.DataFrame({
        "date": dates,
        "open": [100.0, 110.0, 115.0, 120.0, 125.0, 130.0],
        "close": [105.0, 112.0, 118.0, 122.0, 128.0, 132.0],
        "high": [108.0, 113.0, 119.0, 123.0, 129.0, 133.0],
        "low": [99.0, 109.0, 114.0, 119.0, 124.0, 129.0],
        "amount": [1000] * 6,
        "shares": [1000] * 6,
    })
    df_a["ma5"] = bt.compute_ma(df_a["close"], 5)
    df_a["ma20"] = bt.compute_ma(df_a["close"], 20)
    daily_data = {"A": df_a}
    # day4 이후 A는 Top3 제외 (재매수 방지)
    top3_per_day = {d: ["A", "B", "C"] for d in dates[:4]}
    for d in dates[4:]:
        top3_per_day[d] = ["B", "C", "D"]
    trades = bt.simulate_strategy(
        daily_data=daily_data,
        top3_per_day=top3_per_day,
        dates=list(dates),
        rule="HOLD_N",
        hold_n=3,
    )
    # day0 신호 → day1 매수 (110, 1일차)
    # day2 (2일차), day3 (3일차) → day3 종가 마감 시 신호
    # day4 시가 매도 (125)
    assert len(trades) == 1
    t = trades[0]
    assert t["buy_price"] == 110.0
    assert t["sell_price"] == 125.0
    assert t["hold_days"] == 3


# ---------------------------------------------------------------------------
# T6: MA_INIT_STOP 규칙
# ---------------------------------------------------------------------------

def test_ma_init_stop_cuts_loss_when_below_ma_and_drops_n_pct():
    """매수 후 MA 위로 못 올라가고 -N% 이상 하락 → 손절."""
    # 5일치, 매수 후 계속 MA20 아래에서 하락만 함
    dates = pd.to_datetime([f"2026-01-{d:02d}" for d in [2, 3, 6, 7, 8]])
    closes = [100.0, 100.0, 95.0, 92.0, 90.0]  # day1 매수가 100, day3 -8% → 손절 신호
    opens = [100.0, 100.0, 96.0, 93.0, 91.0]
    df_a = pd.DataFrame({
        "date": dates, "open": opens, "close": closes,
        "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "amount": [1000] * 5, "shares": [1000] * 5,
    })
    # MA20은 5일짜리에서 NaN → 절대 위로 못 올라감 → 손절 발동 가능
    df_a["ma5"] = bt.compute_ma(df_a["close"], 5)
    df_a["ma20"] = bt.compute_ma(df_a["close"], 20)
    daily_data = {"A": df_a}
    # day0~1 Top3에 A 포함, day2~ A 제외 (매도 후 재매수 방지)
    top3_per_day = {dates[0]: ["A", "B", "C"], dates[1]: ["A", "B", "C"]}
    for d in dates[2:]:
        top3_per_day[d] = ["B", "C", "D"]
    trades = bt.simulate_strategy(
        daily_data=daily_data, top3_per_day=top3_per_day,
        dates=list(dates), rule="MA_INIT_STOP",
        ma_period=20, stop_pct=-0.05,
    )
    # day1 매수 = 100. day2 close=95 → -5% 손절 신호. day3 시가 매도 = 93
    assert len(trades) == 1
    t = trades[0]
    assert t["buy_price"] == 100.0
    assert t["sell_price"] == 93.0


def test_ma_init_stop_triggers_ma_exit_after_crossing_above():
    """매수 후 MA 위로 올라간 후 MA 이탈 → 매도. 손절선은 비활성화."""
    # 8일치 데이터, MA5 사용
    dates = pd.to_datetime([f"2026-01-{d:02d}" for d in [2, 3, 6, 7, 8, 9, 10, 13]])
    # day1 매수 = 100. day2~5 MA5 위로 상승. day6 MA5 이탈
    closes = [100.0, 102.0, 104.0, 106.0, 108.0, 90.0, 95.0, 100.0]
    opens = [100.0, 101.0, 103.0, 105.0, 107.0, 91.0, 96.0, 99.0]
    df_a = pd.DataFrame({
        "date": dates, "open": opens, "close": closes,
        "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "amount": [1000] * 8, "shares": [1000] * 8,
    })
    df_a["ma5"] = bt.compute_ma(df_a["close"], 5)
    df_a["ma20"] = bt.compute_ma(df_a["close"], 20)
    daily_data = {"A": df_a}
    # day0~5는 A Top3, day6 이후 제외 (매도 후 재매수 방지)
    top3_per_day = {}
    for i, d in enumerate(dates):
        top3_per_day[d] = ["A", "B", "C"] if i <= 5 else ["B", "C", "D"]
    trades = bt.simulate_strategy(
        daily_data=daily_data, top3_per_day=top3_per_day,
        dates=list(dates), rule="MA_INIT_STOP",
        ma_period=5, stop_pct=-0.05,
    )
    # day1 매수=101 (open[1])
    # day4 (close=108) MA5(idx4)=104 → close>=MA → crossed_ma=True
    # day5 (close=90) MA5(idx5)=102 → crossed True 상태에서 close<MA → 신호
    # day6 시가 매도 = 96
    assert len(trades) == 1
    t = trades[0]
    assert t["buy_price"] == 101.0
    assert t["sell_price"] == 96.0


def test_ma_init_stop_no_signal_if_above_ma_and_drop_within_threshold():
    """MA 위에 있으면 -N% 손절 발동 안 함."""
    dates = pd.to_datetime([f"2026-01-{d:02d}" for d in [2, 3, 6, 7, 8, 9]])
    # day1 매수 = 100. close가 MA보다 위에 있는 동안은 손절 발동 안 함
    closes = [100.0, 105.0, 110.0, 115.0, 118.0, 120.0]
    opens = [100.0, 104.0, 109.0, 114.0, 117.0, 119.0]
    df_a = pd.DataFrame({
        "date": dates, "open": opens, "close": closes,
        "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
        "amount": [1000] * 6, "shares": [1000] * 6,
    })
    df_a["ma5"] = bt.compute_ma(df_a["close"], 5)
    df_a["ma20"] = bt.compute_ma(df_a["close"], 20)
    daily_data = {"A": df_a}
    top3_per_day = {d: ["A", "B", "C"] for d in dates}
    trades = bt.simulate_strategy(
        daily_data=daily_data, top3_per_day=top3_per_day,
        dates=list(dates), rule="MA_INIT_STOP",
        ma_period=5, stop_pct=-0.05,
    )
    # 매수 후 계속 상승 → 손절 발동 안 함, MA 이탈도 없음
    # 마지막 날 forced_close
    assert len(trades) == 1
    t = trades[0]
    assert t.get("forced_close") is True


# ---------------------------------------------------------------------------
# T7: 수수료 정확성
# ---------------------------------------------------------------------------

def test_net_return_includes_fees_and_tax():
    """수수료(매수+매도) + 거래세 적용."""
    gross = 0.10  # 10% 총수익
    net = bt.apply_fees(gross)
    # FEE_BUY 0.015% + FEE_SELL 0.015% + TAX_SELL 0.18% = 0.21%
    assert abs(net - (gross - bt.FEE_BUY - bt.FEE_SELL - bt.TAX_SELL)) < 1e-9
