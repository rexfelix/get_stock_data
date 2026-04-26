"""Envelope 매매 전략 백테스트 단위 테스트 (TDD Red 단계).

PRD §9 검증 항목 T1~T6을 모두 커버한다.

이 테스트들은 `backtest_envelope` 모듈이 아직 구현되지 않은 상태에서
모두 실패해야 한다. 3단계(Green) 구현 후 모두 통과해야 한다.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# tests/ 디렉토리에서 부모 디렉토리(research/)를 import path에 추가
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import backtest_envelope as be  # noqa: E402


# ---------------------------------------------------------------------------
# T1: MA 계산 정합성
# ---------------------------------------------------------------------------

def test_ma_calculation_matches_manual_rolling_mean():
    """compute_ma(prices, n)는 pandas rolling().mean()과 동일해야 한다.

    - 입력: 1..50 단조증가 가격
    - n=20일 때, MA[19] = mean(1..20) = 10.5
    - n=20일 때, MA[18] = NaN (윈도우 미달)
    - 모든 인덱스 i ≥ n-1: MA[i] = mean(close[i-n+1 .. i])
    """
    prices = pd.Series(np.arange(1, 51, dtype=float))
    ma = be.compute_ma(prices, 20)

    # 길이 동일
    assert len(ma) == len(prices)
    # 윈도우 미달 구간은 NaN
    assert ma.iloc[:19].isna().all()
    # 첫 유효 값
    assert ma.iloc[19] == pytest.approx(10.5)
    # 마지막 값: mean(31..50) = 40.5
    assert ma.iloc[49] == pytest.approx(40.5)


# ---------------------------------------------------------------------------
# T2: 매수 신호 정합성
# ---------------------------------------------------------------------------

def test_buy_signal_close_below_lower_envelope():
    """매수 신호: close < ma * (1 - pct).

    합성: ma는 항상 100, close는 [101, 100, 95, 89, 90, 88].
    pct=0.10 (10%) → 하단선 = 90.
    close < 90인 날만 True여야 함 → idx=3 (89), idx=5 (88).
    """
    close = pd.Series([101, 100, 95, 89, 90, 88], dtype=float)
    ma = pd.Series([100.0] * 6)
    sig = be.make_buy_signal(close, ma, 0.10)

    expected = pd.Series([False, False, False, True, False, True])
    pd.testing.assert_series_equal(
        sig.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_buy_signal_handles_nan_ma():
    """ma가 NaN인 구간(워밍업)에서는 신호가 False여야 한다."""
    close = pd.Series([50.0, 50.0, 50.0])
    ma = pd.Series([np.nan, np.nan, 100.0])
    sig = be.make_buy_signal(close, ma, 0.10)

    assert sig.iloc[0] == False  # noqa: E712
    assert sig.iloc[1] == False  # noqa: E712
    assert sig.iloc[2] == True  # 50 < 100*(1-0.1)=90  # noqa: E712


# ---------------------------------------------------------------------------
# T3: 매도 신호 정합성
# ---------------------------------------------------------------------------

def test_sell_signal_close_below_ma():
    """매도 신호: close < ma (MA 상향 이력 제약 없음).

    합성: ma=100 일정, close=[105, 99, 100, 98, 101].
    close < 100인 날만 True → idx=1 (99), idx=3 (98).
    """
    close = pd.Series([105, 99, 100, 98, 101], dtype=float)
    ma = pd.Series([100.0] * 5)
    sig = be.make_sell_signal(close, ma)

    expected = pd.Series([False, True, False, True, False])
    pd.testing.assert_series_equal(
        sig.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# T4: 30일 단일 종목 거래 시뮬 정합성
# ---------------------------------------------------------------------------

def _make_single_ticker_df(closes, opens=None, ticker="A"):
    """단일 종목 OHLCV DataFrame 생성. open=close, high=close*1.01, low=close*0.99 기본."""
    n = len(closes)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    if opens is None:
        opens = closes  # 기본은 시가=종가
    return pd.DataFrame({
        "date": dates,
        "ticker": [ticker] * n,
        "name": [ticker] * n,
        "open": opens,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
    })


def test_trade_simulation_buy_sell_roundtrip():
    """첫 매수→첫 매도 페어의 가격/손익 정합성 검증.

    시나리오:
    - day 0~9: close=100 일정 → MA10이 day9에 100으로 산출됨
    - day 10: close=85 (하단=88.65 미만) → 매수 신호 발생
    - day 11: 시가 85 매수 (open=close 가정), 매도 신호도 발생 (85<MA)
    - day 12: 시가 80 매도

    note: 매도 후 같은 종목에 재매수 신호가 발생할 수 있어 trades가 다수가 될 수 있다
    (envelope 평균회귀 매매의 정상 동작). 본 테스트는 **첫 거래**의 정합성만 확인.
    """
    closes = [100.0] * 10 + [85.0, 85.0, 80.0, 80.0]  # 14일
    df = _make_single_ticker_df(closes)

    result = be.simulate(
        df=df,
        ma_n=10,
        pct=0.10,
        slot_capital=1_000_000,
        max_positions=10,
        commission=0.00015,
        sell_commission=0.00015,
        tax=0.0018,
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-12-31"),
    )

    trades = result["trades"]
    assert len(trades) >= 1, "최소 1건의 거래가 발생해야 함"
    t = trades[0]
    assert t["ticker"] == "A"
    assert t["buy_price"] == pytest.approx(85.0)
    assert t["sell_price"] == pytest.approx(80.0)
    # 매도가 < 매수가 → 손실 거래
    assert t["pnl"] < 0


def test_trade_simulation_no_signal_no_trade():
    """매수 신호가 발생하지 않는 시계열에서는 거래가 0건이어야 한다."""
    closes = [100.0] * 30  # 가격 일정 → 신호 없음
    df = _make_single_ticker_df(closes)

    result = be.simulate(
        df=df,
        ma_n=10,
        pct=0.10,
        slot_capital=1_000_000,
        max_positions=10,
        commission=0.00015,
        sell_commission=0.00015,
        tax=0.0018,
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-12-31"),
    )

    assert len(result["trades"]) == 0


# ---------------------------------------------------------------------------
# T5: 포트폴리오 슬롯 제한 + 정렬
# ---------------------------------------------------------------------------

def test_portfolio_slot_cap_with_sort_by_disparity():
    """같은 날 신호 5건 + 빈 슬롯 2개 → 이격률 상위 2건만 진입.

    5종목 모두 day 10에 매수 신호 발생, 이격률 차이를 두어 정렬 검증.
    - A: close=85, ma=100 → 이격 15%
    - B: close=80, ma=100 → 이격 20% (상위)
    - C: close=89, ma=100 → 이격 11%
    - D: close=70, ma=100 → 이격 30% (최상위)
    - E: close=84, ma=100 → 이격 16%

    max_positions=2 → D, B만 매수.
    """
    base_closes = [100.0] * 10
    triggers = {"A": 85.0, "B": 80.0, "C": 89.0, "D": 70.0, "E": 84.0}
    dfs = []
    for tk, trig_close in triggers.items():
        closes = base_closes + [trig_close, trig_close]  # day10 신호, day11 매수
        dfs.append(_make_single_ticker_df(closes, ticker=tk))
    df = pd.concat(dfs, ignore_index=True)

    result = be.simulate(
        df=df,
        ma_n=10,
        pct=0.10,
        slot_capital=1_000_000,
        max_positions=2,
        commission=0.00015,
        sell_commission=0.00015,
        tax=0.0018,
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-12-31"),
    )

    bought_tickers = sorted(t["ticker"] for t in result["trades"])
    # D(이격30%), B(이격20%) 만 매수되어야 함
    assert bought_tickers == ["B", "D"], \
        f"슬롯 정렬 오류: 기대 [B, D], 실제 {bought_tickers}"


# ---------------------------------------------------------------------------
# T6: CAGR 공식 정합성
# ---------------------------------------------------------------------------

def test_cagr_formula_matches_textbook():
    """CAGR = (final/initial) ** (1/years) - 1.

    초기 1억, 종료 2억, 6년 → CAGR = 2 ** (1/6) - 1 ≈ 0.122462...
    """
    initial = 100_000_000
    final = 200_000_000
    years = 6.0

    cagr = be.compute_cagr(initial, final, years)
    expected = 2 ** (1 / 6) - 1

    assert cagr == pytest.approx(expected, rel=1e-9)


def test_cagr_handles_loss():
    """손실 케이스: 1억 → 5천, 2년 → CAGR 음수."""
    cagr = be.compute_cagr(100_000_000, 50_000_000, 2.0)
    expected = 0.5 ** 0.5 - 1  # ≈ -0.2929
    assert cagr == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# T7: fdr 데이터 → 표준 OHLCV 변환 (KODEX 레버리지)
# ---------------------------------------------------------------------------

def test_load_kodex_lev_format(monkeypatch):
    """fdr 결과(mock)를 표준 형식으로 변환하는지 검증.

    표준 컬럼: date(Timestamp), open/high/low/close/volume(int 또는 float),
    ticker='122630', name='KODEX 레버리지'.
    """
    import pandas as pd
    fake_idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], name="Date")
    fake = pd.DataFrame(
        {
            "Open": [12000, 12100, 12050],
            "High": [12200, 12150, 12100],
            "Low": [11900, 11950, 11900],
            "Close": [12100, 12050, 11950],
            "Volume": [1_000_000, 900_000, 1_100_000],
            "Change": [0.01, -0.004, -0.008],
        },
        index=fake_idx,
    )

    def fake_reader(ticker, start=None, end=None):
        assert ticker == "122630"
        return fake

    # be 모듈 내부의 fdr 호출을 monkeypatch
    monkeypatch.setattr(be, "_fdr_reader", fake_reader)

    df = be.load_kodex_lev("2024-01-01")

    expected_cols = {"date", "open", "high", "low", "close", "volume", "ticker", "name"}
    assert set(df.columns) >= expected_cols, f"누락 컬럼: {expected_cols - set(df.columns)}"
    assert len(df) == 3
    assert df["ticker"].unique().tolist() == ["122630"]
    assert df["name"].unique().tolist() == ["KODEX 레버리지"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    # 가격은 숫자형
    for c in ("open", "high", "low", "close", "volume"):
        assert pd.api.types.is_numeric_dtype(df[c]), f"{c}가 숫자형이 아님"
    # 첫 행 값 일치
    first = df.iloc[0]
    assert first["open"] == 12000
    assert first["close"] == 12100


# ---------------------------------------------------------------------------
# T8: ETF 매도세 0% 반영
# ---------------------------------------------------------------------------

def test_simulate_etf_no_tax():
    """동일 매매 시퀀스에서 tax=0과 tax=0.0018 사이의 매도 순익 차이가
    매도금액 × 0.0018 만큼 정확히 발생해야 한다.

    14일 합성, day10에 매수 신호, day11 매수→day12 매도.
    """
    closes = [100.0] * 10 + [85.0, 85.0, 80.0, 80.0]
    df = _make_single_ticker_df(closes, ticker="122630")

    common = dict(
        df=df,
        ma_n=10,
        pct=0.10,
        slot_capital=10_000_000,
        max_positions=1,
        commission=0.00015,
        sell_commission=0.00015,
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-12-31"),
    )

    res_tax = be.simulate(**common, tax=0.0018)
    res_no_tax = be.simulate(**common, tax=0.0)

    assert len(res_tax["trades"]) >= 1 and len(res_no_tax["trades"]) >= 1

    # 첫 거래 비교
    t1 = res_tax["trades"][0]
    t2 = res_no_tax["trades"][0]
    # 동일 매수가/매도가/주식수
    assert t1["buy_price"] == t2["buy_price"]
    assert t1["sell_price"] == t2["sell_price"]
    assert t1["shares"] == t2["shares"]
    # tax=0이면 손익이 정확히 (매도금액 × 0.0018) 만큼 더 크다
    sell_value = t1["shares"] * t1["sell_price"]
    expected_diff = sell_value * 0.0018
    actual_diff = t2["pnl"] - t1["pnl"]
    assert actual_diff == pytest.approx(expected_diff, rel=1e-9), \
        f"매도세 차이 불일치: 기대 {expected_diff}, 실제 {actual_diff}"
