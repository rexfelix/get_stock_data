"""
전략 비교: v4(기준봉+양봉) vs 기준봉+갭하락3% 매수
- 동일 조건: Leaders≥70, 이격도(15일, 92/125), 15MA 매도
- 기준봉+갭하락: 기준봉 조건(이격도92이하+음봉+거래량증가) + 갭하락3%이상 → 종가매수
- 종목당 1천만원 독립 백테스트
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from backtest_crash import (
    ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    START_DATE, END_DATE, load_all_data,
)
from backtest_simple_buy import get_leaders_tickers, calc_indicators_15

MA_PERIOD = 15
CRASH_TH = 92
SURGE_TH = 125
GAP_DOWN_PCT = 0.03  # 시가 기준 3% 갭하락


def run_v4(df: pd.DataFrame, ticker: str, name: str):
    """v4: 기준봉 돌파 + 양봉 즉시 매수"""
    capital = INITIAL_CAPITAL
    position = None
    gijoongbong = None
    above_ma = False

    trades = []
    dates = df.index.tolist()

    for i in range(1, len(df)):
        row = df.iloc[i]
        close = row["close"]
        sma15 = row["sma15"]
        disp = row["disparity15"]
        date = dates[i]

        if pd.isna(sma15):
            continue

        if position is not None:
            sell_reason = None
            if not above_ma and close > sma15:
                above_ma = True
            if close < position["buy_candle_low"]:
                sell_reason = "손절(매수봉저가)"
            elif disp >= SURGE_TH:
                sell_reason = "급등신호"
            elif above_ma and close < sma15:
                sell_reason = "15MA이탈"

            if sell_reason:
                qty = position["quantity"]
                revenue = qty * close
                fee = revenue * FEE_SELL
                tax = revenue * TAX_SELL
                net = revenue - fee - tax
                capital += net
                buy_cost = position["entry_price"] * qty
                buy_fee = buy_cost * FEE_BUY
                pnl = net - buy_cost - buy_fee
                ret = pnl / (buy_cost + buy_fee) * 100
                trades.append({
                    "ticker": ticker, "name": name,
                    "buy_date": position["entry_date"],
                    "buy_price": position["entry_price"],
                    "sell_date": date, "sell_price": close,
                    "quantity": qty, "pnl": pnl,
                    "return_pct": ret, "reason": sell_reason,
                    "buy_type": position["buy_type"],
                })
                position = None
                above_ma = False
        else:
            bought = False
            # 경로A: 기준봉 돌파 매수
            if gijoongbong is not None:
                if close > gijoongbong["open_price"]:
                    max_qty = int(capital / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        cost = max_qty * close
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                            "buy_type": "기준봉돌파",
                        }
                        above_ma = False
                        bought = True
                    gijoongbong = None
            if not bought and gijoongbong is not None:
                if close > sma15:
                    gijoongbong = None

            if not bought and disp <= CRASH_TH:
                is_bearish = close < row["open"]
                is_bullish = close >= row["open"]
                vol_up = row["volume"] > df.iloc[i - 1]["volume"] if i > 0 else False

                # 경로B: 양봉 즉시 매수
                if is_bullish:
                    max_qty = int(capital / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        cost = max_qty * close
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                            "buy_type": "양봉즉시",
                        }
                        above_ma = False
                        bought = True
                # 기준봉 설정: 음봉 + 거래량 증가
                elif is_bearish and vol_up:
                    gijoongbong = {"date": date, "open_price": row["open"]}

        eq = capital + (position["quantity"] * close if position else 0)

    # 미청산
    if position is not None:
        last = df.iloc[-1]
        qty = position["quantity"]
        revenue = qty * last["close"]
        net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
        capital += net
        buy_cost = position["entry_price"] * qty
        buy_fee = buy_cost * FEE_BUY
        pnl = net - buy_cost - buy_fee
        ret = pnl / (buy_cost + buy_fee) * 100
        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": position["entry_date"],
            "buy_price": position["entry_price"],
            "sell_date": dates[-1], "sell_price": last["close"],
            "quantity": qty, "pnl": pnl,
            "return_pct": ret, "reason": "미청산",
            "buy_type": position["buy_type"],
        })

    trades_df = pd.DataFrame(trades)
    end_eq = capital
    total_ret = (end_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return total_ret, trades_df, end_eq


def run_simple(df: pd.DataFrame, ticker: str, name: str):
    """단순 매수: 이격도 92% 이하면 종가 매수"""
    capital = INITIAL_CAPITAL
    position = None
    above_ma = False

    trades = []
    dates = df.index.tolist()

    for i in range(1, len(df)):
        row = df.iloc[i]
        close = row["close"]
        sma15 = row["sma15"]
        disp = row["disparity15"]
        date = dates[i]

        if pd.isna(sma15):
            continue

        if position is not None:
            sell_reason = None
            if not above_ma and close > sma15:
                above_ma = True
            if close < position["buy_candle_low"]:
                sell_reason = "손절(매수봉저가)"
            elif disp >= SURGE_TH:
                sell_reason = "급등신호"
            elif above_ma and close < sma15:
                sell_reason = "15MA이탈"

            if sell_reason:
                qty = position["quantity"]
                revenue = qty * close
                fee = revenue * FEE_SELL
                tax = revenue * TAX_SELL
                net = revenue - fee - tax
                capital += net
                buy_cost = position["entry_price"] * qty
                buy_fee = buy_cost * FEE_BUY
                pnl = net - buy_cost - buy_fee
                ret = pnl / (buy_cost + buy_fee) * 100
                trades.append({
                    "ticker": ticker, "name": name,
                    "buy_date": position["entry_date"],
                    "buy_price": position["entry_price"],
                    "sell_date": date, "sell_price": close,
                    "quantity": qty, "pnl": pnl,
                    "return_pct": ret, "reason": sell_reason,
                })
                position = None
                above_ma = False
        else:
            if disp <= CRASH_TH:
                max_qty = int(capital / (close * (1 + FEE_BUY)))
                if max_qty > 0:
                    cost = max_qty * close
                    fee = cost * FEE_BUY
                    capital -= cost + fee
                    position = {
                        "entry_price": close, "quantity": max_qty,
                        "entry_date": date, "buy_candle_low": row["low"],
                    }
                    above_ma = False

        eq = capital + (position["quantity"] * close if position else 0)

    # 미청산
    if position is not None:
        last = df.iloc[-1]
        qty = position["quantity"]
        revenue = qty * last["close"]
        net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
        capital += net
        buy_cost = position["entry_price"] * qty
        buy_fee = buy_cost * FEE_BUY
        pnl = net - buy_cost - buy_fee
        ret = pnl / (buy_cost + buy_fee) * 100
        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": position["entry_date"],
            "buy_price": position["entry_price"],
            "sell_date": dates[-1], "sell_price": last["close"],
            "quantity": qty, "pnl": pnl,
            "return_pct": ret, "reason": "미청산",
        })

    trades_df = pd.DataFrame(trades)
    end_eq = capital
    total_ret = (end_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return total_ret, trades_df, end_eq


def run_v6(df: pd.DataFrame, ticker: str, name: str):
    """v6: v4 + 갭하락즉시. 기준봉조건+갭하락≥3%→즉시매수, 갭하락<3%→기준봉대기, 양봉→즉시매수"""
    capital = INITIAL_CAPITAL
    position = None
    gijoongbong = None
    above_ma = False

    trades = []
    dates = df.index.tolist()

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        close = row["close"]
        sma15 = row["sma15"]
        disp = row["disparity15"]
        date = dates[i]

        if pd.isna(sma15):
            continue

        if position is not None:
            sell_reason = None
            if not above_ma and close > sma15:
                above_ma = True
            if close < position["buy_candle_low"]:
                sell_reason = "손절(매수봉저가)"
            elif disp >= SURGE_TH:
                sell_reason = "급등신호"
            elif above_ma and close < sma15:
                sell_reason = "15MA이탈"

            if sell_reason:
                qty = position["quantity"]
                revenue = qty * close
                fee = revenue * FEE_SELL
                tax = revenue * TAX_SELL
                net = revenue - fee - tax
                capital += net
                buy_cost = position["entry_price"] * qty
                buy_fee = buy_cost * FEE_BUY
                pnl = net - buy_cost - buy_fee
                ret = pnl / (buy_cost + buy_fee) * 100
                trades.append({
                    "ticker": ticker, "name": name,
                    "buy_date": position["entry_date"],
                    "buy_price": position["entry_price"],
                    "sell_date": date, "sell_price": close,
                    "quantity": qty, "pnl": pnl,
                    "return_pct": ret, "reason": sell_reason,
                    "buy_type": position["buy_type"],
                })
                position = None
                above_ma = False
        else:
            bought = False
            prev_close = prev_row["close"]
            gap_pct = (row["open"] - prev_close) / prev_close if prev_close > 0 else 0
            is_bearish = close < row["open"]
            is_bullish = close >= row["open"]
            vol_up = row["volume"] > prev_row["volume"]

            # 경로A: 기존 기준봉 돌파 매수 (대기 중인 기준봉)
            if gijoongbong is not None:
                if close > gijoongbong["open_price"]:
                    max_qty = int(capital / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        cost = max_qty * close
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                            "buy_type": "기준봉돌파",
                        }
                        above_ma = False
                        bought = True
                    gijoongbong = None
            if not bought and gijoongbong is not None:
                if close > sma15:
                    gijoongbong = None

            if not bought and disp <= CRASH_TH:
                # 경로C: 이격도≤92 + 갭하락≥3% → 즉시 종가 매수 (음봉/거래량 불문)
                if gap_pct <= -GAP_DOWN_PCT:
                    max_qty = int(capital / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        cost = max_qty * close
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                            "buy_type": "갭하락즉시",
                        }
                        above_ma = False
                        bought = True
                    gijoongbong = None
                elif is_bearish and vol_up:
                    # 기준봉 설정, 돌파 대기
                    gijoongbong = {"date": date, "open_price": row["open"]}
                elif is_bullish:
                    # 경로B: 양봉 즉시 매수
                    max_qty = int(capital / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        cost = max_qty * close
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                            "buy_type": "양봉즉시",
                        }
                        above_ma = False
                        bought = True

        eq = capital + (position["quantity"] * close if position else 0)

    # 미청산
    if position is not None:
        last = df.iloc[-1]
        qty = position["quantity"]
        revenue = qty * last["close"]
        net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
        capital += net
        buy_cost = position["entry_price"] * qty
        buy_fee = buy_cost * FEE_BUY
        pnl = net - buy_cost - buy_fee
        ret = pnl / (buy_cost + buy_fee) * 100
        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": position["entry_date"],
            "buy_price": position["entry_price"],
            "sell_date": dates[-1], "sell_price": last["close"],
            "quantity": qty, "pnl": pnl,
            "return_pct": ret, "reason": "미청산",
            "buy_type": position["buy_type"],
        })

    trades_df = pd.DataFrame(trades)
    end_eq = capital
    total_ret = (end_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return total_ret, trades_df, end_eq


def run_v5(df: pd.DataFrame, ticker: str, name: str):
    """v5: 기준봉+갭하락3% → 당일 종가 즉시매수 / 갭하락 없으면 기준봉 돌파 대기"""
    capital = INITIAL_CAPITAL
    position = None
    gijoongbong = None
    above_ma = False

    trades = []
    dates = df.index.tolist()

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        close = row["close"]
        sma15 = row["sma15"]
        disp = row["disparity15"]
        date = dates[i]

        if pd.isna(sma15):
            continue

        if position is not None:
            sell_reason = None
            if not above_ma and close > sma15:
                above_ma = True
            if close < position["buy_candle_low"]:
                sell_reason = "손절(매수봉저가)"
            elif disp >= SURGE_TH:
                sell_reason = "급등신호"
            elif above_ma and close < sma15:
                sell_reason = "15MA이탈"

            if sell_reason:
                qty = position["quantity"]
                revenue = qty * close
                fee = revenue * FEE_SELL
                tax = revenue * TAX_SELL
                net = revenue - fee - tax
                capital += net
                buy_cost = position["entry_price"] * qty
                buy_fee = buy_cost * FEE_BUY
                pnl = net - buy_cost - buy_fee
                ret = pnl / (buy_cost + buy_fee) * 100
                trades.append({
                    "ticker": ticker, "name": name,
                    "buy_date": position["entry_date"],
                    "buy_price": position["entry_price"],
                    "sell_date": date, "sell_price": close,
                    "quantity": qty, "pnl": pnl,
                    "return_pct": ret, "reason": sell_reason,
                    "buy_type": position["buy_type"],
                })
                position = None
                above_ma = False
        else:
            bought = False
            prev_close = prev_row["close"]
            gap_pct = (row["open"] - prev_close) / prev_close if prev_close > 0 else 0
            is_bearish = close < row["open"]
            vol_up = row["volume"] > prev_row["volume"]

            # 경로A: 기존 기준봉 돌파 매수 (대기 중인 기준봉이 있을 때)
            if gijoongbong is not None:
                if close > gijoongbong["open_price"]:
                    max_qty = int(capital / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        cost = max_qty * close
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                            "buy_type": "기준봉돌파",
                        }
                        above_ma = False
                        bought = True
                    gijoongbong = None
            if not bought and gijoongbong is not None:
                if close > sma15:
                    gijoongbong = None

            # 경로B: 기준봉 조건 + 갭하락 3% 이상 → 당일 종가 즉시 매수
            if not bought and disp <= CRASH_TH and is_bearish and vol_up:
                if gap_pct <= -GAP_DOWN_PCT:
                    # 갭하락 크니까 즉시 매수, 기준봉 대기 안 함
                    max_qty = int(capital / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        cost = max_qty * close
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                            "buy_type": "갭하락즉시",
                        }
                        above_ma = False
                        bought = True
                    gijoongbong = None  # 즉시 매수했으므로 기준봉 대기 취소
                else:
                    # 갭하락 3% 미만 → 기존처럼 기준봉 설정 후 돌파 대기
                    gijoongbong = {"date": date, "open_price": row["open"]}

        eq = capital + (position["quantity"] * close if position else 0)

    # 미청산
    if position is not None:
        last = df.iloc[-1]
        qty = position["quantity"]
        revenue = qty * last["close"]
        net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
        capital += net
        buy_cost = position["entry_price"] * qty
        buy_fee = buy_cost * FEE_BUY
        pnl = net - buy_cost - buy_fee
        ret = pnl / (buy_cost + buy_fee) * 100
        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": position["entry_date"],
            "buy_price": position["entry_price"],
            "sell_date": dates[-1], "sell_price": last["close"],
            "quantity": qty, "pnl": pnl,
            "return_pct": ret, "reason": "미청산",
            "buy_type": position["buy_type"],
        })

    trades_df = pd.DataFrame(trades)
    end_eq = capital
    total_ret = (end_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return total_ret, trades_df, end_eq


def run_gap_gijoong(df: pd.DataFrame, ticker: str, name: str):
    """기준봉+갭하락3% 매수: 이격도92이하 + 음봉 + 거래량증가 + 갭하락3%이상 → 종가매수"""
    capital = INITIAL_CAPITAL
    position = None
    above_ma = False

    trades = []
    dates = df.index.tolist()

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        close = row["close"]
        sma15 = row["sma15"]
        disp = row["disparity15"]
        date = dates[i]

        if pd.isna(sma15):
            continue

        if position is not None:
            sell_reason = None
            if not above_ma and close > sma15:
                above_ma = True
            if close < position["buy_candle_low"]:
                sell_reason = "손절(매수봉저가)"
            elif disp >= SURGE_TH:
                sell_reason = "급등신호"
            elif above_ma and close < sma15:
                sell_reason = "15MA이탈"

            if sell_reason:
                qty = position["quantity"]
                revenue = qty * close
                fee = revenue * FEE_SELL
                tax = revenue * TAX_SELL
                net = revenue - fee - tax
                capital += net
                buy_cost = position["entry_price"] * qty
                buy_fee = buy_cost * FEE_BUY
                pnl = net - buy_cost - buy_fee
                ret = pnl / (buy_cost + buy_fee) * 100
                trades.append({
                    "ticker": ticker, "name": name,
                    "buy_date": position["entry_date"],
                    "buy_price": position["entry_price"],
                    "sell_date": date, "sell_price": close,
                    "quantity": qty, "pnl": pnl,
                    "return_pct": ret, "reason": sell_reason,
                })
                position = None
                above_ma = False
        else:
            prev_close = prev_row["close"]
            gap_pct = (row["open"] - prev_close) / prev_close
            is_bearish = close < row["open"]
            vol_up = row["volume"] > prev_row["volume"]

            # 기준봉 조건(이격도92이하 + 음봉 + 거래량증가) + 갭하락3%이상 → 종가매수
            if disp <= CRASH_TH and is_bearish and vol_up and gap_pct <= -GAP_DOWN_PCT:
                max_qty = int(capital / (close * (1 + FEE_BUY)))
                if max_qty > 0:
                    cost = max_qty * close
                    fee = cost * FEE_BUY
                    capital -= cost + fee
                    position = {
                        "entry_price": close, "quantity": max_qty,
                        "entry_date": date, "buy_candle_low": row["low"],
                    }
                    above_ma = False

    # 미청산
    if position is not None:
        last = df.iloc[-1]
        qty = position["quantity"]
        revenue = qty * last["close"]
        net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
        capital += net
        buy_cost = position["entry_price"] * qty
        buy_fee = buy_cost * FEE_BUY
        pnl = net - buy_cost - buy_fee
        ret = pnl / (buy_cost + buy_fee) * 100
        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": position["entry_date"],
            "buy_price": position["entry_price"],
            "sell_date": dates[-1], "sell_price": last["close"],
            "quantity": qty, "pnl": pnl,
            "return_pct": ret, "reason": "미청산",
        })

    trades_df = pd.DataFrame(trades)
    end_eq = capital
    total_ret = (end_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return total_ret, trades_df, end_eq


def summarize(trades_df, label):
    """거래 데이터프레임에서 통계 추출"""
    if trades_df.empty:
        return {}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    tc = trades_df.copy()
    tc["hold_days"] = (pd.to_datetime(tc["sell_date"]) - pd.to_datetime(tc["buy_date"])).dt.days

    return {
        "label": label,
        "n_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "total_pnl": trades_df["pnl"].sum(),
        "avg_return": trades_df["return_pct"].mean(),
        "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0,
        "avg_hold_days": tc["hold_days"].mean(),
        "max_return": trades_df["return_pct"].max(),
        "min_return": trades_df["return_pct"].min(),
    }


def main():
    start_time = time.time()

    # 종목 로딩
    print("[1/3] Leaders 종목 + 데이터 로딩...")
    leaders = get_leaders_tickers(70)
    ticker_list = [t["ticker"] for t in leaders]
    name_map = {t["ticker"]: t["name"] for t in leaders}

    all_data = load_all_data(ticker_list, START_DATE, END_DATE)
    print(f"      {len(all_data)}종목 로딩")

    # 백테스트
    print("[2/3] 세 전략 동시 백테스트...")
    v4_returns = []
    gap_returns = []
    v4_all_trades = []
    gap_all_trades = []

    for ticker in ticker_list:
        if ticker not in all_data:
            continue
        df = calc_indicators_15(all_data[ticker])
        df_test = df.loc[START_DATE:]
        if len(df_test) < 30:
            continue

        name = name_map.get(ticker, ticker)

        ret_v4, trades_v4, eq_v4 = run_v4(df_test, ticker, name)
        ret_gap, trades_gap, eq_gap = run_v6(df_test, ticker, name)

        v4_returns.append({"ticker": ticker, "name": name, "return": ret_v4, "n_trades": len(trades_v4)})
        gap_returns.append({"ticker": ticker, "name": name, "return": ret_gap, "n_trades": len(trades_gap)})

        if not trades_v4.empty:
            v4_all_trades.append(trades_v4)
        if not trades_gap.empty:
            gap_all_trades.append(trades_gap)

    v4_ret_df = pd.DataFrame(v4_returns)
    gap_ret_df = pd.DataFrame(gap_returns)
    v4_trades_df = pd.concat(v4_all_trades, ignore_index=True) if v4_all_trades else pd.DataFrame()
    gap_trades_df = pd.concat(gap_all_trades, ignore_index=True) if gap_all_trades else pd.DataFrame()

    # 결과 출력
    elapsed = time.time() - start_time
    print(f"[3/3] 결과 비교 ({elapsed:.1f}초)\n")

    v4_traded = v4_ret_df[v4_ret_df["n_trades"] > 0]
    gap_traded = gap_ret_df[gap_ret_df["n_trades"] > 0]

    v4_stats = summarize(v4_trades_df, "v4")
    gap_stats = summarize(gap_trades_df, "v6")

    def safe_div(a, b):
        return a / b if b else 0

    print("=" * 75)
    print(f"{'지표':<20} {'v4(기준봉+양봉)':<25} {'v6(v4+갭하락즉시)':<25}")
    print("=" * 75)
    print(f"{'거래 발생 종목':<20} {len(v4_traded):<25} {len(gap_traded):<25}")
    print(f"{'총 거래 건수':<20} {v4_stats.get('n_trades',0):<25} {gap_stats.get('n_trades',0):<25}")

    if not v4_traded.empty:
        print(f"{'종목 평균 수익률':<20} {v4_traded['return'].mean():.2f}%", end="")
    else:
        print(f"{'종목 평균 수익률':<20} {'N/A':<25}", end="")
    if not gap_traded.empty:
        print(f"{'':<15}{gap_traded['return'].mean():.2f}%")
    else:
        print(f"{'':<15}N/A")

    if not v4_traded.empty:
        print(f"{'종목 중위 수익률':<20} {v4_traded['return'].median():.2f}%", end="")
    else:
        print(f"{'종목 중위 수익률':<20} {'N/A':<25}", end="")
    if not gap_traded.empty:
        print(f"{'':<15}{gap_traded['return'].median():.2f}%")
    else:
        print(f"{'':<15}N/A")

    v4_pos = (v4_traded["return"] > 0).sum() if not v4_traded.empty else 0
    g_pos = (gap_traded["return"] > 0).sum() if not gap_traded.empty else 0
    print(f"{'수익/손실 종목':<20} {v4_pos}/{len(v4_traded)-v4_pos}{'':<20} {g_pos}/{len(gap_traded)-g_pos}")

    for label_name, stats in [("v4", v4_stats), ("갭하락", gap_stats)]:
        pass  # printed below

    print(f"{'전체 승률':<20} {v4_stats.get('win_rate',0):.1f}%{'':<20} {gap_stats.get('win_rate',0):.1f}%")
    print(f"{'손익비':<20} {v4_stats.get('profit_factor',0):.2f}{'':<21} {gap_stats.get('profit_factor',0):.2f}")
    print(f"{'총 손익':<20} {v4_stats.get('total_pnl',0):>15,.0f}원   {gap_stats.get('total_pnl',0):>15,.0f}원")
    print(f"{'거래당 평균 수익률':<20} {v4_stats.get('avg_return',0):.2f}%{'':<19} {gap_stats.get('avg_return',0):.2f}%")
    print(f"{'평균 수익(승)':<20} {v4_stats.get('avg_win',0):>15,.0f}원   {gap_stats.get('avg_win',0):>15,.0f}원")
    print(f"{'평균 손실(패)':<20} {v4_stats.get('avg_loss',0):>15,.0f}원   {gap_stats.get('avg_loss',0):>15,.0f}원")
    print(f"{'평균 보유일':<20} {v4_stats.get('avg_hold_days',0):.1f}일{'':<19} {gap_stats.get('avg_hold_days',0):.1f}일")
    v4_exp = safe_div(v4_stats.get('total_pnl',0), v4_stats.get('n_trades',1))
    g_exp = safe_div(gap_stats.get('total_pnl',0), gap_stats.get('n_trades',1))
    print(f"{'거래당 기대수익':<20} {v4_exp:>15,.0f}원   {g_exp:>15,.0f}원")
    print("=" * 75)

    # v4 매수 유형별 분석
    if not v4_trades_df.empty and "buy_type" in v4_trades_df.columns:
        print("\n[v4 매수 유형별 분석]")
        print("-" * 60)
        for btype, grp in v4_trades_df.groupby("buy_type"):
            wins = (grp["pnl"] > 0).sum()
            wr = wins / len(grp) * 100
            gp = grp[grp["pnl"] > 0]["pnl"].sum()
            gl = abs(grp[grp["pnl"] <= 0]["pnl"].sum())
            pf = gp / gl if gl > 0 else float("inf")
            print(f"  {btype}: {len(grp)}건, 승률 {wr:.1f}%, 손익비 {pf:.2f}, "
                  f"평균수익률 {grp['return_pct'].mean():.2f}%, 총손익 {grp['pnl'].sum():,.0f}원")

    # 매도 사유별 비교
    print("\n[매도 사유별 비교]")
    print("-" * 70)
    print(f"{'사유':<15} {'v4 건수':>8} {'v4 평균%':>10} {'v6 건수':>10} {'v6 평균%':>12}")
    print("-" * 70)
    all_reasons = set()
    if not v4_trades_df.empty:
        all_reasons.update(v4_trades_df["reason"].unique())
    if not gap_trades_df.empty:
        all_reasons.update(gap_trades_df["reason"].unique())
    for reason in sorted(all_reasons):
        v4_grp = v4_trades_df[v4_trades_df["reason"] == reason] if not v4_trades_df.empty else pd.DataFrame()
        g_grp = gap_trades_df[gap_trades_df["reason"] == reason] if not gap_trades_df.empty else pd.DataFrame()
        v4_n = len(v4_grp)
        v4_avg = v4_grp["return_pct"].mean() if v4_n > 0 else 0
        g_n = len(g_grp)
        g_avg = g_grp["return_pct"].mean() if g_n > 0 else 0
        print(f"  {reason:<15} {v4_n:>6}   {v4_avg:>8.2f}%     {g_n:>6}   {g_avg:>10.2f}%")

    # 종목별 비교 (거래 발생 종목끼리만)
    merged = v4_ret_df.merge(gap_ret_df, on=["ticker", "name"], suffixes=("_v4", "_gap"))
    both_traded = merged[(merged["n_trades_v4"] > 0) & (merged["n_trades_gap"] > 0)]
    if not both_traded.empty:
        both_traded = both_traded.copy()
        both_traded["diff"] = both_traded["return_v4"] - both_traded["return_gap"]
        v4_better = (both_traded["diff"] > 0).sum()
        gap_better = (both_traded["diff"] < 0).sum()
        tie = (both_traded["diff"] == 0).sum()
        print(f"\n[종목별 승패 (양쪽 모두 거래 발생: {len(both_traded)}종목)]")
        print(f"  v4 우위: {v4_better}종목, v6 우위: {gap_better}종목, 동일: {tie}종목")
        print(f"  v4 평균 차이: {both_traded['diff'].mean():+.2f}%p")

    # v4만 거래, 갭하락만 거래 종목 수
    v4_only = merged[(merged["n_trades_v4"] > 0) & (merged["n_trades_gap"] == 0)]
    gap_only = merged[(merged["n_trades_v4"] == 0) & (merged["n_trades_gap"] > 0)]
    print(f"\n[거래 발생 분포]")
    print(f"  v4만 거래: {len(v4_only)}종목, v6만 거래: {len(gap_only)}종목, 양쪽 모두: {len(both_traded)}종목")

    # v6 매수 유형별 분석
    if not gap_trades_df.empty and "buy_type" in gap_trades_df.columns:
        print("\n[v6 매수 유형별 분석]")
        print("-" * 60)
        for btype, grp in gap_trades_df.groupby("buy_type"):
            wins = (grp["pnl"] > 0).sum()
            wr = wins / len(grp) * 100
            gp = grp[grp["pnl"] > 0]["pnl"].sum()
            gl = abs(grp[grp["pnl"] <= 0]["pnl"].sum())
            pf = gp / gl if gl > 0 else float("inf")
            print(f"  {btype}: {len(grp)}건, 승률 {wr:.1f}%, 손익비 {pf:.2f}, "
                  f"평균수익률 {grp['return_pct'].mean():.2f}%, 총손익 {grp['pnl'].sum():,.0f}원")


if __name__ == "__main__":
    main()
