"""
계단뛰기 매매 백테스트 (research/계단뛰기.md)

시그널 (일봉, 봉 인덱스 (n) = n봉전):
  1) open(2) < close(2)
  2) open(1) > close(2)
  3) low(1) > (open(2)+close(2))/2
  4) open(0) < close(0)
  Branch A: 5) open(1) < close(1)  6) low(0) > (open(1)+close(1))/2  17) close(1) < close(0)
  Branch B: 7) open(1) > close(1)  8) low(0) >= (open(2)+close(2))/2
            9) volume(2) > volume(1)  16) open(1) < close(0)
  10) close(2) > ma5(2)
  11) close(1) > ma5(1)
  12) close(0) > ma5(0)
  13) close(0) > ma120(0)
  14) 시가총액 >= 3000억 (universe 필터)
  15) close >= 2000

매수:
  - 조건 4로 시그널 당일은 항상 양봉 → 다음날 시그널 당일 종가 상향 돌파 시 매수
  - 미체결 시 다음날 만료

매도 (3-파라미터, 최적화 대상):
  - Tgt%: 익절 (고가가 entry*(1+Tgt) 도달 시 limit 청산)
  - SL%:  손절 (저가가 entry*(1-SL)  도달 시 stop 청산)
  - Min%: 이익보전 (고가가 entry*(1+Min) 한 번이라도 도달 후
                    종가가 entry*(1+Min) 하향 시 entry*(1+Min)에 청산)
  - 우선순위: target -> stop -> trailing

테스트 기간: 2025-01-01 ~ today
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_crash import (
    ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    load_all_data,
)
from backtest_3bar import get_mcap_tickers


START_DATE = "2025-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

MCAP_THRESHOLD = 3e11      # 3000억
PRICE_FLOOR = 2000          # 종가 ≥ 2000원

DEFAULT_TGT = 7.0
DEFAULT_MIN = 3.0
DEFAULT_SL = 3.0


# ──────────────────────────────────────────────
# 지표
# ──────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma120"] = df["close"].rolling(window=120).mean()
    return df


# ──────────────────────────────────────────────
# 계단뛰기 시그널
# ──────────────────────────────────────────────
def find_signals(df: pd.DataFrame) -> list[int]:
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["volume"].values
    ma5 = df["ma5"].values
    ma120 = df["ma120"].values

    signals = []
    for i in range(2, len(df)):
        # 결측 체크
        if (np.isnan(ma5[i]) or np.isnan(ma5[i-1]) or np.isnan(ma5[i-2])
                or np.isnan(ma120[i])):
            continue

        # 1) open(2) < close(2)
        if not (o[i-2] < c[i-2]):
            continue
        # 2) open(1) > close(2)
        if not (o[i-1] > c[i-2]):
            continue
        # 3) low(1) > (open(2)+close(2))/2
        if not (l[i-1] > (o[i-2] + c[i-2]) / 2):
            continue
        # 4) open(0) < close(0)
        if not (o[i] < c[i]):
            continue

        mid2 = (o[i-2] + c[i-2]) / 2
        mid1 = (o[i-1] + c[i-1]) / 2

        # Branch A: 5 && 6 && 17
        a = (o[i-1] < c[i-1]) and (l[i] > mid1) and (c[i-1] < c[i])
        # Branch B: 7 && 8 && 9 && 16
        b = (o[i-1] > c[i-1]) and (l[i] >= mid2) and (v[i-2] > v[i-1]) and (o[i-1] < c[i])
        if not (a or b):
            continue

        # 10) close(2) > ma5(2), 11) close(1) > ma5(1), 12) close(0) > ma5(0)
        if not (c[i-2] > ma5[i-2] and c[i-1] > ma5[i-1] and c[i] > ma5[i]):
            continue
        # 13) close(0) > ma120(0)
        if not (c[i] > ma120[i]):
            continue
        # 15) close >= PRICE_FLOOR
        if c[i] < PRICE_FLOOR:
            continue

        signals.append(i)
    return signals


# ──────────────────────────────────────────────
# 백테스트 (종목 1개)
# ──────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, ticker: str, name: str,
                 tgt_pct: float, min_pct: float, sl_pct: float,
                 exit_mode: str = "tpsl"):
    """다음날 시그널종가 상향돌파 매수.

    exit_mode:
      - "tpsl"   : 익절 Tgt% + 손절 SL% + 이익보전 Min% (기존)
      - "ma5"    : 익절 Tgt% + MA5 종가이탈 손절 + 이익보전 Min% (sl_pct 미사용)
      - "sameday": 매수 당일 종가에 무조건 매도 (모든 % 미사용)
    """
    capital = INITIAL_CAPITAL
    position = None
    pending_entry = None  # {"trigger_price": float, "entry_idx": int}
    trades = []
    equity_curve = []
    dates = df.index.tolist()

    signals = set(find_signals(df))
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    ma5_arr = df["ma5"].values

    def close_trade(sell_price, date, reason):
        nonlocal capital, position
        qty = position["quantity"]
        entry = position["entry_price"]
        revenue = qty * sell_price
        fee = revenue * FEE_SELL
        tax = revenue * TAX_SELL
        net = revenue - fee - tax
        capital += net
        buy_cost = entry * qty
        buy_fee = buy_cost * FEE_BUY
        pnl = net - buy_cost - buy_fee
        ret = pnl / (buy_cost + buy_fee) * 100
        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": position["entry_date"],
            "buy_price": entry,
            "sell_date": date, "sell_price": sell_price,
            "quantity": qty, "pnl": pnl,
            "return_pct": ret, "reason": reason,
        })
        position = None

    for i in range(len(df)):
        date = dates[i]
        bar_open = o[i]
        bar_high = h[i]
        bar_low = l[i]
        bar_close = c[i]
        ma5_val = ma5_arr[i]

        # ── 1) 매도 판정 (전일 이전부터 보유 중인 포지션) ──
        if position is not None and position.get("entry_idx") != i:
            entry = position["entry_price"]
            target_price = entry * (1 + tgt_pct / 100)
            min_price = entry * (1 + min_pct / 100)

            sell_price = None
            reason = None

            if exit_mode == "tpsl":
                stop_price = entry * (1 - sl_pct / 100)
                if bar_open >= target_price:
                    sell_price, reason = bar_open, "익절(갭)"
                elif bar_open <= stop_price:
                    sell_price, reason = bar_open, "손절(갭)"
                elif bar_high >= target_price:
                    sell_price, reason = target_price, "익절"
                elif bar_low <= stop_price:
                    sell_price, reason = stop_price, "손절"
                else:
                    if bar_high >= min_price:
                        position["trail_armed"] = True
                    if position["trail_armed"] and bar_close < min_price:
                        sell_price, reason = bar_close, "이익보전"

            elif exit_mode == "ma5":
                # 익절(갭) → 익절(intra) → MA5 종가이탈 → 이익보전 트레일
                if bar_open >= target_price:
                    sell_price, reason = bar_open, "익절(갭)"
                elif bar_high >= target_price:
                    sell_price, reason = target_price, "익절"
                elif not np.isnan(ma5_val) and bar_close < ma5_val:
                    sell_price, reason = bar_close, "MA5이탈"
                else:
                    if bar_high >= min_price:
                        position["trail_armed"] = True
                    if position["trail_armed"] and bar_close < min_price:
                        sell_price, reason = bar_close, "이익보전"

            if sell_price is not None:
                close_trade(sell_price, date, reason)

        # ── 2) 매수 (대기 중인 트리거 처리) ──
        if position is None and pending_entry is not None:
            trig = pending_entry["trigger_price"]
            if i == pending_entry["entry_idx"]:
                if bar_open >= trig:
                    buy_price = bar_open
                    triggered = True
                elif bar_high >= trig:
                    buy_price = trig
                    triggered = True
                else:
                    triggered = False
                if triggered:
                    max_qty = int(capital / (buy_price * (1 + FEE_BUY)))
                    if max_qty > 0:
                        cost = max_qty * buy_price
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = {
                            "entry_price": buy_price,
                            "quantity": max_qty,
                            "entry_date": date,
                            "entry_idx": i,
                            "trail_armed": False,
                        }
                pending_entry = None
            elif i > pending_entry["entry_idx"]:
                pending_entry = None

        # ── 2b) 당일 종가 매도 (sameday 모드) ──
        if exit_mode == "sameday" and position is not None and position.get("entry_idx") == i:
            close_trade(bar_close, date, "당일종가")

        # ── 3) 시그널 등록 (다음봉 매수 예약) ──
        if i in signals and position is None and pending_entry is None:
            if i + 1 < len(df):
                pending_entry = {
                    "trigger_price": bar_close,
                    "entry_idx": i + 1,
                }

        # ── 4) 자산 기록 ──
        eq = capital + (position["quantity"] * bar_close if position else 0)
        equity_curve.append({"date": date, "equity": eq})

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

    return capital, trades, equity_curve, len(signals)


# ──────────────────────────────────────────────
# 단일 파라미터 풀-유니버스 실행
# ──────────────────────────────────────────────
def run_universe(tgt_pct: float, min_pct: float, sl_pct: float,
                 ticker_list: list[str], name_map: dict,
                 all_data: dict, start_date: str, end_date: str,
                 verbose: bool = True, exit_mode: str = "tpsl"):
    summaries = []
    all_trades = []
    n_signals_total = 0

    for ticker in ticker_list:
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        df = calc_indicators(df)
        df_test = df.loc[start_date:end_date]
        if len(df_test) < 5:
            continue
        nm = name_map.get(ticker, ticker)
        cap, trades, equity, n_sig = run_backtest(
            df_test, ticker, nm, tgt_pct, min_pct, sl_pct,
            exit_mode=exit_mode,
        )
        n_signals_total += n_sig
        if trades:
            t_df = pd.DataFrame(trades)
            wins = (t_df["pnl"] > 0).sum()
            summaries.append({
                "ticker": ticker, "name": nm,
                "n_signals": n_sig,
                "n_trades": len(trades),
                "final_equity": cap if not trades else equity[-1]["equity"],
                "win_rate": wins / len(trades) * 100,
                "total_pnl": t_df["pnl"].sum(),
                "avg_return": t_df["return_pct"].mean(),
            })
            all_trades.append(t_df)
        else:
            summaries.append({
                "ticker": ticker, "name": nm,
                "n_signals": n_sig,
                "n_trades": 0,
                "final_equity": INITIAL_CAPITAL,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_return": 0,
            })

    sum_df = pd.DataFrame(summaries)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return sum_df, trades_df, n_signals_total


def _metrics(trades_df: pd.DataFrame, sum_df: pd.DataFrame) -> dict:
    m = {"n_stocks": len(sum_df), "n_traded": int((sum_df["n_trades"] > 0).sum()) if not sum_df.empty else 0,
         "n_trades": int(len(trades_df))}
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        m["win_rate"] = len(wins) / len(trades_df) * 100
        gp = wins["pnl"].sum() if len(wins) else 0
        gl = abs(losses["pnl"].sum()) if len(losses) else 0
        m["pf"] = gp / gl if gl > 0 else float("inf")
        m["total_pnl"] = trades_df["pnl"].sum()
        m["avg_ret"] = trades_df["return_pct"].mean()
        tc = trades_df.copy()
        tc["hd"] = (pd.to_datetime(tc["sell_date"]) - pd.to_datetime(tc["buy_date"])).dt.days
        m["hold_days"] = tc["hd"].mean()
        # 종목 평균 수익률 (per-stock 단리 합산 / 거래종목 수)
        traded = sum_df[sum_df["n_trades"] > 0]
        if not traded.empty:
            traded = traded.assign(stock_ret=(traded["final_equity"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100)
            m["stock_avg_ret"] = traded["stock_ret"].mean()
            m["stock_med_ret"] = traded["stock_ret"].median()
        else:
            m["stock_avg_ret"] = m["stock_med_ret"] = 0
    else:
        m.update({"win_rate": 0, "pf": 0, "total_pnl": 0, "avg_ret": 0,
                  "hold_days": 0, "stock_avg_ret": 0, "stock_med_ret": 0})
    return m


# ──────────────────────────────────────────────
# 유니버스 로드 (시총 ≥ 3000억)
# ──────────────────────────────────────────────
def load_universe():
    print(f"[1/3] 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억 유니버스 산정...")
    mcap_map = get_mcap_tickers(MCAP_THRESHOLD)
    print(f"        {len(mcap_map)}종목 (mcap 기반 추정)")

    with ENGINE.connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT ticker, name FROM stocks ORDER BY ticker"
        )).fetchall()
    name_map = {r[0]: r[1] for r in rows}

    ticker_list = sorted(t for t in mcap_map.keys() if t in name_map)
    return ticker_list, name_map


def load_all(ticker_list: list[str], start: str, end: str) -> dict:
    print(f"[2/3] 데이터 로딩 ({start} ~ {end})...")
    all_data: dict[str, pd.DataFrame] = {}
    batch = 500
    for i in range(0, len(ticker_list), batch):
        chunk = ticker_list[i:i + batch]
        d = load_all_data(chunk, start, end)
        all_data.update(d)
    print(f"        {len(all_data)}종목 로딩 완료")
    return all_data


# ──────────────────────────────────────────────
# 리포트 (단일 파라미터)
# ──────────────────────────────────────────────
def make_single_report(sum_df, trades_df, n_signals_total,
                       tgt: float, min_p: float, sl: float, elapsed: float,
                       exit_mode: str = "tpsl") -> str:
    m = _metrics(trades_df, sum_df)
    lines = []
    exit_desc = {
        "tpsl": f"익절 {tgt}% / 손절 {sl}% / 이익보전 {min_p}%",
        "ma5": f"익절 {tgt}% / MA5 종가이탈 손절 / 이익보전 {min_p}%",
        "sameday": "매수 당일 종가 무조건 매도",
    }[exit_mode]
    title_tag = {"tpsl": "Tgt/Min/SL", "ma5": "Tgt/Min/MA5", "sameday": "당일종가"}[exit_mode]
    lines.append(f"# 계단뛰기 백테스트 ({title_tag})\n")
    lines.append("## 전략 개요\n")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **유니버스**: 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억 (추정), 종가 ≥ {PRICE_FLOOR:,}원")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append(f"- **매수**: 시그널 다음봉 시그널종가 상향 돌파 (만료 1봉)")
    lines.append(f"- **매도**: {exit_desc}\n")

    lines.append("## 성과 요약\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 유니버스 종목 | {m['n_stocks']:,} |")
    lines.append(f"| 거래 발생 종목 | {m['n_traded']:,} |")
    lines.append(f"| 총 시그널 | {n_signals_total:,} |")
    lines.append(f"| 총 거래 | {m['n_trades']:,} |")
    if m["n_trades"]:
        lines.append(f"| 종목 평균 수익률 | {m['stock_avg_ret']:.2f}% |")
        lines.append(f"| 종목 중위 수익률 | {m['stock_med_ret']:.2f}% |")
        lines.append(f"| 전체 승률 | {m['win_rate']:.1f}% |")
        lines.append(f"| 손익비 (PF) | {m['pf']:.2f}" + (" |" if m['pf'] != float('inf') else ' (∞) |'))
        lines.append(f"| 총 손익 | {m['total_pnl']:,.0f}원 |")
        lines.append(f"| 거래당 평균 수익률 | {m['avg_ret']:.2f}% |")
        lines.append(f"| 평균 보유일 | {m['hold_days']:.1f}일 |")
    lines.append("")

    # 사유 분포
    if not trades_df.empty:
        lines.append("## 매도 사유 분포\n")
        lines.append("| 사유 | 건수 | 평균 수익률 | 승률 | 총 손익 |")
        lines.append("|------|------|------------|------|---------|")
        for reason, grp in trades_df.groupby("reason"):
            n = len(grp)
            wr = (grp["pnl"] > 0).sum() / n * 100
            lines.append(f"| {reason} | {n:,} | {grp['return_pct'].mean():.2f}% | {wr:.1f}% | {grp['pnl'].sum():,.0f} |")
        lines.append("")

    # TOP 10
    if not sum_df.empty:
        traded = sum_df[sum_df["n_trades"] > 0].copy()
        if not traded.empty:
            traded["stock_ret"] = (traded["final_equity"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            lines.append("## TOP 10 종목\n")
            lines.append("| # | 종목 | 코드 | 수익률 | 거래 | 승률 |")
            lines.append("|---|------|------|--------|------|------|")
            for j, (_, r) in enumerate(traded.nlargest(10, "stock_ret").iterrows()):
                lines.append(f"| {j+1} | {r['name']} | {r['ticker']} | {r['stock_ret']:.2f}% | {r['n_trades']} | {r['win_rate']:.0f}% |")
            lines.append("")
            lines.append("## BOTTOM 10 종목\n")
            lines.append("| # | 종목 | 코드 | 수익률 | 거래 | 승률 |")
            lines.append("|---|------|------|--------|------|------|")
            for j, (_, r) in enumerate(traded.nsmallest(10, "stock_ret").iterrows()):
                lines.append(f"| {j+1} | {r['name']} | {r['ticker']} | {r['stock_ret']:.2f}% | {r['n_trades']} | {r['win_rate']:.0f}% |")
            lines.append("")

    # 전체 거래 (최대 200건)
    if not trades_df.empty:
        lines.append("## 전체 거래 기록 (최근 200건)\n")
        lines.append("| # | 종목 | 매수일 | 매수가 | 매도일 | 매도가 | 수익률 | 사유 |")
        lines.append("|---|------|--------|--------|--------|--------|--------|------|")
        td = trades_df.sort_values("sell_date").tail(200)
        for j, (_, t) in enumerate(td.iterrows()):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            lines.append(f"| {j+1} | {t['name']} | {bd} | {t['buy_price']:,.0f} | {sd} | {t['sell_price']:,.0f} | {t['return_pct']:.2f}% | {t['reason']} |")
        lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- 실행 시간: {elapsed:.2f}초")
    lines.append(f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 리포트 (옵티마이즈)
# ──────────────────────────────────────────────
def make_opt_report(rows: list[dict], elapsed: float, exit_mode: str = "tpsl") -> str:
    df = pd.DataFrame(rows)
    lines = []
    title_tag = {"tpsl": "Tgt/Min/SL", "ma5": "Tgt/Min/MA5손절"}[exit_mode]
    stop_label = "SL" if exit_mode == "tpsl" else "MA5"
    stop_desc = "고정 손절 %" if exit_mode == "tpsl" else "MA5 종가이탈 손절"
    lines.append(f"# 계단뛰기 백테스트 — 매도 최적화 ({title_tag})\n")
    lines.append("## 전략 개요\n")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **유니버스**: 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억, 종가 ≥ {PRICE_FLOOR:,}원")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append("- **매수**: 시그널 다음봉 시그널종가 상향 돌파")
    lines.append(f"- **매도 후보**: 익절(Tgt) × 이익보전(Min) × {stop_desc}")
    lines.append("")

    def stop_cell(sl_val: float) -> str:
        return f"{sl_val:.1f}%" if exit_mode == "tpsl" else "MA5"

    # 전체 결과
    lines.append("## 전체 그리드 결과\n")
    lines.append(f"| # | Tgt | Min | {stop_label} | 거래 | 승률 | PF | 총손익 | 거래당% | 종목평균% | 보유일 |")
    lines.append("|---|----:|----:|---:|-----:|-----:|----:|--------:|--------:|----------:|------:|")
    df_sorted = df.sort_values("total_pnl", ascending=False).reset_index(drop=True)
    for j, r in df_sorted.iterrows():
        pf_s = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "∞"
        lines.append(
            f"| {j+1} | {r['tgt']:.1f}% | {r['min']:.1f}% | {stop_cell(r['sl'])} "
            f"| {int(r['n_trades']):,} | {r['win_rate']:.1f}% | {pf_s} "
            f"| {r['total_pnl']:,.0f} | {r['avg_ret']:.2f}% | {r['stock_avg_ret']:.2f}% | {r['hold_days']:.1f} |"
        )
    lines.append("")

    # TOP 10 by metric
    for metric, asc, title in [
        ("total_pnl", False, "총손익 TOP 10"),
        ("pf", False, "손익비 TOP 10"),
        ("stock_avg_ret", False, "종목 평균 수익률 TOP 10"),
    ]:
        d = df.copy()
        if metric == "pf":
            d = d[d["pf"] != float("inf")]
        d = d.sort_values(metric, ascending=asc).head(10).reset_index(drop=True)
        lines.append(f"## {title}\n")
        lines.append(f"| # | Tgt | Min | {stop_label} | 거래 | 승률 | PF | 총손익 | 거래당% | 종목평균% |")
        lines.append("|---|----:|----:|---:|-----:|-----:|----:|--------:|--------:|----------:|")
        for j, r in d.iterrows():
            pf_s = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "∞"
            lines.append(
                f"| {j+1} | {r['tgt']:.1f}% | {r['min']:.1f}% | {stop_cell(r['sl'])} "
                f"| {int(r['n_trades']):,} | {r['win_rate']:.1f}% | {pf_s} "
                f"| {r['total_pnl']:,.0f} | {r['avg_ret']:.2f}% | {r['stock_avg_ret']:.2f}% |"
            )
        lines.append("")

    # 베스트 추천
    best_pnl = df.sort_values("total_pnl", ascending=False).iloc[0]
    df_pf = df[df["pf"] != float("inf")]
    best_pf = df_pf.sort_values("pf", ascending=False).iloc[0] if not df_pf.empty else best_pnl
    lines.append("## 최적 파라미터 추천\n")
    lines.append(f"- **총손익 최대**: Tgt={best_pnl['tgt']:.1f}%, Min={best_pnl['min']:.1f}%, {stop_label}={stop_cell(best_pnl['sl'])} → 총손익 {best_pnl['total_pnl']:,.0f}원, PF {best_pnl['pf']:.2f}, 승률 {best_pnl['win_rate']:.1f}%")
    lines.append(f"- **손익비 최대**: Tgt={best_pf['tgt']:.1f}%, Min={best_pf['min']:.1f}%, {stop_label}={stop_cell(best_pf['sl'])} → PF {best_pf['pf']:.2f}, 총손익 {best_pf['total_pnl']:,.0f}원, 거래 {int(best_pf['n_trades'])}건")
    lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- 실행 시간: {elapsed:.2f}초")
    lines.append(f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def _exit_label(exit_mode: str) -> str:
    return {"tpsl": "Tgt/Min/SL", "ma5": "Tgt/Min/MA5", "sameday": "당일종가"}[exit_mode]


def _report_suffix(exit_mode: str) -> str:
    return {"tpsl": "", "ma5": "_ma5", "sameday": "_sameday"}[exit_mode]


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def run_single(tgt: float, min_p: float, sl: float, exit_mode: str = "tpsl"):
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    ticker_list, name_map = load_universe()
    all_data = load_all(ticker_list, START_DATE, END_DATE)

    label = _exit_label(exit_mode)
    if exit_mode == "sameday":
        print(f"[3/3] 백테스트 ({label})...")
    elif exit_mode == "ma5":
        print(f"[3/3] 백테스트 ({label}: Tgt={tgt}% / Min={min_p}% / MA5손절)...")
    else:
        print(f"[3/3] 백테스트 ({label}: Tgt={tgt}% / Min={min_p}% / SL={sl}%)...")
    sum_df, trades_df, n_sig = run_universe(
        tgt, min_p, sl, ticker_list, name_map, all_data, START_DATE, END_DATE,
        exit_mode=exit_mode,
    )
    elapsed = time.time() - start_time

    n_traded = int((sum_df["n_trades"] > 0).sum()) if not sum_df.empty else 0
    print(f"        시그널 {n_sig:,}건, 거래종목 {n_traded}, 총거래 {len(trades_df):,}건")

    report = make_single_report(sum_df, trades_df, n_sig, tgt, min_p, sl, elapsed, exit_mode=exit_mode)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"backtest_stair_jump{_report_suffix(exit_mode)}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트: {out} ({elapsed:.2f}초)")


def run_optimize(exit_mode: str = "tpsl"):
    """매도 파라미터 그리드 최적화. exit_mode='tpsl' 또는 'ma5'."""
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    ticker_list, name_map = load_universe()
    all_data = load_all(ticker_list, START_DATE, END_DATE)

    tgt_grid = [5.0, 7.0, 10.0, 15.0, 20.0]
    min_grid = [1.0, 2.0, 3.0, 5.0]
    if exit_mode == "tpsl":
        sl_grid = [2.0, 3.0, 4.0, 5.0]
        combos = [(t, mn, s) for t in tgt_grid for mn in min_grid for s in sl_grid if mn < t]
    elif exit_mode == "ma5":
        # SL%은 MA5로 대체 → sl 차원 제거
        combos = [(t, mn, 0.0) for t in tgt_grid for mn in min_grid if mn < t]
    else:
        raise ValueError(f"optimize not supported for exit_mode={exit_mode}")

    print(f"[3/3] 그리드 최적화 ({_exit_label(exit_mode)}): {len(combos)} 조합")
    rows = []
    for k, (t, mn, s) in enumerate(combos, 1):
        sub_t = time.time()
        sum_df, trades_df, n_sig = run_universe(
            t, mn, s, ticker_list, name_map, all_data, START_DATE, END_DATE,
            verbose=False, exit_mode=exit_mode,
        )
        m = _metrics(trades_df, sum_df)
        row = {
            "tgt": t, "min": mn, "sl": s,
            "n_signals": n_sig,
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "pf": m["pf"],
            "total_pnl": m["total_pnl"],
            "avg_ret": m["avg_ret"],
            "stock_avg_ret": m["stock_avg_ret"],
            "stock_med_ret": m["stock_med_ret"],
            "hold_days": m["hold_days"],
        }
        rows.append(row)
        sub_e = time.time() - sub_t
        pf_s = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "∞"
        s_tag = f"S={s:.1f}" if exit_mode == "tpsl" else "MA5"
        print(f"  [{k:>3}/{len(combos)}] T={t:.1f} M={mn:.1f} {s_tag} | "
              f"거래 {m['n_trades']:,} 승률 {m['win_rate']:.1f}% PF {pf_s} "
              f"PNL {m['total_pnl']:>14,.0f} ({sub_e:.1f}s)")

    elapsed = time.time() - start_time
    report = make_opt_report(rows, elapsed, exit_mode=exit_mode)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"backtest_stair_jump_optimize{_report_suffix(exit_mode)}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트: {out} ({elapsed:.2f}초)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="계단뛰기 매매 백테스트")
    parser.add_argument("--optimize", action="store_true",
                        help="매도 파라미터 그리드 최적화")
    parser.add_argument("--exit", dest="exit_mode", choices=["tpsl", "ma5", "sameday"],
                        default="tpsl", help="매도 방식 (default tpsl)")
    parser.add_argument("--tgt", type=float, default=DEFAULT_TGT, help="익절 %% (default 7)")
    parser.add_argument("--min", dest="min_p", type=float, default=DEFAULT_MIN, help="이익보전 %% (default 3)")
    parser.add_argument("--sl", type=float, default=DEFAULT_SL, help="손절 %% (default 3)")
    args = parser.parse_args()

    if args.optimize:
        if args.exit_mode == "sameday":
            print("sameday 모드는 최적화 파라미터 없음 → 단일 실행으로 처리")
            run_single(args.tgt, args.min_p, args.sl, exit_mode="sameday")
        else:
            run_optimize(exit_mode=args.exit_mode)
    else:
        run_single(args.tgt, args.min_p, args.sl, exit_mode=args.exit_mode)
