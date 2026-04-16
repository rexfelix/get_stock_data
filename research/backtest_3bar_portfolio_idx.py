"""
3-bar 포트폴리오 백테스트 + 지수 필터 비교
- 유니버스: KOSPI200 ∪ 시총≥3조
- 슬롯: 최대 10종목, 거래대금 상위 우선
- 매도: MA5이탈 (원규칙, 기본)
- 지수 필터 4가지 비교:
    F0: 필터 없음 (baseline)
    F1: KOSPI > MA60
    F2: KOSPI > MA20 AND MA20 상승
    F3: KOSPI 60일 수익률 > 0
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_3bar import calc_indicators, find_signals, get_mcap_tickers, load_amount_data, ENGINE
from backtest_crash import (
    load_all_data, get_kospi200_tickers, FEE_BUY, FEE_SELL, TAX_SELL,
)

START_DATE = "2023-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
MAX_POSITIONS = 10
SLOT_CAPITAL = 10_000_000
INITIAL_CAPITAL = MAX_POSITIONS * SLOT_CAPITAL

EXIT_STRATEGY = "MA5이탈"


# ──────────────────────────────────────────────
# 지수 데이터 + 필터
# ──────────────────────────────────────────────
def load_kospi_filters():
    """KOSPI 일봉 로드하고 4가지 필터 시리즈 계산."""
    q = """
        SELECT date, close FROM market_indices
        WHERE symbol='^KS11' AND date >= '2022-01-01'
        ORDER BY date ASC
    """
    df = pd.read_sql(q, ENGINE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma20_up"] = df["ma20"] > df["ma20"].shift(1)
    df["ret60"] = df["close"] / df["close"].shift(60) - 1

    filters = {
        "F0_없음":        pd.Series(True, index=df.index),
        "F1_MA60위":      df["close"] > df["ma60"],
        "F2_MA20위_상승": (df["close"] > df["ma20"]) & df["ma20_up"],
        "F3_60일수익양": df["ret60"] > 0,
    }
    return df, filters


# ──────────────────────────────────────────────
# 유니버스
# ──────────────────────────────────────────────
def build_universe():
    kospi200 = get_kospi200_tickers()
    with ENGINE.connect() as conn:
        db_rows = conn.execute(text("SELECT DISTINCT ticker FROM stocks")).fetchall()
    db_tickers = set(r[0] for r in db_rows)
    k200 = {t["ticker"]: t["name"] for t in kospi200 if t["ticker"] in db_tickers}
    mcap = get_mcap_tickers(3e12)
    with ENGINE.connect() as conn:
        name_rows = conn.execute(text("SELECT DISTINCT ticker, name FROM stocks")).fetchall()
    all_names = {r[0]: r[1] for r in name_rows}
    merged = sorted(set(k200) | set(mcap))
    name_map = {t: k200.get(t) or all_names.get(t, t) for t in merged}
    return merged, name_map


# ──────────────────────────────────────────────
# 포트폴리오 엔진 (지수 필터 적용)
# ──────────────────────────────────────────────
def run_portfolio(ind_data, amount_data, ticker_list, name_map,
                   index_filter: pd.Series, filter_name: str):
    all_dates = set()
    for df in ind_data.values():
        all_dates.update(df.index)
    all_dates = sorted(d for d in all_dates
                       if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE))

    signals_by_date = {}
    for t in ticker_list:
        if t not in ind_data:
            continue
        df = ind_data[t]
        sigs = find_signals(df)
        for i in sigs:
            d = df.index[i]
            if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE):
                signals_by_date.setdefault(d, []).append(t)

    positions = {}
    cash = INITIAL_CAPITAL
    trades = []
    equity_curve = []
    blocked_days = 0
    signal_days = 0

    for d in all_dates:
        # (A) 매도 먼저
        to_close = []
        for t, pos in positions.items():
            if t not in ind_data or d not in ind_data[t].index:
                continue
            df_t = ind_data[t]
            try:
                i = df_t.index.get_loc(d)
            except KeyError:
                continue
            if isinstance(i, slice):
                i = i.start
            if i <= pos["entry_idx"]:
                continue
            c = df_t["close"].iloc[i]
            ma5 = df_t["sma5"].iloc[i]
            if not np.isnan(ma5):
                if c > ma5:
                    pos["ma5_crossed"] = True
                elif pos["ma5_crossed"] and c < ma5:
                    # 매도 체결
                    qty = pos["quantity"]
                    revenue = qty * c
                    net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
                    buy_cost = pos["entry_price"] * qty
                    buy_fee = buy_cost * FEE_BUY
                    pnl = net - buy_cost - buy_fee
                    ret_pct = pnl / (buy_cost + buy_fee) * 100
                    trades.append({
                        "ticker": t, "name": name_map.get(t, t),
                        "buy_date": pos["entry_date"], "buy_price": pos["entry_price"],
                        "sell_date": d, "sell_price": c, "quantity": qty,
                        "pnl": pnl, "return_pct": ret_pct, "reason": "MA5이탈",
                    })
                    cash += net
                    to_close.append(t)
        for t in to_close:
            del positions[t]

        # (B) 지수 필터 체크
        idx_ok = True
        if d in index_filter.index:
            idx_ok = bool(index_filter.loc[d])

        # (C) 매수 (필터 통과 시)
        if d in signals_by_date:
            signal_days += 1
            if not idx_ok:
                blocked_days += 1
            elif len(positions) < MAX_POSITIONS:
                slots_free = MAX_POSITIONS - len(positions)
                candidates = [c for c in signals_by_date[d] if c not in positions]
                cand_amt = []
                for c in candidates:
                    amt = 0
                    if c in amount_data and d in amount_data[c].index:
                        v = amount_data[c].loc[d, "amount"]
                        amt = float(v) if pd.notna(v) else 0
                    cand_amt.append((c, amt))
                cand_amt.sort(key=lambda x: -x[1])

                for t, _ in cand_amt:
                    if slots_free == 0 or cash < SLOT_CAPITAL * 0.9:
                        break
                    df_t = ind_data[t]
                    try:
                        i = df_t.index.get_loc(d)
                    except KeyError:
                        continue
                    if isinstance(i, slice):
                        i = i.start
                    close_p = df_t["close"].iloc[i]
                    qty = int(SLOT_CAPITAL / (close_p * (1 + FEE_BUY)))
                    if qty <= 0:
                        continue
                    cost = qty * close_p
                    fee = cost * FEE_BUY
                    if cash < cost + fee:
                        continue
                    cash -= (cost + fee)
                    positions[t] = {
                        "entry_price": close_p, "quantity": qty, "entry_date": d,
                        "entry_idx": i, "ma5_crossed": False,
                    }
                    slots_free -= 1

        # (D) 자산 평가
        pos_val = 0
        for t, pos in positions.items():
            if t in ind_data and d in ind_data[t].index:
                c_now = ind_data[t]["close"].loc[d]
                if isinstance(c_now, pd.Series):
                    c_now = c_now.iloc[-1]
                pos_val += pos["quantity"] * c_now
        equity_curve.append({"date": d, "equity": cash + pos_val,
                              "n_positions": len(positions)})

    # 미청산
    last_d = all_dates[-1] if all_dates else pd.Timestamp(END_DATE)
    for t, pos in list(positions.items()):
        if t in ind_data and last_d in ind_data[t].index:
            last_close = ind_data[t]["close"].loc[last_d]
            if isinstance(last_close, pd.Series):
                last_close = last_close.iloc[-1]
            qty = pos["quantity"]
            revenue = qty * last_close
            net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
            buy_cost = pos["entry_price"] * qty
            buy_fee = buy_cost * FEE_BUY
            pnl = net - buy_cost - buy_fee
            ret_pct = pnl / (buy_cost + buy_fee) * 100
            trades.append({
                "ticker": t, "name": name_map.get(t, t),
                "buy_date": pos["entry_date"], "buy_price": pos["entry_price"],
                "sell_date": last_d, "sell_price": last_close, "quantity": qty,
                "pnl": pnl, "return_pct": ret_pct, "reason": "미청산",
            })
            cash += net

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("date") if equity_curve else pd.DataFrame()
    return trades_df, equity_df, {"signal_days": signal_days, "blocked_days": blocked_days}


def summarize(trades_df, equity_df, label=""):
    final_eq = equity_df["equity"].iloc[-1] if not equity_df.empty else INITIAL_CAPITAL
    total_ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    if not equity_df.empty:
        dur_days = (equity_df.index[-1] - equity_df.index[0]).days
        years = dur_days / 365.25
        cagr = ((final_eq / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
        roll_max = equity_df["equity"].cummax()
        dd = equity_df["equity"] / roll_max - 1
        mdd = dd.min() * 100
        avg_pos = equity_df["n_positions"].mean()
    else:
        cagr = mdd = avg_pos = 0
    m = {"label": label, "total_return": total_ret, "cagr": cagr,
         "final_equity": final_eq, "mdd": mdd,
         "n_trades": len(trades_df), "avg_positions": avg_pos}
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        m["win_rate"] = len(wins) / len(trades_df) * 100
        gp = wins["pnl"].sum() if len(wins) > 0 else 0
        gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        m["pf"] = gp / gl if gl > 0 else float("inf")
        m["avg_trade_ret"] = trades_df["return_pct"].mean()
        tc = trades_df.copy()
        tc["hd"] = (pd.to_datetime(tc["sell_date"]) - pd.to_datetime(tc["buy_date"])).dt.days
        m["hold_days"] = tc["hd"].mean()
    else:
        m["win_rate"] = m["pf"] = m["avg_trade_ret"] = m["hold_days"] = 0
    return m


def year_metrics(equity_df, trades_df, year):
    eq = equity_df[equity_df.index.year == year]
    if eq.empty:
        return None
    yr_ret = (eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1) * 100
    roll_max = eq["equity"].cummax()
    dd = eq["equity"] / roll_max - 1
    mdd = dd.min() * 100
    t = trades_df[(pd.to_datetime(trades_df["buy_date"]).dt.year == year)]
    n_tr = len(t)
    wr = (t["pnl"] > 0).mean() * 100 if n_tr > 0 else 0
    return {"year": year, "return": yr_ret, "mdd": mdd,
            "n_trades": n_tr, "win_rate": wr}


def main():
    t0 = time.time()
    print("[1/4] 유니버스 구성...")
    tickers, name_map = build_universe()
    print(f"     {len(tickers)}종목")

    print("[2/4] KOSPI 지수 + 필터 계산...")
    idx_df, filters = load_kospi_filters()
    # 각 필터의 유효일 수 (기간 내)
    period_mask = (idx_df.index >= START_DATE) & (idx_df.index <= END_DATE)
    n_total = period_mask.sum()
    print(f"     KOSPI 영업일 {n_total}일 (기간 내)")
    for fname, fs in filters.items():
        fs_period = fs[period_mask]
        ok_ratio = fs_period.sum() / n_total * 100
        print(f"       {fname}: 통과율 {ok_ratio:.1f}%")

    print("[3/4] 데이터 로딩 (OHLCV + 거래대금)...")
    all_data = {}
    amt_data = {}
    for i in range(0, len(tickers), 500):
        batch = tickers[i:i+500]
        all_data.update(load_all_data(batch, START_DATE, END_DATE))
        amt_data.update(load_amount_data(batch, START_DATE, END_DATE))
    ind_data = {}
    for t, df in all_data.items():
        df = df[~df.index.duplicated(keep="last")]
        if t in amt_data:
            df = df.join(amt_data[t], how="left")
        ind_data[t] = calc_indicators(df, apply_amount_filter=False)

    print(f"[4/4] 4개 지수필터 비교 백테스트 (MA5이탈)...")
    results = {}
    for fname, fs in filters.items():
        print(f"  ▸ {fname}...", end=" ", flush=True)
        ts = time.time()
        tdf, edf, stats = run_portfolio(ind_data, amt_data, tickers, name_map,
                                          fs, fname)
        m = summarize(tdf, edf, fname)
        m.update(stats)
        results[fname] = {"trades": tdf, "equity": edf, "summary": m}
        pf = "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}"
        print(f"CAGR {m['cagr']:.1f}%, 총 {m['total_return']:.1f}%, "
              f"거래 {m['n_trades']}건, PF {pf}, MDD {m['mdd']:.1f}% ({time.time()-ts:.1f}s)")

    elapsed = time.time() - t0

    # 리포트
    lines = []
    lines.append("# 3-bar 포트폴리오 백테스트 + 지수 필터 비교\n")
    lines.append("## 개요\n")
    lines.append(f"- **유니버스**: KOSPI200 ∪ 시총≥3조 ({len(tickers)}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **초기자본**: {INITIAL_CAPITAL:,}원 (슬롯 {MAX_POSITIONS}개)")
    lines.append(f"- **매수**: 거래대금 순 상위 / **매도**: {EXIT_STRATEGY} (원규칙)")
    lines.append("")
    lines.append("### 지수 필터 정의\n")
    lines.append("| 코드 | 조건 | 설명 |")
    lines.append("|---|---|---|")
    lines.append("| F0_없음 | 항상 True | 기준선 (필터 없음) |")
    lines.append("| F1_MA60위 | KOSPI 종가 > KOSPI MA60 | 장기 추세 상승장만 매수 |")
    lines.append("| F2_MA20위_상승 | KOSPI > MA20 AND MA20↑ | 종목 매수 조건을 시장에 적용 |")
    lines.append("| F3_60일수익양 | KOSPI 60일 수익률 > 0 | 모멘텀 양수 구간만 매수 |")
    lines.append("")

    # 필터 통과율
    lines.append("### 기간 내 필터 통과율\n")
    lines.append("| 필터 | 통과 영업일 | 통과율 |")
    lines.append("|---|---|---|")
    for fname, fs in filters.items():
        fs_period = fs[period_mask]
        ok = int(fs_period.sum())
        lines.append(f"| {fname} | {ok}일 / {n_total}일 | {ok/n_total*100:.1f}% |")
    lines.append("")

    # 종합 비교
    lines.append("## 지수 필터 종합 비교\n")
    lines.append("| 필터 | 총수익률 | CAGR | 최종자본 | MDD | 거래수 | 승률 | 손익비 | 시그널일 | 차단일 | 평균포지션 |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for fname in filters:
        m = results[fname]["summary"]
        pf = "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}"
        lines.append(
            f"| {fname} | {m['total_return']:.2f}% | {m['cagr']:.2f}% "
            f"| {m['final_equity']:,.0f}원 | {m['mdd']:.2f}% "
            f"| {m['n_trades']}건 | {m['win_rate']:.1f}% | {pf} "
            f"| {m['signal_days']}일 | {m['blocked_days']}일 | {m['avg_positions']:.1f}/10 |"
        )
    lines.append("")

    # 연도별 (필터별)
    for fname in filters:
        lines.append(f"## 필터 · {fname} — 연도별 성과\n")
        lines.append("| 연도 | 수익률 | MDD | 거래수 | 승률 |")
        lines.append("|---|:---:|:---:|:---:|:---:|")
        eq_df = results[fname]["equity"]
        tr_df = results[fname]["trades"]
        for yr in [2023, 2024, 2025, 2026]:
            ym = year_metrics(eq_df, tr_df, yr)
            if ym is None:
                continue
            lines.append(f"| {yr} | {ym['return']:.2f}% | {ym['mdd']:.2f}% "
                         f"| {ym['n_trades']}건 | {ym['win_rate']:.1f}% |")
        lines.append("")

    # 최고 필터 선정
    best = max(filters, key=lambda f: results[f]["summary"]["cagr"])
    lines.append(f"## 결론\n")
    lines.append(f"- **최고 CAGR 필터**: {best} ({results[best]['summary']['cagr']:.2f}%)")
    for fname in filters:
        m = results[fname]["summary"]
        lines.append(
            f"- {fname}: CAGR {m['cagr']:.2f}%, MDD {m['mdd']:.2f}%, "
            f"거래 {m['n_trades']}건"
        )
    lines.append("")

    lines.append(f"## 실행 정보\n- 실행시간: {elapsed:.1f}초")
    lines.append(f"- 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    out = os.path.join(os.path.dirname(__file__), "results",
                       "backtest_3bar_portfolio_idx.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n리포트: {out}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
