"""
3-bar 매매규칙 포트폴리오 백테스트
- 유니버스: KOSPI200 ∪ 시총≥3조
- 슬롯: 최대 10종목 동시 보유
- 매수 우선순위: 시그널 발생일 당일 거래대금 상위
- 슬롯당 고정 1천만원 (초기자본 = 10 × 1천만 = 1억)
- 5개 매도 전략 비교
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_3bar import (
    ENGINE, EXIT_STRATEGIES, calc_indicators, find_signals,
    get_mcap_tickers, load_amount_data,
)
from backtest_crash import (
    load_all_data, get_kospi200_tickers,
    FEE_BUY, FEE_SELL, TAX_SELL,
)

START_DATE = "2023-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
MAX_POSITIONS = 10
SLOT_CAPITAL = 10_000_000
INITIAL_CAPITAL = MAX_POSITIONS * SLOT_CAPITAL  # 1억


# ──────────────────────────────────────────────
# 유니버스 구성
# ──────────────────────────────────────────────
def build_universe():
    print("[1/4] KOSPI200 ∪ 시총≥3조 유니버스...")
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
    print(f"     {len(merged)}종목")
    return merged, name_map


# ──────────────────────────────────────────────
# 포트폴리오 백테스트 엔진
# ──────────────────────────────────────────────
def run_portfolio(ind_data, amount_data, ticker_list, name_map,
                  exit_strategy: str):
    # 전체 거래일 집합 (유니버스에 존재하는 모든 날짜)
    all_dates = set()
    for df in ind_data.values():
        all_dates.update(df.index)
    all_dates = sorted(d for d in all_dates if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE))

    # 시그널 사전 계산: date -> [ticker]
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

    # 실행
    positions = {}  # ticker -> {entry_price, qty, entry_date, ma5_crossed, high_since_entry, entry_idx_map}
    cash = INITIAL_CAPITAL  # 대기 자본
    trades = []
    equity_curve = []

    for d in all_dates:
        # (A) 매도 판정 먼저
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
            c = df_t["close"].iloc[i]
            h = df_t["high"].iloc[i]
            lw = df_t["low"].iloc[i]
            o = df_t["open"].iloc[i]
            ma5 = df_t["sma5"].iloc[i]
            atr = df_t["atr14"].iloc[i]

            entry_idx = pos["entry_idx"]
            if i <= entry_idx:
                continue  # 진입일은 매도 판정 스킵
            hold_days = i - entry_idx

            sell_price = None
            reason = None

            if exit_strategy == "MA5이탈":
                if not np.isnan(ma5):
                    if c > ma5:
                        pos["ma5_crossed"] = True
                    elif pos["ma5_crossed"] and c < ma5:
                        sell_price = c
                        reason = "MA5이탈"

            elif exit_strategy == "익일시가":
                if hold_days >= 1:
                    sell_price = o
                    reason = "익일시가"

            elif exit_strategy == "3%손절7%익절":
                ep = pos["entry_price"]
                stop = ep * 0.97
                tp = ep * 1.07
                if lw <= stop:
                    sell_price = min(o, stop)
                    reason = "-3%손절"
                elif h >= tp:
                    sell_price = o if o >= tp else tp
                    reason = "+7%익절"

            elif exit_strategy == "ATR2x":
                pos["high_since_entry"] = max(pos["high_since_entry"], h)
                if not np.isnan(atr):
                    stop = pos["high_since_entry"] - 2 * atr
                    if lw <= stop:
                        sell_price = min(o, stop)
                        reason = "ATR2x이탈"

            elif exit_strategy == "3봉저가이탈":
                if i >= 3:
                    swing_low = df_t["low"].iloc[i-3:i].min()
                    if lw <= swing_low:
                        sell_price = min(o, swing_low)
                        reason = "3봉저가이탈"

            if sell_price is not None:
                qty = pos["quantity"]
                revenue = qty * sell_price
                fee = revenue * FEE_SELL
                tax = revenue * TAX_SELL
                net = revenue - fee - tax
                buy_cost = pos["entry_price"] * qty
                buy_fee = buy_cost * FEE_BUY
                pnl = net - buy_cost - buy_fee
                ret_pct = pnl / (buy_cost + buy_fee) * 100
                trades.append({
                    "ticker": t, "name": name_map.get(t, t),
                    "buy_date": pos["entry_date"], "buy_price": pos["entry_price"],
                    "sell_date": d, "sell_price": sell_price,
                    "quantity": qty, "pnl": pnl, "return_pct": ret_pct,
                    "reason": reason,
                })
                cash += net
                to_close.append(t)
        for t in to_close:
            del positions[t]

        # (B) 매수 판정: 당일 시그널 중 거래대금 상위
        if d in signals_by_date and len(positions) < MAX_POSITIONS:
            slots_free = MAX_POSITIONS - len(positions)
            candidates = [c for c in signals_by_date[d] if c not in positions]
            # 당일 거래대금 조회
            cand_amt = []
            for c in candidates:
                amt = 0
                if c in amount_data and d in amount_data[c].index:
                    v = amount_data[c].loc[d, "amount"]
                    amt = float(v) if pd.notna(v) else 0
                cand_amt.append((c, amt))
            cand_amt.sort(key=lambda x: -x[1])  # 거래대금 내림차순

            for t, amt in cand_amt:
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
                    "entry_idx": i, "ma5_crossed": False, "high_since_entry": close_p,
                }
                slots_free -= 1

        # (C) 자산 평가
        pos_val = 0
        for t, pos in positions.items():
            if t in ind_data and d in ind_data[t].index:
                c_now = ind_data[t]["close"].loc[d]
                if isinstance(c_now, pd.Series):
                    c_now = c_now.iloc[-1]
                pos_val += pos["quantity"] * c_now
        equity_curve.append({"date": d, "equity": cash + pos_val,
                              "n_positions": len(positions)})

    # 미청산 청산 (마지막 날 종가)
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
                "sell_date": last_d, "sell_price": last_close,
                "quantity": qty, "pnl": pnl, "return_pct": ret_pct,
                "reason": "미청산",
            })
            cash += net

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("date") if equity_curve else pd.DataFrame()
    return trades_df, equity_df


# ──────────────────────────────────────────────
# 요약 지표
# ──────────────────────────────────────────────
def summarize(trades_df, equity_df, label=""):
    final_eq = equity_df["equity"].iloc[-1] if not equity_df.empty else INITIAL_CAPITAL
    total_ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # CAGR
    if not equity_df.empty:
        dur_days = (equity_df.index[-1] - equity_df.index[0]).days
        years = dur_days / 365.25
        cagr = ((final_eq / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
    else:
        cagr = 0

    # MDD
    if not equity_df.empty:
        roll_max = equity_df["equity"].cummax()
        dd = equity_df["equity"] / roll_max - 1
        mdd = dd.min() * 100
    else:
        mdd = 0

    m = {"label": label, "total_return": total_ret, "cagr": cagr,
         "final_equity": final_eq, "mdd": mdd,
         "n_trades": len(trades_df)}
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        m["win_rate"] = len(wins) / len(trades_df) * 100
        gp = wins["pnl"].sum() if len(wins) > 0 else 0
        gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        m["pf"] = gp / gl if gl > 0 else float("inf")
        m["total_pnl"] = trades_df["pnl"].sum()
        m["avg_trade_ret"] = trades_df["return_pct"].mean()
        tc = trades_df.copy()
        tc["hd"] = (pd.to_datetime(tc["sell_date"]) - pd.to_datetime(tc["buy_date"])).dt.days
        m["hold_days"] = tc["hd"].mean()
    else:
        m["win_rate"] = m["pf"] = m["total_pnl"] = m["avg_trade_ret"] = m["hold_days"] = 0

    # 평균/최대 포지션 수
    if not equity_df.empty and "n_positions" in equity_df.columns:
        m["avg_positions"] = equity_df["n_positions"].mean()
        m["max_positions"] = equity_df["n_positions"].max()
    else:
        m["avg_positions"] = m["max_positions"] = 0

    return m


def year_split_metrics(equity_df, trades_df, year):
    eq = equity_df[equity_df.index.year == year]
    if eq.empty:
        return None
    start_eq = eq["equity"].iloc[0]
    end_eq = eq["equity"].iloc[-1]
    yr_ret = (end_eq / start_eq - 1) * 100
    roll_max = eq["equity"].cummax()
    dd = eq["equity"] / roll_max - 1
    mdd = dd.min() * 100
    t = trades_df[(pd.to_datetime(trades_df["buy_date"]).dt.year == year)]
    n_tr = len(t)
    wr = (t["pnl"] > 0).mean() * 100 if n_tr > 0 else 0
    return {"year": year, "return": yr_ret, "mdd": mdd,
            "n_trades": n_tr, "win_rate": wr}


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    t0 = time.time()
    tickers, name_map = build_universe()

    print(f"[2/4] 데이터 로딩 (OHLCV + 거래대금)...")
    all_data = {}
    amt_data = {}
    for i in range(0, len(tickers), 500):
        batch = tickers[i:i+500]
        all_data.update(load_all_data(batch, START_DATE, END_DATE))
        amt_data.update(load_amount_data(batch, START_DATE, END_DATE))

    print(f"[3/4] 지표 계산 (amount merge)...")
    ind_data = {}
    for t, df in all_data.items():
        df = df[~df.index.duplicated(keep="last")]
        if t in amt_data:
            df = df.join(amt_data[t], how="left")
        ind_data[t] = calc_indicators(df, apply_amount_filter=False)

    print(f"[4/4] 5개 매도전략 포트폴리오 백테스트...")
    results = {}
    for ex in EXIT_STRATEGIES:
        print(f"  ▸ {ex}...", end=" ", flush=True)
        ts = time.time()
        trades_df, equity_df = run_portfolio(ind_data, amt_data, tickers,
                                              name_map, ex)
        m = summarize(trades_df, equity_df, ex)
        results[ex] = {"trades": trades_df, "equity": equity_df, "summary": m}
        pf_str = "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}"
        print(f"CAGR {m['cagr']:.1f}%, 총수익 {m['total_return']:.1f}%, "
              f"거래 {m['n_trades']}건, 승률 {m['win_rate']:.1f}%, PF {pf_str}, "
              f"MDD {m['mdd']:.1f}% ({time.time()-ts:.1f}s)")

    elapsed = time.time() - t0

    # ── 리포트 생성 ──
    lines = []
    lines.append("# 3-bar 매매규칙 포트폴리오 백테스트\n")
    lines.append("## 전략 개요\n")
    lines.append(f"- **유니버스**: KOSPI200 ∪ 시총≥3조 ({len(tickers)}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **초기자본**: {INITIAL_CAPITAL:,}원 (슬롯 {MAX_POSITIONS}개 × 슬롯당 {SLOT_CAPITAL:,}원)")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append("")
    lines.append("### 매수 로직\n")
    lines.append(f"- 당일 3-bar 시그널 발생 종목 중 **거래대금 순 상위 N개 선택** (N = 빈 슬롯 수)")
    lines.append(f"- 최대 동시 보유 {MAX_POSITIONS}종목, 슬롯 비면 자본은 현금으로 대기")
    lines.append("")

    # 매도전략 비교표
    lines.append("## 매도전략 종합 비교\n")
    lines.append("| 매도전략 | 총수익률 | CAGR | 최종자본 | MDD | 거래수 | 승률 | 손익비 | 평균보유 | 평균포지션 |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for ex in EXIT_STRATEGIES:
        m = results[ex]["summary"]
        pf = "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}"
        lines.append(
            f"| {ex} | {m['total_return']:.2f}% | {m['cagr']:.2f}% "
            f"| {m['final_equity']:,.0f}원 | {m['mdd']:.2f}% "
            f"| {m['n_trades']}건 | {m['win_rate']:.1f}% | {pf} "
            f"| {m['hold_days']:.1f}일 | {m['avg_positions']:.1f}/{int(m['max_positions'])} |"
        )
    lines.append("")

    # 연도별 (매도전략별)
    for ex in EXIT_STRATEGIES:
        lines.append(f"## 매도전략 · {ex} — 연도별 성과\n")
        lines.append("| 연도 | 수익률 | MDD | 거래수 | 승률 |")
        lines.append("|------|:---:|:---:|:---:|:---:|")
        eq_df = results[ex]["equity"]
        tr_df = results[ex]["trades"]
        for yr in [2023, 2024, 2025, 2026]:
            ym = year_split_metrics(eq_df, tr_df, yr)
            if ym is None:
                continue
            lines.append(f"| {yr} | {ym['return']:.2f}% | {ym['mdd']:.2f}% "
                         f"| {ym['n_trades']}건 | {ym['win_rate']:.1f}% |")
        lines.append("")

    # 최고 전략 TOP 거래
    best_ex = max(EXIT_STRATEGIES,
                  key=lambda e: results[e]["summary"]["total_return"])
    lines.append(f"## 최고 성과 전략 · {best_ex} — TOP 10 거래\n")
    bt = results[best_ex]["trades"]
    if not bt.empty:
        top = bt.nlargest(10, "return_pct")
        lines.append("| # | 종목 | 매수일 | 매도일 | 매수가 | 매도가 | 수익률 | 사유 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for j, (_, r) in enumerate(top.iterrows()):
            lines.append(
                f"| {j+1} | {r['name']} | {pd.Timestamp(r['buy_date']).strftime('%Y-%m-%d')} "
                f"| {pd.Timestamp(r['sell_date']).strftime('%Y-%m-%d')} "
                f"| {r['buy_price']:,.0f} | {r['sell_price']:,.0f} "
                f"| {r['return_pct']:.2f}% | {r['reason']} |"
            )
    lines.append("")

    # 워스트 TOP (손실)
    lines.append(f"## 최고 성과 전략 · {best_ex} — 최악 손실 TOP 10\n")
    if not bt.empty:
        worst = bt.nsmallest(10, "return_pct")
        lines.append("| # | 종목 | 매수일 | 매도일 | 수익률 | 사유 |")
        lines.append("|---|---|---|---|---|---|")
        for j, (_, r) in enumerate(worst.iterrows()):
            lines.append(
                f"| {j+1} | {r['name']} | {pd.Timestamp(r['buy_date']).strftime('%Y-%m-%d')} "
                f"| {pd.Timestamp(r['sell_date']).strftime('%Y-%m-%d')} "
                f"| {r['return_pct']:.2f}% | {r['reason']} |"
            )
    lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- **실행 시간**: {elapsed:.2f}초")
    lines.append(f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    out = os.path.join(os.path.dirname(__file__), "results",
                       "backtest_3bar_portfolio.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n리포트: {out}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
