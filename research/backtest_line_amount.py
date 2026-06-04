"""
종가가 기준선(BBI vs MA20)을 상/하향 돌파하는 매매 비교 + 거래대금 10일 1500억 필터
====================================================================================

기준선만 교체하고 나머지(거래대금 필터, 슬롯, 수수료, 자본)는 완전 동일하게 비교.
  - BBI  = (MA3 + MA6 + MA12 + MA24) / 4
  - MA20 = 종가 20일 단순이동평균

매수: 후보(최근 10일 거래대금 모두 1500억↑) 중 당일 종가가 기준선 상향돌파 → 종가 매수
매도: 보유 중 당일 종가가 기준선 하향돌파 → 종가 전량 매도
"""

import os
import time
import numpy as np
import pandas as pd

from backtest_larry_mtl import stats, PER_TRADE, MAX_POSITIONS, COMMISSION, SELL_COMMISSION, TAX
from backtest_bbi import perf_metrics, yearly_report, INIT_CAPITAL
from backtest_bbi_amount import (
    load_data_with_amount, simulate as simulate_amount,
    START_DATE, AMOUNT_THRESHOLD, AMOUNT_WINDOW,
)


def make_line(c: pd.Series, kind: str) -> pd.Series:
    if kind == "BBI":
        return (c.rolling(3).mean() + c.rolling(6).mean()
                + c.rolling(12).mean() + c.rolling(24).mean()) / 4.0
    if kind == "MA20":
        return c.rolling(20).mean()
    raise ValueError(kind)


def calc_signals(gdf: pd.DataFrame, kind: str) -> pd.DataFrame:
    g = gdf.sort_values("date").reset_index(drop=True)
    c = g["close"]
    line = make_line(c, kind)
    g["bbi"] = line  # simulate가 참조하는 컬럼명 재사용
    prev_c, prev_line = c.shift(1), line.shift(1)
    g["bull_cross"] = (prev_c <= prev_line) & (c > line) & line.notna() & prev_line.notna()
    g["bear_cross"] = (prev_c >= prev_line) & (c < line) & line.notna() & prev_line.notna()
    g["amt_min10"] = g["amount"].rolling(AMOUNT_WINDOW).min()
    g["amt_ok"] = g["amt_min10"] >= AMOUNT_THRESHOLD
    return g


def run(stocks_raw: dict, kind: str):
    # backtest_bbi_amount.simulate 와 동일 로직이되, 기준선만 kind로 교체.
    stocks = {t: calc_signals(g, kind) for t, g in stocks_raw.items()}

    all_dates = set()
    for g in stocks.values():
        all_dates.update(g["date"].values)
    all_dates = sorted(d for d in all_dates if d >= np.datetime64(START_DATE))

    stock_data = {t: {"df": g, "idx": {d: i for i, d in enumerate(g["date"].values)}}
                  for t, g in stocks.items()}

    cash = float(INIT_CAPITAL)
    positions, trades, equity_curve, skipped = {}, [], [], 0

    def close_pos(ticker, sell_price, date, reason):
        nonlocal cash
        pos = positions.pop(ticker)
        qty = pos["qty"]
        cost = pos["entry_price"] * qty
        revenue = sell_price * qty
        fee = cost * COMMISSION + revenue * (SELL_COMMISSION + TAX)
        pnl = revenue - cost - fee
        cash += revenue - revenue * (SELL_COMMISSION + TAX)
        trades.append({
            "ticker": ticker, "name": pos["name"],
            "buy_date": pd.Timestamp(pos["entry_date"]), "sell_date": pd.Timestamp(date),
            "buy_price": pos["entry_price"], "sell_price": sell_price, "qty": qty, "pnl": pnl,
            "return_pct": pnl / cost * 100 if cost > 0 else 0,
            "hold_days": (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
            "reason": reason,
        })

    for date in all_dates:
        for ticker in list(positions.keys()):
            sd = stock_data[ticker]; i = sd["idx"].get(date)
            if i is None:
                continue
            row = sd["df"].iloc[i]
            if row["bear_cross"]:
                close_pos(ticker, row["close"], date, "하향돌파")

        cands = []
        for ticker, sd in stock_data.items():
            if ticker in positions:
                continue
            i = sd["idx"].get(date)
            if i is None:
                continue
            row = sd["df"].iloc[i]
            if row["bull_cross"] and bool(row["amt_ok"]):
                cands.append((ticker, row["close"], row["name"], row["amt_min10"]))
        cands.sort(key=lambda x: x[3], reverse=True)

        slots = MAX_POSITIONS - len(positions)
        if len(cands) > slots:
            skipped += len(cands) - max(slots, 0)
        for ticker, price, name, _ in cands[:max(slots, 0)]:
            if price <= 0:
                continue
            qty = int(PER_TRADE / price)
            if qty <= 0:
                continue
            buy_cost = price * qty; buy_fee = buy_cost * COMMISSION
            if cash < buy_cost + buy_fee:
                continue
            cash -= buy_cost + buy_fee
            positions[ticker] = {"qty": qty, "entry_price": price, "entry_date": date, "name": name}

        holdings_val = 0.0
        for ticker, pos in positions.items():
            sd = stock_data[ticker]; i = sd["idx"].get(date)
            px = sd["df"].iloc[i]["close"] if i is not None else pos["entry_price"]
            holdings_val += px * pos["qty"]
        equity_curve.append((pd.Timestamp(date), cash + holdings_val))

    last_date = all_dates[-1]
    for ticker in list(positions.keys()):
        sd = stock_data[ticker]; i = sd["idx"].get(last_date)
        row = sd["df"].iloc[i] if i is not None else sd["df"].iloc[-1]
        close_pos(ticker, row["close"], row["date"], "미청산")

    eq = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return pd.DataFrame(trades), eq, skipped


def main():
    print("데이터 로딩...")
    t0 = time.time()
    df, k200 = load_data_with_amount()
    stocks_raw = {t: g.reset_index(drop=True) for t, g in df.groupby("ticker") if t in k200}
    print(f"  로드 {time.time()-t0:.1f}초, KOSPI200 {len(stocks_raw)}종목")

    results = {}
    for kind in ["BBI", "MA20"]:
        t0 = time.time()
        tdf, eq, skipped = run(stocks_raw, kind)
        pm = perf_metrics(eq); s = stats(tdf)
        results[kind] = (tdf, eq, skipped, pm, s)
        print(f"  {kind}: {time.time()-t0:.1f}초, 거래 {len(tdf)}건, 스킵 {skipped}")

    # 비교 리포트
    L = ["# BBI vs MA20 돌파매매 비교 (거래대금 10일 1500억 필터)\n",
         "## 공통 규칙\n",
         f"- 매수후보: 최근 {AMOUNT_WINDOW}일 거래대금 모두 1500억↑ + 당일 종가 기준선 **상향돌파**",
         "- 매도: 보유 중 당일 종가 기준선 **하향돌파** → 종가 전량",
         f"- KOSPI200, 최대 {MAX_POSITIONS}종목, 1거래 {PER_TRADE/10000:.0f}만원, 초기 {INIT_CAPITAL/10000:.0f}만원",
         f"- 기간 {START_DATE} ~ {results['BBI'][1].index[-1].date()}, 수수료 매수{COMMISSION*100:.3f}%+매도{SELL_COMMISSION*100:.3f}%+세금{TAX*100:.2f}%\n",
         "---\n", "## 핵심 비교\n",
         "| 지표 | BBI | MA20 |", "|---|---:|---:|"]
    keys = [("총수익률", lambda pm, s: f"{pm['total_ret']*100:+.2f}%"),
            ("CAGR", lambda pm, s: f"{pm['cagr']*100:+.2f}%"),
            ("MDD", lambda pm, s: f"{pm['mdd']*100:.2f}%"),
            ("Calmar", lambda pm, s: f"{pm['calmar']:.2f}"),
            ("거래수", lambda pm, s: f"{s['total']:,}"),
            ("승률(%)", lambda pm, s: f"{s['win_rate']:.1f}"),
            ("평균수익(%)", lambda pm, s: f"{s['avg_ret']:+.2f}"),
            ("평균이익(%)", lambda pm, s: f"{s['avg_win']:+.2f}"),
            ("평균손실(%)", lambda pm, s: f"{s['avg_loss']:+.2f}"),
            ("손익비", lambda pm, s: f"{s['pf']:.2f}"),
            ("평균보유일", lambda pm, s: f"{s['avg_hold']:.1f}")]
    for label, fn in keys:
        b = fn(results["BBI"][3], results["BBI"][4])
        m = fn(results["MA20"][3], results["MA20"][4])
        L.append(f"| {label} | {b} | {m} |")
    L.append("")
    for kind in ["BBI", "MA20"]:
        tdf = results[kind][0]
        L.append(f"\n---\n\n## {kind} 연도별\n")
        L.append(yearly_report(tdf))

    report = "\n".join(L) + "\n"
    os.makedirs("results", exist_ok=True)
    path = "results/backtest_bbi_vs_ma20.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n결과 저장: {path}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
