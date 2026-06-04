"""
BBI가 MA20을 상/하향 돌파하는 2선 교차 매매 + 거래대금 10일 1500억 필터
========================================================================

기존 backtest_bbi.py / backtest_line_amount.py 는 "종가가 기준선을 돌파"하는 방식.
본 스크립트는 사용자 요청대로 **두 이동평균선의 교차(BBI vs MA20)** 를 신호로 사용.

  BBI  = (MA3 + MA6 + MA12 + MA24) / 4   (종가 기반 복합 이동평균)
  MA20 = 종가 20일 단순이동평균

  매수: 전일 BBI ≤ 전일 MA20  AND  당일 BBI > 당일 MA20  (BBI가 MA20 상향돌파) → 당일 종가 매수
  매도: 전일 BBI ≥ 전일 MA20  AND  당일 BBI < 당일 MA20  (BBI가 MA20 하향돌파) → 당일 종가 전량 매도

매수 후보: 최근 10거래일 거래대금이 "모두" 1500억원 이상인 종목 (= 10일 rolling min ≥ 1500억)
포트폴리오: KOSPI200, 최대 10종목(K-Tide 10 = 10/10), 1거래 300만원, 초기자본 3,000만원
기간: 2023-01-01 ~ 현재 (MA24 워밍업 위해 2022-01-01부터 로드)
거래대금 단위: stocks.amount = 백만원 → 1500억 = 150,000
"""

import os
import time
import numpy as np
import pandas as pd

from backtest_larry_mtl import (
    stats, PER_TRADE, MAX_POSITIONS, COMMISSION, SELL_COMMISSION, TAX,
)
from backtest_bbi import perf_metrics, yearly_report, INIT_CAPITAL
from backtest_bbi_amount import (
    load_data_with_amount, START_DATE, AMOUNT_THRESHOLD, AMOUNT_WINDOW,
)


def calc_signals(gdf: pd.DataFrame) -> pd.DataFrame:
    """BBI vs MA20 2선 교차 신호 + 거래대금 필터."""
    g = gdf.sort_values("date").reset_index(drop=True)
    c = g["close"]
    bbi = (c.rolling(3).mean() + c.rolling(6).mean()
           + c.rolling(12).mean() + c.rolling(24).mean()) / 4.0
    ma20 = c.rolling(20).mean()
    prev_bbi, prev_ma20 = bbi.shift(1), ma20.shift(1)

    g["bull_cross"] = ((prev_bbi <= prev_ma20) & (bbi > ma20)
                       & bbi.notna() & ma20.notna()
                       & prev_bbi.notna() & prev_ma20.notna())
    g["bear_cross"] = ((prev_bbi >= prev_ma20) & (bbi < ma20)
                       & bbi.notna() & ma20.notna()
                       & prev_bbi.notna() & prev_ma20.notna())

    g["amt_min10"] = g["amount"].rolling(AMOUNT_WINDOW).min()
    g["amt_ok"] = g["amt_min10"] >= AMOUNT_THRESHOLD
    return g


def simulate(stocks_raw: dict):
    stocks = {t: calc_signals(g) for t, g in stocks_raw.items()}

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
        # 1) 매도(하향돌파) 먼저
        for ticker in list(positions.keys()):
            sd = stock_data[ticker]; i = sd["idx"].get(date)
            if i is None:
                continue
            row = sd["df"].iloc[i]
            if row["bear_cross"]:
                close_pos(ticker, row["close"], date, "하향돌파")

        # 2) 매수 후보: 거래대금 10/10 OK + BBI가 MA20 상향돌파
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
        cands.sort(key=lambda x: x[3], reverse=True)  # 10일 최소거래대금 큰 순 우선

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
            positions[ticker] = {"qty": qty, "entry_price": price,
                                 "entry_date": date, "name": name}

        # 3) 일별 평가금액
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

    print("시뮬레이션...")
    t0 = time.time()
    tdf, eq, skipped = simulate(stocks_raw)
    print(f"  완료: {time.time()-t0:.1f}초, 거래 {len(tdf)}건, 스킵 {skipped}건")

    if tdf.empty:
        print("거래 없음")
        return

    pm = perf_metrics(eq)
    s = stats(tdf)

    header = [
        "# BBI×MA20 교차매매 + 거래대금 10일 1500억 필터 (10/10)\n",
        "## 규칙\n",
        "- **BBI = (MA3 + MA6 + MA12 + MA24) / 4**, **MA20 = 종가 20일 단순이동평균**",
        "- 매수: 전일 BBI ≤ 전일 MA20 **AND** 당일 BBI > 당일 MA20 (BBI가 MA20 **상향돌파**) → 당일 종가 매수",
        "- 매도: 전일 BBI ≥ 전일 MA20 **AND** 당일 BBI < 당일 MA20 (BBI가 MA20 **하향돌파**) → 당일 종가 전량 매도",
        f"- **매수 후보**: 최근 {AMOUNT_WINDOW}거래일 거래대금이 **모두 1500억원 이상**인 종목",
        "- 후보 과다 시 10일 최소거래대금 큰 순 우선",
        f"- 대상: KOSPI200, 최대 {MAX_POSITIONS}종목, 1거래당 {PER_TRADE/10000:.0f}만원, 초기자본 {INIT_CAPITAL/10000:.0f}만원",
        f"- 기간: {START_DATE} ~ {eq.index[-1].date()} (MA24 워밍업 2022-01-01부터 로드)",
        f"- 수수료: 매수 {COMMISSION*100:.3f}% + 매도 {SELL_COMMISSION*100:.3f}% + 세금 {TAX*100:.2f}%",
        f"- 슬롯부족 스킵 시그널: {skipped:,}건\n",
        "---\n",
        "## 포트폴리오 성과 (일별 평가금액 기준)\n",
        "| 지표 | 값 |",
        "|---|---:|",
        f"| 초기자본 | {pm['init']:,.0f}원 |",
        f"| 최종자본 | {pm['final']:,.0f}원 |",
        f"| 총수익률 | {pm['total_ret']*100:+.2f}% |",
        f"| CAGR | {pm['cagr']*100:+.2f}% |",
        f"| MDD | {pm['mdd']*100:.2f}% |",
        f"| Calmar | {pm['calmar']:.2f} |",
        f"| 운용기간 | {pm['years']:.2f}년 |\n",
        "## 거래 통계\n",
        "| 지표 | 값 |",
        "|---|---:|",
        f"| 총 거래수 | {s['total']:,} |",
        f"| 승률(%) | {s['win_rate']:.1f} |",
        f"| 평균 수익률(%) | {s['avg_ret']:+.2f} |",
        f"| 중간값 수익률(%) | {s['med_ret']:+.2f} |",
        f"| 평균이익(%) | {s['avg_win']:+.2f} |",
        f"| 평균손실(%) | {s['avg_loss']:+.2f} |",
        f"| 손익비(평균이익/평균손실) | {s['pf']:.2f} |",
        f"| 평균 보유일 | {s['avg_hold']:.1f} |\n",
        "---\n",
    ]
    report = "\n".join(header) + yearly_report(tdf) + "\n"

    os.makedirs("results", exist_ok=True)
    path = "results/backtest_bbi_ma20_cross.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n결과 저장: {path}\n")
    print("\n".join(header))
    print(yearly_report(tdf))


if __name__ == "__main__":
    main()
