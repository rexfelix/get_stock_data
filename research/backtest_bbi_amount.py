"""
BBI 돌파매매 + 거래대금 필터(10일 모두 1500억 이상) 백테스트
================================================================

BBI = (MA3 + MA6 + MA12 + MA24) / 4

매수 후보 조건: 최근 10거래일 거래대금이 "모두" 1500억원 이상인 종목
  (= 10일 거래대금 rolling min >= 1500억)
매수: 위 후보 중 당일 종가가 BBI 상향돌파 → 당일 종가 매수
매도: 보유 중 당일 종가가 BBI 하향돌파 → 당일 종가 전량 매도 (거래대금 무관)

포트폴리오: KOSPI200, 최대 10종목, 1거래당 300만원, 초기자본 3,000만원
기간: 2023-01-01 ~ 현재 (워밍업 2022-01-01부터 로드)
거래대금 단위: stocks.amount = 백만원 → 1500억 = 150,000
"""

import os
import time
import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_larry_mtl import (
    ENGINE, stats,
    PER_TRADE, MAX_POSITIONS, COMMISSION, SELL_COMMISSION, TAX,
)
from backtest_bbi import calc_bbi, perf_metrics, yearly_report, INIT_CAPITAL

START_DATE = "2023-01-01"
AMOUNT_THRESHOLD = 150_000   # 1500억원 (백만원 단위)
AMOUNT_WINDOW = 10           # 최근 10거래일 모두


def load_data_with_amount():
    with ENGINE.connect() as conn:
        k200 = set(pd.read_sql(text("SELECT ticker FROM kospi200_members"), conn)["ticker"])
        df = pd.read_sql(text("""
            SELECT date, open, high, low, close, amount, ticker, name
            FROM stocks WHERE date >= '2022-01-01'
            ORDER BY ticker, date
        """), conn)
    df["date"] = pd.to_datetime(df["date"])
    return df, k200


def calc_signals(gdf: pd.DataFrame) -> pd.DataFrame:
    g = calc_bbi(gdf)
    # 최근 10거래일 거래대금이 모두 임계 이상 → rolling min
    g["amt_min10"] = g["amount"].rolling(AMOUNT_WINDOW).min()
    g["amt_ok"] = g["amt_min10"] >= AMOUNT_THRESHOLD
    return g


def simulate(stocks: dict[str, pd.DataFrame], start_date: str):
    all_dates = set()
    for g in stocks.values():
        all_dates.update(g["date"].values)
    all_dates = sorted(d for d in all_dates if d >= np.datetime64(start_date))

    stock_data = {}
    for ticker, g in stocks.items():
        g = calc_signals(g)
        stock_data[ticker] = {"df": g, "idx": {d: i for i, d in enumerate(g["date"].values)}}

    cash = float(INIT_CAPITAL)
    positions = {}
    trades = []
    equity_curve = []
    skipped = 0

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
            "buy_date": pd.Timestamp(pos["entry_date"]),
            "sell_date": pd.Timestamp(date),
            "buy_price": pos["entry_price"], "sell_price": sell_price,
            "qty": qty, "pnl": pnl,
            "return_pct": pnl / cost * 100 if cost > 0 else 0,
            "hold_days": (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
            "reason": reason,
        })

    for date in all_dates:
        # 1) 매도(하향돌파)
        for ticker in list(positions.keys()):
            sd = stock_data[ticker]
            i = sd["idx"].get(date)
            if i is None:
                continue
            row = sd["df"].iloc[i]
            if row["bear_cross"]:
                close_pos(ticker, row["close"], date, "하향돌파")

        # 2) 매수 후보: 거래대금 10/10 OK + 상향돌파
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

        # 후보 과다 시 10일 최소거래대금 큰 순으로 우선 (임의성 완화)
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
            buy_cost = price * qty
            buy_fee = buy_cost * COMMISSION
            if cash < buy_cost + buy_fee:
                continue
            cash -= buy_cost + buy_fee
            positions[ticker] = {"qty": qty, "entry_price": price,
                                 "entry_date": date, "name": name}

        # 3) 일별 평가금액
        holdings_val = 0.0
        for ticker, pos in positions.items():
            sd = stock_data[ticker]
            i = sd["idx"].get(date)
            px = sd["df"].iloc[i]["close"] if i is not None else pos["entry_price"]
            holdings_val += px * pos["qty"]
        equity_curve.append((pd.Timestamp(date), cash + holdings_val))

    last_date = all_dates[-1]
    for ticker in list(positions.keys()):
        sd = stock_data[ticker]
        i = sd["idx"].get(last_date)
        row = sd["df"].iloc[i] if i is not None else sd["df"].iloc[-1]
        close_pos(ticker, row["close"], row["date"], "미청산")

    eq = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return pd.DataFrame(trades), eq, skipped


def main():
    print("데이터 로딩...")
    t0 = time.time()
    df, k200 = load_data_with_amount()
    print(f"  로드: {time.time()-t0:.1f}초")

    stocks = {}
    for ticker, g in df.groupby("ticker"):
        if ticker not in k200:
            continue
        stocks[ticker] = g.reset_index(drop=True)
    print(f"  KOSPI200 종목: {len(stocks):,}개")

    print("시뮬레이션...")
    t0 = time.time()
    tdf, eq, skipped = simulate(stocks, START_DATE)
    print(f"  완료: {time.time()-t0:.1f}초, 거래 {len(tdf)}건, 스킵 {skipped}건")

    if tdf.empty:
        print("거래 없음 (거래대금 필터로 후보 부재)")
        return

    pm = perf_metrics(eq)
    s = stats(tdf)

    header = [
        "# BBI 돌파매매 + 거래대금 10일 1500억 필터 백테스트\n",
        "## 규칙\n",
        "- **BBI = (MA3 + MA6 + MA12 + MA24) / 4** (종가)",
        f"- **매수 후보**: 최근 {AMOUNT_WINDOW}거래일 거래대금이 **모두 1500억원 이상**인 종목",
        "- 매수: 후보 중 당일 종가가 BBI **상향돌파** → 당일 종가 매수",
        "- 매도: 보유 중 당일 종가가 BBI **하향돌파** → 당일 종가 전량 매도 (거래대금 무관)",
        "- 후보 과다 시 10일 최소거래대금 큰 순 우선",
        f"- 대상: KOSPI200, 최대 {MAX_POSITIONS}종목, 1거래당 {PER_TRADE/10000:.0f}만원, 초기자본 {INIT_CAPITAL/10000:.0f}만원",
        f"- 기간: {START_DATE} ~ {eq.index[-1].date()}",
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
        f"| 손익비 | {s['pf']:.2f} |",
        f"| 평균 보유일 | {s['avg_hold']:.1f} |\n",
        "---\n",
    ]
    report = "\n".join(header) + yearly_report(tdf) + "\n"

    os.makedirs("results", exist_ok=True)
    path = "results/backtest_bbi_amount.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n결과 저장: {path}\n")
    print("\n".join(header))
    print(yearly_report(tdf))


if __name__ == "__main__":
    main()
