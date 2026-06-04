"""
BBI(Bull & Bear Index) 상향돌파 매수 / 하향돌파 매도 백테스트
================================================================

BBI = (MA3 + MA6 + MA12 + MA24) / 4

규칙(모두 당일 종가 확정 → 룩어헤드 없음):
  - 매수: 전일 종가 <= 전일 BBI  AND  당일 종가 > 당일 BBI  (상향돌파) → 당일 종가 매수
  - 매도: 전일 종가 >= 전일 BBI  AND  당일 종가 < 당일 BBI  (하향돌파) → 당일 종가 전량 매도

포트폴리오:
  - KOSPI200, 최대 10종목 동시보유, 1거래당 300만원(고정), 초기자본 3,000만원
  - 일별 평가금액(현금+보유종목 종가평가) 추적 → CAGR / MDD / Calmar 정확 산출
  - 기간: 2023-01-01 ~ 현재 (MA24 워밍업 위해 2022-01-01부터 로드)
"""

import os
import time
import numpy as np
import pandas as pd

from backtest_larry_mtl import (
    ENGINE, load_data, stats,
    PER_TRADE, MAX_POSITIONS, COMMISSION, SELL_COMMISSION, TAX,
)

START_DATE = "2023-01-01"
INIT_CAPITAL = MAX_POSITIONS * PER_TRADE  # 3,000만원


def calc_bbi(gdf: pd.DataFrame) -> pd.DataFrame:
    """BBI 및 돌파 신호 컬럼 생성 (티커별 정렬된 df 전제)."""
    g = gdf.sort_values("date").reset_index(drop=True)
    c = g["close"]
    bbi = (c.rolling(3).mean() + c.rolling(6).mean()
           + c.rolling(12).mean() + c.rolling(24).mean()) / 4.0
    g["bbi"] = bbi
    prev_c = c.shift(1)
    prev_bbi = bbi.shift(1)
    g["bull_cross"] = (prev_c <= prev_bbi) & (c > bbi) & bbi.notna() & prev_bbi.notna()
    g["bear_cross"] = (prev_c >= prev_bbi) & (c < bbi) & bbi.notna() & prev_bbi.notna()
    return g


def simulate(stocks: dict[str, pd.DataFrame], start_date: str):
    all_dates = set()
    for g in stocks.values():
        all_dates.update(g["date"].values)
    all_dates = sorted(d for d in all_dates if d >= np.datetime64(start_date))

    stock_data = {}
    for ticker, g in stocks.items():
        g = calc_bbi(g)
        stock_data[ticker] = {
            "df": g,
            "idx": {d: i for i, d in enumerate(g["date"].values)},
        }

    cash = float(INIT_CAPITAL)
    positions = {}          # ticker -> dict(qty, entry_price, entry_date, name)
    trades = []
    equity_curve = []       # (date, equity)
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
        # 1) 매도(하향돌파) 먼저
        for ticker in list(positions.keys()):
            sd = stock_data[ticker]
            i = sd["idx"].get(date)
            if i is None:
                continue
            row = sd["df"].iloc[i]
            if row["bear_cross"]:
                close_pos(ticker, row["close"], date, "하향돌파")

        # 2) 매수(상향돌파) 후보
        cands = []
        for ticker, sd in stock_data.items():
            if ticker in positions:
                continue
            i = sd["idx"].get(date)
            if i is None:
                continue
            row = sd["df"].iloc[i]
            if row["bull_cross"]:
                cands.append((ticker, row["close"], row["name"]))

        slots = MAX_POSITIONS - len(positions)
        if len(cands) > slots:
            skipped += len(cands) - max(slots, 0)
        for ticker, price, name in cands[:max(slots, 0)]:
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
            positions[ticker] = {
                "qty": qty, "entry_price": price,
                "entry_date": date, "name": name,
            }

        # 3) 일별 평가금액
        holdings_val = 0.0
        for ticker, pos in positions.items():
            sd = stock_data[ticker]
            i = sd["idx"].get(date)
            px = sd["df"].iloc[i]["close"] if i is not None else pos["entry_price"]
            holdings_val += px * pos["qty"]
        equity_curve.append((pd.Timestamp(date), cash + holdings_val))

    # 미청산 종결(마지막 종가)
    last_date = all_dates[-1]
    for ticker in list(positions.keys()):
        sd = stock_data[ticker]
        i = sd["idx"].get(last_date)
        row = sd["df"].iloc[i] if i is not None else sd["df"].iloc[-1]
        close_pos(ticker, row["close"], row["date"], "미청산")

    eq = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return pd.DataFrame(trades), eq, skipped


def perf_metrics(eq: pd.DataFrame) -> dict:
    e = eq["equity"]
    years = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    roll_max = e.cummax()
    dd = e / roll_max - 1
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else float("inf")
    return {
        "init": e.iloc[0], "final": e.iloc[-1],
        "total_ret": e.iloc[-1] / e.iloc[0] - 1,
        "cagr": cagr, "mdd": mdd, "calmar": calmar, "years": years,
    }


def yearly_report(df: pd.DataFrame) -> str:
    df = df.copy()
    df["year"] = df["sell_date"].dt.year
    lines = ["### 연도별 성과 (실현 매도일 기준)\n"]
    lines.append("| 연도 | 거래수 | 승률(%) | 평균수익(%) | 최대수익(%) | 최대손실(%) | 손익비 | 총손익(원) | 자본대비(%) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y]
        s = stats(sub)
        lines.append(
            f"| {y} | {s['total']:,} | {s['win_rate']:.1f} "
            f"| {s['avg_ret']:+.2f} | {sub['return_pct'].max():+.2f} "
            f"| {sub['return_pct'].min():+.2f} | {s['pf']:.2f} "
            f"| {s['total_pnl']:+,.0f} | {s['total_pnl']/INIT_CAPITAL*100:+.2f} |"
        )
    s = stats(df)
    lines.append(
        f"| **전체** | **{s['total']:,}** | **{s['win_rate']:.1f}** "
        f"| **{s['avg_ret']:+.2f}** | **{df['return_pct'].max():+.2f}** "
        f"| **{df['return_pct'].min():+.2f}** | **{s['pf']:.2f}** "
        f"| **{s['total_pnl']:+,.0f}** | **{s['total_pnl']/INIT_CAPITAL*100:+.2f}** |"
    )
    return "\n".join(lines)


def main():
    print("데이터 로딩...")
    t0 = time.time()
    df, k200 = load_data(kospi200_only=True)
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
        print("거래 없음")
        return

    pm = perf_metrics(eq)
    s = stats(tdf)

    header = [
        "# BBI 상향돌파 매수 / 하향돌파 매도 백테스트\n",
        "## 규칙\n",
        "- **BBI = (MA3 + MA6 + MA12 + MA24) / 4** (종가 기준)",
        "- 매수: 전일 종가 ≤ 전일 BBI **AND** 당일 종가 > 당일 BBI (상향돌파) → 당일 종가 매수",
        "- 매도: 전일 종가 ≥ 전일 BBI **AND** 당일 종가 < 당일 BBI (하향돌파) → 당일 종가 전량 매도",
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
    path = "results/backtest_bbi.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n결과 저장: {path}\n")
    print("\n".join(header))
    print(yearly_report(tdf))


if __name__ == "__main__":
    main()
