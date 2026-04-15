"""STH 상향돌파 매수 / STL 하향돌파 매도 백테스트.

- 진입: 당일 종가 > 최근 확정 STH 고가 AND 전일 종가 <= STH 고가
- 청산: 당일 저가 <= 최근 확정 STL 저가 → 해당 STL 저가에 전량 매도
- 1건당 300만원, 최대 10종목, KOSPI200, 2023-01-01~현재
"""

import os
import time
import pandas as pd
from backtest_larry_mtl import (
    load_data, find_swing_points, stats,
    PER_TRADE, COMMISSION, SELL_COMMISSION, TAX,
)

MAX_POSITIONS = 10


def precompute(gdf):
    stls, sths = find_swing_points(gdf)
    return {"stls": stls, "sths": sths}


def simulate(stocks, start_date):
    all_dates = set()
    for gdf in stocks.values():
        all_dates.update(gdf["date"].values)
    all_dates = sorted(d for d in all_dates if d >= pd.Timestamp(start_date))

    precomp = {}
    stock_data = {}
    for ticker, gdf in stocks.items():
        gdf = gdf.reset_index(drop=True)
        precomp[ticker] = precompute(gdf)
        date_to_idx = {d: i for i, d in enumerate(gdf["date"].values)}
        stock_data[ticker] = {"df": gdf, "date_to_idx": date_to_idx}

    positions = []
    trades = []
    skipped = 0

    def close_pos(pos, sell_price, date, reason):
        qty = pos["qty"]
        cost = pos["entry_price"] * qty
        revenue = sell_price * qty
        fee = cost * COMMISSION + revenue * (SELL_COMMISSION + TAX)
        pnl = revenue - cost - fee
        ret = pnl / cost * 100 if cost > 0 else 0
        trades.append({
            "ticker": pos["ticker"], "name": pos["name"],
            "buy_date": pd.Timestamp(pos["entry_date"]),
            "sell_date": pd.Timestamp(date),
            "buy_price": pos["entry_price"], "sell_price": sell_price,
            "qty": qty, "pnl": pnl, "return_pct": ret,
            "hold_days": (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
            "reason": reason,
        })

    for date in all_dates:
        # 청산 체크
        closed = []
        for pos in positions:
            td = stock_data[pos["ticker"]]
            idx = td["date_to_idx"].get(date)
            if idx is None:
                continue
            row = td["df"].iloc[idx]
            pc = precomp[pos["ticker"]]

            # 최근 확정 STL (진입일 이후 새로 생긴 것 우선, 없으면 진입 시점 것)
            latest_stl_low = pos["initial_stop"]
            for stl_pivot, stl_conf in pc["stls"]:
                if stl_conf <= idx:
                    latest_stl_low = td["df"].iloc[stl_pivot]["low"]

            if row["low"] <= latest_stl_low:
                close_pos(pos, latest_stl_low, date, "STL이탈")
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        # 신규 진입
        held = {p["ticker"] for p in positions}
        cands = []
        for ticker, td in stock_data.items():
            if ticker in held:
                continue
            idx = td["date_to_idx"].get(date)
            if idx is None or idx < 1:
                continue
            row = td["df"].iloc[idx]
            prev = td["df"].iloc[idx - 1]
            pc = precomp[ticker]

            latest_sth_high = None
            for sth_pivot, sth_conf in pc["sths"]:
                if sth_conf <= idx:
                    latest_sth_high = td["df"].iloc[sth_pivot]["high"]
            if latest_sth_high is None:
                continue
            latest_stl_low = None
            for stl_pivot, stl_conf in pc["stls"]:
                if stl_conf <= idx:
                    latest_stl_low = td["df"].iloc[stl_pivot]["low"]
            if latest_stl_low is None:
                continue

            if prev["close"] <= latest_sth_high and row["close"] > latest_sth_high:
                if row["close"] <= latest_stl_low:
                    continue
                cands.append({
                    "ticker": ticker, "name": row["name"],
                    "entry_price": row["close"],
                    "initial_stop": latest_stl_low,
                    "date": date,
                })

        slots = MAX_POSITIONS - len(positions)
        if len(cands) > slots:
            skipped += len(cands) - max(slots, 0)
        for cand in cands[:max(slots, 0)]:
            qty = int(PER_TRADE / cand["entry_price"])
            if qty <= 0:
                continue
            positions.append({
                "ticker": cand["ticker"], "name": cand["name"],
                "qty": qty, "entry_price": cand["entry_price"],
                "entry_date": date, "initial_stop": cand["initial_stop"],
            })

    # 미청산 종결
    for pos in positions:
        td = stock_data[pos["ticker"]]
        last_row = td["df"].iloc[-1]
        close_pos(pos, last_row["close"], last_row["date"], "미청산")

    return trades, skipped


def yearly_report(df):
    df = df.copy()
    df["year"] = df["sell_date"].dt.year
    CAPITAL = MAX_POSITIONS * PER_TRADE
    lines = ["## 연도별 성과 (실현 매도일 기준)\n"]
    lines.append("| 연도 | 거래수 | 승률(%) | 평균수익(%) | 최대수익(%) | 최대손실(%) | 손익비 | 총손익(원) | 자본대비(%) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y]
        s = stats(sub)
        lines.append(
            f"| {y} | {s['total']:,} | {s['win_rate']:.1f} "
            f"| {s['avg_ret']:+.2f} | {sub['return_pct'].max():+.2f} "
            f"| {sub['return_pct'].min():+.2f} | {s['pf']:.2f} "
            f"| {s['total_pnl']:+,.0f} | {s['total_pnl']/CAPITAL*100:+.2f} |"
        )
    s = stats(df)
    lines.append(
        f"| **전체** | **{s['total']:,}** | **{s['win_rate']:.1f}** "
        f"| **{s['avg_ret']:+.2f}** | **{df['return_pct'].max():+.2f}** "
        f"| **{df['return_pct'].min():+.2f}** | **{s['pf']:.2f}** "
        f"| **{s['total_pnl']:+,.0f}** | **{s['total_pnl']/CAPITAL*100:+.2f}** |"
    )
    return "\n".join(lines)


def main():
    print("데이터 로딩...")
    t0 = time.time()
    df, k200 = load_data(kospi200_only=True)
    print(f"  로드: {time.time()-t0:.1f}초")

    stocks = {}
    for ticker, gdf in df.groupby("ticker"):
        if ticker not in k200:
            continue
        stocks[ticker] = gdf.reset_index(drop=True)
    print(f"  KOSPI200 종목: {len(stocks):,}개")

    print("시뮬레이션...")
    t0 = time.time()
    trades, skipped = simulate(stocks, "2023-01-01")
    print(f"  완료: {time.time()-t0:.1f}초, 거래 {len(trades)}건, 스킵 {skipped}건")

    if not trades:
        print("거래 없음")
        return

    tdf = pd.DataFrame(trades)

    header = [
        "# STH 돌파 매수 / STL 이탈 매도 백테스트\n",
        "## 규칙\n",
        "- 진입: 최근 확정 STH 고가 상향돌파 (전일 종가 <= STH 고가 AND 당일 종가 > STH 고가) → 당일 종가 매수",
        "- 청산: 보유 중 최근 확정 STL 저가 하향이탈 → 해당 STL 저가에 전량 매도",
        f"- 최대 동시 보유: {MAX_POSITIONS}종목, 1건당 {PER_TRADE/10000:.0f}만원",
        "- 대상: KOSPI200, 2023-01-01 ~ 현재",
        f"- 수수료: 매수 {COMMISSION*100:.3f}% + 매도 {SELL_COMMISSION*100:.3f}% + 세금 {TAX*100:.2f}%",
        f"- 슬롯부족 스킵: {skipped}건\n",
        "---\n",
    ]
    report = "\n".join(header) + yearly_report(tdf) + "\n"
    os.makedirs("results", exist_ok=True)
    path = "results/backtest_sth_stl.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n결과 저장: {path}")
    print("\n" + yearly_report(tdf))


if __name__ == "__main__":
    main()
