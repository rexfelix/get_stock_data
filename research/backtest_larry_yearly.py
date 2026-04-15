"""래리 MTL/MTH 스윙 매매 연도별 백테스트 (R%>=7% 필터 + 최대 10종목 + 300만원/건)."""

import os
import time
import pandas as pd
from backtest_larry_mtl import (
    load_data, precompute, stats,
    _close_position, _record_partial,
    PER_TRADE, COMMISSION, SELL_COMMISSION, TAX,
)

MAX_POSITIONS = 10
R_PCT_MIN = 7.0  # R% >= 7% 필터
CAPITAL = MAX_POSITIONS * PER_TRADE  # 3,000만원 베이스 자본


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

    for date in all_dates:
        closed = []
        for pos in positions:
            td = stock_data[pos["ticker"]]
            idx = td["date_to_idx"].get(date)
            if idx is None:
                continue
            row = td["df"].iloc[idx]
            pc = precomp[pos["ticker"]]

            latest_stl_low = pos["initial_stop"]
            for stl_pivot, stl_conf in pc["stls"]:
                if stl_conf <= idx and td["df"].iloc[stl_pivot]["date"] > pos["entry_date"]:
                    latest_stl_low = td["df"].iloc[stl_pivot]["low"]

            if row["low"] <= pos["initial_stop"]:
                _close_position(pos, pos["initial_stop"], date, "손절", trades)
                closed.append(pos)
                continue

            r = pos["entry_price"] - pos["initial_stop"]
            target_2r = pos["entry_price"] + 2 * r
            if not pos["took_profit_1"] and r > 0 and row["high"] >= target_2r:
                half_qty = pos["qty_remaining"] // 2
                if half_qty > 0:
                    _record_partial(pos, target_2r, date, "1차익절(2R)", half_qty, trades)
                    pos["qty_remaining"] -= half_qty
                pos["took_profit_1"] = True

            if latest_stl_low > pos["initial_stop"] and row["low"] <= latest_stl_low:
                _close_position(pos, latest_stl_low, date, "STL이탈", trades)
                closed.append(pos)
                continue

        for c in closed:
            positions.remove(c)

        held = {p["ticker"] for p in positions}
        cands = []

        for ticker, td in stock_data.items():
            if ticker in held:
                continue
            idx = td["date_to_idx"].get(date)
            if idx is None:
                continue
            row = td["df"].iloc[idx]
            pc = precomp[ticker]

            latest_mtl = None
            for mtl_pivot, mtl_conf in pc["mtls"]:
                if mtl_conf <= idx:
                    latest_mtl = (mtl_pivot, mtl_conf)
            if latest_mtl is None:
                continue

            latest_mth = None
            for mth_pivot, mth_conf in pc["mths"]:
                if mth_conf <= idx and mth_pivot > latest_mtl[0]:
                    latest_mth = (mth_pivot, mth_conf)
            if latest_mth is None:
                continue

            mth_high = td["df"].iloc[latest_mth[0]]["high"]
            prev_idx = idx - 1
            if prev_idx < 0:
                continue
            prev_close = td["df"].iloc[prev_idx]["close"]

            if prev_close <= mth_high and row["close"] > mth_high:
                latest_stl_low = None
                for stl_pivot, stl_conf in pc["stls"]:
                    if stl_conf <= idx:
                        latest_stl_low = td["df"].iloc[stl_pivot]["low"]
                if latest_stl_low is None:
                    continue
                entry = row["close"]
                stop = latest_stl_low - 1
                if entry <= 0 or entry <= stop:
                    continue
                r_pct = (entry - stop) / entry * 100
                if r_pct < R_PCT_MIN:
                    continue
                cands.append({
                    "ticker": ticker, "name": row["name"],
                    "entry_price": entry, "initial_stop": stop,
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
                "qty_full": qty, "qty_remaining": qty,
                "entry_price": cand["entry_price"],
                "entry_date": date,
                "initial_stop": cand["initial_stop"],
                "took_profit_1": False,
            })

    for pos in positions:
        td = stock_data[pos["ticker"]]
        last_row = td["df"].iloc[-1]
        _close_position(pos, last_row["close"], last_row["date"], "미청산", trades)

    return trades, skipped


def yearly_report(df: pd.DataFrame) -> str:
    df = df.copy()
    df["year"] = df["sell_date"].dt.year  # 실현 기준
    lines = []
    lines.append("## 연도별 성과 (실현 매도일 기준)\n")
    lines.append("| 연도 | 거래수 | 승률(%) | 평균수익(%) | 최대수익(%) | 최대손실(%) | 손익비 | 총손익(원) | 자본대비수익률(%) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y]
        s = stats(sub)
        max_p = sub["return_pct"].max()
        max_l = sub["return_pct"].min()
        cap_ret = s["total_pnl"] / CAPITAL * 100
        lines.append(
            f"| {y} | {s['total']:,} | {s['win_rate']:.1f} "
            f"| {s['avg_ret']:+.2f} | {max_p:+.2f} | {max_l:+.2f} "
            f"| {s['pf']:.2f} | {s['total_pnl']:+,.0f} | {cap_ret:+.2f} |"
        )
    # 전체
    s = stats(df)
    max_p = df["return_pct"].max()
    max_l = df["return_pct"].min()
    cap_ret = s["total_pnl"] / CAPITAL * 100
    lines.append(
        f"| **전체** | **{s['total']:,}** | **{s['win_rate']:.1f}** "
        f"| **{s['avg_ret']:+.2f}** | **{max_p:+.2f}** | **{max_l:+.2f}** "
        f"| **{s['pf']:.2f}** | **{s['total_pnl']:+,.0f}** | **{cap_ret:+.2f}** |"
    )
    lines.append(f"\n※ 자본대비수익률 = 연도별 총손익 / 베이스 자본({CAPITAL/10000:.0f}만원) × 100")
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
        "# 래리 MTL/MTH 스윙 연도별 백테스트\n",
        "## 규칙\n",
        f"- 대상: KOSPI200, 진입 필터 R% >= {R_PCT_MIN}%",
        f"- 최대 동시 보유: {MAX_POSITIONS}종목, 1거래당 {PER_TRADE/10000:.0f}만원 (고정)",
        f"- 베이스 자본: {CAPITAL/10000:.0f}만원",
        f"- 수수료: 매수 {COMMISSION*100:.3f}% + 매도 {SELL_COMMISSION*100:.3f}% + 세금 {TAX*100:.2f}%",
        "- 청산: 손절(STL-1원), 2R 50% 익절, STL 이탈 잔량",
        "- 기간: 2023-01-01 ~ 현재\n",
        f"- 슬롯부족 스킵 시그널: {skipped}건\n",
        "---\n",
    ]

    report = "\n".join(header) + yearly_report(tdf) + "\n"
    os.makedirs("results", exist_ok=True)
    path = "results/backtest_larry_yearly.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n결과 저장: {path}")
    print("\n" + yearly_report(tdf))


if __name__ == "__main__":
    main()
