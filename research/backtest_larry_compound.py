"""래리 MTL/MTH 복리 백테스트: 3000만원으로 시작, 매 진입 시 현재 자본/10 투입."""

import os
import time
import pandas as pd
from backtest_larry_mtl import (
    load_data, precompute, stats,
    COMMISSION, SELL_COMMISSION, TAX,
)

INITIAL_CAPITAL = 30_000_000
MAX_POSITIONS = 10
R_PCT_MIN = 7.0


def _fee_buy(cost):
    return cost * COMMISSION


def _fee_sell(revenue):
    return revenue * (SELL_COMMISSION + TAX)


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

    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    equity_curve = []
    skipped = 0

    def close_pos(pos, sell_price, date, reason, qty):
        nonlocal cash
        cost = pos["entry_price"] * qty
        revenue = sell_price * qty
        fee = _fee_buy(cost) + _fee_sell(revenue)
        pnl = revenue - cost - fee
        cash += revenue - _fee_sell(revenue)
        ret = pnl / (cost + _fee_buy(cost)) * 100 if cost > 0 else 0
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
                close_pos(pos, pos["initial_stop"], date, "손절", pos["qty_remaining"])
                closed.append(pos)
                continue

            r = pos["entry_price"] - pos["initial_stop"]
            target_2r = pos["entry_price"] + 2 * r
            if not pos["took_profit_1"] and r > 0 and row["high"] >= target_2r:
                half_qty = pos["qty_remaining"] // 2
                if half_qty > 0:
                    close_pos(pos, target_2r, date, "1차익절(2R)", half_qty)
                    pos["qty_remaining"] -= half_qty
                pos["took_profit_1"] = True

            if latest_stl_low > pos["initial_stop"] and row["low"] <= latest_stl_low:
                close_pos(pos, latest_stl_low, date, "STL이탈", pos["qty_remaining"])
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
                    "entry_price": entry, "initial_stop": stop, "date": date,
                })

        # 현재 총자산 추정 (보유 포지션 당일 종가 평가)
        holdings_value = 0
        for pos in positions:
            td = stock_data[pos["ticker"]]
            idx = td["date_to_idx"].get(date)
            if idx is not None:
                holdings_value += td["df"].iloc[idx]["close"] * pos["qty_remaining"]
            else:
                holdings_value += pos["entry_price"] * pos["qty_remaining"]
        total_equity = cash + holdings_value

        slots = MAX_POSITIONS - len(positions)
        if len(cands) > slots:
            skipped += len(cands) - max(slots, 0)
        # 1건당 투자금: 총자산 / 10 (복리)
        per_trade = total_equity / MAX_POSITIONS
        for cand in cands[:max(slots, 0)]:
            qty = int(per_trade / cand["entry_price"])
            if qty <= 0:
                continue
            cost = cand["entry_price"] * qty
            buy_fee = _fee_buy(cost)
            if cash < cost + buy_fee:
                # 현금 부족 → 건너뜀
                continue
            cash -= (cost + buy_fee)
            positions.append({
                "ticker": cand["ticker"], "name": cand["name"],
                "qty_full": qty, "qty_remaining": qty,
                "entry_price": cand["entry_price"],
                "entry_date": date,
                "initial_stop": cand["initial_stop"],
                "took_profit_1": False,
            })

        # equity 기록
        holdings_value = 0
        for pos in positions:
            td = stock_data[pos["ticker"]]
            idx = td["date_to_idx"].get(date)
            if idx is not None:
                holdings_value += td["df"].iloc[idx]["close"] * pos["qty_remaining"]
            else:
                holdings_value += pos["entry_price"] * pos["qty_remaining"]
        equity_curve.append({
            "date": pd.Timestamp(date), "cash": cash,
            "holdings": holdings_value, "equity": cash + holdings_value,
        })

    # 미청산 종결
    for pos in positions:
        td = stock_data[pos["ticker"]]
        last_row = td["df"].iloc[-1]
        close_pos(pos, last_row["close"], last_row["date"], "미청산", pos["qty_remaining"])

    return trades, equity_curve, skipped


def yearly_report(trades_df, eq_df):
    trades_df = trades_df.copy()
    trades_df["year"] = trades_df["sell_date"].dt.year
    eq_df = eq_df.copy()
    eq_df["year"] = eq_df["date"].dt.year

    lines = []
    lines.append("## 연도별 성과 (복리 운용)\n")
    lines.append("| 연도 | 시작자본 | 종료자본 | 연수익률(%) | 거래수 | 승률(%) | 평균수익(%) | 최대수익(%) | 최대손실(%) | 손익비 | 연손익(원) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    prev_end = INITIAL_CAPITAL
    for y in sorted(eq_df["year"].unique()):
        ye = eq_df[eq_df["year"] == y]
        start_eq = prev_end
        end_eq = ye["equity"].iloc[-1]
        year_ret = (end_eq / start_eq - 1) * 100
        year_pnl = end_eq - start_eq

        sub = trades_df[trades_df["year"] == y]
        if len(sub) > 0:
            s = stats(sub)
            max_p = sub["return_pct"].max()
            max_l = sub["return_pct"].min()
            lines.append(
                f"| {y} | {start_eq:,.0f} | {end_eq:,.0f} | {year_ret:+.2f} "
                f"| {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} "
                f"| {max_p:+.2f} | {max_l:+.2f} | {s['pf']:.2f} | {year_pnl:+,.0f} |"
            )
        else:
            lines.append(
                f"| {y} | {start_eq:,.0f} | {end_eq:,.0f} | {year_ret:+.2f} "
                f"| 0 | - | - | - | - | - | {year_pnl:+,.0f} |"
            )
        prev_end = end_eq

    # 전체
    s = stats(trades_df)
    max_p = trades_df["return_pct"].max()
    max_l = trades_df["return_pct"].min()
    final_eq = eq_df["equity"].iloc[-1]
    total_ret = (final_eq / INITIAL_CAPITAL - 1) * 100
    total_pnl = final_eq - INITIAL_CAPITAL
    lines.append(
        f"| **전체** | **{INITIAL_CAPITAL:,.0f}** | **{final_eq:,.0f}** | **{total_ret:+.2f}** "
        f"| **{s['total']:,}** | **{s['win_rate']:.1f}** | **{s['avg_ret']:+.2f}** "
        f"| **{max_p:+.2f}** | **{max_l:+.2f}** | **{s['pf']:.2f}** | **{total_pnl:+,.0f}** |"
    )

    # MDD
    eq_series = eq_df.set_index("date")["equity"]
    peak = eq_series.cummax()
    dd = (eq_series / peak - 1) * 100
    mdd = dd.min()
    mdd_date = dd.idxmin()

    # 연환산 CAGR
    days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days
    years = days / 365.25
    cagr = (final_eq / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else 0

    lines.append("")
    lines.append("## 요약\n")
    lines.append(f"- 초기 자본: {INITIAL_CAPITAL:,}원")
    lines.append(f"- 최종 자본: {final_eq:,.0f}원")
    lines.append(f"- 누적 수익률: {total_ret:+.2f}%")
    lines.append(f"- CAGR: {cagr*100:+.2f}% (운용 {years:.2f}년)")
    lines.append(f"- MDD: {mdd:.2f}% ({mdd_date.strftime('%Y-%m-%d')})")
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

    print("시뮬레이션(복리)...")
    t0 = time.time()
    trades, eq_curve, skipped = simulate(stocks, "2023-01-01")
    print(f"  완료: {time.time()-t0:.1f}초, 거래 {len(trades)}건, 스킵 {skipped}건")

    if not trades:
        print("거래 없음")
        return

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(eq_curve)

    header = [
        "# 래리 MTL/MTH 복리 운용 백테스트\n",
        "## 규칙\n",
        f"- 초기 자본: {INITIAL_CAPITAL:,}원",
        f"- 최대 동시 보유: {MAX_POSITIONS}종목",
        "- **1건당 투자금: 진입 시점 총자산 / 10 (복리)**",
        f"- 대상: KOSPI200, 진입 필터 R% >= {R_PCT_MIN}%",
        "- 청산: 손절(STL-1원), 2R 50% 익절, STL 이탈 잔량",
        f"- 수수료: 매수 {COMMISSION*100:.3f}% + 매도 {SELL_COMMISSION*100:.3f}% + 세금 {TAX*100:.2f}%",
        "- 기간: 2023-01-01 ~ 현재\n",
        f"- 슬롯부족 스킵: {skipped}건\n",
        "---\n",
    ]

    report = "\n".join(header) + yearly_report(tdf, edf) + "\n"
    os.makedirs("results", exist_ok=True)
    path = "results/backtest_larry_compound.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n결과 저장: {path}")
    print("\n" + yearly_report(tdf, edf))


if __name__ == "__main__":
    main()
