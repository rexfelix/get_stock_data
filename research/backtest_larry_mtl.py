"""
래리 윌리엄스 MTL/MTH 기반 스윙 매매 백테스트
- STL/STH → MTL/MTH 계층 구조
- 진입: MTL 확정 후 MTH 확정 → 종가가 MTH 고가 상향돌파 시 당일 종가 매수
- 손절: 진입 시 최근 STL 저가 - 1원
- 1차 익절: 2R 도달 시 50% 청산
- 최종 청산: 가장 최근 STL 저가 이탈 시 전량 청산
- 포트폴리오: 최대 10종목, 1거래당 300만원
- 기간: 2023 ~ 현재
"""

import os
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

ENGINE = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

PER_TRADE = 3_000_000  # 1거래당 300만원
MAX_POSITIONS = 10
COMMISSION = 0.00015
SELL_COMMISSION = 0.00015
TAX = 0.0018


def load_data(kospi200_only: bool) -> tuple[pd.DataFrame, set | None]:
    with ENGINE.connect() as conn:
        k200 = None
        if kospi200_only:
            k200 = set(
                pd.read_sql(text("SELECT ticker FROM kospi200_members"), conn)["ticker"]
            )
        df = pd.read_sql(text("""
            SELECT date, open, high, low, close, ticker, name
            FROM stocks WHERE date >= '2022-01-01'
            ORDER BY ticker, date
        """), conn)
    df["date"] = pd.to_datetime(df["date"])
    return df, k200


def find_swing_points(gdf: pd.DataFrame) -> tuple[list, list]:
    """STL, STH 리스트 반환. 각 항목은 (pivot_idx, confirmed_idx)."""
    low = gdf["low"].values
    high = gdf["high"].values
    n = len(gdf)

    stls = []  # (pivot_idx, confirmed_idx)
    sths = []

    for i in range(1, n - 1):
        # STL: Low[i] < Low[i-1] and Low[i] < Low[i+1], confirmed at i+1
        if low[i] < low[i - 1] and low[i] < low[i + 1]:
            stls.append((i, i + 1))
        # STH: High[i] > High[i-1] and High[i] > High[i+1], confirmed at i+1
        if high[i] > high[i - 1] and high[i] > high[i + 1]:
            sths.append((i, i + 1))

    return stls, sths


def find_medium_points(swing_points: list, values: np.ndarray,
                       is_low: bool) -> list:
    """MTL 또는 MTH 리스트 반환. (pivot_idx, confirmed_idx)
    MTL: 연속 3 STL 중 가운데가 가장 낮음 → confirmed = 3번째 STL의 confirmed
    MTH: 연속 3 STH 중 가운데가 가장 높음 → confirmed = 3번째 STH의 confirmed
    """
    mts = []
    for k in range(2, len(swing_points)):
        a_idx = swing_points[k - 2][0]
        b_idx = swing_points[k - 1][0]
        c_idx = swing_points[k][0]
        confirmed = swing_points[k][1]  # 3번째 ST가 확정되는 날

        if is_low:
            if values[b_idx] < values[a_idx] and values[b_idx] < values[c_idx]:
                mts.append((b_idx, confirmed))
        else:
            if values[b_idx] > values[a_idx] and values[b_idx] > values[c_idx]:
                mts.append((b_idx, confirmed))

    return mts


def precompute(gdf: pd.DataFrame) -> dict:
    """종목별 사전 계산: STL, STH, MTL, MTH + 날짜별 인덱스."""
    low = gdf["low"].values
    high = gdf["high"].values

    stls, sths = find_swing_points(gdf)
    mtls = find_medium_points(stls, low, is_low=True)
    mths = find_medium_points(sths, high, is_low=False)

    return {
        "stls": stls, "sths": sths,
        "mtls": mtls, "mths": mths,
    }


def simulate(stocks: dict[str, pd.DataFrame], start_date: str,
             max_pos: int = MAX_POSITIONS) -> tuple[list[dict], pd.DataFrame, int]:
    """포트폴리오 시뮬레이션. returns (trades, daily_pos_df, skipped_count)"""
    # 모든 거래일 수집
    all_dates = set()
    for gdf in stocks.values():
        all_dates.update(gdf["date"].values)
    all_dates = sorted(all_dates)
    start = pd.Timestamp(start_date)
    all_dates = [d for d in all_dates if d >= start]

    # 종목별 사전 계산
    precomp = {}
    stock_data = {}
    for ticker, gdf in stocks.items():
        gdf = gdf.reset_index(drop=True)
        pc = precompute(gdf)
        precomp[ticker] = pc
        date_to_idx = {d: i for i, d in enumerate(gdf["date"].values)}
        stock_data[ticker] = {"df": gdf, "date_to_idx": date_to_idx}

    # 포지션 관리
    positions = []
    all_trades = []
    daily_pos_counts = []  # (date, count)
    skipped = 0  # 슬롯 부족으로 놓친 시그널

    for date in all_dates:
        # 1) 기존 포지션 체크 (손절/익절/STL trailing stop)
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
                    stl_low_val = td["df"].iloc[stl_pivot]["low"]
                    latest_stl_low = stl_low_val

            if row["low"] <= pos["initial_stop"]:
                sell_price = pos["initial_stop"]
                _close_position(pos, sell_price, date, "손절", all_trades)
                closed.append(pos)
                continue

            r = pos["entry_price"] - pos["initial_stop"]
            target_2r = pos["entry_price"] + 2 * r
            if not pos["took_profit_1"] and r > 0 and row["high"] >= target_2r:
                half_qty = pos["qty_remaining"] // 2
                if half_qty > 0:
                    _record_partial(pos, target_2r, date, "1차익절(2R)", half_qty, all_trades)
                    pos["qty_remaining"] -= half_qty
                pos["took_profit_1"] = True

            if latest_stl_low > pos["initial_stop"] and row["low"] <= latest_stl_low:
                sell_price = latest_stl_low
                _close_position(pos, sell_price, date, "STL이탈", all_trades)
                closed.append(pos)
                continue

        for c in closed:
            positions.remove(c)

        # 2) 신규 매수 시그널 탐색
        held_tickers = {p["ticker"] for p in positions}
        candidates = []

        for ticker, td in stock_data.items():
            if ticker in held_tickers:
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
                candidates.append({
                    "ticker": ticker,
                    "name": row["name"],
                    "entry_price": row["close"],
                    "initial_stop": latest_stl_low - 1,
                    "date": date,
                })

        # 슬롯 제한 내 매수
        valid_cands = [c for c in candidates
                       if c["entry_price"] > 0 and c["entry_price"] > c["initial_stop"]]
        if max_pos > 0:
            slots = max_pos - len(positions)
            if len(valid_cands) > slots:
                skipped += len(valid_cands) - max(slots, 0)
            for cand in valid_cands[:max(slots, 0)]:
                qty = int(PER_TRADE / cand["entry_price"])
                if qty <= 0:
                    continue
                positions.append({
                    "ticker": cand["ticker"],
                    "name": cand["name"],
                    "qty_full": qty,
                    "qty_remaining": qty,
                    "entry_price": cand["entry_price"],
                    "entry_date": date,
                    "initial_stop": cand["initial_stop"],
                    "took_profit_1": False,
                })
        else:
            # 무제한
            for cand in valid_cands:
                qty = int(PER_TRADE / cand["entry_price"])
                if qty <= 0:
                    continue
                positions.append({
                    "ticker": cand["ticker"],
                    "name": cand["name"],
                    "qty_full": qty,
                    "qty_remaining": qty,
                    "entry_price": cand["entry_price"],
                    "entry_date": date,
                    "initial_stop": cand["initial_stop"],
                    "took_profit_1": False,
                })

        daily_pos_counts.append({"date": date, "pos_count": len(positions)})

    # 미청산 포지션 처리
    for pos in positions:
        td = stock_data[pos["ticker"]]
        last_row = td["df"].iloc[-1]
        _close_position(pos, last_row["close"], last_row["date"], "미청산", all_trades)

    pos_df = pd.DataFrame(daily_pos_counts)
    pos_df["date"] = pd.to_datetime(pos_df["date"])
    return all_trades, pos_df, skipped


def _close_position(pos, sell_price, date, reason, trades):
    if pos["qty_remaining"] <= 0:
        return
    cost = pos["entry_price"] * pos["qty_remaining"]
    revenue = sell_price * pos["qty_remaining"]
    fee = cost * COMMISSION + revenue * (SELL_COMMISSION + TAX)
    pnl = revenue - cost - fee
    ret = pnl / cost * 100 if cost > 0 else 0
    trades.append({
        "ticker": pos["ticker"],
        "name": pos["name"],
        "buy_date": pd.Timestamp(pos["entry_date"]),
        "sell_date": pd.Timestamp(date),
        "buy_price": pos["entry_price"],
        "sell_price": sell_price,
        "qty": pos["qty_remaining"],
        "pnl": pnl,
        "return_pct": ret,
        "hold_days": (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
        "reason": reason,
    })


def _record_partial(pos, sell_price, date, reason, qty, trades):
    cost = pos["entry_price"] * qty
    revenue = sell_price * qty
    fee = cost * COMMISSION + revenue * (SELL_COMMISSION + TAX)
    pnl = revenue - cost - fee
    ret = pnl / cost * 100 if cost > 0 else 0
    trades.append({
        "ticker": pos["ticker"],
        "name": pos["name"],
        "buy_date": pd.Timestamp(pos["entry_date"]),
        "sell_date": pd.Timestamp(date),
        "buy_price": pos["entry_price"],
        "sell_price": sell_price,
        "qty": qty,
        "pnl": pnl,
        "return_pct": ret,
        "hold_days": (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
        "reason": reason,
    })


def stats(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {}
    wins = (df["return_pct"] > 0).sum()
    losses = total - wins
    win_rate = wins / total * 100
    avg_ret = df["return_pct"].mean()
    med_ret = df["return_pct"].median()
    avg_win = df.loc[df["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
    avg_loss = df.loc[df["return_pct"] <= 0, "return_pct"].mean() if losses > 0 else 0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    avg_hold = df["hold_days"].mean()
    total_pnl = df["pnl"].sum()
    return {
        "total": total, "wins": wins, "losses": losses, "win_rate": win_rate,
        "avg_ret": avg_ret, "med_ret": med_ret, "avg_win": avg_win,
        "avg_loss": avg_loss, "pf": pf, "avg_hold": avg_hold,
        "total_pnl": total_pnl,
    }


def generate_report(df: pd.DataFrame, pos_df: pd.DataFrame,
                    label: str, skipped: int) -> str:
    s = stats(df)
    if not s:
        return f"## {label}\n\n거래 없음\n"

    lines = [f"## {label}\n"]
    lines.append("### 전체 통계\n")
    lines.append("| 지표 | 값 |")
    lines.append("|---|---:|")
    lines.append(f"| 총 거래수 | {s['total']:,} |")
    lines.append(f"| 승률(%) | {s['win_rate']:.1f} |")
    lines.append(f"| 평균 수익률(%) | {s['avg_ret']:+.2f} |")
    lines.append(f"| 중간값 수익률(%) | {s['med_ret']:+.2f} |")
    lines.append(f"| 평균이익(%) | {s['avg_win']:+.2f} |")
    lines.append(f"| 평균손실(%) | {s['avg_loss']:+.2f} |")
    lines.append(f"| 손익비 | {s['pf']:.2f} |")
    lines.append(f"| 평균 보유일 | {s['avg_hold']:.1f} |")
    lines.append(f"| 총 손익 | {s['total_pnl']:+,.0f}원 |")
    if skipped > 0:
        lines.append(f"| 슬롯부족 스킵 시그널 | {skipped:,} |")

    # 보유종목수 통계 (연도별)
    if not pos_df.empty:
        pos_df["year"] = pos_df["date"].dt.year
        lines.append("\n### 연도별 보유종목수\n")
        lines.append("| 연도 | 평균 | 최대 | 최대 기간 | 최소 | 최소 기간 |")
        lines.append("|---:|---:|---:|---|---:|---|")
        for y in sorted(pos_df["year"].unique()):
            yp = pos_df[pos_df["year"] == y]
            avg_p = yp["pos_count"].mean()
            max_p = yp["pos_count"].max()
            min_p = yp["pos_count"].min()
            # 최대 종목수 기간
            max_rows = yp[yp["pos_count"] == max_p]
            max_start = pd.Timestamp(max_rows["date"].iloc[0]).strftime("%m/%d")
            max_end = pd.Timestamp(max_rows["date"].iloc[-1]).strftime("%m/%d")
            max_period = f"{max_start}~{max_end}" if max_start != max_end else max_start
            # 최소 종목수 기간
            min_rows = yp[yp["pos_count"] == min_p]
            min_start = pd.Timestamp(min_rows["date"].iloc[0]).strftime("%m/%d")
            min_end = pd.Timestamp(min_rows["date"].iloc[-1]).strftime("%m/%d")
            min_period = f"{min_start}~{min_end}" if min_start != min_end else min_start
            lines.append(
                f"| {y} | {avg_p:.1f} | {max_p} | {max_period} "
                f"| {min_p} | {min_period} |"
            )

    # 청산 사유별 통계
    lines.append("\n### 청산 사유별\n")
    lines.append("| 사유 | 거래수 | 승률(%) | 평균수익(%) | 총손익 |")
    lines.append("|---|---:|---:|---:|---:|")
    for reason in ["1차익절(2R)", "STL이탈", "손절", "미청산"]:
        sub = df[df["reason"] == reason]
        if len(sub) == 0:
            continue
        rs = stats(sub)
        lines.append(
            f"| {reason} | {rs['total']:,} | {rs['win_rate']:.1f} "
            f"| {rs['avg_ret']:+.2f} | {rs['total_pnl']:+,.0f}원 |"
        )

    # 연도별
    df = df.copy()
    df["buy_year"] = df["buy_date"].dt.year
    lines.append("\n### 연도별 통계\n")
    lines.append("| 연도 | 거래수 | 승률(%) | 평균수익(%) | 중간값(%) | 손익비 | 총손익 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for y in sorted(df["buy_year"].unique()):
        sub = df[df["buy_year"] == y]
        ys = stats(sub)
        lines.append(
            f"| {y} | {ys['total']:,} | {ys['win_rate']:.1f} "
            f"| {ys['avg_ret']:+.2f} | {ys['med_ret']:+.2f} "
            f"| {ys['pf']:.2f} | {ys['total_pnl']:+,.0f}원 |"
        )

    # Top/Bottom 20
    lines.append("\n### 수익률 상위 (Top 20)\n")
    lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) | 사유 |")
    lines.append("|---|---|---|---:|---:|---|")
    for _, r in df.nlargest(20, "return_pct").iterrows():
        lines.append(
            f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
            f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
            f"| {r['return_pct']:+.1f} | {r['reason']} |"
        )

    lines.append("\n### 수익률 하위 (Bottom 20)\n")
    lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) | 사유 |")
    lines.append("|---|---|---|---:|---:|---|")
    for _, r in df.nsmallest(20, "return_pct").iterrows():
        lines.append(
            f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
            f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
            f"| {r['return_pct']:+.1f} | {r['reason']} |"
        )

    return "\n".join(lines)


def main():
    report_lines = ["# 래리 윌리엄스 MTL/MTH 스윙 매매 백테스트\n"]
    report_lines.append("## 매매 규칙\n")
    report_lines.append("- **STL**: Low[i] < Low[i-1] and Low[i] < Low[i+1]")
    report_lines.append("- **STH**: High[i] > High[i-1] and High[i] > High[i+1]")
    report_lines.append("- **MTL**: 연속 3 STL 중 가운데 저가가 가장 낮음")
    report_lines.append("- **MTH**: 연속 3 STH 중 가운데 고가가 가장 높음")
    report_lines.append("- **진입**: MTL 확정 후 MTH 확정 → 종가가 MTH 고가 상향돌파 시 당일 종가 매수")
    report_lines.append("- **손절**: 진입 시 최근 STL 저가 - 1원")
    report_lines.append("- **1차 익절**: 2R 도달 시 50% 청산 (R = 진입가 - 손절가)")
    report_lines.append("- **최종 청산**: 보유 중 가장 최근 STL 저가 이탈 시 전량")
    report_lines.append(f"- **투자금**: 1거래당 {PER_TRADE/10000:.0f}만원")
    report_lines.append(f"- **수수료**: 매수 {COMMISSION*100:.3f}% + 매도 {SELL_COMMISSION*100:.3f}% + 세금 {TAX*100:.2f}%")
    report_lines.append("- **기간**: 2023 ~ 현재\n")
    report_lines.append("---\n")

    # KOSPI200 데이터 한번만 로드
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

    # 시나리오: max10 vs 무제한
    scenarios = [
        ("KOSPI200 (최대 10종목)", 10),
        ("KOSPI200 (무제한)", 0),
    ]

    for label, max_pos in scenarios:
        print(f"\n{'='*50}")
        print(f"[{label}] 시뮬레이션 실행...")
        t0 = time.time()
        trades, pos_df, skipped = simulate(stocks, "2023-01-01", max_pos=max_pos)
        elapsed = time.time() - t0
        print(f"  완료: {elapsed:.1f}초")

        if not trades:
            print("  거래 없음!")
            report_lines.append(f"\n## {label}\n\n거래 없음\n")
            continue

        tdf = pd.DataFrame(trades)
        s = stats(tdf)
        print(f"  {label}: {s['total']:,}건, 승률 {s['win_rate']:.1f}%, "
              f"평균수익 {s['avg_ret']:+.2f}%, 손익비 {s['pf']:.2f}, "
              f"총손익 {s['total_pnl']:+,.0f}원, 스킵 {skipped}건")

        report_lines.append(
            f"\n{generate_report(tdf, pos_df, label, skipped)}\n"
        )

    report = "\n".join(report_lines)
    os.makedirs("results", exist_ok=True)
    with open("results/backtest_larry_mtl.md", "w") as f:
        f.write(report + "\n")
    print(f"\n결과 저장: results/backtest_larry_mtl.md")


if __name__ == "__main__":
    main()
