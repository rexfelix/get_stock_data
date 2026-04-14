"""
래리 MTL/MTH 전략 - 진입 시점 필터 분석
무제한 KOSPI200 시뮬의 각 거래에 진입 시점 지표를 태깅하고
어떤 조건의 종목이 수익률이 높은지 비교한다.

테스트 필터:
1. 종가 vs MA20/MA60/MA120 위치
2. 이동평균선 정배열 (MA20>MA60, MA20>MA60>MA120)
3. 거래대금 1500억 연속일수 (10/10, 20/20)
4. R 크기 (진입가-손절가 비율)
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

PER_TRADE = 3_000_000
COMMISSION = 0.00015
SELL_COMMISSION = 0.00015
TAX = 0.0018
AMOUNT_THRESHOLD = 150_000  # 1500억 = 150,000 백만원


def load_data():
    with ENGINE.connect() as conn:
        k200 = set(
            pd.read_sql(text("SELECT ticker FROM kospi200_members"), conn)["ticker"]
        )
        df = pd.read_sql(text("""
            SELECT s.date, s.open, s.high, s.low, s.close, s.ticker, s.name,
                   sa.amount
            FROM stocks s
            LEFT JOIN stock_all sa ON s.ticker = sa.ticker AND s.date = sa.date
            WHERE s.date >= '2022-01-01'
            ORDER BY s.ticker, s.date
        """), conn)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df, k200


def prepare_stock(gdf: pd.DataFrame) -> pd.DataFrame:
    """종목별 지표 사전 계산"""
    gdf = gdf.copy()
    gdf["ma20"] = gdf["close"].rolling(20).mean()
    gdf["ma60"] = gdf["close"].rolling(60).mean()
    gdf["ma120"] = gdf["close"].rolling(120).mean()
    gdf["above_1500"] = (gdf["amount"] >= AMOUNT_THRESHOLD).astype(int)
    gdf["above_10d"] = gdf["above_1500"].rolling(10, min_periods=10).sum()
    gdf["above_20d"] = gdf["above_1500"].rolling(20, min_periods=20).sum()
    return gdf


def find_swing_points(gdf):
    low = gdf["low"].values
    high = gdf["high"].values
    n = len(gdf)
    stls, sths = [], []
    for i in range(1, n - 1):
        if low[i] < low[i - 1] and low[i] < low[i + 1]:
            stls.append((i, i + 1))
        if high[i] > high[i - 1] and high[i] > high[i + 1]:
            sths.append((i, i + 1))
    return stls, sths


def find_medium_points(swing_points, values, is_low):
    mts = []
    for k in range(2, len(swing_points)):
        a_idx = swing_points[k - 2][0]
        b_idx = swing_points[k - 1][0]
        c_idx = swing_points[k][0]
        confirmed = swing_points[k][1]
        if is_low:
            if values[b_idx] < values[a_idx] and values[b_idx] < values[c_idx]:
                mts.append((b_idx, confirmed))
        else:
            if values[b_idx] > values[a_idx] and values[b_idx] > values[c_idx]:
                mts.append((b_idx, confirmed))
    return mts


def simulate_with_tags(stocks: dict) -> list[dict]:
    """무제한 포지션 시뮬레이션 + 진입 시점 지표 태깅"""
    all_dates = set()
    for gdf in stocks.values():
        all_dates.update(gdf["date"].values)
    all_dates = sorted(d for d in all_dates if d >= pd.Timestamp("2023-01-01"))

    precomp = {}
    stock_data = {}
    for ticker, gdf in stocks.items():
        gdf = gdf.reset_index(drop=True)
        stls, sths = find_swing_points(gdf)
        mtls = find_medium_points(stls, gdf["low"].values, True)
        mths = find_medium_points(sths, gdf["high"].values, False)
        precomp[ticker] = {"stls": stls, "sths": sths, "mtls": mtls, "mths": mths}
        stock_data[ticker] = {"df": gdf,
                              "date_to_idx": {d: i for i, d in enumerate(gdf["date"].values)}}

    positions = []
    all_trades = []

    for date in all_dates:
        # 1) 기존 포지션 체크
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
                _close(pos, pos["initial_stop"], date, "손절", all_trades)
                closed.append(pos)
                continue

            r = pos["entry_price"] - pos["initial_stop"]
            target_2r = pos["entry_price"] + 2 * r
            if not pos["took_profit_1"] and r > 0 and row["high"] >= target_2r:
                half = pos["qty_rem"] // 2
                if half > 0:
                    _partial(pos, target_2r, date, "1차익절(2R)", half, all_trades)
                    pos["qty_rem"] -= half
                pos["took_profit_1"] = True

            if latest_stl_low > pos["initial_stop"] and row["low"] <= latest_stl_low:
                _close(pos, latest_stl_low, date, "STL이탈", all_trades)
                closed.append(pos)
                continue

        for c in closed:
            positions.remove(c)

        # 2) 신규 시그널
        held = {p["ticker"] for p in positions}
        for ticker, td in stock_data.items():
            if ticker in held:
                continue
            idx = td["date_to_idx"].get(date)
            if idx is None or idx < 1:
                continue
            df = td["df"]
            row = df.iloc[idx]
            pc = precomp[ticker]

            latest_mtl = None
            for p, c in pc["mtls"]:
                if c <= idx:
                    latest_mtl = (p, c)
            if not latest_mtl:
                continue

            latest_mth = None
            for p, c in pc["mths"]:
                if c <= idx and p > latest_mtl[0]:
                    latest_mth = (p, c)
            if not latest_mth:
                continue

            mth_high = df.iloc[latest_mth[0]]["high"]
            prev_close = df.iloc[idx - 1]["close"]
            if not (prev_close <= mth_high and row["close"] > mth_high):
                continue

            latest_stl_low = None
            for sp, sc in pc["stls"]:
                if sc <= idx:
                    latest_stl_low = df.iloc[sp]["low"]
            if latest_stl_low is None:
                continue

            entry_price = row["close"]
            stop = latest_stl_low - 1
            if entry_price <= 0 or entry_price <= stop:
                continue
            qty = int(PER_TRADE / entry_price)
            if qty <= 0:
                continue

            # 진입 시점 지표 태깅
            ma20 = row.get("ma20", np.nan)
            ma60 = row.get("ma60", np.nan)
            ma120 = row.get("ma120", np.nan)
            above_10d = row.get("above_10d", np.nan)
            above_20d = row.get("above_20d", np.nan)

            r_pct = (entry_price - stop) / entry_price * 100

            tags = {
                "above_ma20": not np.isnan(ma20) and entry_price > ma20,
                "above_ma60": not np.isnan(ma60) and entry_price > ma60,
                "above_ma120": not np.isnan(ma120) and entry_price > ma120,
                "ma20_above_ma60": not np.isnan(ma20) and not np.isnan(ma60) and ma20 > ma60,
                "ma_triple": (not np.isnan(ma20) and not np.isnan(ma60) and not np.isnan(ma120)
                              and ma20 > ma60 > ma120),
                "amount_10_10": not np.isnan(above_10d) and above_10d >= 10,
                "amount_20_20": not np.isnan(above_20d) and above_20d >= 20,
                "r_pct": r_pct,
            }

            positions.append({
                "ticker": ticker, "name": row["name"],
                "qty_full": qty, "qty_rem": qty,
                "entry_price": entry_price, "entry_date": date,
                "initial_stop": stop, "took_profit_1": False,
                "tags": tags,
            })

    for pos in positions:
        td = stock_data[pos["ticker"]]
        last = td["df"].iloc[-1]
        _close(pos, last["close"], last["date"], "미청산", all_trades)

    return all_trades


def _close(pos, price, date, reason, trades):
    if pos["qty_rem"] <= 0:
        return
    cost = pos["entry_price"] * pos["qty_rem"]
    rev = price * pos["qty_rem"]
    fee = cost * COMMISSION + rev * (SELL_COMMISSION + TAX)
    pnl = rev - cost - fee
    ret = pnl / cost * 100 if cost > 0 else 0
    rec = {
        "ticker": pos["ticker"], "name": pos["name"],
        "buy_date": pd.Timestamp(pos["entry_date"]),
        "sell_date": pd.Timestamp(date),
        "buy_price": pos["entry_price"], "sell_price": price,
        "qty": pos["qty_rem"], "pnl": pnl, "return_pct": ret,
        "hold_days": (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
        "reason": reason,
    }
    rec.update(pos.get("tags", {}))
    trades.append(rec)


def _partial(pos, price, date, reason, qty, trades):
    cost = pos["entry_price"] * qty
    rev = price * qty
    fee = cost * COMMISSION + rev * (SELL_COMMISSION + TAX)
    pnl = rev - cost - fee
    ret = pnl / cost * 100 if cost > 0 else 0
    rec = {
        "ticker": pos["ticker"], "name": pos["name"],
        "buy_date": pd.Timestamp(pos["entry_date"]),
        "sell_date": pd.Timestamp(date),
        "buy_price": pos["entry_price"], "sell_price": price,
        "qty": qty, "pnl": pnl, "return_pct": ret,
        "hold_days": (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
        "reason": reason,
    }
    rec.update(pos.get("tags", {}))
    trades.append(rec)


def filter_stats(df, mask, label):
    yes = df[mask]
    no = df[~mask]
    def s(sub):
        n = len(sub)
        if n == 0:
            return {"n": 0}
        wins = (sub["return_pct"] > 0).sum()
        return {
            "n": n,
            "win_rate": wins / n * 100,
            "avg_ret": sub["return_pct"].mean(),
            "med_ret": sub["return_pct"].median(),
            "avg_win": sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0,
            "avg_loss": sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if wins < n else 0,
            "pf": abs(sub.loc[sub["return_pct"] > 0, "return_pct"].mean() /
                      sub.loc[sub["return_pct"] <= 0, "return_pct"].mean())
                  if wins > 0 and wins < n else float("inf") if wins == n else 0,
            "total_pnl": sub["pnl"].sum(),
        }
    return label, s(yes), s(no)


def main():
    print("데이터 로딩...")
    t0 = time.time()
    df, k200 = load_data()
    print(f"  로드: {time.time()-t0:.1f}초")

    stocks = {}
    for ticker, gdf in df.groupby("ticker"):
        if ticker not in k200:
            continue
        stocks[ticker] = prepare_stock(gdf.reset_index(drop=True))
    print(f"  KOSPI200: {len(stocks)}종목")

    print("시뮬레이션 실행...")
    t0 = time.time()
    trades = simulate_with_tags(stocks)
    print(f"  완료: {time.time()-t0:.1f}초, {len(trades):,}건")

    tdf = pd.DataFrame(trades)

    # ── 필터별 성과 비교 ──
    filters = [
        ("종가 > MA20", tdf["above_ma20"] == True),
        ("종가 > MA60", tdf["above_ma60"] == True),
        ("종가 > MA120", tdf["above_ma120"] == True),
        ("MA20 > MA60", tdf["ma20_above_ma60"] == True),
        ("정배열 (MA20>MA60>MA120)", tdf["ma_triple"] == True),
        ("거래대금 10/10일", tdf["amount_10_10"] == True),
        ("거래대금 20/20일", tdf["amount_20_20"] == True),
        ("R% < 3%", tdf["r_pct"] < 3),
        ("R% < 5%", tdf["r_pct"] < 5),
        ("R% < 7%", tdf["r_pct"] < 7),
    ]

    lines = ["# 래리 MTL/MTH 전략 - 진입 필터 분석 (KOSPI200, 무제한)\n"]
    lines.append("각 필터를 만족하는 거래 vs 만족하지 않는 거래의 성과 비교\n")
    lines.append("---\n")

    lines.append("## 필터별 성과 비교\n")
    lines.append("| 필터 | 구분 | 거래수 | 승률(%) | 평균수익(%) | 중간값(%) | 손익비 | 총손익 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for label, mask in filters:
        _, yes_s, no_s = filter_stats(tdf, mask, label)
        for tag, s in [("O", yes_s), ("X", no_s)]:
            if s["n"] == 0:
                lines.append(f"| {label} | {tag} | 0 | - | - | - | - | - |")
            else:
                lines.append(
                    f"| {label} | {tag} | {s['n']:,} | {s['win_rate']:.1f} "
                    f"| {s['avg_ret']:+.2f} | {s['med_ret']:+.2f} "
                    f"| {s['pf']:.2f} | {s['total_pnl']/10000:+,.0f}만원 |"
                )

    # 복합 필터
    lines.append("\n## 복합 필터 성과\n")
    combos = [
        ("MA20위 + MA20>MA60", (tdf["above_ma20"] == True) & (tdf["ma20_above_ma60"] == True)),
        ("정배열 + 거래대금10/10", (tdf["ma_triple"] == True) & (tdf["amount_10_10"] == True)),
        ("정배열 + 거래대금20/20", (tdf["ma_triple"] == True) & (tdf["amount_20_20"] == True)),
        ("MA20위 + 거래대금10/10", (tdf["above_ma20"] == True) & (tdf["amount_10_10"] == True)),
        ("MA20위 + 거래대금20/20", (tdf["above_ma20"] == True) & (tdf["amount_20_20"] == True)),
        ("MA20위 + R%<5%", (tdf["above_ma20"] == True) & (tdf["r_pct"] < 5)),
        ("정배열 + R%<5%", (tdf["ma_triple"] == True) & (tdf["r_pct"] < 5)),
        ("정배열 + 거래대금10/10 + R%<5%",
         (tdf["ma_triple"] == True) & (tdf["amount_10_10"] == True) & (tdf["r_pct"] < 5)),
    ]

    lines.append("| 필터 | 거래수 | 승률(%) | 평균수익(%) | 중간값(%) | 손익비 | 총손익 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    # 기준선: 필터 없음
    base_s = filter_stats(tdf, tdf.index == tdf.index, "전체")[1]
    lines.append(
        f"| **(기준) 필터 없음** | **{base_s['n']:,}** | **{base_s['win_rate']:.1f}** "
        f"| **{base_s['avg_ret']:+.2f}** | **{base_s['med_ret']:+.2f}** "
        f"| **{base_s['pf']:.2f}** | **{base_s['total_pnl']/10000:+,.0f}만원** |"
    )

    for label, mask in combos:
        sub = tdf[mask]
        n = len(sub)
        if n == 0:
            lines.append(f"| {label} | 0 | - | - | - | - | - |")
            continue
        wins = (sub["return_pct"] > 0).sum()
        wr = wins / n * 100
        avg = sub["return_pct"].mean()
        med = sub["return_pct"].median()
        avg_w = sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
        avg_l = sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if wins < n else 0
        pf = abs(avg_w / avg_l) if avg_l != 0 else float("inf")
        pnl = sub["pnl"].sum()
        lines.append(
            f"| {label} | {n:,} | {wr:.1f} "
            f"| {avg:+.2f} | {med:+.2f} "
            f"| {pf:.2f} | {pnl/10000:+,.0f}만원 |"
        )

    # R% 구간별 성과
    lines.append("\n## R% 구간별 성과 (손절폭 비율)\n")
    lines.append("| R% 구간 | 거래수 | 승률(%) | 평균수익(%) | 중간값(%) | 손익비 | 총손익 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    bins = [(0, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 100)]
    for lo, hi in bins:
        mask = (tdf["r_pct"] >= lo) & (tdf["r_pct"] < hi)
        sub = tdf[mask]
        n = len(sub)
        if n == 0:
            continue
        wins = (sub["return_pct"] > 0).sum()
        wr = wins / n * 100
        avg = sub["return_pct"].mean()
        med = sub["return_pct"].median()
        avg_w = sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
        avg_l = sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if wins < n else 0
        pf = abs(avg_w / avg_l) if avg_l != 0 else float("inf")
        pnl = sub["pnl"].sum()
        lines.append(
            f"| {lo}~{hi}% | {n:,} | {wr:.1f} "
            f"| {avg:+.2f} | {med:+.2f} "
            f"| {pf:.2f} | {pnl/10000:+,.0f}만원 |"
        )

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs("results", exist_ok=True)
    with open("results/analyze_larry_filters.md", "w") as f:
        f.write(report + "\n")
    print("\n결과 저장: results/analyze_larry_filters.md")


if __name__ == "__main__":
    main()
