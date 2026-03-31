"""
Williams %R(9) + CCI(14) 복합 시그널 백테스트
- 매수: WR(9) crossup(-80) AND CCI(14) crossup(-100) 동시
- 매도 비교:
  A) MA 이탈 (5/10/20)
  B) 래리식 단기고점 저가 이탈
- 손절: 매수일 저가 이탈
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


def calc_wr(df: pd.DataFrame, period: int = 9) -> pd.Series:
    """Williams %R"""
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return (hh - df["close"]) / (hh - ll) * (-100)


def calc_cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Commodity Channel Index"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)


def load_stocks() -> dict[str, pd.DataFrame]:
    query = text("""
        SELECT date, open, high, low, close, ticker
        FROM stocks WHERE date >= '2022-11-01'
        ORDER BY ticker, date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return {t: g.reset_index(drop=True) for t, g in df.groupby("ticker")}


def simulate_trades_swing_high(df: pd.DataFrame, ticker: str = "") -> list[dict]:
    """매도: 래리식 단기고점 캔들의 저가 이탈"""
    if len(df) < 20:
        return []

    df = df.copy()
    df["wr"] = calc_wr(df, 9)
    df["cci"] = calc_cci(df, 14)

    high = df["high"]
    df["is_swing_high"] = (high > high.shift(1)) & (high > high.shift(-1))

    trades = []
    in_position = False
    buy_price = buy_date = stop_loss = 0
    trailing_stop = 0.0  # 단기고점 캔들의 저가

    for i in range(1, len(df)):
        wr_prev = df.at[i - 1, "wr"]
        wr_curr = df.at[i, "wr"]
        cci_prev = df.at[i - 1, "cci"]
        cci_curr = df.at[i, "cci"]
        row = df.iloc[i]

        if any(pd.isna(v) for v in [wr_prev, wr_curr, cci_prev, cci_curr]):
            continue

        if not in_position:
            wr_cross = wr_prev < -80 and wr_curr >= -80
            cci_cross = cci_prev < -100 and cci_curr >= -100
            if wr_cross and cci_cross:
                buy_price = row["close"]
                buy_date = row["date"]
                stop_loss = row["low"]
                trailing_stop = 0.0
                in_position = True
        else:
            # 1) 손절
            if row["low"] <= stop_loss:
                ret = (stop_loss - buy_price) / buy_price * 100
                trades.append({
                    "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
                    "sell_date": row["date"], "sell_price": stop_loss,
                    "return_pct": ret, "hold_days": (row["date"] - buy_date).days,
                    "reason": "손절",
                })
                in_position = False
                continue

            # 2) 단기고점 확정(i-1) → 트레일링 스톱 갱신
            if i >= 2 and df.iloc[i - 1]["is_swing_high"]:
                swing_high_low = df.iloc[i - 1]["low"]
                if swing_high_low > trailing_stop:
                    trailing_stop = swing_high_low

            # 3) 트레일링 스톱 이탈 매도
            if trailing_stop > 0 and row["low"] <= trailing_stop:
                ret = (trailing_stop - buy_price) / buy_price * 100
                trades.append({
                    "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
                    "sell_date": row["date"], "sell_price": trailing_stop,
                    "return_pct": ret, "hold_days": (row["date"] - buy_date).days,
                    "reason": "고점저가이탈",
                })
                in_position = False
                continue

    if in_position:
        last = df.iloc[-1]
        ret = (last["close"] - buy_price) / buy_price * 100
        trades.append({
            "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
            "sell_date": last["date"], "sell_price": last["close"],
            "return_pct": ret, "hold_days": (last["date"] - buy_date).days,
            "reason": "미청산",
        })
    return trades


def simulate_trades(df: pd.DataFrame, ticker: str = "", ma_period: int = 5) -> list[dict]:
    if len(df) < 20:
        return []

    df = df.copy()
    df["wr"] = calc_wr(df, 9)
    df["cci"] = calc_cci(df, 14)
    df["ma"] = df["close"].rolling(ma_period).mean()

    trades = []
    in_position = False
    buy_price = 0.0
    buy_date = None
    stop_loss = 0.0
    above_ma = False

    for i in range(1, len(df)):
        wr_prev = df.at[i - 1, "wr"]
        wr_curr = df.at[i, "wr"]
        cci_prev = df.at[i - 1, "cci"]
        cci_curr = df.at[i, "cci"]
        row = df.iloc[i]
        ma = row["ma"]

        if any(pd.isna(v) for v in [wr_prev, wr_curr, cci_prev, cci_curr]):
            continue

        if not in_position:
            # 매수: WR crossup(-80) AND CCI crossup(-100)
            wr_cross = wr_prev < -80 and wr_curr >= -80
            cci_cross = cci_prev < -100 and cci_curr >= -100
            if wr_cross and cci_cross:
                buy_price = row["close"]
                buy_date = row["date"]
                stop_loss = row["low"]
                above_ma = False
                if pd.notna(ma) and row["close"] > ma:
                    above_ma = True
                in_position = True
        else:
            # 1) 손절: 매수일 저가 이탈
            if row["low"] <= stop_loss:
                ret = (stop_loss - buy_price) / buy_price * 100
                trades.append({
                    "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
                    "sell_date": row["date"], "sell_price": stop_loss,
                    "return_pct": ret, "hold_days": (row["date"] - buy_date).days,
                    "reason": "손절",
                })
                in_position = False
                continue

            # 2) MA 이탈 매도
            if pd.notna(ma):
                if row["close"] > ma:
                    above_ma = True
                if above_ma and row["close"] < ma:
                    ret = (row["close"] - buy_price) / buy_price * 100
                    trades.append({
                        "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
                        "sell_date": row["date"], "sell_price": row["close"],
                        "return_pct": ret, "hold_days": (row["date"] - buy_date).days,
                        "reason": f"{ma_period}일선이탈",
                    })
                    in_position = False
                    continue

    # 미청산
    if in_position:
        last = df.iloc[-1]
        ret = (last["close"] - buy_price) / buy_price * 100
        trades.append({
            "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
            "sell_date": last["date"], "sell_price": last["close"],
            "return_pct": ret, "hold_days": (last["date"] - buy_date).days,
            "reason": "미청산",
        })

    return trades


def analyze(trades_df: pd.DataFrame) -> str:
    trades_df = trades_df.copy()
    trades_df["buy_year"] = pd.to_datetime(trades_df["buy_date"]).dt.year

    lines = []
    sections = [("전체 기간 종합", trades_df)]
    for y in sorted(trades_df["buy_year"].unique()):
        sections.append((f"{y}년", trades_df[trades_df["buy_year"] == y]))

    for label, sub in sections:
        if sub.empty:
            continue

        total = len(sub)
        wins = (sub["return_pct"] > 0).sum()
        losses = total - wins
        win_rate = wins / total * 100
        avg_ret = sub["return_pct"].mean()
        med_ret = sub["return_pct"].median()
        avg_win = sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
        avg_loss = sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if losses > 0 else 0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        avg_hold = sub["hold_days"].mean()

        lines.append(f"\n## {label}\n")
        lines.append("| 지표 | 값 |")
        lines.append("|---|---|")
        lines.append(f"| 총 거래수 | {total:,} |")
        lines.append(f"| 승리 / 패배 | {wins:,} / {losses:,} |")
        lines.append(f"| **승률** | **{win_rate:.1f}%** |")
        lines.append(f"| 평균 수익률 | {avg_ret:.2f}% |")
        lines.append(f"| 중간값 수익률 | {med_ret:.2f}% |")
        lines.append(f"| 평균 이익 (승리) | +{avg_win:.2f}% |")
        lines.append(f"| 평균 손실 (패배) | {avg_loss:.2f}% |")
        lines.append(f"| **손익비** | **{pf:.2f}** |")
        lines.append(f"| 평균 보유일수 | {avg_hold:.1f}일 |")

        # 수익률 분포
        lines.append(f"\n### 수익률 분포\n")
        bins = [-np.inf, -20, -10, -5, 0, 5, 10, 20, np.inf]
        labels_b = ["<-20%", "-20~-10%", "-10~-5%", "-5~0%", "0~5%", "5~10%", "10~20%", ">20%"]
        sub_c = sub.copy()
        sub_c["ret_bin"] = pd.cut(sub_c["return_pct"], bins=bins, labels=labels_b, right=False)
        dist = sub_c["ret_bin"].value_counts().reindex(labels_b, fill_value=0)

        lines.append("| 수익률 구간 | 건수 | 비율 |")
        lines.append("|---|---:|---:|")
        for lbl in labels_b:
            cnt = dist[lbl]
            lines.append(f"| {lbl} | {cnt:,} | {cnt/total*100:.1f}% |")

    return "\n".join(lines)


def run_one(stocks: dict, ma_period: int) -> pd.DataFrame:
    """MA 기간 1개로 전 종목 시뮬레이션"""
    all_trades = []
    for ticker, df in stocks.items():
        all_trades.extend(simulate_trades(df, ticker, ma_period))
    tdf = pd.DataFrame(all_trades)
    if tdf.empty:
        return tdf
    tdf["buy_year"] = pd.to_datetime(tdf["buy_date"]).dt.year
    return tdf[tdf["buy_year"] >= 2023]


def stats_row(sub: pd.DataFrame) -> dict:
    """통계 한 줄 계산"""
    total = len(sub)
    if total == 0:
        return {}
    wins = (sub["return_pct"] > 0).sum()
    losses = total - wins
    avg_w = sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
    avg_l = sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if losses > 0 else 0
    pf = abs(avg_w / avg_l) if avg_l != 0 else 0
    reasons = sub["reason"].value_counts()
    return {
        "total": total, "wins": wins,
        "win_rate": wins / total * 100,
        "avg_ret": sub["return_pct"].mean(),
        "med_ret": sub["return_pct"].median(),
        "avg_win": avg_w, "avg_loss": avg_l, "pf": pf,
        "hold": sub["hold_days"].mean(),
        "sl_pct": reasons.get("손절", 0) / total * 100,
    }


def main():
    print("데이터 로딩...")
    stocks = load_stocks()
    print(f"  {len(stocks):,}종목 로드")

    os.makedirs("results", exist_ok=True)
    all_results = {}

    for ma in [5, 10, 20]:
        print(f"\n{'='*60}")
        print(f"  MA{ma} 매도 테스트")
        print(f"{'='*60}")
        start = time.time()
        tdf = run_one(stocks, ma)
        elapsed = time.time() - start
        print(f"  {len(tdf):,}건, {elapsed:.1f}초")
        all_results[ma] = tdf

    # 래리식 단기고점 저가 이탈 매도
    print(f"\n{'='*60}")
    print(f"  고점저가이탈 매도 테스트")
    print(f"{'='*60}")
    start = time.time()
    swing_trades = []
    for ticker, df in stocks.items():
        swing_trades.extend(simulate_trades_swing_high(df, ticker))
    stdf = pd.DataFrame(swing_trades)
    if not stdf.empty:
        stdf["buy_year"] = pd.to_datetime(stdf["buy_date"]).dt.year
        stdf = stdf[stdf["buy_year"] >= 2023]
    elapsed = time.time() - start
    print(f"  {len(stdf):,}건, {elapsed:.1f}초")
    all_results["고점저가"] = stdf

    # 비교 테이블 출력
    print(f"\n{'='*80}")
    print("  MA 매도 기간별 비교 (전체 기간)")
    print(f"{'='*80}")

    lines = []
    methods = [5, 10, 20, "고점저가"]
    method_labels = ["MA5", "MA10", "MA20", "고점저가"]

    lines.append("\n## 매도 방법별 비교 (전체 기간 종합)\n")
    lines.append("| 지표 | " + " | ".join(method_labels) + " |")
    lines.append("|---|" + "|".join("---:" for _ in methods) + "|")

    stats = {m: stats_row(tdf) for m, tdf in all_results.items()}

    for label, key, fmt in [
        ("거래수", "total", "{:,}"),
        ("**승률**", "win_rate", "**{:.1f}%**"),
        ("평균수익률", "avg_ret", "{:+.2f}%"),
        ("중간값수익률", "med_ret", "{:+.2f}%"),
        ("평균이익", "avg_win", "+{:.2f}%"),
        ("평균손실", "avg_loss", "{:.2f}%"),
        ("**손익비**", "pf", "**{:.2f}**"),
        ("평균보유일", "hold", "{:.1f}일"),
        ("손절비율", "sl_pct", "{:.1f}%"),
    ]:
        vals = " | ".join(fmt.format(stats[m].get(key, 0)) for m in methods)
        lines.append(f"| {label} | {vals} |")
        print(f"  {label}: " + " / ".join(f"{ml}={fmt.format(stats[m].get(key,0))}" for m, ml in zip(methods, method_labels)))

    # 연도별 비교
    lines.append("\n## 연도별 비교\n")

    for metric_label, metric_key, fmt in [
        ("평균 수익률 (%)", "avg_ret", "{:+.2f}"),
        ("승률 (%)", "win_rate", "{:.1f}"),
        ("손익비", "pf", "{:.2f}"),
    ]:
        lines.append(f"\n### {metric_label}\n")
        lines.append("| 연도 | " + " | ".join(method_labels) + " |")
        lines.append("|:---:|" + "|".join("---:" for _ in methods) + "|")

        all_years = sorted(set(y for tdf in all_results.values() for y in tdf["buy_year"].unique()))
        for year in all_years:
            vals = []
            for m in methods:
                sub = all_results[m]
                sub_y = sub[sub["buy_year"] == year]
                s = stats_row(sub_y)
                vals.append(fmt.format(s.get(metric_key, 0)) if s else "-")
            lines.append(f"| {year} | {' | '.join(vals)} |")

    report = "\n".join(lines)
    print(report)

    # 각 MA별 상세 리포트도 포함
    full_report = report
    for ma in [5, 10, 20]:
        full_report += f"\n\n---\n\n# MA{ma} 상세\n"
        full_report += analyze(all_results[ma])

    with open("results/backtest_wr_cci.md", "w") as f:
        f.write("# WR(9)+CCI(14) 매수 + MA 매도 최적화 백테스트\n\n")
        f.write("## 매매 규칙\n\n")
        f.write("- **매수**: WR(9) crossup(-80) AND CCI(14) crossup(-100) 동시\n")
        f.write("- **매도**: N일선 올라탄 후 종가 이탈 (N=5,10,20 비교)\n")
        f.write("- **손절**: 매수일 저가 이탈\n\n")
        f.write("---\n")
        f.write(full_report)
        f.write("\n")

    print("\n결과 저장: results/backtest_wr_cci.md")


if __name__ == "__main__":
    main()
