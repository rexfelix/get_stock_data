"""
하락장악형 캔들 역추세 매매 백테스트 v3
- 하락장악형: 전일 고가 < 당일 고가 AND 전일 저가 > 당일 저가 AND 음봉
- 매수: 하락장악형 다음날 시가 < 장악형 저가이고, 시가+range 돌파 시 매수
  (range = 장악형 캔들 가격폭 × 50%)
- 손절: 매수일(기준봉) 저가 이탈
- 매도: 5일선 위로 올라탄 후, 종가가 5일선 아래로 이탈하면 종가 매도
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


def load_stocks() -> dict[str, pd.DataFrame]:
    query = text("""
        SELECT date, open, high, low, close, volume, ticker
        FROM stocks WHERE date >= '2022-12-01'
        ORDER BY ticker, date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return {t: g.reset_index(drop=True) for t, g in df.groupby("ticker")}


def simulate_trades(df: pd.DataFrame, ticker: str = "") -> list[dict]:
    n = len(df)
    if n < 10:
        return []

    # 5일 이동평균
    df = df.copy()
    df["ma5"] = df["close"].rolling(5).mean()

    trades = []
    in_position = False
    buy_price = 0.0
    buy_date = None
    stop_loss = 0.0
    above_ma5 = False  # 5일선 위에 올라탄 적 있는지

    i = 2
    while i < n:
        row = df.iloc[i]

        if in_position:
            ma5 = row["ma5"]

            # 1) 손절: 매수일(기준봉) 저가 이탈
            if row["low"] <= stop_loss:
                sell_price = stop_loss
                sell_date = row["date"]
                ret = (sell_price - buy_price) / buy_price * 100
                hold_days = (sell_date - buy_date).days
                trades.append({
                    "ticker": ticker,
                    "buy_date": buy_date,
                    "buy_price": buy_price,
                    "sell_date": sell_date,
                    "sell_price": sell_price,
                    "return_pct": ret,
                    "hold_days": hold_days,
                    "reason": "손절",
                })
                in_position = False
                i += 1
                continue

            # 2) 5일선 체크
            if pd.notna(ma5):
                if row["close"] > ma5:
                    above_ma5 = True

                # 5일선 올라탄 후 종가가 5일선 아래로 이탈 → 종가 매도
                if above_ma5 and row["close"] < ma5:
                    sell_price = row["close"]
                    sell_date = row["date"]
                    ret = (sell_price - buy_price) / buy_price * 100
                    hold_days = (sell_date - buy_date).days
                    trades.append({
                        "ticker": ticker,
                        "buy_date": buy_date,
                        "buy_price": buy_price,
                        "sell_date": sell_date,
                        "sell_price": sell_price,
                        "return_pct": ret,
                        "hold_days": hold_days,
                        "reason": "5일선이탈",
                    })
                    in_position = False
                    i += 1
                    continue

            i += 1

        else:
            # 하락장악형 캔들 확인
            prev = df.iloc[i - 1]
            before = df.iloc[i - 2]

            is_engulfing = (
                prev["high"] > before["high"] and
                prev["low"] < before["low"] and
                prev["close"] < prev["open"]
            )

            if is_engulfing:
                engulf_low = prev["low"]
                engulf_range = (prev["high"] - prev["low"]) * 0.5
                trigger = row["open"] + engulf_range

                # 시가가 장악형 저가보다 낮고, 시가+range 돌파 시 매수
                if row["open"] < engulf_low and row["high"] >= trigger and trigger > 0:
                    buy_price = trigger
                    buy_date = row["date"]
                    stop_loss = row["low"]  # 매수일(기준봉) 저가
                    above_ma5 = False
                    in_position = True
                    if pd.notna(row["ma5"]) and row["close"] > row["ma5"]:
                        above_ma5 = True
                    i += 1
                    continue

            i += 1

    # 미청산
    if in_position:
        last = df.iloc[-1]
        ret = (last["close"] - buy_price) / buy_price * 100
        hold_days = (last["date"] - buy_date).days
        trades.append({
            "ticker": ticker,
            "buy_date": buy_date,
            "buy_price": buy_price,
            "sell_date": last["date"],
            "sell_price": last["close"],
            "return_pct": ret,
            "hold_days": hold_days,
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
        losses = (sub["return_pct"] <= 0).sum()
        win_rate = wins / total * 100

        avg_ret = sub["return_pct"].mean()
        med_ret = sub["return_pct"].median()
        avg_win = sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
        avg_loss = sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if losses > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        avg_hold = sub["hold_days"].mean()

        reason_counts = sub["reason"].value_counts()

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
        lines.append(f"| **손익비** | **{profit_factor:.2f}** |")
        lines.append(f"| 평균 보유일수 | {avg_hold:.1f}일 |")

        for reason, cnt in reason_counts.items():
            lines.append(f"| 매도사유: {reason} | {cnt:,} ({cnt/total*100:.1f}%) |")

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


def main():
    print("데이터 로딩...")
    stocks = load_stocks()
    print(f"  {len(stocks):,}종목 로드")

    print("매매 시뮬레이션...")
    all_trades = []
    start = time.time()

    for i, (ticker, df) in enumerate(stocks.items()):
        if len(df) < 10:
            continue
        trades = simulate_trades(df, ticker)
        all_trades.extend(trades)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(stocks)}] 거래: {len(all_trades):,}건 | {(i+1)/elapsed:.0f}종목/초")

    elapsed = time.time() - start
    print(f"  완료: {len(all_trades):,}건, {elapsed:.1f}초")

    trades_df = pd.DataFrame(all_trades)
    trades_df["buy_year"] = pd.to_datetime(trades_df["buy_date"]).dt.year
    trades_df = trades_df[
        (trades_df["buy_year"] >= 2023)
        & (trades_df["buy_price"] > 0)
        & np.isfinite(trades_df["return_pct"])
    ]
    print(f"  2023~2026 유효 거래: {len(trades_df):,}건")

    print("\n분석 중...")
    report = analyze(trades_df)
    print(report)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_bearish_engulfing.md", "w") as f:
        f.write("# 하락장악형 캔들 역추세 매매 백테스트 v3\n\n")
        f.write("## 매매 규칙\n")
        f.write("- **하락장악형**: 전일 고가 < 당일 고가 AND 전일 저가 > 당일 저가 AND 음봉\n")
        f.write("- **매수**: 하락장악형 다음날 시가 < 장악형 저가이고, 시가+range 돌파 시\n")
        f.write("  - range = 장악형 캔들 가격폭 × 50%\n")
        f.write("- **손절**: 매수일(기준봉) 저가 이탈\n")
        f.write("- **매도**: 5일선 위로 올라탄 후, 종가가 5일선 아래로 이탈하면 종가 매도\n\n")
        f.write("---\n")
        f.write(report)
        f.write("\n")

    print("\n결과 저장: results/backtest_bearish_engulfing.md")

    # 삼성전자 검증
    if "005930" in stocks:
        samsung = trades_df[trades_df["ticker"] == "005930"].tail(10)
        print("\n=== 삼성전자 매매 로그 (최근 10건) ===")
        for _, t in samsung.iterrows():
            print(
                f"  {t['buy_date'].strftime('%Y-%m-%d')} 매수@{t['buy_price']:,.0f} → "
                f"{t['sell_date'].strftime('%Y-%m-%d')} 매도@{t['sell_price']:,.0f} "
                f"({t['return_pct']:+.2f}%, {t['hold_days']}일, {t['reason']})"
            )


if __name__ == "__main__":
    main()
