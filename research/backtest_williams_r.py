"""
윌리엄스 %R 지표 백테스트
- WR = (highest(H,14) - C) / (highest(H,14) - lowest(L,14)) * (-100)
- 매수: WR이 -80을 상향 돌파 (crossup)
- 매도: WR이 -20을 하향 돌파 (crossdown)
- 대상: stocks 테이블 전 종목
- 연도별: 2023, 2024, 2025, 2026
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

PERIOD = 14
UL = -80  # 매수 라인
DL = -20  # 매도 라인


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


def calc_williams_r(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hh = df["high"].rolling(PERIOD).max()
    ll = df["low"].rolling(PERIOD).min()
    df["wr"] = (hh - df["close"]) / (hh - ll) * (-100)
    return df


def simulate_trades(df: pd.DataFrame, ticker: str = "") -> list[dict]:
    df = calc_williams_r(df)
    n = len(df)
    trades = []

    in_position = False
    buy_price = 0.0
    buy_date = None

    for i in range(1, n):
        wr_prev = df.at[i - 1, "wr"]
        wr_curr = df.at[i, "wr"]
        row = df.iloc[i]

        if pd.isna(wr_prev) or pd.isna(wr_curr):
            continue

        if not in_position:
            # crossup(wr, -80): 전일 < -80 이고 금일 >= -80
            if wr_prev < UL and wr_curr >= UL:
                buy_price = row["close"]
                buy_date = row["date"]
                in_position = True
        else:
            # crossdown(wr, -20): 전일 > -20 이고 금일 <= -20
            if wr_prev > DL and wr_curr <= DL:
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
                })
                in_position = False

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
        if len(df) < PERIOD + 5:
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
    trades_df = trades_df[trades_df["buy_year"] >= 2023]
    print(f"  2023~2026 거래: {len(trades_df):,}건")

    print("\n분석 중...")
    report = analyze(trades_df)
    print(report)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_williams_r.md", "w") as f:
        f.write("# 윌리엄스 %R 백테스트\n\n")
        f.write("## 공식\n")
        f.write("```\n")
        f.write("WR = (highest(H,14) - C) / (highest(H,14) - lowest(L,14)) × (-100)\n")
        f.write("```\n\n")
        f.write("## 매매 규칙\n")
        f.write("- **매수**: WR이 -80을 상향 돌파 (crossup)\n")
        f.write("- **매도**: WR이 -20을 하향 돌파 (crossdown)\n")
        f.write("- **체결가**: 크로스 발생일 종가\n\n")
        f.write("---\n")
        f.write(report)
        f.write("\n")

    print("\n결과 저장: results/backtest_williams_r.md")

    # 삼성전자 검증
    if "005930" in stocks:
        samsung = trades_df[trades_df["ticker"] == "005930"].tail(10)
        print("\n=== 삼성전자 매매 로그 (최근 10건) ===")
        for _, t in samsung.iterrows():
            print(
                f"  {t['buy_date'].strftime('%Y-%m-%d')} 매수@{t['buy_price']:,.0f} → "
                f"{t['sell_date'].strftime('%Y-%m-%d')} 매도@{t['sell_price']:,.0f} "
                f"({t['return_pct']:+.2f}%, {t['hold_days']}일)"
            )


if __name__ == "__main__":
    main()
