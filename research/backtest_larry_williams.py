"""
래리 윌리엄스 매매법 백테스트
- 단기저점/고점 식별 후 돌파 매수, 트레일링 스톱 매도
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


def load_stocks() -> dict[str, pd.DataFrame]:
    """전 종목 OHLCV 로드 (2022-12-01~ MA 여유분 포함)"""
    query = text("""
        SELECT date, open, high, low, close, volume, ticker
        FROM stocks
        WHERE date >= '2022-12-01'
        ORDER BY ticker, date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return {ticker: grp.reset_index(drop=True) for ticker, grp in df.groupby("ticker")}


def find_swing_points(df: pd.DataFrame) -> pd.DataFrame:
    """단기 고점/저점 식별"""
    df = df.copy()
    low = df["low"]
    high = df["high"]

    df["is_swing_low"] = (low < low.shift(1)) & (low < low.shift(-1))
    df["is_swing_high"] = (high > high.shift(1)) & (high > high.shift(-1))
    return df


def simulate_trades(df: pd.DataFrame, ticker: str = "") -> list[dict]:
    """종목 1개 매매 시뮬레이션"""
    df = find_swing_points(df)
    n = len(df)
    trades = []

    in_position = False
    buy_price = 0.0
    buy_date = None
    initial_stop = 0.0
    trailing_stop = 0.0

    i = 0
    while i < n:
        row = df.iloc[i]

        if in_position:
            # 1) 손절/트레일링 스톱 체크 (당일 저가가 스톱 이하)
            current_stop = max(initial_stop, trailing_stop)
            if row["low"] <= current_stop:
                sell_price = current_stop
                sell_date = row["date"]
                ret = (sell_price - buy_price) / buy_price * 100
                hold_days = (sell_date - buy_date).days
                reason = "손절" if trailing_stop <= initial_stop else "트레일링스톱"
                trades.append({
                    "ticker": ticker,
                    "buy_date": buy_date,
                    "buy_price": buy_price,
                    "sell_date": sell_date,
                    "sell_price": sell_price,
                    "return_pct": ret,
                    "hold_days": hold_days,
                    "reason": reason,
                })
                in_position = False
                i += 1
                continue

            # 2) 새로운 단기저점 확정 시 트레일링 스톱 갱신
            #    i-1이 단기저점이면 (i에서 확정)
            if i >= 1 and df.iloc[i - 1]["is_swing_low"]:
                new_stop = df.iloc[i - 1]["low"]
                if new_stop > trailing_stop:
                    trailing_stop = new_stop

            i += 1

        else:
            # 매수 시도: i-1이 단기저점이면 (i에서 확정), i에서 매수 시도
            if i >= 1 and df.iloc[i - 1]["is_swing_low"]:
                swing_low_row = df.iloc[i - 1]
                range_val = (swing_low_row["high"] - swing_low_row["low"]) * 0.5
                trigger = row["open"] + range_val

                if row["high"] >= trigger and trigger > 0:
                    buy_price = trigger
                    buy_date = row["date"]
                    initial_stop = swing_low_row["low"]
                    trailing_stop = 0.0
                    in_position = True

            i += 1

    # 미청산 포지션 종가로 청산
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
    """통계 분석 및 마크다운 표 생성"""
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
        win_rate = wins / total * 100 if total > 0 else 0

        avg_ret = sub["return_pct"].mean()
        med_ret = sub["return_pct"].median()
        avg_win = sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
        avg_loss = sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if losses > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        avg_hold = sub["hold_days"].mean()

        # 매도 사유별
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
        labels = ["<-20%", "-20~-10%", "-10~-5%", "-5~0%", "0~5%", "5~10%", "10~20%", ">20%"]
        sub_copy = sub.copy()
        sub_copy["ret_bin"] = pd.cut(sub_copy["return_pct"], bins=bins, labels=labels, right=False)
        dist = sub_copy["ret_bin"].value_counts().reindex(labels, fill_value=0)

        lines.append("| 수익률 구간 | 건수 | 비율 |")
        lines.append("|---|---:|---:|")
        for lbl in labels:
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
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(stocks)}] 거래: {len(all_trades):,}건 | {rate:.0f}종목/초")

    elapsed = time.time() - start
    print(f"  완료: {len(all_trades):,}건, {elapsed:.1f}초")

    if not all_trades:
        print("거래 없음")
        return

    trades_df = pd.DataFrame(all_trades)

    # 2023~2026만 분석 (2022년 말 데이터는 MA 계산용)
    trades_df["buy_year"] = pd.to_datetime(trades_df["buy_date"]).dt.year
    trades_df = trades_df[trades_df["buy_year"] >= 2023]
    print(f"  2023~2026 거래: {len(trades_df):,}건")

    print("\n분석 중...")
    report = analyze(trades_df)
    print(report)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_larry_williams.md", "w") as f:
        f.write("# 래리 윌리엄스 매매법 백테스트\n\n")
        f.write("## 매매 규칙\n")
        f.write("- **단기저점**: 앞뒤날 저가보다 가운데날 저가가 더 낮은 날\n")
        f.write("- **매수**: 단기저점 다음날, 시가 + range(단기저점일 가격폭의 50%) 돌파 시\n")
        f.write("- **손절**: 기준 단기저점 이탈\n")
        f.write("- **매도**: 매수 후 새 단기저점의 저가 이탈 (트레일링 스톱)\n\n")
        f.write("---\n")
        f.write(report)
        f.write("\n")

    print("\n결과 저장: results/backtest_larry_williams.md")

    # 삼성전자 매매 로그 (검증용)
    if "005930" in stocks:
        print("\n=== 삼성전자 매매 로그 (최근 10건) ===")
        samsung = trades_df[trades_df["ticker"] == "005930"].tail(10)
        for _, t in samsung.iterrows():
            print(
                f"  {t['buy_date'].strftime('%Y-%m-%d')} 매수@{t['buy_price']:.0f} → "
                f"{t['sell_date'].strftime('%Y-%m-%d')} 매도@{t['sell_price']:.0f} "
                f"({t['return_pct']:+.2f}%, {t['hold_days']}일, {t['reason']})"
            )


if __name__ == "__main__":
    main()
