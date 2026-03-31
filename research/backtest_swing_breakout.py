"""
스윙 돌파 매매 백테스트 v2
- 매수: 단기저점 확정 후, 다음봉 시가 + range 상향 돌파
  range = (저점캔들 고가 + 저가) / 2
  trigger = 다음봉 시가 + range
- 매도: 단기고점 캔들의 저가를 하향 돌파 (체결가 = 고점 캔들 저가)
- 손절: 매수일 저가 이탈

단기저점: day[i-1].low > day[i].low < day[i+1].low
단기고점: day[i-1].high < day[i].high > day[i+1].high
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
        SELECT date, open, high, low, close, ticker
        FROM stocks WHERE date >= '2022-12-01'
        ORDER BY ticker, date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return {t: g.reset_index(drop=True) for t, g in df.groupby("ticker")}


def simulate_trades(df: pd.DataFrame, ticker: str = "") -> list[dict]:
    n = len(df)
    if n < 5:
        return []

    low = df["low"]
    high = df["high"]
    is_swing_low = (low < low.shift(1)) & (low < low.shift(-1))
    is_swing_high = (high > high.shift(1)) & (high > high.shift(-1))

    trades = []
    in_position = False
    buy_price = 0.0
    buy_date = None
    stop_loss = 0.0

    # 최신 확정된 스윙포인트 추적
    last_swing_low_range = 0.0  # (저점캔들 고가+저가)/2
    last_swing_high_low = 0.0   # 매도 트리거: 고점 캔들의 저가
    has_buy_signal = False
    has_sell_signal = False
    buy_signal_day = -1         # 저점 확정일 (다음봉에서 매수 시도)

    i = 2
    while i < n:
        row = df.iloc[i]

        # i-1이 단기저점으로 확정 (i에서 확인)
        if is_swing_low.iloc[i - 1]:
            swing_candle = df.iloc[i - 1]
            last_swing_low_range = (swing_candle["high"] - swing_candle["low"]) / 2
            has_buy_signal = True
            buy_signal_day = i  # 확정일 = i, 매수 시도일 = i (다음봉 = i)

        # i-1이 단기고점으로 확정 (i에서 확인)
        if is_swing_high.iloc[i - 1]:
            last_swing_high_low = df.iloc[i - 1]["low"]
            if in_position:
                has_sell_signal = True

        if in_position:
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
                has_sell_signal = False
                i += 1
                continue

            # 2) 고점 저가 하향 돌파 매도
            if has_sell_signal and last_swing_high_low > 0 and row["low"] <= last_swing_high_low:
                sell_price = last_swing_high_low
                ret = (sell_price - buy_price) / buy_price * 100
                trades.append({
                    "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
                    "sell_date": row["date"], "sell_price": sell_price,
                    "return_pct": ret, "hold_days": (row["date"] - buy_date).days,
                    "reason": "고점저가이탈",
                })
                in_position = False
                has_sell_signal = False
                i += 1
                continue

            i += 1

        else:
            # 매수: 저점 확정일(i)에서 시가 + range 돌파
            if has_buy_signal and buy_signal_day == i and last_swing_low_range > 0:
                trigger = row["open"] + last_swing_low_range
                if row["high"] >= trigger and trigger > 0:
                    buy_price = trigger
                    buy_date = row["date"]
                    stop_loss = row["low"]  # 매수일 저가
                    has_buy_signal = False
                    has_sell_signal = False
                    last_swing_high_low = 0.0
                    in_position = True
                    i += 1
                    continue

            i += 1

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
        lines.append(f"| **손익비** | **{pf:.2f}** |")
        lines.append(f"| 평균 보유일수 | {avg_hold:.1f}일 |")
        for reason, cnt in reason_counts.items():
            lines.append(f"| 매도사유: {reason} | {cnt:,} ({cnt/total*100:.1f}%) |")

        bins = [-np.inf, -20, -10, -5, 0, 5, 10, 20, np.inf]
        labels_b = ["<-20%", "-20~-10%", "-10~-5%", "-5~0%", "0~5%", "5~10%", "10~20%", ">20%"]
        sub_c = sub.copy()
        sub_c["ret_bin"] = pd.cut(sub_c["return_pct"], bins=bins, labels=labels_b, right=False)
        dist = sub_c["ret_bin"].value_counts().reindex(labels_b, fill_value=0)

        lines.append(f"\n### 수익률 분포\n")
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
    with open("results/backtest_swing_breakout.md", "w") as f:
        f.write("# 스윙 돌파 매매 백테스트 v2\n\n")
        f.write("## 매매 규칙\n\n")
        f.write("- **매수**: 단기저점 확정 후, 다음봉 시가 + (저점캔들 고가-저가)/2 돌파 시 매수\n")
        f.write("- **매도**: 단기고점 캔들의 저가를 하향 돌파 (체결가 = 고점 캔들 저가)\n")
        f.write("- **손절**: 매수일 저가 이탈\n\n")
        f.write("---\n")
        f.write(report)
        f.write("\n")

    print("\n결과 저장: results/backtest_swing_breakout.md")

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
