"""
거래대금 20일 연속 1500억 이상 + 이평선 정배열 홀딩 백테스트
- 매수: 최근 20일 모두 거래대금 1500억 이상 AND 종가>MA20>MA60 → 당일 시가 매수
- 보유: 두 조건 모두 유지
- 매도: 거래대금 조건 이탈 OR 종가<MA20 → 당일 종가 매도
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

AMOUNT_THRESHOLD = 150_000  # 1500억원 = 150,000 백만원
LOOKBACK = 20
MIN_DAYS = 20  # 20일 모두 충족
FEE_BUY = 0.00015
FEE_SELL = 0.00015
TAX_SELL = 0.0018


def load_data() -> dict[str, pd.DataFrame]:
    query = text("""
        SELECT s.date, s.open, s.close, s.ticker, s.name, sa.amount
        FROM stocks s
        JOIN stock_all sa ON s.ticker = sa.ticker AND s.date = sa.date
        WHERE s.date >= '2022-06-01'
        ORDER BY s.ticker, s.date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    out = {}
    for t, g in df.groupby("ticker"):
        g = g.reset_index(drop=True)
        g["above"] = (g["amount"] >= AMOUNT_THRESHOLD).astype(int)
        g["above_count"] = g["above"].rolling(LOOKBACK, min_periods=LOOKBACK).sum()
        g["ma20"] = g["close"].rolling(20, min_periods=20).mean()
        g["ma60"] = g["close"].rolling(60, min_periods=60).mean()
        out[t] = g
    return out


def simulate(df: pd.DataFrame, ticker: str) -> list[dict]:
    n = len(df)
    if n < 60 + 1:
        return []

    above_count = df["above_count"].values
    close = df["close"].values
    ma20 = df["ma20"].values
    ma60 = df["ma60"].values
    trades = []
    in_position = False
    buy_price = 0.0
    buy_date = None
    entry_name = ""

    for i in range(60, n):
        row = df.iloc[i]
        cnt = above_count[i]
        c = close[i]
        m20 = ma20[i]
        m60 = ma60[i]

        if not in_position:
            # 매수: 거래대금 20일 모두 충족 AND 종가>MA20>MA60
            if (not np.isnan(cnt) and cnt >= MIN_DAYS
                    and row["open"] > 0
                    and not np.isnan(m20) and not np.isnan(m60)
                    and c > m20 > m60):
                buy_price = row["open"]
                buy_date = row["date"]
                entry_name = row["name"]
                in_position = True
        else:
            # 매도: 거래대금 이탈 OR 종가<MA20
            if (np.isnan(cnt) or cnt < MIN_DAYS
                    or np.isnan(m20) or c < m20):
                sell_price = row["close"]
                gross_ret = (sell_price - buy_price) / buy_price
                net_ret = gross_ret - FEE_BUY - FEE_SELL - TAX_SELL
                trades.append({
                    "ticker": ticker,
                    "name": entry_name,
                    "buy_date": buy_date,
                    "buy_price": buy_price,
                    "sell_date": row["date"],
                    "sell_price": sell_price,
                    "return_pct": net_ret * 100,
                    "hold_days": (row["date"] - buy_date).days,
                })
                in_position = False

    if in_position:
        last = df.iloc[-1]
        gross_ret = (last["close"] - buy_price) / buy_price
        net_ret = gross_ret - FEE_BUY - FEE_SELL - TAX_SELL
        trades.append({
            "ticker": ticker,
            "name": entry_name,
            "buy_date": buy_date,
            "buy_price": buy_price,
            "sell_date": last["date"],
            "sell_price": last["close"],
            "return_pct": net_ret * 100,
            "hold_days": (last["date"] - buy_date).days,
            "reason": "미청산",
        })

    return trades


def stats(sub: pd.DataFrame) -> dict:
    total = len(sub)
    if total == 0:
        return {}
    wins = (sub["return_pct"] > 0).sum()
    losses = total - wins
    win_rate = wins / total * 100
    avg_ret = sub["return_pct"].mean()
    med_ret = sub["return_pct"].median()
    avg_win = sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
    avg_loss = sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if losses > 0 else 0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    avg_hold = sub["hold_days"].mean()
    total_return = (1 + sub["return_pct"] / 100).prod()
    return {
        "total": total, "wins": wins, "losses": losses, "win_rate": win_rate,
        "avg_ret": avg_ret, "med_ret": med_ret, "avg_win": avg_win,
        "avg_loss": avg_loss, "pf": pf, "avg_hold": avg_hold,
        "total_return": total_return,
    }


def main():
    print("데이터 로딩...")
    t0 = time.time()
    stocks = load_data()
    print(f"  {len(stocks):,}종목 로드 ({time.time()-t0:.1f}초)")

    all_trades = []
    for ticker, df in stocks.items():
        all_trades.extend(simulate(df, ticker))

    if not all_trades:
        print("거래 없음!")
        return

    df = pd.DataFrame(all_trades)
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df["sell_date"] = pd.to_datetime(df["sell_date"])
    df["buy_year"] = df["buy_date"].dt.year
    df = df[(df["buy_year"] >= 2023) & np.isfinite(df["return_pct"])]

    s = stats(df)
    print(f"\n전체: {s['total']:,}건, 승률 {s['win_rate']:.1f}%, "
          f"평균수익 {s['avg_ret']:+.2f}%, 손익비 {s['pf']:.2f}, "
          f"누적 {s['total_return']:.2f}x")

    # 리포트 생성
    lines = ["# 거래대금 20일 연속 1500억 + 이평선 정배열 홀딩 백테스트\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **매수**: 최근 20일 모두 거래대금 1,500억원 이상 AND 종가 > MA20 > MA60 → 당일 시가 매수")
    lines.append("- **보유**: 두 조건 모두 유지")
    lines.append("- **매도**: 거래대금 조건 이탈 OR 종가 < MA20 → 당일 종가 매도")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% + 매도 {FEE_SELL*100:.3f}% + 세금 {TAX_SELL*100:.2f}%")
    lines.append("- **기간**: 2023 ~ 현재\n")
    lines.append("---")

    # 전체 통계
    lines.append("\n## 전체 통계\n")
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
    lines.append(f"| 누적수익률(배수) | {s['total_return']:.2f}x |")

    # 연도별 통계
    lines.append("\n## 연도별 통계\n")
    lines.append("| 연도 | 거래수 | 승률(%) | 평균수익(%) | 중간값(%) | 손익비 | 평균보유일 | 누적수익(x) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for y in sorted(df["buy_year"].unique()):
        sub = df[df["buy_year"] == y]
        ys = stats(sub)
        lines.append(
            f"| {y} | {ys['total']:,} | {ys['win_rate']:.1f} "
            f"| {ys['avg_ret']:+.2f} | {ys['med_ret']:+.2f} "
            f"| {ys['pf']:.2f} | {ys['avg_hold']:.1f} "
            f"| {ys['total_return']:.2f} |"
        )

    # 상위/하위 거래
    lines.append("\n## 수익률 상위 거래 (Top 20)\n")
    lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
    lines.append("|---|---|---|---:|---:|")
    for _, r in df.nlargest(20, "return_pct").iterrows():
        lines.append(
            f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
            f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
            f"| {r['return_pct']:+.1f} |"
        )

    lines.append("\n## 수익률 하위 거래 (Bottom 20)\n")
    lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
    lines.append("|---|---|---|---:|---:|")
    for _, r in df.nsmallest(20, "return_pct").iterrows():
        lines.append(
            f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
            f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
            f"| {r['return_pct']:+.1f} |"
        )

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_amount_hold_20_ma.md", "w") as f:
        f.write(report)
        f.write("\n")
    print("\n결과 저장: results/backtest_amount_hold_20_ma.md")


if __name__ == "__main__":
    main()
