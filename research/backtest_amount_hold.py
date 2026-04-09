"""
거래대금 20일 중 N일 1500억 이상 매매 백테스트 - 파라미터 최적화
- 매수: 최근 20일 중 거래대금 1500억 이상인 날이 N일 이상 → 당일 시가 매수
- 보유: 조건 유지
- 매도: 조건 이탈 (20일 중 1500억 이상 일수가 N일 미만) → 당일 종가 매도
- 파라미터 스윕: N = 10 ~ 20
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
        # 미리 20일 rolling count 계산 (1500억 이상인 날 수)
        g["above"] = (g["amount"] >= AMOUNT_THRESHOLD).astype(int)
        g["above_count"] = g["above"].rolling(LOOKBACK, min_periods=LOOKBACK).sum()
        out[t] = g
    return out


def simulate(df: pd.DataFrame, ticker: str, min_days: int) -> list[dict]:
    n = len(df)
    if n < LOOKBACK + 1:
        return []

    above_count = df["above_count"].values
    above = df["above"].values

    trades = []
    in_position = False
    buy_price = 0.0
    buy_date = None
    entry_name = ""

    for i in range(LOOKBACK, n):
        row = df.iloc[i]
        cnt = above_count[i]

        if not in_position:
            if not np.isnan(cnt) and cnt >= min_days and row["open"] > 0:
                buy_price = row["open"]
                buy_date = row["date"]
                entry_name = row["name"]
                in_position = True
        else:
            # 조건 이탈: 20일 중 1500억 이상 일수가 min_days 미만
            if np.isnan(cnt) or cnt < min_days:
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


def run_sweep(stocks: dict, min_days: int) -> pd.DataFrame:
    all_trades = []
    for ticker, df in stocks.items():
        all_trades.extend(simulate(df, ticker, min_days))
    if not all_trades:
        return pd.DataFrame()
    df = pd.DataFrame(all_trades)
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df["sell_date"] = pd.to_datetime(df["sell_date"])
    df["buy_year"] = df["buy_date"].dt.year
    df = df[(df["buy_year"] >= 2023) & np.isfinite(df["return_pct"])]
    return df


def top_trades_for_best(df: pd.DataFrame, n: int = 20) -> list[str]:
    if df.empty:
        return []
    lines = ["\n## 수익률 상위 거래 (Top 20)\n"]
    lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
    lines.append("|---|---|---|---:|---:|")
    top = df.nlargest(n, "return_pct")
    for _, r in top.iterrows():
        lines.append(
            f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
            f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
            f"| {r['return_pct']:+.1f} |"
        )
    return lines


def worst_trades_for_best(df: pd.DataFrame, n: int = 20) -> list[str]:
    if df.empty:
        return []
    lines = ["\n## 수익률 하위 거래 (Bottom 20)\n"]
    lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
    lines.append("|---|---|---|---:|---:|")
    bottom = df.nsmallest(n, "return_pct")
    for _, r in bottom.iterrows():
        lines.append(
            f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
            f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
            f"| {r['return_pct']:+.1f} |"
        )
    return lines


def main():
    print("데이터 로딩...")
    stocks = load_data()
    print(f"  {len(stocks):,}종목 로드")

    results = []
    yearly_results = {}  # min_days -> {year: stats}
    best_df = {}  # min_days -> DataFrame

    for min_days in range(10, 21):
        t0 = time.time()
        df = run_sweep(stocks, min_days)
        elapsed = time.time() - t0
        s = stats(df)
        if s:
            results.append({"min_days": min_days, **s})
            best_df[min_days] = df
            # 연도별 통계
            yearly = {}
            for y in sorted(df["buy_year"].unique()):
                sub = df[df["buy_year"] == y]
                yearly[y] = stats(sub)
            yearly_results[min_days] = yearly
        print(f"  min_days={min_days}: {s.get('total', 0):>4}건, "
              f"승률 {s.get('win_rate', 0):>5.1f}%, "
              f"평균수익 {s.get('avg_ret', 0):>+7.2f}%, "
              f"손익비 {s.get('pf', 0):>5.2f}, "
              f"누적 {s.get('total_return', 0):>8.2f}x | {elapsed:.1f}초")

    if not results:
        print("결과 없음!")
        return

    # 리포트 생성
    lines = ["# 거래대금 20일 중 N일 1500억 이상 - 파라미터 최적화\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **매수**: 최근 20일 중 거래대금 1,500억원 이상인 날이 N일 이상 → 당일 시가 매수")
    lines.append("- **보유**: 조건 유지")
    lines.append("- **매도**: 20일 중 1,500억 이상 일수가 N일 미만으로 하락 → 당일 종가 매도")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% + 매도 {FEE_SELL*100:.3f}% + 세금 {TAX_SELL*100:.2f}%")
    lines.append("- **기간**: 2023 ~ 현재")
    lines.append("- **파라미터 스윕**: N = 10 ~ 20\n")
    lines.append("---")

    # 종합 비교 테이블
    lines.append("\n## 파라미터 비교 (전체 기간)\n")
    lines.append("| N일 | 거래수 | 승률(%) | 평균수익(%) | 중간값(%) | 평균이익(%) | 평균손실(%) | 손익비 | 평균보유일 | 누적수익(x) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['min_days']} | {r['total']:,} | {r['win_rate']:.1f} "
            f"| {r['avg_ret']:+.2f} | {r['med_ret']:+.2f} "
            f"| {r['avg_win']:+.2f} | {r['avg_loss']:+.2f} "
            f"| {r['pf']:.2f} | {r['avg_hold']:.1f} "
            f"| {r['total_return']:.2f} |"
        )

    # 연도별 비교 테이블
    all_years = sorted(set(y for yd in yearly_results.values() for y in yd))
    for y in all_years:
        lines.append(f"\n## {y}년 비교\n")
        lines.append("| N일 | 거래수 | 승률(%) | 평균수익(%) | 손익비 | 누적수익(x) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for min_days in range(10, 21):
            yd = yearly_results.get(min_days, {})
            s = yd.get(y, {})
            if s:
                lines.append(
                    f"| {min_days} | {s['total']:,} | {s['win_rate']:.1f} "
                    f"| {s['avg_ret']:+.2f} | {s['pf']:.2f} "
                    f"| {s['total_return']:.2f} |"
                )
            else:
                lines.append(f"| {min_days} | 0 | - | - | - | - |")

    # 최적 파라미터 상세
    # 누적수익률 기준 best (미청산 제외 버전도 참고)
    best_by_cum = max(results, key=lambda x: x["total_return"])
    best_by_avg = max(results, key=lambda x: x["avg_ret"])
    best_by_med = max(results, key=lambda x: x["med_ret"])
    best_by_pf = max(results, key=lambda x: x["pf"] if x["pf"] != float("inf") else 0)

    lines.append("\n## 최적 파라미터 요약\n")
    lines.append(f"- **누적수익률 최고**: N = {best_by_cum['min_days']}일 → {best_by_cum['total_return']:.2f}x")
    lines.append(f"- **평균수익률 최고**: N = {best_by_avg['min_days']}일 → {best_by_avg['avg_ret']:+.2f}%")
    lines.append(f"- **중간값수익률 최고**: N = {best_by_med['min_days']}일 → {best_by_med['med_ret']:+.2f}%")
    lines.append(f"- **손익비 최고**: N = {best_by_pf['min_days']}일 → {best_by_pf['pf']:.2f}")

    # 최적 N의 상세 거래 내역
    best_n = best_by_med["min_days"]
    lines.append(f"\n---\n\n# 최적 파라미터 상세: N = {best_n}일 (중간값 수익률 기준)")
    bdf = best_df[best_n]
    s = stats(bdf)
    lines.append(f"\n| 지표 | 값 |")
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

    lines += top_trades_for_best(bdf)
    lines += worst_trades_for_best(bdf)

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_amount_hold.md", "w") as f:
        f.write(report)
        f.write("\n")
    print("\n결과 저장: results/backtest_amount_hold.md")


if __name__ == "__main__":
    main()
