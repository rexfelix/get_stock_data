"""
스윙 돌파 매매 포트폴리오 시뮬레이션
- 초기 자본: 1,000만원
- KOSPI200 종목 대상
- 매일 신규 매수 종목수로 현금을 균등 배분
- 실전 규칙: 돌파 매수 → 당일 음봉 종가 청산 / 고점저가 이탈 매도 / 매수일 저가 손절
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

ENGINE = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

INITIAL_CAPITAL = 10_000_000
COMMISSION = 0.00015  # 매수 수수료 0.015%
TAX = 0.0018  # 매도 세금 0.18% (2024~ 코스피)
SELL_COMMISSION = 0.00015


def load_data():
    with ENGINE.connect() as conn:
        k200 = set(pd.read_sql(text("SELECT ticker FROM kospi200_members"), conn)["ticker"])
        df = pd.read_sql(text("""
            SELECT date, open, high, low, close, ticker
            FROM stocks WHERE date >= '2022-12-01'
            ORDER BY date, ticker
        """), conn)
    df["date"] = pd.to_datetime(df["date"])
    return df, k200


def precompute_signals(df: pd.DataFrame, k200: set) -> dict:
    """종목별 스윙포인트 사전 계산"""
    signals = {}
    for ticker, gdf in df.groupby("ticker"):
        if ticker not in k200:
            continue
        gdf = gdf.reset_index(drop=True)
        low = gdf["low"]
        high = gdf["high"]
        gdf["is_swing_low"] = (low < low.shift(1)) & (low < low.shift(-1))
        gdf["is_swing_high"] = (high > high.shift(1)) & (high > high.shift(-1))
        signals[ticker] = gdf
    return signals


def simulate():
    print("데이터 로딩...")
    df, k200 = load_data()
    print(f"  종목수: {len(k200)}, 전체행: {len(df):,}")

    print("스윙포인트 계산...")
    signals = precompute_signals(df, k200)
    print(f"  {len(signals)}종목 준비")

    # 날짜별 데이터 인덱싱
    dates = sorted(df["date"].unique())
    dates = [d for d in dates if d >= pd.Timestamp("2023-01-01")]

    # 종목별 인덱스 매핑 (날짜 → 행 인덱스)
    ticker_date_idx = {}
    for ticker, gdf in signals.items():
        date_to_idx = dict(zip(gdf["date"], gdf.index))
        ticker_date_idx[ticker] = date_to_idx

    # 포트폴리오 상태
    cash = INITIAL_CAPITAL
    positions = {}  # ticker -> {shares, buy_price, buy_date, stop_loss, first_day, has_sell, last_sh_low}
    daily_equity = []
    trade_log = []

    # 종목별 마지막 스윙 저점/고점 추적
    last_swing_low = {}   # ticker -> {high, low}
    last_swing_high = {}  # ticker -> {low}
    buy_signals = {}      # ticker -> True (매수 대기)

    for di, today in enumerate(dates):
        # 1) 스윙포인트 업데이트 + 매수/매도 시그널 수집
        new_buy_candidates = []  # (ticker, trigger_price, stop_price)
        sell_actions = []  # (ticker, reason, price)

        for ticker, gdf in signals.items():
            idx_map = ticker_date_idx[ticker]
            if today not in idx_map:
                continue
            idx = idx_map[today]
            if idx < 2:
                continue

            row = gdf.iloc[idx]
            prev = gdf.iloc[idx - 1]

            # 단기저점 확정 (어제가 저점)
            if prev["is_swing_low"]:
                last_swing_low[ticker] = {"high": prev["high"], "low": prev["low"]}
                buy_signals[ticker] = True

            # 단기고점 확정 (어제가 고점)
            if prev["is_swing_high"]:
                last_swing_high[ticker] = {"low": prev["low"]}

            # 보유 중인 종목 매도 체크
            if ticker in positions:
                pos = positions[ticker]

                # 첫날: 음봉 청산
                if pos["first_day"]:
                    pos["first_day"] = False
                    if row["close"] < row["open"]:
                        sell_actions.append((ticker, "당일음봉", row["close"]))
                        continue

                # 손절
                if row["low"] <= pos["stop_loss"]:
                    sell_actions.append((ticker, "손절", pos["stop_loss"]))
                    continue

                # 고점저가 이탈
                if ticker in last_swing_high:
                    sh_low = last_swing_high[ticker]["low"]
                    if row["low"] <= sh_low:
                        sell_actions.append((ticker, "고점저가", sh_low))
                        continue

            # 매수 후보
            elif ticker in buy_signals and buy_signals[ticker]:
                if ticker in last_swing_low:
                    sl = last_swing_low[ticker]
                    trigger = sl["high"]
                    if row["high"] >= trigger and trigger > 0:
                        new_buy_candidates.append((ticker, trigger, sl["low"], row))

        # 2) 매도 실행
        for ticker, reason, price in sell_actions:
            if ticker not in positions:
                continue
            pos = positions[ticker]
            proceeds = pos["shares"] * price
            cost = proceeds * (TAX + SELL_COMMISSION)
            cash += proceeds - cost

            ret = (price - pos["buy_price"]) / pos["buy_price"] * 100
            trade_log.append({
                "buy_date": pos["buy_date"], "sell_date": today,
                "ticker": ticker, "buy_price": pos["buy_price"],
                "sell_price": price, "shares": pos["shares"],
                "return_pct": ret, "reason": reason,
                "profit": proceeds - cost - pos["invested"],
            })
            del positions[ticker]

        # 3) 매수 실행 (현금을 신규 종목수로 균등 배분)
        # 이미 보유 중인 종목은 제외
        buy_candidates = [(t, p, s, r) for t, p, s, r in new_buy_candidates if t not in positions]

        if buy_candidates and cash > 0:
            per_stock = cash / len(buy_candidates)

            for ticker, trigger, stop, row in buy_candidates:
                if per_stock < trigger:
                    continue
                shares = int(per_stock / trigger)
                if shares <= 0:
                    continue

                cost = shares * trigger
                commission = cost * COMMISSION
                total_cost = cost + commission

                if total_cost > cash:
                    shares = int((cash / (1 + COMMISSION)) / trigger)
                    if shares <= 0:
                        continue
                    cost = shares * trigger
                    commission = cost * COMMISSION
                    total_cost = cost + commission

                cash -= total_cost
                positions[ticker] = {
                    "shares": shares,
                    "buy_price": trigger,
                    "buy_date": today,
                    "stop_loss": stop,
                    "first_day": True,
                    "invested": total_cost,
                }
                buy_signals[ticker] = False

        # 4) 일일 평가
        stock_value = 0
        for ticker, pos in positions.items():
            idx_map = ticker_date_idx[ticker]
            if today in idx_map:
                idx = idx_map[today]
                close = signals[ticker].iloc[idx]["close"]
                stock_value += pos["shares"] * close
            else:
                stock_value += pos["shares"] * pos["buy_price"]

        total_equity = cash + stock_value
        daily_equity.append({
            "date": today,
            "cash": cash,
            "stock_value": stock_value,
            "total_equity": total_equity,
            "positions": len(positions),
            "return_pct": (total_equity / INITIAL_CAPITAL - 1) * 100,
        })

    # 미청산 포지션 마지막날 종가로 청산
    last_date = dates[-1]
    for ticker, pos in list(positions.items()):
        idx_map = ticker_date_idx[ticker]
        if last_date in idx_map:
            idx = idx_map[last_date]
            price = signals[ticker].iloc[idx]["close"]
        else:
            price = pos["buy_price"]
        proceeds = pos["shares"] * price
        cost = proceeds * (TAX + SELL_COMMISSION)
        cash += proceeds - cost
        ret = (price - pos["buy_price"]) / pos["buy_price"] * 100
        trade_log.append({
            "buy_date": pos["buy_date"], "sell_date": last_date,
            "ticker": ticker, "buy_price": pos["buy_price"],
            "sell_price": price, "shares": pos["shares"],
            "return_pct": ret, "reason": "미청산",
            "profit": proceeds - cost - pos["invested"],
        })

    return pd.DataFrame(daily_equity), pd.DataFrame(trade_log)


def report(equity_df, trades_df):
    lines = []

    # 최종 성과
    final = equity_df.iloc[-1]
    total_ret = final["return_pct"]
    years = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days / 365.25
    cagr = ((1 + total_ret / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

    # MDD
    equity_df["peak"] = equity_df["total_equity"].cummax()
    equity_df["drawdown"] = (equity_df["total_equity"] / equity_df["peak"] - 1) * 100
    mdd = equity_df["drawdown"].min()

    lines.append("## 최종 성과\n")
    lines.append("| 지표 | 값 |")
    lines.append("|---|---|")
    lines.append(f"| 초기 자본 | {INITIAL_CAPITAL:,.0f}원 |")
    lines.append(f"| 최종 자산 | {final['total_equity']:,.0f}원 |")
    lines.append(f"| **총 수익률** | **{total_ret:+.2f}%** |")
    lines.append(f"| **CAGR** | **{cagr:.2f}%** |")
    lines.append(f"| **MDD** | **{mdd:.2f}%** |")
    lines.append(f"| 운용 기간 | {years:.1f}년 |")
    lines.append(f"| 총 거래수 | {len(trades_df):,} |")
    lines.append(f"| 총 수익금 | {final['total_equity'] - INITIAL_CAPITAL:+,.0f}원 |")

    # 연도별
    equity_df["year"] = equity_df["date"].dt.year
    lines.append("\n## 연도별 성과\n")
    lines.append("| 연도 | 시작자산 | 종료자산 | 수익률 | MDD | 거래수 |")
    lines.append("|:---:|---:|---:|---:|---:|---:|")

    for year in sorted(equity_df["year"].unique()):
        ydf = equity_df[equity_df["year"] == year]
        start_eq = ydf.iloc[0]["total_equity"]
        end_eq = ydf.iloc[-1]["total_equity"]
        yr_ret = (end_eq / start_eq - 1) * 100
        yr_peak = ydf["total_equity"].cummax()
        yr_dd = (ydf["total_equity"] / yr_peak - 1).min() * 100
        yr_trades = len(trades_df[pd.to_datetime(trades_df["buy_date"]).dt.year == year])
        lines.append(f"| {year} | {start_eq:,.0f} | {end_eq:,.0f} | {yr_ret:+.2f}% | {yr_dd:.2f}% | {yr_trades:,} |")

    # 거래 통계
    lines.append("\n## 거래 통계\n")
    total = len(trades_df)
    wins = (trades_df["return_pct"] > 0).sum()
    lines.append("| 지표 | 값 |")
    lines.append("|---|---|")
    lines.append(f"| 총 거래수 | {total:,} |")
    lines.append(f"| 승률 | {wins/total*100:.1f}% |")
    lines.append(f"| 평균 수익률 | {trades_df['return_pct'].mean():+.2f}% |")
    lines.append(f"| 평균 수익금 | {trades_df['profit'].mean():+,.0f}원 |")
    lines.append(f"| 총 수익금 | {trades_df['profit'].sum():+,.0f}원 |")

    reason_counts = trades_df["reason"].value_counts()
    for reason, cnt in reason_counts.items():
        sub = trades_df[trades_df["reason"] == reason]
        lines.append(f"| {reason} | {cnt}건 ({cnt/total*100:.1f}%), 평균 {sub['return_pct'].mean():+.2f}% |")

    # 월별 수익률
    equity_df["ym"] = equity_df["date"].dt.to_period("M")
    monthly = equity_df.groupby("ym").agg(
        start=("total_equity", "first"),
        end=("total_equity", "last"),
    )
    monthly["ret"] = (monthly["end"] / monthly["start"] - 1) * 100

    lines.append("\n## 월별 수익률 (%)\n")
    lines.append("| 월 | 수익률 | 자산 |")
    lines.append("|---|---:|---:|")
    for ym, row in monthly.iterrows():
        lines.append(f"| {ym} | {row['ret']:+.2f}% | {row['end']:,.0f} |")

    return "\n".join(lines)


def main():
    equity_df, trades_df = simulate()

    print(f"\n거래: {len(trades_df):,}건")
    print(f"최종 자산: {equity_df.iloc[-1]['total_equity']:,.0f}원")

    result = report(equity_df, trades_df)
    print(result)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_portfolio_sim.md", "w") as f:
        f.write("# 스윙 돌파 매매 포트폴리오 시뮬레이션\n\n")
        f.write("- 초기 자본: 1,000만원\n")
        f.write("- 대상: KOSPI200 종목\n")
        f.write("- 수수료: 매수 0.015%, 매도 0.015% + 세금 0.18%\n")
        f.write("- 매수: 저점고가 돌파, 당일 음봉 종가 청산\n")
        f.write("- 매도: 고점저가 이탈 / 매수일 저가 손절\n")
        f.write("- 자금배분: 당일 신규 매수 종목수로 현금 균등 배분\n\n")
        f.write("---\n\n")
        f.write(result)
        f.write("\n")

    print("\n결과 저장: results/backtest_portfolio_sim.md")


if __name__ == "__main__":
    main()
