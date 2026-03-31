"""
전일 가격변동폭 돌파 매매 백테스트
- 매수: 당일 시가 + 전일(고가-저가)/2 돌파 시 매수 (체결가 = 시가+range)
- 매도: 다음날 시가에 매도
- 대상: KOSPI200, 전 종목
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


def load_data():
    with ENGINE.connect() as conn:
        k200 = set(pd.read_sql(text("SELECT ticker FROM kospi200_members"), conn)["ticker"])
        df = pd.read_sql(text("""
            SELECT date, open, high, low, close, ticker
            FROM stocks WHERE date >= '2022-12-01'
            ORDER BY ticker, date
        """), conn)
    df["date"] = pd.to_datetime(df["date"])
    return df, k200


def simulate_group(stocks_dict):
    all_trades = []
    for ticker, df in stocks_dict.items():
        df = df.reset_index(drop=True)
        n = len(df)
        for i in range(1, n - 1):
            prev = df.iloc[i - 1]
            row = df.iloc[i]
            nxt = df.iloc[i + 1]

            range_val = (prev["high"] - prev["low"]) / 2
            trigger = row["open"] + range_val

            if trigger <= 0:
                continue

            # 당일 고가가 트리거 이상이면 매수
            if row["high"] >= trigger:
                buy_price = trigger
                sell_price = nxt["open"]
                ret = (sell_price - buy_price) / buy_price * 100
                all_trades.append({
                    "ticker": ticker,
                    "buy_date": row["date"],
                    "sell_date": nxt["date"],
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "return_pct": ret,
                })

    return pd.DataFrame(all_trades)


def stats(sub, label):
    total = len(sub)
    if total == 0:
        return
    wins = (sub["return_pct"] > 0).sum()
    wr = wins / total * 100
    avg = sub["return_pct"].mean()
    med = sub["return_pct"].median()
    avg_w = sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
    avg_l = sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if total - wins > 0 else 0
    pf = abs(avg_w / avg_l) if avg_l != 0 else 0
    print(f"{label:>12}: {total:>8,}건 | 승률={wr:5.1f}% | 평균={avg:+6.2f}% | 중간={med:+6.2f}% | 이익={avg_w:+6.2f}% | 손실={avg_l:+6.2f}% | 손익비={pf:.2f}")


def main():
    print("데이터 로딩...")
    df, k200 = load_data()
    stocks_all = {t: g for t, g in df.groupby("ticker")}
    stocks_k200 = {t: g for t, g in stocks_all.items() if t in k200}
    stocks_non = {t: g for t, g in stocks_all.items() if t not in k200}
    print(f"  KOSPI200: {len(stocks_k200)}종목, 비KOSPI200: {len(stocks_non)}종목")

    for label, stocks_dict in [("KOSPI200", stocks_k200), ("전종목", stocks_all)]:
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")
        start = time.time()
        tdf = simulate_group(stocks_dict)
        tdf["buy_year"] = pd.to_datetime(tdf["buy_date"]).dt.year
        tdf = tdf[(tdf["buy_year"] >= 2023) & (tdf["buy_price"] > 0) & np.isfinite(tdf["return_pct"])]
        print(f"  {len(tdf):,}건, {time.time()-start:.1f}초")

        stats(tdf, "전체")
        for y in sorted(tdf["buy_year"].unique()):
            stats(tdf[tdf["buy_year"] == y], str(y))


if __name__ == "__main__":
    main()
