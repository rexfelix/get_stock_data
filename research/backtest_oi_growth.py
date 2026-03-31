"""
영업이익 증가율별 주가 수익률 백테스트 (v2 - financial_annual 사용)
- 데이터: financial_annual (2020~2025 실적)
- 전년 영업이익 증가율 → 다음 해 4월 매수 → N일 후 수익률
- 매수년도: 2021, 2022, 2023, 2024, 2025
- 구간: 적자전환, 적자지속, <-50%, -50~-20%, -20~0%, 0~20%, 20~50%, 50~100%, >100%, 흑자전환
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB = os.getenv("DB_NAME")
USER = os.getenv("DB_USER")
PW = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
ENGINE = create_engine(f"postgresql://{USER}:{PW}@{HOST}:{PORT}/{DB}")

N_DAYS = [5, 10, 20, 60, 120]


def load_data():
    """재무 + 주가 데이터 로드"""
    with ENGINE.connect() as conn:
        fin = pd.read_sql(text("""
            SELECT ticker, year, operating_income, revenue, net_income,
                   roe, operating_margin, gross_margin, eps
            FROM financial_annual
            WHERE is_estimate = false
            ORDER BY ticker, year
        """), conn)

        stocks = pd.read_sql(text("""
            SELECT date, close, ticker FROM stocks
            WHERE date >= '2021-01-01'
            ORDER BY ticker, date
        """), conn)

    stocks["date"] = pd.to_datetime(stocks["date"])
    return fin, stocks


def classify_oi_growth(prev_oi, curr_oi):
    """영업이익 증가율 구간 분류"""
    if prev_oi is None or curr_oi is None:
        return None
    if prev_oi <= 0 and curr_oi > 0:
        return "흑자전환"
    elif prev_oi > 0 and curr_oi <= 0:
        return "적자전환"
    elif prev_oi <= 0 and curr_oi <= 0:
        return "적자지속"
    else:
        growth = (curr_oi - prev_oi) / prev_oi * 100
        if growth < -50:
            return "<-50%"
        elif growth < -20:
            return "-50~-20%"
        elif growth < 0:
            return "-20~0%"
        elif growth < 20:
            return "0~20%"
        elif growth < 50:
            return "20~50%"
        elif growth < 100:
            return "50~100%"
        else:
            return ">100%"


def calc_oi_growth(fin: pd.DataFrame) -> pd.DataFrame:
    """연도별 영업이익 증가율 계산"""
    fin = fin.drop_duplicates(subset=["ticker", "year"], keep="first")
    pivot = fin.pivot(index="ticker", columns="year", values="operating_income")

    results = []
    year_pairs = [(y, y + 1) for y in range(2020, 2025)]  # 2020→21, ..., 2024→25

    for prev_y, curr_y in year_pairs:
        if prev_y not in pivot.columns or curr_y not in pivot.columns:
            continue
        df = pivot[[prev_y, curr_y]].dropna().copy()
        df.columns = ["prev_oi", "curr_oi"]
        df["oi_group"] = [classify_oi_growth(p, c) for p, c in zip(df["prev_oi"], df["curr_oi"])]
        df["growth"] = np.where(
            df["prev_oi"] > 0,
            (df["curr_oi"] - df["prev_oi"]) / df["prev_oi"] * 100,
            np.nan,
        )
        df["fin_year"] = curr_y
        df = df.reset_index()
        results.append(df[["ticker", "fin_year", "prev_oi", "curr_oi", "growth", "oi_group"]])

    return pd.concat(results, ignore_index=True)


def calc_returns(stocks: pd.DataFrame, oi_growth: pd.DataFrame) -> pd.DataFrame:
    """4월 첫 거래일 매수 → N일 후 수익률"""
    # 종목별 인덱스 구조로 변환 (속도 최적화)
    stocks_grouped = {ticker: grp.reset_index(drop=True) for ticker, grp in stocks.groupby("ticker")}

    all_results = []

    for _, row in oi_growth.iterrows():
        ticker = row["ticker"]
        buy_year = int(row["fin_year"]) + 1

        tdf = stocks_grouped.get(ticker)
        if tdf is None:
            continue

        # 해당 연도 4월 이후 데이터
        mask = (tdf["date"].dt.year == buy_year) & (tdf["date"].dt.month >= 4)
        apr_idx = tdf.index[mask]
        if len(apr_idx) == 0:
            continue

        buy_pos = apr_idx[0]
        buy_price = tdf.at[buy_pos, "close"]
        buy_date = tdf.at[buy_pos, "date"]

        result = {
            "ticker": ticker,
            "fin_year": row["fin_year"],
            "buy_year": buy_year,
            "buy_date": buy_date,
            "oi_group": row["oi_group"],
            "growth": row["growth"],
        }

        for n in N_DAYS:
            target_pos = buy_pos + n
            if target_pos < len(tdf):
                result[f"ret_{n}d"] = tdf.at[target_pos, "close"] / buy_price - 1
            else:
                result[f"ret_{n}d"] = None

        all_results.append(result)

    return pd.DataFrame(all_results)


GROUP_ORDER = ["적자전환", "적자지속", "<-50%", "-50~-20%", "-20~0%",
               "0~20%", "20~50%", "50~100%", ">100%", "흑자전환"]


def analyze(df: pd.DataFrame) -> str:
    """영업이익 증가율 구간별 수익률 분석"""
    lines = []

    sections = [("전체 기간 종합", df)]
    for y in sorted(df["buy_year"].unique()):
        sections.append((f"매수년도 {y}년", df[df["buy_year"] == y]))

    for label, sub in sections:
        if sub.empty:
            continue

        lines.append(f"\n## {label}\n")

        # 평균 수익률
        lines.append("### 평균 수익률 (%)\n")
        header = "| 영업이익 구간 | " + " | ".join(f"{n}일후" for n in N_DAYS) + " | 건수 |"
        sep = "|---|" + "|".join("---:" for _ in N_DAYS) + "|---:|"
        lines.append(header)
        lines.append(sep)

        for grp in GROUP_ORDER:
            g = sub[sub["oi_group"] == grp]
            cnt = len(g)
            if cnt == 0:
                continue
            vals = " | ".join(
                f"{g[f'ret_{n}d'].mean()*100:.2f}" if g[f"ret_{n}d"].notna().sum() > 0 else "-"
                for n in N_DAYS
            )
            bold = "**" if grp in ("0~20%", "20~50%") else ""
            lines.append(f"| {bold}{grp}{bold} | {vals} | {cnt} |")

        # 상승 확률
        lines.append("\n### 상승 확률 (%)\n")
        lines.append(header)
        lines.append(sep)

        for grp in GROUP_ORDER:
            g = sub[sub["oi_group"] == grp]
            cnt = len(g)
            if cnt == 0:
                continue
            parts = []
            for n in N_DAYS:
                valid = g[f"ret_{n}d"].dropna()
                if len(valid) == 0:
                    parts.append("-")
                else:
                    parts.append(f"{(valid > 0).mean()*100:.1f}")
            vals = " | ".join(parts)
            bold = "**" if grp in ("0~20%", "20~50%") else ""
            lines.append(f"| {bold}{grp}{bold} | {vals} | {cnt} |")

    return "\n".join(lines)


def main():
    print("데이터 로딩...")
    fin, stocks = load_data()
    print(f"  재무: {len(fin)}건 ({fin['ticker'].nunique()}종목), 주가: {len(stocks):,}건")

    print("영업이익 증가율 계산...")
    oi_growth = calc_oi_growth(fin)
    print(f"  {len(oi_growth)}건")
    print(oi_growth.groupby("fin_year")["oi_group"].value_counts().unstack(fill_value=0))

    print("\nN일 후 수익률 계산...")
    results = calc_returns(stocks, oi_growth)
    print(f"  {len(results)}건 (매수년도: {sorted(results['buy_year'].unique())})")

    print("\n분석 중...")
    report = analyze(results)
    print(report)

    # 저장
    os.makedirs("results", exist_ok=True)
    with open("results/backtest_oi_growth.md", "w") as f:
        f.write("# 영업이익 증가율별 주가 수익률 분석 (v2)\n\n")
        f.write("- **매수 시점**: 결산년도 다음 해 4월 첫 거래일\n")
        f.write("- **영업이익 증가율**: 전년 대비 YoY\n")
        f.write("- **매수년도**: 2021, 2022, 2023, 2024, 2025\n")
        f.write("- **대상**: 전 종목 (financial_annual 테이블)\n\n")
        f.write("---\n")
        f.write(report)
        f.write("\n")

    print("\n결과 저장: results/backtest_oi_growth.md")


if __name__ == "__main__":
    main()
