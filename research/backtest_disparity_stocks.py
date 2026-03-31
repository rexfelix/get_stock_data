"""
개별종목 5일선 이격도에 따른 N일 후 평균 수익률 & 상승 확률 분석
- 대상: stocks 테이블 전 종목
- 연도별: 2023, 2024, 2025, 2026
- 이격도 구간: <97, 97~98, 98~99, 99~100, 100~101, 101~102, 102~103, >=103
- N일: 1, 2, 3, 4, 5, 10, 20
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

# ── 설정 ──
YEARS = [2023, 2024, 2025, 2026]
N_DAYS = [1, 2, 3, 4, 5, 10, 20]
DISPARITY_BINS = [-np.inf, 97, 98, 99, 100, 101, 102, 103, np.inf]
DISPARITY_LABELS = ["<97", "97~98", "98~99", "99~100", "100~101", "101~102", "102~103", ">=103"]


def load_all_stocks() -> pd.DataFrame:
    """stocks 테이블에서 2022년부터 로드 (MA5 계산을 위해 여유분 포함)"""
    query = text("""
        SELECT date, close, ticker FROM stocks
        WHERE date >= '2022-01-01'
        ORDER BY ticker, date
    """)
    print("DB에서 데이터 로딩 중...")
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  총 {len(df):,}행, {df['ticker'].nunique():,}종목 로드")
    return df


def calc_disparity_and_returns(df: pd.DataFrame) -> pd.DataFrame:
    """종목별 5일선 이격도 및 N일 후 수익률 계산"""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 종목별 MA5 계산
    df["ma5"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).mean())
    df["disparity"] = (df["close"] / df["ma5"]) * 100

    # 종목별 N일 후 수익률 계산
    for n in N_DAYS:
        df[f"ret_{n}d"] = df.groupby("ticker")["close"].transform(lambda x: x.shift(-n) / x - 1)

    df["year"] = df["date"].dt.year
    # 2023년 이후만 분석 대상
    df = df[df["year"] >= 2023].copy()
    df["disp_bin"] = pd.cut(df["disparity"], bins=DISPARITY_BINS, labels=DISPARITY_LABELS, right=False)
    return df.dropna(subset=["disparity"])


def analyze(df: pd.DataFrame, title: str) -> str:
    """연도별 × 이격도구간별 × N일후 수익률/상승확률 분석"""
    lines = []
    lines.append(f"\n## {title}")

    for year in YEARS:
        ydf = df[df["year"] == year]
        if ydf.empty:
            lines.append(f"\n### {year}년: 데이터 없음")
            continue

        n_tickers = ydf["ticker"].nunique()
        lines.append(f"\n### {year}년 (관측수: {len(ydf):,}건, {n_tickers:,}종목)")

        # 평균 수익률 테이블
        lines.append(f"\n#### 평균 수익률 (%)\n")
        lines.append(f"| 이격도 | " + " | ".join(f"{n}일후" for n in N_DAYS) + " | 건수 |")
        lines.append(f"|--------|" + "|".join("------:" for _ in N_DAYS) + "|-----:|")

        for label in DISPARITY_LABELS:
            bdf = ydf[ydf["disp_bin"] == label]
            cnt = len(bdf)
            if cnt == 0:
                vals = " | ".join("-" for _ in N_DAYS)
            else:
                vals = " | ".join(
                    f"{bdf[f'ret_{n}d'].mean()*100:.2f}" if bdf[f"ret_{n}d"].notna().sum() > 0 else "-"
                    for n in N_DAYS
                )
            lines.append(f"| {label} | {vals} | {cnt:,} |")

        # 상승 확률 테이블
        lines.append(f"\n#### 상승 확률 (%)\n")
        lines.append(f"| 이격도 | " + " | ".join(f"{n}일후" for n in N_DAYS) + " | 건수 |")
        lines.append(f"|--------|" + "|".join("------:" for _ in N_DAYS) + "|-----:|")

        for label in DISPARITY_LABELS:
            bdf = ydf[ydf["disp_bin"] == label]
            cnt = len(bdf)
            if cnt == 0:
                vals = " | ".join("-" for _ in N_DAYS)
            else:
                parts = []
                for n in N_DAYS:
                    valid = bdf[f"ret_{n}d"].dropna()
                    if len(valid) == 0:
                        parts.append("-")
                    else:
                        parts.append(f"{(valid > 0).mean() * 100:.1f}")
                vals = " | ".join(parts)
            lines.append(f"| {label} | {vals} | {cnt:,} |")

    # 전체 기간 종합
    adf = df[df["year"].isin(YEARS)]
    n_tickers = adf["ticker"].nunique()
    lines.append(f"\n### 전체 기간 종합 (2023~2026, {n_tickers:,}종목)")

    lines.append(f"\n#### 평균 수익률 (%)\n")
    lines.append(f"| 이격도 | " + " | ".join(f"{n}일후" for n in N_DAYS) + " | 건수 |")
    lines.append(f"|--------|" + "|".join("------:" for _ in N_DAYS) + "|-----:|")

    for label in DISPARITY_LABELS:
        bdf = adf[adf["disp_bin"] == label]
        cnt = len(bdf)
        if cnt == 0:
            vals = " | ".join("-" for _ in N_DAYS)
        else:
            vals = " | ".join(
                f"{bdf[f'ret_{n}d'].mean()*100:.2f}" if bdf[f"ret_{n}d"].notna().sum() > 0 else "-"
                for n in N_DAYS
            )
        bold = "**" if label == "<97" else ""
        lines.append(f"| {bold}{label}{bold} | {vals} | {cnt:,} |")

    lines.append(f"\n#### 상승 확률 (%)\n")
    lines.append(f"| 이격도 | " + " | ".join(f"{n}일후" for n in N_DAYS) + " | 건수 |")
    lines.append(f"|--------|" + "|".join("------:" for _ in N_DAYS) + "|-----:|")

    for label in DISPARITY_LABELS:
        bdf = adf[adf["disp_bin"] == label]
        cnt = len(bdf)
        if cnt == 0:
            vals = " | ".join("-" for _ in N_DAYS)
        else:
            parts = []
            for n in N_DAYS:
                valid = bdf[f"ret_{n}d"].dropna()
                if len(valid) == 0:
                    parts.append("-")
                else:
                    parts.append(f"{(valid > 0).mean() * 100:.1f}")
            vals = " | ".join(parts)
        bold = "**" if label == "<97" else ""
        lines.append(f"| {bold}{label}{bold} | {vals} | {cnt:,} |")

    return "\n".join(lines)


def main():
    df = load_all_stocks()
    print("이격도 및 수익률 계산 중...")
    df = calc_disparity_and_returns(df)
    print(f"  분석 대상: {len(df):,}건")

    result = analyze(df, "개별종목 전체")
    print(result)

    # 결과 파일 저장
    os.makedirs("results", exist_ok=True)
    with open("results/backtest_disparity_stocks.md", "w") as f:
        f.write("# 개별종목 5일선 이격도별 N일 후 수익률 분석\n\n")
        f.write("- **이격도** = (종가 / 5일이평) x 100\n")
        f.write("- **분석 기간**: 2023 ~ 2026 (연도별 구분)\n")
        f.write("- **대상**: stocks 테이블 전 종목\n\n")
        f.write("---\n")
        f.write(result)
        f.write("\n")

    print("\n결과 저장 완료: results/backtest_disparity_stocks.md")


if __name__ == "__main__":
    main()
