"""
5일선 이격도에 따른 N일 후 평균 수익률 & 상승 확률 분석
- 대상: KOSPI, KOSDAQ 지수
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
INDICES = {"^KS11": "KOSPI", "^KQ11": "KOSDAQ"}


def load_index_data(symbol: str) -> pd.DataFrame:
    query = text("""
        SELECT date, close FROM market_indices
        WHERE symbol = :sym
        ORDER BY date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn, params={"sym": symbol})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def calc_disparity_and_returns(df: pd.DataFrame) -> pd.DataFrame:
    """5일선 이격도 및 N일 후 수익률 계산"""
    df = df.copy()
    df["ma5"] = df["close"].rolling(5).mean()
    df["disparity"] = (df["close"] / df["ma5"]) * 100

    for n in N_DAYS:
        df[f"ret_{n}d"] = df["close"].shift(-n) / df["close"] - 1

    df["year"] = df["date"].dt.year
    df["disp_bin"] = pd.cut(df["disparity"], bins=DISPARITY_BINS, labels=DISPARITY_LABELS, right=False)
    return df.dropna(subset=["disparity"])


def analyze(df: pd.DataFrame, index_name: str) -> str:
    """연도별 × 이격도구간별 × N일후 수익률/상승확률 분석"""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  {index_name} — 5일선 이격도별 N일 후 수익률 분석")
    lines.append(f"{'='*80}")

    for year in YEARS:
        ydf = df[df["year"] == year]
        if ydf.empty:
            lines.append(f"\n## {year}년: 데이터 없음")
            continue

        lines.append(f"\n## {year}년 (관측수: {len(ydf)}일)")

        # 평균 수익률 테이블
        lines.append(f"\n### 평균 수익률 (%)")
        header = f"{'이격도구간':<12}" + "".join(f"{'  '+str(n)+'일후':>8}" for n in N_DAYS) + f"{'  건수':>7}"
        lines.append(header)
        lines.append("-" * len(header))

        for label in DISPARITY_LABELS:
            bdf = ydf[ydf["disp_bin"] == label]
            cnt = len(bdf)
            if cnt == 0:
                vals = "".join(f"{'   -':>8}" for _ in N_DAYS)
            else:
                vals = "".join(
                    f"{bdf[f'ret_{n}d'].mean()*100:8.2f}" if bdf[f"ret_{n}d"].notna().sum() > 0 else f"{'   -':>8}"
                    for n in N_DAYS
                )
            lines.append(f"{label:<12}{vals}{cnt:>7}")

        # 상승 확률 테이블
        lines.append(f"\n### 상승 확률 (%)")
        header = f"{'이격도구간':<12}" + "".join(f"{'  '+str(n)+'일후':>8}" for n in N_DAYS) + f"{'  건수':>7}"
        lines.append(header)
        lines.append("-" * len(header))

        for label in DISPARITY_LABELS:
            bdf = ydf[ydf["disp_bin"] == label]
            cnt = len(bdf)
            if cnt == 0:
                vals = "".join(f"{'   -':>8}" for _ in N_DAYS)
            else:
                vals = ""
                for n in N_DAYS:
                    col = f"ret_{n}d"
                    valid = bdf[col].dropna()
                    if len(valid) == 0:
                        vals += f"{'   -':>8}"
                    else:
                        prob = (valid > 0).mean() * 100
                        vals += f"{prob:8.1f}"
            lines.append(f"{label:<12}{vals}{cnt:>7}")

    # 전체 기간 종합
    lines.append(f"\n## 전체 기간 종합 (2023~2026)")

    lines.append(f"\n### 평균 수익률 (%)")
    adf = df[df["year"].isin(YEARS)]
    header = f"{'이격도구간':<12}" + "".join(f"{'  '+str(n)+'일후':>8}" for n in N_DAYS) + f"{'  건수':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for label in DISPARITY_LABELS:
        bdf = adf[adf["disp_bin"] == label]
        cnt = len(bdf)
        if cnt == 0:
            vals = "".join(f"{'   -':>8}" for _ in N_DAYS)
        else:
            vals = "".join(
                f"{bdf[f'ret_{n}d'].mean()*100:8.2f}" if bdf[f"ret_{n}d"].notna().sum() > 0 else f"{'   -':>8}"
                for n in N_DAYS
            )
        lines.append(f"{label:<12}{vals}{cnt:>7}")

    lines.append(f"\n### 상승 확률 (%)")
    lines.append(header)
    lines.append("-" * len(header))

    for label in DISPARITY_LABELS:
        bdf = adf[adf["disp_bin"] == label]
        cnt = len(bdf)
        if cnt == 0:
            vals = "".join(f"{'   -':>8}" for _ in N_DAYS)
        else:
            vals = ""
            for n in N_DAYS:
                col = f"ret_{n}d"
                valid = bdf[col].dropna()
                if len(valid) == 0:
                    vals += f"{'   -':>8}"
                else:
                    prob = (valid > 0).mean() * 100
                    vals += f"{prob:8.1f}"
        lines.append(f"{label:<12}{vals}{cnt:>7}")

    return "\n".join(lines)


def main():
    all_results = []

    for symbol, name in INDICES.items():
        print(f"\n{name} 데이터 로딩...")
        df = load_index_data(symbol)
        print(f"  총 {len(df)}행 로드")

        df = calc_disparity_and_returns(df)
        result = analyze(df, name)
        all_results.append(result)
        print(result)

    # 결과 파일 저장
    output = "\n".join(all_results)
    os.makedirs("results", exist_ok=True)
    with open("results/backtest_disparity.md", "w") as f:
        f.write("# 5일선 이격도별 N일 후 수익률 분석 (KOSPI / KOSDAQ)\n")
        f.write(f"분석 기간: 2023~2026 (연도별 구분)\n")
        f.write(f"이격도 = (종가 / 5일이평) × 100\n\n")
        f.write("```\n")
        f.write(output)
        f.write("\n```\n")

    print("\n\n결과 저장 완료: results/backtest_disparity.md")


if __name__ == "__main__":
    main()
