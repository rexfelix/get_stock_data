"""
성장주 퀄리티 필터 백테스트
조건:
  - EPS YoY ≥ +15% (QoQ 대체, 연간 데이터 한계)
  - TTM EPS > 0
  - 매출 YoY ≥ +10%
  - 영업이익 YoY ≥ +15%
매수: 결산년도 다음 해 4월 첫 거래일
비교: 조건 충족 종목 평균 vs 동기간 KOSPI/KOSDAQ 지수
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

ENGINE = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

N_DAYS = [5, 10, 20, 60, 120]


def load_data():
    with ENGINE.connect() as conn:
        fin = pd.read_sql(text("""
            SELECT ticker, year, revenue, operating_income, net_income, eps, roe
            FROM financial_annual
            WHERE is_estimate = false
            ORDER BY ticker, year
        """), conn)

        stocks = pd.read_sql(text("""
            SELECT date, close, ticker FROM stocks
            WHERE date >= '2021-01-01'
            ORDER BY ticker, date
        """), conn)

        indices = pd.read_sql(text("""
            SELECT date, close, symbol FROM market_indices
            WHERE date >= '2021-01-01'
            ORDER BY symbol, date
        """), conn)

    for df in [stocks, indices]:
        df["date"] = pd.to_datetime(df["date"])
    return fin, stocks, indices


def screen_stocks(fin: pd.DataFrame) -> pd.DataFrame:
    """각 연도에 성장주 조건을 만족하는 종목 스크리닝"""
    fin = fin.drop_duplicates(subset=["ticker", "year"], keep="first")
    fin = fin.sort_values(["ticker", "year"])

    records = []
    year_pairs = [(y, y + 1) for y in range(2020, 2025)]

    for prev_y, curr_y in year_pairs:
        prev = fin[fin["year"] == prev_y].set_index("ticker")
        curr = fin[fin["year"] == curr_y].set_index("ticker")

        common = prev.index.intersection(curr.index)
        if len(common) == 0:
            continue

        p = prev.loc[common]
        c = curr.loc[common]

        df = pd.DataFrame({
            "ticker": common,
            "fin_year": curr_y,
            "eps_curr": c["eps"].values,
            "eps_prev": p["eps"].values,
            "rev_curr": c["revenue"].values,
            "rev_prev": p["revenue"].values,
            "oi_curr": c["operating_income"].values,
            "oi_prev": p["operating_income"].values,
            "roe": c["roe"].values,
        })

        # EPS가 없으므로 당기순이익(지배주주)으로 대체
        df["ni_curr"] = c["net_income"].values
        df["ni_prev"] = p["net_income"].values

        # 증가율 계산 (EPS → 당기순이익 대체)
        df["eps_yoy"] = np.where(
            (df["ni_prev"] > 0), (df["ni_curr"] - df["ni_prev"]) / df["ni_prev"] * 100, np.nan
        )
        df["rev_yoy"] = np.where(
            (df["rev_prev"] > 0), (df["rev_curr"] - df["rev_prev"]) / df["rev_prev"] * 100, np.nan
        )
        df["oi_yoy"] = np.where(
            (df["oi_prev"] > 0), (df["oi_curr"] - df["oi_prev"]) / df["oi_prev"] * 100, np.nan
        )

        # 조건 적용 (EPS → 당기순이익 대체)
        mask = (
            (df["eps_yoy"] >= 15) &   # 당기순이익 YoY ≥ 15%
            (df["ni_curr"] > 0) &      # 당기순이익 > 0
            (df["rev_yoy"] >= 10) &    # 매출 YoY ≥ 10%
            (df["oi_yoy"] >= 15)       # 영업이익 YoY ≥ 15%
        )

        passed = df[mask].copy()
        all_stocks = df.copy()  # 비교용 전체

        for _, row in passed.iterrows():
            records.append({
                "ticker": row["ticker"],
                "fin_year": curr_y,
                "buy_year": curr_y + 1,
                "group": "조건충족",
                "eps_yoy": row["eps_yoy"],
                "rev_yoy": row["rev_yoy"],
                "oi_yoy": row["oi_yoy"],
                "roe": row["roe"],
            })

        # 조건 미충족 (비교 그룹)
        failed = df[~mask].copy()
        for _, row in failed.iterrows():
            records.append({
                "ticker": row["ticker"],
                "fin_year": curr_y,
                "buy_year": curr_y + 1,
                "group": "미충족",
                "eps_yoy": row["eps_yoy"],
                "rev_yoy": row["rev_yoy"],
                "oi_yoy": row["oi_yoy"],
                "roe": row["roe"],
            })

    return pd.DataFrame(records)


def calc_stock_returns(stocks: pd.DataFrame, screened: pd.DataFrame) -> pd.DataFrame:
    """종목별 4월 매수 후 N일 수익률"""
    stocks_grouped = {t: g.reset_index(drop=True) for t, g in stocks.groupby("ticker")}
    results = []

    for _, row in screened.iterrows():
        ticker = row["ticker"]
        buy_year = row["buy_year"]

        tdf = stocks_grouped.get(ticker)
        if tdf is None:
            continue

        mask = (tdf["date"].dt.year == buy_year) & (tdf["date"].dt.month >= 4)
        apr_idx = tdf.index[mask]
        if len(apr_idx) == 0:
            continue

        buy_pos = apr_idx[0]
        buy_price = tdf.at[buy_pos, "close"]
        buy_date = tdf.at[buy_pos, "date"]

        r = {**row.to_dict(), "buy_date": buy_date}
        for n in N_DAYS:
            tp = buy_pos + n
            if tp < len(tdf):
                r[f"ret_{n}d"] = tdf.at[tp, "close"] / buy_price - 1
            else:
                r[f"ret_{n}d"] = None
        results.append(r)

    return pd.DataFrame(results)


def calc_index_returns(indices: pd.DataFrame) -> pd.DataFrame:
    """지수의 4월 첫 거래일 기준 N일 수익률"""
    results = []
    for symbol, name in [("^KS11", "KOSPI"), ("^KQ11", "KOSDAQ")]:
        idf = indices[indices["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        for buy_year in range(2022, 2026):
            mask = (idf["date"].dt.year == buy_year) & (idf["date"].dt.month >= 4)
            apr_idx = idf.index[mask]
            if len(apr_idx) == 0:
                continue
            buy_pos = apr_idx[0]
            buy_price = idf.at[buy_pos, "close"]
            buy_date = idf.at[buy_pos, "date"]

            r = {"index": name, "buy_year": buy_year, "buy_date": buy_date}
            for n in N_DAYS:
                tp = buy_pos + n
                if tp < len(idf):
                    r[f"ret_{n}d"] = (idf.at[tp, "close"] / buy_price - 1) * 100
                else:
                    r[f"ret_{n}d"] = None
            results.append(r)

    return pd.DataFrame(results)


def report(stock_ret: pd.DataFrame, idx_ret: pd.DataFrame) -> str:
    lines = []

    # 조건 충족 종목 통계
    passed = stock_ret[stock_ret["group"] == "조건충족"]
    failed = stock_ret[stock_ret["group"] == "미충족"]

    lines.append("\n## 조건 충족 종목 현황\n")
    lines.append("| 결산년도 | 매수년도 | 조건충족 | 미충족 | 비율 |")
    lines.append("|:---:|:---:|---:|---:|---:|")
    for fy in sorted(passed["fin_year"].unique()):
        p = len(passed[passed["fin_year"] == fy])
        f = len(failed[failed["fin_year"] == fy])
        lines.append(f"| {fy} | {fy+1} | {p} | {f} | {p/(p+f)*100:.1f}% |")
    total_p = len(passed)
    total_f = len(failed)
    lines.append(f"| **합계** | | **{total_p}** | **{total_f}** | **{total_p/(total_p+total_f)*100:.1f}%** |")

    # 연도별 비교 테이블
    for buy_year in sorted(stock_ret["buy_year"].unique()):
        p = passed[passed["buy_year"] == buy_year]
        f = failed[failed["buy_year"] == buy_year]
        ki = idx_ret[(idx_ret["buy_year"] == buy_year) & (idx_ret["index"] == "KOSPI")]
        kq = idx_ret[(idx_ret["buy_year"] == buy_year) & (idx_ret["index"] == "KOSDAQ")]

        lines.append(f"\n## {buy_year}년 4월 매수 (결산: {buy_year-1}년)\n")

        # 평균 수익률
        lines.append("### 평균 수익률 (%)\n")
        lines.append("| 그룹 | " + " | ".join(f"{n}일후" for n in N_DAYS) + " | 건수 |")
        lines.append("|---|" + "|".join("---:" for _ in N_DAYS) + "|---:|")

        for label, sub in [("**조건충족**", p), ("미충족", f)]:
            if sub.empty:
                continue
            vals = " | ".join(
                f"{sub[f'ret_{n}d'].mean()*100:.2f}" if sub[f"ret_{n}d"].notna().sum() > 0 else "-"
                for n in N_DAYS
            )
            lines.append(f"| {label} | {vals} | {len(sub)} |")

        if not ki.empty:
            vals = " | ".join(f"{ki.iloc[0].get(f'ret_{n}d', 0):.2f}" if ki.iloc[0].get(f"ret_{n}d") else "-" for n in N_DAYS)
            lines.append(f"| KOSPI | {vals} | 1 |")
        if not kq.empty:
            vals = " | ".join(f"{kq.iloc[0].get(f'ret_{n}d', 0):.2f}" if kq.iloc[0].get(f"ret_{n}d") else "-" for n in N_DAYS)
            lines.append(f"| KOSDAQ | {vals} | 1 |")

        # 상승 확률
        lines.append("\n### 상승 확률 (%)\n")
        lines.append("| 그룹 | " + " | ".join(f"{n}일후" for n in N_DAYS) + " | 건수 |")
        lines.append("|---|" + "|".join("---:" for _ in N_DAYS) + "|---:|")

        for label, sub in [("**조건충족**", p), ("미충족", f)]:
            if sub.empty:
                continue
            parts = []
            for n in N_DAYS:
                v = sub[f"ret_{n}d"].dropna()
                parts.append(f"{(v>0).mean()*100:.1f}" if len(v) > 0 else "-")
            lines.append(f"| {label} | {' | '.join(parts)} | {len(sub)} |")

    # 전체 기간 종합
    lines.append("\n## 전체 기간 종합 (2022~2025 매수)\n")
    lines.append("### 평균 수익률 (%)\n")
    lines.append("| 그룹 | " + " | ".join(f"{n}일후" for n in N_DAYS) + " | 건수 |")
    lines.append("|---|" + "|".join("---:" for _ in N_DAYS) + "|---:|")

    for label, sub in [("**조건충족**", passed), ("미충족", failed)]:
        vals = " | ".join(
            f"{sub[f'ret_{n}d'].mean()*100:.2f}" if sub[f"ret_{n}d"].notna().sum() > 0 else "-"
            for n in N_DAYS
        )
        lines.append(f"| {label} | {vals} | {len(sub)} |")

    # 지수 평균
    for idx_name in ["KOSPI", "KOSDAQ"]:
        idf = idx_ret[idx_ret["index"] == idx_name]
        vals = " | ".join(f"{idf[f'ret_{n}d'].mean():.2f}" if idf[f"ret_{n}d"].notna().sum() > 0 else "-" for n in N_DAYS)
        lines.append(f"| {idx_name}(평균) | {vals} | {len(idf)} |")

    lines.append("\n### 상승 확률 (%)\n")
    lines.append("| 그룹 | " + " | ".join(f"{n}일후" for n in N_DAYS) + " | 건수 |")
    lines.append("|---|" + "|".join("---:" for _ in N_DAYS) + "|---:|")

    for label, sub in [("**조건충족**", passed), ("미충족", failed)]:
        parts = []
        for n in N_DAYS:
            v = sub[f"ret_{n}d"].dropna()
            parts.append(f"{(v>0).mean()*100:.1f}" if len(v) > 0 else "-")
        lines.append(f"| {label} | {' | '.join(parts)} | {len(sub)} |")

    # 조건충족 종목의 재무 프로필
    lines.append("\n## 조건충족 종목 재무 프로필 (평균)\n")
    lines.append("| 결산년도 | EPS YoY | 매출 YoY | 영업이익 YoY | ROE | 종목수 |")
    lines.append("|:---:|---:|---:|---:|---:|---:|")
    for fy in sorted(passed["fin_year"].unique()):
        sub = passed[passed["fin_year"] == fy]
        lines.append(
            f"| {fy} | {sub['eps_yoy'].mean():.1f}% | {sub['rev_yoy'].mean():.1f}% "
            f"| {sub['oi_yoy'].mean():.1f}% | {sub['roe'].mean():.1f}% | {len(sub)} |"
        )

    return "\n".join(lines)


def main():
    print("데이터 로딩...")
    fin, stocks, indices = load_data()
    print(f"  재무: {len(fin)}건, 주가: {len(stocks):,}건, 지수: {len(indices):,}건")

    print("성장주 스크리닝...")
    screened = screen_stocks(fin)
    passed = screened[screened["group"] == "조건충족"]
    print(f"  조건충족: {len(passed)}건 / 전체: {len(screened)}건")
    print(passed.groupby("fin_year").size())

    print("\n종목 수익률 계산...")
    stock_ret = calc_stock_returns(stocks, screened)
    print(f"  {len(stock_ret)}건")

    print("지수 수익률 계산...")
    idx_ret = calc_index_returns(indices)
    print(idx_ret)

    print("\n리포트 생성...")
    result = report(stock_ret, idx_ret)
    print(result)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_growth_quality.md", "w") as f:
        f.write("# 성장주 퀄리티 필터 백테스트\n\n")
        f.write("## 조건\n")
        f.write("- EPS YoY ≥ +15% (연간, QoQ 대체)\n")
        f.write("- TTM EPS > 0\n")
        f.write("- 매출 YoY ≥ +10%\n")
        f.write("- 영업이익 YoY ≥ +15%\n\n")
        f.write("## 매수 규칙\n")
        f.write("- 결산년도 다음 해 4월 첫 거래일 매수\n")
        f.write("- 비교: 조건충족 vs 미충족 vs KOSPI/KOSDAQ\n\n")
        f.write("---\n")
        f.write(result)
        f.write("\n")

    print("\n결과 저장: results/backtest_growth_quality.md")


if __name__ == "__main__":
    main()
