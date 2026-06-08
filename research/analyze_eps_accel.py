"""
EPS 상승률의 가속도(델타)와 주가 추세의 상관관계 검증.

가설:
  영업이익 상승률의 가속도(2차 미분)는 주가 추세 상승과 양(+)의 상관을 갖는다(실증).
  EPS 상승률의 가속도도 동일하게 양의 상관을 갖는가?

정의:
  EPS 상승률(t)   g(t) = (EPS_t - EPS_{t-1}) / EPS_{t-1}
  델타(가속도)     d(t) = g(t) - g(t-1)
  (분모 왜곡 방지: EPS_{t-1} > 0 AND EPS_{t-2} > 0 인 표본만 사용)

주가 추세(3개 창, 각 회계연도 t 기준):
  ret_prev : 연도 t-1 한 해 수익률 (주가 선행 가설 검증)
  ret_curr : 연도 t   한 해 수익률 (동행)
  ret_fwd  : t+1년 4월초 ~ t+2년 3월말 (실적확정 후 매매가능 수익률)

벤치마크: 동일 방법으로 영업이익(operating_income) 가속도 / EPS 상승률(1차) 비교.
"""
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from scipy import stats

load_dotenv("/Volumes/SSD/project/py/invest/data_center/.env")
ENGINE = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

WINSOR = 0.01  # 1%/99% 윈저화


# ---------------------------------------------------------------- 데이터 로드
def load_fundamentals() -> pd.DataFrame:
    """연간 실적(actual only, quarter=0)."""
    with ENGINE.connect() as c:
        df = pd.read_sql(text("""
            SELECT ticker, year, eps, operating_income
            FROM financial_annual
            WHERE quarter = 0 AND is_estimate = false
        """), c)
    return df.sort_values(["ticker", "year"]).reset_index(drop=True)


def load_prices() -> pd.DataFrame:
    with ENGINE.connect() as c:
        df = pd.read_sql(text(
            "SELECT ticker, date, close FROM stocks WHERE close > 0"), c)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def load_kospi200() -> set:
    with ENGINE.connect() as c:
        return set(pd.read_sql(text(
            "SELECT ticker FROM kospi200_members"), c)["ticker"])


# ----------------------------------------------------- 연말/특정일 종가 헬퍼
def price_on_or_before(g: pd.DataFrame, asof: pd.Timestamp):
    """asof 이전(포함) 가장 가까운 종가. (g: 단일 ticker 정렬된 프레임)"""
    sub = g[g["date"] <= asof]
    if sub.empty:
        return np.nan
    # 너무 오래된(60일 초과) 데이터면 결측 처리(상폐/거래정지 대응)
    last = sub.iloc[-1]
    if (asof - last["date"]).days > 60:
        return np.nan
    return last["close"]


def build_return_table(prices: pd.DataFrame) -> dict:
    """ticker -> 정렬된 price 프레임 (빠른 asof 조회용)."""
    return {t: g.reset_index(drop=True) for t, g in prices.groupby("ticker")}


def ret_between(pmap, ticker, d0, d1):
    g = pmap.get(ticker)
    if g is None:
        return np.nan
    p0 = price_on_or_before(g, d0)
    p1 = price_on_or_before(g, d1)
    if not (p0 and p1) or np.isnan(p0) or np.isnan(p1) or p0 <= 0:
        return np.nan
    return p1 / p0 - 1.0


# ----------------------------------------------------------- 가속도 계산
def compute_accel(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """col 의 상승률 g(t) 와 가속도 d(t) 를 ticker별로 계산.
    분모(전년·전전년 값) 양수 표본만 유효."""
    rows = []
    for tk, g in df.groupby("ticker"):
        g = g.set_index("year")[col]
        years = g.index.tolist()
        for y in years:
            v0 = g.get(y - 2)      # t-2
            v1 = g.get(y - 1)      # t-1
            v2 = g.get(y)          # t
            if v0 is None or v1 is None or v2 is None:
                continue
            if pd.isna(v0) or pd.isna(v1) or pd.isna(v2):
                continue
            if v0 <= 0 or v1 <= 0:   # 분모 양수 제약
                continue
            g_t = (v2 - v1) / v1
            g_t1 = (v1 - v0) / v0
            rows.append({
                "ticker": tk, "year": y,
                f"{col}_g": g_t,          # 상승률 (1차)
                f"{col}_accel": g_t - g_t1,  # 가속도 (델타)
            })
    return pd.DataFrame(rows)


def winsorize(s: pd.Series, p=WINSOR) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


# --------------------------------------------------------------- 상관 분석
def corr_report(d: pd.DataFrame, xcol: str, ycol: str):
    sub = d[[xcol, ycol]].dropna()
    if len(sub) < 10:
        return None
    x = winsorize(sub[xcol]).values
    y = sub[ycol].values
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return dict(n=len(sub), pearson=pr, p_pearson=pp,
               spearman=sr, p_spearman=sp)


def quintile_report(d: pd.DataFrame, xcol: str, ycol: str, q=5):
    sub = d[[xcol, ycol, "year"]].dropna().copy()
    if len(sub) < 25:
        return None
    # 연도별 분위 (각 연도 cross-section 내 순위로 묶음)
    sub["bin"] = sub.groupby("year")[xcol].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False))
    g = sub.groupby("bin")[ycol].agg(["mean", "median", "count"])
    return g


def fmt_corr(label, r):
    if r is None:
        return f"  {label:<14} n부족"
    star = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    return (f"  {label:<14} n={r['n']:5d}  "
            f"Pearson={r['pearson']:+.4f}{star(r['p_pearson']):<3} "
            f"Spearman={r['spearman']:+.4f}{star(r['p_spearman']):<3}")


# ---------------------------------------------------------------------- main
def main():
    print("데이터 로드 중...")
    fund = load_fundamentals()
    prices = load_prices()
    k200 = load_kospi200()
    pmap = build_return_table(prices)

    # 가속도 계산 (EPS, 영업이익)
    eps = compute_accel(fund[["ticker", "year", "eps"]], "eps")
    oi = compute_accel(fund[["ticker", "year", "operating_income"]], "operating_income")

    d = eps.merge(oi, on=["ticker", "year"], how="outer")

    # 수익률 3개 창 계산
    print("수익률 계산 중...")
    recs = []
    for _, row in d.iterrows():
        tk, y = row["ticker"], int(row["year"])
        # 연말 기준일
        prev0 = pd.Timestamp(y - 2, 12, 31)
        prev1 = pd.Timestamp(y - 1, 12, 31)
        curr1 = pd.Timestamp(y, 12, 31)
        fwd0 = pd.Timestamp(y + 1, 4, 1)
        fwd1 = pd.Timestamp(y + 2, 3, 31)
        recs.append({
            "ticker": tk, "year": y,
            "ret_prev": ret_between(pmap, tk, prev0, prev1),
            "ret_curr": ret_between(pmap, tk, prev1, curr1),
            "ret_fwd": ret_between(pmap, tk, fwd0, fwd1),
        })
    rdf = pd.DataFrame(recs)
    d = d.merge(rdf, on=["ticker", "year"], how="left")
    d["k200"] = d["ticker"].isin(k200)

    # 분석 대상 연도 분포
    print("\n" + "=" * 78)
    print("표본 분포 (가속도 계산 가능 + 분모 양수 필터 후)")
    print("=" * 78)
    print(d.groupby("year").agg(
        eps_accel=("eps_accel", "count"),
        oi_accel=("operating_income_accel", "count"),
        k200=("k200", "sum"),
    ).to_string())

    windows = [("ret_prev", "전년(주가선행)"),
               ("ret_curr", "당년(동행)"),
               ("ret_fwd", "익년4월~(매매가능)")]

    for universe, mask in [("전체 유니버스", d["eps_accel"].notna()),
                           ("KOSPI200", d["k200"])]:
        sub = d[mask]
        print("\n" + "=" * 78)
        print(f"[{universe}]  상관분석  (*** p<.01, ** p<.05, * p<.10)")
        print("=" * 78)
        for ycol, yname in windows:
            print(f"\n── 주가창: {yname} ({ycol}) ──")
            print(fmt_corr("EPS 가속도", corr_report(sub, "eps_accel", ycol)))
            print(fmt_corr("EPS 상승률1차", corr_report(sub, "eps_g", ycol)))
            print(fmt_corr("영업익 가속도", corr_report(sub, "operating_income_accel", ycol)))
            print(fmt_corr("영업익 상승률1차", corr_report(sub, "operating_income_g", ycol)))

    # 5분위 (전체 유니버스, 핵심 창)
    print("\n" + "=" * 78)
    print("[전체 유니버스] EPS 가속도 5분위 정렬 → 평균 수익률 (단조성 검증)")
    print("  Q0=가속도 최저(감속) ... Q4=가속도 최고(가속)")
    print("=" * 78)
    sub = d[d["eps_accel"].notna()]
    for ycol, yname in windows:
        q = quintile_report(sub, "eps_accel", ycol)
        if q is None:
            continue
        print(f"\n── 주가창: {yname} ({ycol}) ──")
        qf = q.copy()
        qf["mean"] = (qf["mean"] * 100).round(2)
        qf["median"] = (qf["median"] * 100).round(2)
        print(qf.to_string())
        spread = (q["mean"].iloc[-1] - q["mean"].iloc[0]) * 100
        print(f"  Q4-Q0 스프레드: {spread:+.2f}%p")

    # 연도별 단면 상관 (강건성)
    print("\n" + "=" * 78)
    print("[전체 유니버스] 연도별 단면 상관 (EPS 가속도 vs 당년수익률)")
    print("=" * 78)
    for y, g in sub.groupby("year"):
        r = corr_report(g, "eps_accel", "ret_curr")
        if r:
            print(f"  {y}: " + fmt_corr("", r).strip())

    # 증분 설명력: ret ~ EPS상승률(1차) + EPS가속도 (표준화 OLS)
    print("\n" + "=" * 78)
    print("[전체 유니버스] 증분 설명력 OLS  (z-표준화 계수, HC 보정 t)")
    print("  ret ~ b1·EPS상승률1차 + b2·EPS가속도  → b2 유의하면 가속도 독립 기여")
    print("=" * 78)

    def zscore(s):
        return (s - s.mean()) / s.std(ddof=0)

    def ols2(reg, gcol, acol, ycol, glabel, alabel):
        reg = reg[[gcol, acol, ycol]].dropna().copy()
        if len(reg) < 50:
            return
        reg[gcol] = zscore(winsorize(reg[gcol]))
        reg[acol] = zscore(winsorize(reg[acol]))
        X = np.column_stack([np.ones(len(reg)), reg[gcol], reg[acol]])
        y = reg[ycol].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        S = (X * resid[:, None]).T @ (X * resid[:, None])
        cov = XtX_inv @ S @ XtX_inv
        tval = beta / np.sqrt(np.diag(cov))
        sig = lambda t: ('독립 +유의' if t > 1.96 else '독립 -유의' if t < -1.96 else '비유의')
        print(f"  n={len(reg):5d}  {glabel} b1={beta[1]:+.4f}(t{tval[1]:+.2f})"
              f"   {alabel} b2={beta[2]:+.4f}(t{tval[2]:+.2f}) ← {sig(tval[2])}")

    for ycol, yname in windows:
        print(f"\n── 주가창: {yname} ({ycol}) ──")
        ols2(sub, "eps_g", "eps_accel", ycol, "EPS상승률1차", "EPS가속도  ")
        ols2(sub, "operating_income_g", "operating_income_accel", ycol,
             "영업익상승1차", "영업익가속도")

    d.to_parquet("/tmp/eps_accel.parquet")
    print("\n저장: /tmp/eps_accel.parquet")
    return d


if __name__ == "__main__":
    main()
