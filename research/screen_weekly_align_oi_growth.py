"""
주봉 정배열(MA4>13>26>52) + 차년도 예측 영업이익 성장률 >= 200% 스크리닝

- 주봉 변환: 일봉 → W-FRI 리샘플링 (close 마지막 값)
- 정배열: 최신 주봉에서 MA4 > MA13 > MA26 > MA52
- 영업이익 성장률: 2026E 연간(quarter=0, is_estimate=true) / 2025 실적 (quarter=0)
- 둘 다 양수여야 함 (적자전환·흑자전환 별도 표시)
- 2026E 가 여러 건이면 collected_at 최신
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

GROWTH_THRESHOLD = 2.0   # 200% (= 3.0x)
PREV_YEAR = 2025
EST_YEAR = 2026
PRICE_START = "2024-01-01"  # MA52 + 여유


def load_oi_growth() -> pd.DataFrame:
    with ENGINE.connect() as conn:
        prev = pd.read_sql(text(f"""
            SELECT ticker, operating_income AS oi_prev
            FROM financial_annual
            WHERE year = {PREV_YEAR}
              AND quarter = 0
              AND is_estimate = false
              AND operating_income IS NOT NULL
        """), conn)

        est = pd.read_sql(text(f"""
            SELECT DISTINCT ON (ticker)
                   ticker, operating_income AS oi_est, collected_at
            FROM financial_annual
            WHERE year = {EST_YEAR}
              AND quarter = 0
              AND is_estimate = true
              AND operating_income IS NOT NULL
            ORDER BY ticker, collected_at DESC
        """), conn)

    prev = prev[~prev["oi_prev"].isna()]
    est = est[~est["oi_est"].isna()]

    df = prev.merge(est, on="ticker", how="inner")

    def growth(p, c):
        if pd.isna(p) or pd.isna(c):
            return np.nan
        if p <= 0 and c > 0:
            return np.inf      # 흑자전환
        if p > 0 and c <= 0:
            return -np.inf     # 적자전환
        if p <= 0 and c <= 0:
            return np.nan      # 적자지속
        return (c - p) / p

    df["growth"] = [growth(p, c) for p, c in zip(df["oi_prev"], df["oi_est"])]
    return df


def load_weekly_aligned(tickers: list[str]) -> pd.DataFrame:
    """선택된 티커들의 최신 주봉 MA 정배열 여부"""
    if not tickers:
        return pd.DataFrame()

    placeholders = ",".join(f"'{t}'" for t in tickers)
    with ENGINE.connect() as conn:
        px = pd.read_sql(text(f"""
            SELECT date, ticker, close, name
            FROM stocks
            WHERE ticker IN ({placeholders})
              AND date >= '{PRICE_START}'
            ORDER BY ticker, date
        """), conn)

    px["date"] = pd.to_datetime(px["date"])
    rows = []
    for tkr, g in px.groupby("ticker"):
        g = g.set_index("date").sort_index()
        w = g["close"].resample("W-FRI").last().dropna()
        if len(w) < 52:
            continue
        ma4 = w.rolling(4).mean()
        ma13 = w.rolling(13).mean()
        ma26 = w.rolling(26).mean()
        ma52 = w.rolling(52).mean()

        last_date = w.index[-1]
        m4, m13, m26, m52 = ma4.iloc[-1], ma13.iloc[-1], ma26.iloc[-1], ma52.iloc[-1]
        if pd.isna(m52):
            continue
        aligned = (m4 > m13 > m26 > m52)
        name = g["name"].dropna().iloc[-1] if g["name"].notna().any() else ""
        rows.append({
            "ticker": tkr,
            "name": name,
            "last_week": last_date.date(),
            "close": int(w.iloc[-1]),
            "ma4": m4, "ma13": m13, "ma26": m26, "ma52": m52,
            "aligned": aligned,
        })
    return pd.DataFrame(rows)


def fmt_growth(g):
    if g == np.inf:
        return "흑자전환"
    if g == -np.inf:
        return "적자전환"
    if pd.isna(g):
        return "N/A"
    return f"{g*100:+.1f}%"


def main():
    oi = load_oi_growth()
    print(f"[1] OI 데이터 매칭: {len(oi):,} 종목 (prev={PREV_YEAR} 실적, est={EST_YEAR} 예측)")

    # 200% 이상 성장 (흑자전환 포함)
    cand = oi[(oi["growth"] >= GROWTH_THRESHOLD) | (oi["growth"] == np.inf)].copy()
    print(f"[2] 영업이익 성장률 >= {GROWTH_THRESHOLD*100:.0f}% 종목: {len(cand):,}")

    if cand.empty:
        return

    tickers = cand["ticker"].tolist()
    wk = load_weekly_aligned(tickers)
    print(f"[3] 주봉 52주 이상 데이터 보유: {len(wk):,}")

    aligned = wk[wk["aligned"]].copy()
    print(f"[4] 주봉 MA4>13>26>52 정배열: {len(aligned):,}")

    out = aligned.merge(cand[["ticker", "oi_prev", "oi_est", "growth"]], on="ticker", how="left")
    # 성장률 큰 순 (흑자전환 = inf 가 위로)
    out["_sort"] = out["growth"].replace({np.inf: 1e18, -np.inf: -1e18})
    out = out.sort_values("_sort", ascending=False).drop(columns=["_sort"])

    print("\n" + "=" * 130)
    print(f"{'ticker':<8} {'name':<22} {'last_week':<12} {'close':>8} "
          f"{'MA4':>9} {'MA13':>9} {'MA26':>9} {'MA52':>9} "
          f"{'OI(2025)':>14} {'OI(2026E)':>14} {'growth':>12}")
    print("-" * 130)
    for _, r in out.iterrows():
        print(f"{r['ticker']:<8} {str(r['name'])[:20]:<22} {str(r['last_week']):<12} "
              f"{r['close']:>8,} "
              f"{r['ma4']:>9,.0f} {r['ma13']:>9,.0f} {r['ma26']:>9,.0f} {r['ma52']:>9,.0f} "
              f"{r['oi_prev']:>14,.0f} {r['oi_est']:>14,.0f} {fmt_growth(r['growth']):>12}")
    print("=" * 130)
    print(f"총 {len(out):,} 종목")

    out_path = os.path.join(os.path.dirname(__file__), "results",
                            "screen_weekly_align_oi_growth.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.assign(growth=[fmt_growth(g) for g in out["growth"]]).to_csv(out_path, index=False)
    print(f"[CSV] {out_path}")


if __name__ == "__main__":
    main()
