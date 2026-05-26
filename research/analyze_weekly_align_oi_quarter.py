"""
주봉 정배열(MA4>13>26>52) 발생 시점 기준 — 전/현재/후 분기 중
어느 분기에 영업이익이 급격히 증가했는지 통계 분석.

- 이벤트: 정배열 진입(직전 4주 비정배열 → 첫 정배열 주)
- 매핑: 이벤트 주가 속한 회계 분기 = current(Q)
- 성장: g(Q) = (OI(Q) - OI(Q-1)) / max(|OI(Q-1)|, 1)  (QoQ)
- 분석 대상: g_prev (Q-1), g_curr (Q), g_next (Q+1)
- 사용 가능 분기 데이터: 2024Q4 ~ 2026Q1 (2024Q3, 2026Q1 부분 커버)

Outputs (results/):
  weekly_align_oi_quarter_events.csv
  weekly_align_oi_quarter_summary.md
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

PRICE_START = "2023-06-01"   # 52주 + 여유 (정배열 진입 검출용)
SURGE_THRESHOLD = 0.50       # +50% QoQ = "급격한 증가"
NON_ALIGN_LOOKBACK = 4       # 이벤트 직전 N주 비정배열 조건

# 이벤트 적합 분기: Q-2 ~ Q+1 데이터 보유
ELIGIBLE_QUARTERS = [
    (2025, 1), (2025, 2), (2025, 3), (2025, 4),
]
# 부분커버(2024Q3=614, 2026Q1=1154) 포함 허용


def week_to_quarter(date: pd.Timestamp) -> tuple[int, int]:
    """주봉 마지막일(금) → 회계 분기(year, quarter 1-4)"""
    return (date.year, (date.month - 1) // 3 + 1)


def offset_quarter(yq: tuple[int, int], k: int) -> tuple[int, int]:
    y, q = yq
    n = y * 4 + (q - 1) + k
    return (n // 4, n % 4 + 1)


def load_prices() -> pd.DataFrame:
    print("[1] stocks 일봉 로드...")
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT date, ticker, close, name
            FROM stocks
            WHERE date >= '{PRICE_START}' AND close IS NOT NULL
            ORDER BY ticker, date
        """), conn)
    df["date"] = pd.to_datetime(df["date"])
    print(f"    rows={len(df):,}  tickers={df['ticker'].nunique():,}")
    return df


def load_quarterly_oi() -> pd.DataFrame:
    """단일분기 영업이익(억원) - 실적만 (is_estimate=false)"""
    print("[2] 분기 OI 로드...")
    with ENGINE.connect() as conn:
        df = pd.read_sql(text("""
            SELECT ticker, year, quarter, operating_income AS oi
            FROM financial_annual
            WHERE quarter > 0 AND is_estimate = false
              AND operating_income IS NOT NULL
        """), conn)
    print(f"    rows={len(df):,}  tickers={df['ticker'].nunique():,}")
    return df


def detect_alignment_events(prices: pd.DataFrame) -> pd.DataFrame:
    """티커별 주봉 변환 → MA 정배열 진입 이벤트 추출"""
    print("[3] 주봉 변환 + 정배열 진입 검출...")
    events = []
    tickers = prices["ticker"].unique()
    for i, tkr in enumerate(tickers):
        if i and i % 500 == 0:
            print(f"    {i:,}/{len(tickers):,} ...")
        g = prices[prices["ticker"] == tkr].set_index("date").sort_index()
        if len(g) < 260:
            continue
        w = g["close"].resample("W-FRI").last().dropna()
        if len(w) < 52 + NON_ALIGN_LOOKBACK + 1:
            continue
        ma4 = w.rolling(4).mean()
        ma13 = w.rolling(13).mean()
        ma26 = w.rolling(26).mean()
        ma52 = w.rolling(52).mean()
        aligned = (ma4 > ma13) & (ma13 > ma26) & (ma26 > ma52)

        a = aligned.values
        for j in range(52 + NON_ALIGN_LOOKBACK, len(a)):
            if not a[j]:
                continue
            # 직전 NON_ALIGN_LOOKBACK 주 모두 비정배열 → 진입
            if all(not a[k] for k in range(j - NON_ALIGN_LOOKBACK, j)):
                week_date = w.index[j]
                events.append({
                    "ticker": tkr,
                    "event_week": week_date.date(),
                    "close": float(w.iloc[j]),
                    "ma4": float(ma4.iloc[j]),
                    "ma13": float(ma13.iloc[j]),
                    "ma26": float(ma26.iloc[j]),
                    "ma52": float(ma52.iloc[j]),
                })
    ev = pd.DataFrame(events)
    if ev.empty:
        return ev
    ev["event_week"] = pd.to_datetime(ev["event_week"])
    ev["y"] = ev["event_week"].dt.year
    ev["q"] = (ev["event_week"].dt.month - 1) // 3 + 1
    print(f"    총 이벤트={len(ev):,}")
    return ev


def attach_oi_growth(events: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    """각 이벤트에 prev/curr/next OI + growth 부착"""
    print("[4] OI growth 계산...")
    oi_idx = oi.set_index(["ticker", "year", "quarter"])["oi"].to_dict()

    def get(t, yq):
        return oi_idx.get((t, yq[0], yq[1]), np.nan)

    def growth(curr, prev):
        if pd.isna(curr) or pd.isna(prev):
            return np.nan
        denom = abs(prev) if abs(prev) > 1 else 1.0
        return (curr - prev) / denom

    rows = []
    for _, r in events.iterrows():
        Q = (int(r["y"]), int(r["q"]))
        Qm2 = offset_quarter(Q, -2)
        Qm1 = offset_quarter(Q, -1)
        Qp1 = offset_quarter(Q, +1)
        oi_m2 = get(r["ticker"], Qm2)
        oi_m1 = get(r["ticker"], Qm1)
        oi_c  = get(r["ticker"], Q)
        oi_p1 = get(r["ticker"], Qp1)
        g_prev = growth(oi_m1, oi_m2)   # prev quarter QoQ
        g_curr = growth(oi_c,  oi_m1)   # current quarter QoQ
        g_next = growth(oi_p1, oi_c)    # next quarter QoQ
        rows.append({
            **r.to_dict(),
            "Q": f"{Q[0]}Q{Q[1]}",
            "oi_Qm2": oi_m2, "oi_Qm1": oi_m1, "oi_Q": oi_c, "oi_Qp1": oi_p1,
            "g_prev": g_prev, "g_curr": g_curr, "g_next": g_next,
        })
    df = pd.DataFrame(rows)
    # 3분기 growth 모두 보유한 케이스만
    df = df.dropna(subset=["g_prev", "g_curr", "g_next"])
    print(f"    growth 3개 모두 유효: {len(df):,}")
    return df


def classify_burst(df: pd.DataFrame) -> pd.DataFrame:
    """max growth 분기 분류 + 임계 충족 여부"""
    g_cols = ["g_prev", "g_curr", "g_next"]
    df = df.copy()
    df["max_growth"] = df[g_cols].max(axis=1)
    df["burst_q"] = df[g_cols].idxmax(axis=1).map(
        {"g_prev": "prev", "g_curr": "current", "g_next": "next"}
    )
    df["is_surge"] = df["max_growth"] >= SURGE_THRESHOLD
    return df


def summarize(df: pd.DataFrame) -> str:
    out = []
    out.append("# 주봉 정배열 진입 vs 영업이익 급증 분기 통계\n")
    out.append(f"- 분석일: 2026-05-26  /  이벤트 표본: {len(df):,}건")
    out.append(f"- 이벤트 정의: 직전 {NON_ALIGN_LOOKBACK}주 비정배열 → 첫 정배열 주")
    out.append(f"- 성장률: g(Q) = (OI(Q) - OI(Q-1)) / |OI(Q-1)|  (QoQ)")
    out.append(f"- 급증 임계: max(g_prev, g_curr, g_next) >= {SURGE_THRESHOLD*100:.0f}%")
    out.append("- 사용 분기: 2024Q3 ~ 2026Q1 (Q-2 ~ Q+1 4분기 필요)\n")

    out.append("## 1) 이벤트 분포 (event_week 분기별)\n")
    qdist = df["Q"].value_counts().sort_index()
    out.append("| Q | events |")
    out.append("|---|--------|")
    for q, n in qdist.items():
        out.append(f"| {q} | {n:,} |")
    out.append("")

    out.append("## 2) burst 분기 분포 (max growth 기준)\n")
    bdist = df["burst_q"].value_counts()
    total = len(df)
    out.append("| burst | count | share |")
    out.append("|-------|-------|-------|")
    for k in ["prev", "current", "next"]:
        n = int(bdist.get(k, 0))
        out.append(f"| {k} | {n:,} | {n/total*100:.1f}% |")
    out.append("")

    out.append(f"## 3) 급증(≥{SURGE_THRESHOLD*100:.0f}%) 이벤트 한정\n")
    surge = df[df["is_surge"]]
    out.append(f"- 급증 이벤트: {len(surge):,}건  ({len(surge)/total*100:.1f}%)\n")
    if not surge.empty:
        bdist2 = surge["burst_q"].value_counts()
        tot2 = len(surge)
        out.append("| burst | count | share |")
        out.append("|-------|-------|-------|")
        for k in ["prev", "current", "next"]:
            n = int(bdist2.get(k, 0))
            out.append(f"| {k} | {n:,} | {n/tot2*100:.1f}% |")
        out.append("")

    out.append("## 4) 성장률 분포 (평균/중앙값)\n")
    out.append("| 분기 | mean | median | p75 | p90 |")
    out.append("|------|------|--------|-----|-----|")
    for col, lbl in [("g_prev", "prev"), ("g_curr", "current"), ("g_next", "next")]:
        s = df[col]
        out.append(
            f"| {lbl} | {s.mean()*100:+.1f}% | {s.median()*100:+.1f}% | "
            f"{s.quantile(0.75)*100:+.1f}% | {s.quantile(0.90)*100:+.1f}% |"
        )
    out.append("")

    out.append("## 5) 분기별 burst 분포 교차표 (event_Q × burst_q)\n")
    ct = pd.crosstab(df["Q"], df["burst_q"], normalize="index") * 100
    cols = [c for c in ["prev", "current", "next"] if c in ct.columns]
    out.append("| event_Q | prev% | current% | next% | N |")
    out.append("|---------|-------|----------|-------|---|")
    cnt = df.groupby("Q").size()
    for q in ct.index:
        row = ct.loc[q]
        out.append(
            f"| {q} | {row.get('prev', 0):.1f} | {row.get('current', 0):.1f} | "
            f"{row.get('next', 0):.1f} | {cnt.loc[q]} |"
        )
    out.append("")

    out.append("## 6) Top 10 급증 이벤트 (max_growth 내림차순)\n")
    top = df.sort_values("max_growth", ascending=False).head(10)
    out.append("| ticker | event_week | Q | burst | g_prev | g_curr | g_next |")
    out.append("|--------|------------|---|-------|--------|--------|--------|")
    for _, r in top.iterrows():
        out.append(
            f"| {r['ticker']} | {r['event_week'].date()} | {r['Q']} | {r['burst_q']} | "
            f"{r['g_prev']*100:+.0f}% | {r['g_curr']*100:+.0f}% | {r['g_next']*100:+.0f}% |"
        )
    out.append("")

    return "\n".join(out)


def main():
    prices = load_prices()
    oi = load_quarterly_oi()
    events = detect_alignment_events(prices)
    if events.empty:
        print("이벤트 없음")
        return
    df = attach_oi_growth(events, oi)
    df = classify_burst(df)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "weekly_align_oi_quarter_events.csv")
    df.to_csv(csv_path, index=False)
    print(f"[CSV] {csv_path}")

    md = summarize(df)
    md_path = os.path.join(results_dir, "weekly_align_oi_quarter_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"[MD] {md_path}")
    print("\n" + md[:2000])


if __name__ == "__main__":
    main()
