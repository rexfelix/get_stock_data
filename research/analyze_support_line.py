"""
상승추세 종목의 '일시적 하락 시 자주 지지받는 선' 탐색
========================================================

목적: 주봉 정배열(상승추세) 종목이 일시적으로 밀릴 때, 어떤 일봉 지지선
      (BBI, MAxx)을 어느 정도(%) 아래에서 터치하면 가장 자주 '재상승(지지)'
      하는지 통계적으로 탐색.

정의
----
- 상승추세(=두봉): 주봉(W-FRI) MA4 > MA13 > MA26 > MA52 정배열이
  '직전 2주 연속' 성립.  각 일자에는 merge_asof(backward)로 '직전 완성 주봉'
  기준값을 매핑 → 룩어헤드 없음.
- 지지선 후보(일봉): BBI=(MA3+MA6+MA12+MA24)/4, MA5, MA10, MA20, MA60, MA120
  각 라인 L 에 offset 을 곱해 후보선 = L*(1+offset),  offset ∈ {0,-3,-5,-7,-10,-15%}
- 터치(touch): 상승추세 상태에서 '전일 저가 > 후보선' 이고 '당일 저가 <= 후보선'
  (위에서 내려와 선을 처음 닿는 사건. 머무는 동안 중복 카운트 방지).
- 지지성공(rebound): 터치일 저가 대비, 이후 N거래일 내 고가가 +X% 도달.
  기본 N=10, X=5%.  지지율 = 성공 / 터치.

집계: 전종목 / KOSPI200 두 유니버스.
"""

import time
import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_larry_mtl import ENGINE

BASES = ["BBI", "MA5", "MA10", "MA20", "MA60", "MA120"]
OFFSETS = [0.0, -0.03, -0.05, -0.07, -0.10, -0.15]
N_BASE, X_BASE = 10, 0.05          # 기본 재상승 판정창/임계
N_GRID = [5, 10, 20]
X_GRID = [0.03, 0.05, 0.10]
WARMUP_START = "2019-01-02"        # 주봉 MA52 워밍업 위해 전체 로드


def load():
    with ENGINE.connect() as conn:
        k200 = set(pd.read_sql(text("SELECT ticker FROM kospi200_members"), conn)["ticker"])
        df = pd.read_sql(text(f"""
            SELECT date, open, high, low, close, ticker
            FROM stocks WHERE date >= '{WARMUP_START}'
            ORDER BY ticker, date
        """), conn)
    df["date"] = pd.to_datetime(df["date"])
    return df, k200


def fwd_max_high(high: np.ndarray, n: int) -> np.ndarray:
    """각 t에 대해 t+1..t+n 구간 고가 최댓값 (look-ahead 의도적, 미래수익 판정용)."""
    s = pd.Series(high).shift(-1)          # t+1 시작
    rev = s[::-1].rolling(n, min_periods=1).max()[::-1]
    return rev.to_numpy()


def build_lines(g: pd.DataFrame) -> dict[str, np.ndarray]:
    c = g["close"]
    bbi = (c.rolling(3).mean() + c.rolling(6).mean()
           + c.rolling(12).mean() + c.rolling(24).mean()) / 4.0
    return {
        "BBI": bbi.to_numpy(),
        "MA5": c.rolling(5).mean().to_numpy(),
        "MA10": c.rolling(10).mean().to_numpy(),
        "MA20": c.rolling(20).mean().to_numpy(),
        "MA60": c.rolling(60).mean().to_numpy(),
        "MA120": c.rolling(120).mean().to_numpy(),
    }


def uptrend_flag(g: pd.DataFrame) -> np.ndarray:
    """주봉 MA4>13>26>52 가 2주 연속 → 각 일자 직전 완성주봉 기준 boolean."""
    w = g.set_index("date")["close"].resample("W-FRI").last().dropna()
    if len(w) < 53:
        return np.zeros(len(g), dtype=bool)
    ma4, ma13 = w.rolling(4).mean(), w.rolling(13).mean()
    ma26, ma52 = w.rolling(26).mean(), w.rolling(52).mean()
    aligned = (ma4 > ma13) & (ma13 > ma26) & (ma26 > ma52)
    aligned2 = aligned & aligned.shift(1)            # 두봉(2주 연속)
    wk = pd.DataFrame({"week_end": w.index, "up": aligned2.fillna(False).values})
    m = pd.merge_asof(g[["date"]], wk, left_on="date",
                      right_on="week_end", direction="backward")
    return m["up"].fillna(False).to_numpy()


def analyze():
    print("로딩...")
    t0 = time.time()
    df, k200 = load()
    print(f"  {time.time()-t0:.1f}s, {df['ticker'].nunique()} 종목")

    # 집계 컨테이너: universe -> (base,offset) -> [touch, success_baseNX]
    agg = {u: {(b, o): np.zeros(2, dtype=np.int64)
               for b in BASES for o in OFFSETS} for u in ("ALL", "K200")}
    # 헤드라인 라인(MA10,BBI) × offset(-0.05,-0.10) 의 N×X 민감도
    head_keys = [(b, o) for b in ("MA10", "BBI") for o in (-0.05, -0.10)]
    sens = {u: {(b, o, n, x): np.zeros(2, dtype=np.int64)
                for (b, o) in head_keys for n in N_GRID for x in X_GRID}
            for u in ("ALL", "K200")}

    t0 = time.time()
    ng = 0
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 130:
            continue
        up = uptrend_flag(g)
        if not up.any():
            continue
        low = g["low"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        lines = build_lines(g)
        fwd = {n: fwd_max_high(high, n) for n in N_GRID}   # N별 미래고가
        low_prev = np.concatenate([[np.nan], low[:-1]])

        unis = ["ALL"] + (["K200"] if ticker in k200 else [])

        for b in BASES:
            base = lines[b]
            for o in OFFSETS:
                L = base * (1.0 + o)
                Lprev = np.concatenate([[np.nan], L[:-1]])
                touch = up & (low <= L) & (low_prev > Lprev) & np.isfinite(L)
                if not touch.any():
                    continue
                # 기본 N,X 성공
                succ = touch & (fwd[N_BASE] >= low * (1.0 + X_BASE))
                tc, sc = int(touch.sum()), int(succ.sum())
                for u in unis:
                    agg[u][(b, o)] += (tc, sc)
                # 민감도(헤드라인만)
                if (b, o) in head_keys:
                    for n in N_GRID:
                        for x in X_GRID:
                            s2 = touch & (fwd[n] >= low * (1.0 + x))
                            for u in unis:
                                sens[u][(b, o, n, x)] += (int(touch.sum()), int(s2.sum()))
        ng += 1
        if ng % 500 == 0:
            print(f"  {ng} 종목... {time.time()-t0:.0f}s")
    print(f"  분석 {ng} 종목, {time.time()-t0:.0f}s")
    return agg, sens


def grid_table(agg_u: dict) -> str:
    """offset 행 × base 열, 각 셀 = 지지율%(터치수)."""
    lines = ["| offset \\\\ line | " + " | ".join(BASES) + " |",
             "|---|" + "---:|" * len(BASES)]
    for o in OFFSETS:
        cells = []
        for b in BASES:
            tc, sc = agg_u[(b, o)]
            cells.append(f"{sc/tc*100:.1f}% ({tc:,})" if tc else "–")
        lines.append(f"| {o*100:+.0f}% | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_table(agg_u: dict, min_touch: int) -> str:
    rows = []
    for (b, o), (tc, sc) in agg_u.items():
        if tc >= min_touch:
            rows.append((sc/tc, tc, b, o))
    rows.sort(reverse=True)
    out = ["| 순위 | 라인 | offset | 지지율 | 터치수 | 성공수 |",
           "|---:|---|---:|---:|---:|---:|"]
    for i, (rate, tc, b, o) in enumerate(rows[:12], 1):
        sc = round(rate*tc)
        out.append(f"| {i} | {b} | {o*100:+.0f}% | {rate*100:.1f}% | {tc:,} | {sc:,} |")
    return "\n".join(out)


def sens_table(sens_u: dict) -> str:
    out = ["| 라인 | offset | N | X | 지지율 | 터치수 |",
           "|---|---:|---:|---:|---:|---:|"]
    for (b, o, n, x), (tc, sc) in sens_u.items():
        out.append(f"| {b} | {o*100:+.0f}% | {n} | {x*100:.0f}% | "
                   f"{(sc/tc*100 if tc else 0):.1f}% | {tc:,} |")
    return "\n".join(out)


def main():
    agg, sens = analyze()
    parts = ["# 상승추세 종목의 지지선 탐색 — 일봉 BBI/MA × offset%\n",
             "## 정의\n",
             "- 상승추세(두봉): 주봉 MA4>13>26>52 정배열이 **직전 2주 연속** "
             "(각 일자=직전 완성 주봉 기준, 룩어헤드 없음)",
             "- 후보 지지선: BBI/MA5/10/20/60/120 × offset(선*(1+offset))",
             "- 터치: 상승추세 중 전일저가>선 & 당일저가<=선 (위에서 처음 닿는 사건)",
             f"- 지지성공(기본): 터치일 저가 대비 이후 **{N_BASE}거래일 내 고가 +{X_BASE*100:.0f}%** 도달",
             f"- 기간: {WARMUP_START} ~ 2026-06-23, 전종목 4,079개\n", "---\n"]

    for u, label in (("ALL", "전종목"), ("K200", "KOSPI200")):
        parts += [f"## [{label}] 지지율 그리드  (셀 = 지지율% (터치수)), N={N_BASE} X={X_BASE*100:.0f}%\n",
                  grid_table(agg[u]), "\n",
                  f"### [{label}] 상위 조합 (터치수 ≥ {200 if u=='ALL' else 30})\n",
                  best_table(agg[u], 200 if u == "ALL" else 30), "\n"]

    parts += ["---\n", "## 헤드라인 비교: MA10 vs BBI (N×X 민감도)\n"]
    for u, label in (("ALL", "전종목"), ("K200", "KOSPI200")):
        parts += [f"### [{label}]\n", sens_table(sens[u]), "\n"]

    report = "\n".join(parts) + "\n"
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/analyze_support_line.md", "w") as f:
        f.write(report)
    print("\n저장: results/analyze_support_line.md\n")
    print(report)


if __name__ == "__main__":
    main()
