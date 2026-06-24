"""
상승추세 종목 — 당일 종가 하락폭별 '다음날 반등' 통계
=====================================================
질문: 주봉 정배열(상승추세) 종목에서 당일 종가가 전일 종가 대비 몇 % 하락하면
      다음날 반등이 잘 나오는가?

정의
----
- 상승추세(두봉): 주봉 MA4>13>26>52 정배열 2주 연속(직전 완성 주봉 기준, 룩어헤드 없음)
- 신호일 t: 당일수익률 r = close_t / close_{t-1} - 1  (당일 종가 확정)
- 하락폭 버킷별로, 다음날 t+1 지표:
    · 다음날 상승확률 = P(close_{t+1} > close_t)
    · 다음날 종가수익률 평균/중앙값 = close_{t+1}/close_t - 1
    · 다음날 장중 +2%/+3% 도달확률 = P(high_{t+1} >= close_t*(1+k))
    · 다음날 고가/저가 평균%
- baseline = 상승추세 전체 일자(하락폭 무관)의 동일 지표
집계: 전종목 / KOSPI200
"""
import time
import numpy as np
import pandas as pd
from analyze_support_line import load, uptrend_flag

# 당일수익률 버킷 (하한, 상한)  단위 비율
BUCKETS = [(-99, -0.10), (-0.10, -0.07), (-0.07, -0.05), (-0.05, -0.03),
           (-0.03, -0.02), (-0.02, -0.01), (-0.01, 0.0)]
LABELS = {(-99,-0.10):"≤ -10%", (-0.10,-0.07):"-10~-7%", (-0.07,-0.05):"-7~-5%",
          (-0.05,-0.03):"-5~-3%", (-0.03,-0.02):"-3~-2%", (-0.02,-0.01):"-2~-1%",
          (-0.01,0.0):"-1~0%"}


def main():
    print("로딩..."); t0 = time.time()
    df, k200 = load()
    print(f"  {time.time()-t0:.0f}s")

    # universe -> bucket('ALL'=baseline 포함) -> dict of accumulator lists
    def new_acc():
        return {b: {"ret": [], "hi": []} for b in BUCKETS + ["BASE"]}
    store = {"ALL": new_acc(), "K200": new_acc()}

    t0 = time.time(); ng = 0
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 130:
            continue
        up = uptrend_flag(g)
        if not up.any():
            continue
        close = g["close"].to_numpy(float)
        high = g["high"].to_numpy(float)
        prev = np.concatenate([[np.nan], close[:-1]])
        r = close / prev - 1.0
        nxt_close = np.concatenate([close[1:], [np.nan]])
        nxt_high = np.concatenate([high[1:], [np.nan]])
        nret = nxt_close / close - 1.0
        nhi = nxt_high / close - 1.0
        valid = up & np.isfinite(r) & np.isfinite(nret)

        unis = ["ALL"] + (["K200"] if ticker in k200 else [])
        for u in unis:
            acc = store[u]
            acc["BASE"]["ret"].append(nret[valid])
            acc["BASE"]["hi"].append(nhi[valid])
            for b in BUCKETS:
                lo, hi = b
                m = valid & (r >= lo) & (r < hi)
                if m.any():
                    acc[b]["ret"].append(nret[m])
                    acc[b]["hi"].append(nhi[m])
        ng += 1
        if ng % 800 == 0:
            print(f"  {ng}... {time.time()-t0:.0f}s")
    print(f"  {ng} 종목, {time.time()-t0:.0f}s")

    def row(name, ret, hi):
        n = len(ret)
        if n == 0:
            return f"| {name} | 0 | – | – | – | – | – |"
        p_up = (ret > 0).mean() * 100
        p2 = (hi >= 0.02).mean() * 100
        p3 = (hi >= 0.03).mean() * 100
        return (f"| {name} | {n:,} | {p_up:.1f}% | {ret.mean()*100:+.2f}% | "
                f"{np.median(ret)*100:+.2f}% | {p2:.1f}% | {p3:.1f}% |")

    parts = ["# 상승추세 종목 — 당일 하락폭별 다음날 반등 통계\n",
             "- 상승추세 = 주봉 MA4>13>26>52 정배열 2주 연속(직전 완성 주봉, 룩어헤드 없음)",
             "- 신호: 당일 종가수익률 r = close_t/close_{t-1}-1, 다음날 t+1 측정",
             "- 기간 2019-01-02~2026-06-23, 전종목 4,079개\n",
             "지표: 다음날상승확률=P(다음종가>당일종가), 평균/중앙=다음날 종가수익률, "
             "+2%/+3%=다음날 장중 고가가 당일종가 대비 도달확률\n", "---\n"]

    for u, label in (("ALL", "전종목"), ("K200", "KOSPI200")):
        acc = store[u]
        parts += [f"## [{label}]\n",
                  "| 당일 하락폭 | 표본수 | 다음날 상승확률 | 다음날 평균 | 다음날 중앙 | 장중+2% | 장중+3% |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for b in BUCKETS:
            ret = np.concatenate(acc[b]["ret"]) if acc[b]["ret"] else np.array([])
            hi = np.concatenate(acc[b]["hi"]) if acc[b]["hi"] else np.array([])
            parts.append(row(LABELS[b], ret, hi))
        bret = np.concatenate(acc["BASE"]["ret"])
        bhi = np.concatenate(acc["BASE"]["hi"])
        parts.append(row("**baseline(전체)**", bret, bhi))
        parts.append("")

    rep = "\n".join(parts) + "\n"
    with open("results/analyze_nextday_rebound.md", "w") as f:
        f.write(rep)
    print("\n저장: results/analyze_nextday_rebound.md\n")
    print(rep)


if __name__ == "__main__":
    main()
