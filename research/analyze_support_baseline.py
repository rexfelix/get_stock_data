"""
지지선 분석의 baseline(통제군) — '기계적 착시' 검증
====================================================
상승추세(주봉 2주연속 정배열) 일자에서, 터치 조건 없이도 '이후 10일 내 +5%'가
얼마나 흔한가?  + 저가가 선 아래로 깊이 내려갈수록 반등률이 자동으로 오르는가?

이 baseline 보다 '특정 선 터치'의 지지율이 유의하게 높아야 그 선이 의미를 가짐.
"""
import time
import numpy as np
import pandas as pd
from sqlalchemy import text
from backtest_larry_mtl import ENGINE
from analyze_support_line import (load, uptrend_flag, build_lines, fwd_max_high,
                                  N_BASE, X_BASE)


def main():
    print("로딩..."); t0 = time.time()
    df, k200 = load()
    print(f"  {time.time()-t0:.0f}s")

    # 누적: 전체 상승추세일 / 하락한 상승추세일(저가가 전일보다 낮음)
    tot = np.zeros(2, np.int64)      # [days, success]
    down = np.zeros(2, np.int64)
    # 저가의 BBI 대비 위치 버킷별 반등률  (편차 = low/BBI - 1)
    BUCKETS = [(-99, -0.12), (-0.12, -0.08), (-0.08, -0.05), (-0.05, -0.02),
               (-0.02, 0.0), (0.0, 0.02), (0.02, 99)]
    bk = {b: np.zeros(2, np.int64) for b in BUCKETS}
    bk_ma10 = {b: np.zeros(2, np.int64) for b in BUCKETS}

    t0 = time.time(); ng = 0
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 130:
            continue
        up = uptrend_flag(g)
        if not up.any():
            continue
        low = g["low"].to_numpy(float)
        fwd = fwd_max_high(g["high"].to_numpy(float), N_BASE)
        succ = fwd >= low * (1.0 + X_BASE)
        lines = build_lines(g)
        bbi, ma10 = lines["BBI"], lines["MA10"]
        low_prev = np.concatenate([[np.nan], low[:-1]])

        u = up & np.isfinite(fwd)
        tot += (int(u.sum()), int((u & succ).sum()))
        d = u & (low < low_prev)
        down += (int(d.sum()), int((d & succ).sum()))

        dev_bbi = low / bbi - 1.0
        dev_m10 = low / ma10 - 1.0
        for (lo, hi) in BUCKETS:
            mb = u & np.isfinite(dev_bbi) & (dev_bbi >= lo) & (dev_bbi < hi)
            bk[(lo, hi)] += (int(mb.sum()), int((mb & succ).sum()))
            mm = u & np.isfinite(dev_m10) & (dev_m10 >= lo) & (dev_m10 < hi)
            bk_ma10[(lo, hi)] += (int(mm.sum()), int((mm & succ).sum()))
        ng += 1
        if ng % 800 == 0:
            print(f"  {ng}... {time.time()-t0:.0f}s")
    print(f"  {ng} 종목, {time.time()-t0:.0f}s")

    def rate(a): return f"{a[1]/a[0]*100:.1f}%" if a[0] else "–"
    out = ["# Baseline 통제군 — 상승추세일의 무조건 반등률 (N=10, X=5%)\n",
           f"- 전체 상승추세 일수: **{tot[0]:,}**, 이후 10일내 +5% 도달: **{rate(tot)}**",
           f"- 그중 '하락일(저가<전일저가)'만: {down[0]:,}일, 반등률 **{rate(down)}**\n",
           "→ 이 값이 baseline. 특정 선 터치 지지율이 이보다 높아야 '선'이 의미.\n",
           "## 저가의 선-대비 편차 버킷별 반등률 (편차 = 저가/선 - 1)\n",
           "| 편차 버킷 | BBI 일수 | BBI 반등률 | MA10 일수 | MA10 반등률 |",
           "|---|---:|---:|---:|---:|"]
    labels = {(-99,-0.12):"≤ -12%", (-0.12,-0.08):"-12~-8%", (-0.08,-0.05):"-8~-5%",
              (-0.05,-0.02):"-5~-2%", (-0.02,0.0):"-2~0%", (0.0,0.02):"0~+2%",
              (0.02,99):"≥ +2%"}
    for b in BUCKETS:
        out.append(f"| {labels[b]} | {bk[b][0]:,} | {rate(bk[b])} | "
                   f"{bk_ma10[b][0]:,} | {rate(bk_ma10[b])} |")
    rep = "\n".join(out) + "\n"
    with open("results/analyze_support_baseline.md", "w") as f:
        f.write(rep)
    print("\n저장: results/analyze_support_baseline.md\n")
    print(rep)


if __name__ == "__main__":
    main()
