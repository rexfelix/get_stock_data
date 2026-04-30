"""
N (lookback) × K (슬롯) 매트릭스 백테스트.

매수: 1500억 N/N
매도: LIST_EXIT
자본: 진짜 K슬롯 모델

매트릭스: N=[3,5,7,10,15] × K=[3,5,7,10,15]
"""
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402
from backtest_5d_amount_filter import compute_5d_filter_signals  # noqa: E402
from backtest_5d_realistic_k import equity_real_k  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_n_k_matrix.md"

N_VALUES = [3, 5, 7, 10, 15]
K_VALUES = [3, 5, 7, 10, 15]


def main():
    print("=" * 60)
    print("N×K 매트릭스 백테스트 (1500억 N/N + K슬롯 + LIST_EXIT)")
    print("=" * 60)

    print("[1] 데이터 로드...")
    k200 = bt.load_kospi200_tickers()
    snapshot = bt.load_market_cap_snapshot()
    price_df = bt.load_price_data(k200["ticker"].tolist())
    daily_data = bt.build_daily_data(price_df, snapshot)
    panel = bt.build_daily_indicator_panel(daily_data, "amount")
    print(f"    {len(daily_data)}종목 시계열")

    # N별 신호 사전 계산
    print(f"\n[2] N별 신호 생성...")
    signals_by_n = {}
    for n in N_VALUES:
        signals = compute_5d_filter_signals(daily_data, lookback=n, top_k=200)
        n_per_day = [len(v) for v in signals.values()]
        signals_by_n[n] = signals
        print(f"  N={n:2d}: {len(signals)}일치, 평균 {np.mean(n_per_day):.2f}, max {max(n_per_day)}")

    # 매트릭스
    print(f"\n[3] N×K 매트릭스 백테스트")
    print("-" * 80)

    results = {}
    for n in N_VALUES:
        signals = signals_by_n[n]
        for k in K_VALUES:
            t0 = time.time()
            trades, _ = bt.run_backtest(
                daily_data, panel, signals,
                rule="LIST_EXIT", slots=k,
                max_concurrent=k,
            )
            eq = equity_real_k(trades, K=k)
            stats = bt.compute_stats(trades)
            stats.update(eq)
            cagr = stats.get("cagr", 0)
            mdd = stats.get("mdd", 0)
            calmar = abs(cagr / mdd) if mdd != 0 else 0
            results[(n, k)] = {
                "n": n, "k": k, "stats": stats, "calmar": calmar,
                "trades": trades,
            }
        print(f"  N={n:2d} 완료 ({time.time()-t0:.1f}s)")

    # 매트릭스 출력
    print(f"\n[4] CAGR / MDD / Calmar 매트릭스")
    print("-" * 80)
    print(f"  Calmar 매트릭스:")
    print(f"    {'N\\K':>6}", " ".join(f"{k:>7}" for k in K_VALUES))
    for n in N_VALUES:
        row = []
        for k in K_VALUES:
            cal = results[(n, k)]["calmar"]
            row.append(f"{cal:>7.2f}")
        print(f"    {n:>6}", " ".join(row))

    print(f"\n  CAGR 매트릭스 (%):")
    print(f"    {'N\\K':>6}", " ".join(f"{k:>7}" for k in K_VALUES))
    for n in N_VALUES:
        row = []
        for k in K_VALUES:
            c = results[(n, k)]["stats"].get("cagr", 0)
            row.append(f"{c:>+7.1f}")
        print(f"    {n:>6}", " ".join(row))

    print(f"\n  MDD 매트릭스 (%):")
    print(f"    {'N\\K':>6}", " ".join(f"{k:>7}" for k in K_VALUES))
    for n in N_VALUES:
        row = []
        for k in K_VALUES:
            m = results[(n, k)]["stats"].get("mdd", 0)
            row.append(f"{m:>+7.2f}")
        print(f"    {n:>6}", " ".join(row))

    # 최우수 찾기
    best_calmar = max(results.values(), key=lambda r: r["calmar"])
    best_cagr = max(results.values(), key=lambda r: r["stats"].get("cagr", 0))
    best_mdd = max(results.values(), key=lambda r: r["stats"].get("mdd", -999))

    print(f"\n  Calmar 최고: N={best_calmar['n']}/{best_calmar['n']} K={best_calmar['k']} → "
          f"Calmar {best_calmar['calmar']:.2f}, "
          f"CAGR {best_calmar['stats']['cagr']:+.1f}%, MDD {best_calmar['stats']['mdd']:+.1f}%")
    print(f"  CAGR 최고: N={best_cagr['n']}/{best_cagr['n']} K={best_cagr['k']} → "
          f"CAGR {best_cagr['stats']['cagr']:+.1f}%, "
          f"MDD {best_cagr['stats']['mdd']:+.1f}%, Calmar {best_cagr['calmar']:.2f}")
    print(f"  MDD 최저: N={best_mdd['n']}/{best_mdd['n']} K={best_mdd['k']} → "
          f"MDD {best_mdd['stats']['mdd']:+.1f}%, "
          f"CAGR {best_mdd['stats']['cagr']:+.1f}%, Calmar {best_mdd['calmar']:.2f}")

    # 리포트
    print(f"\n[5] 리포트 생성...")
    lines = ["# N×K 매트릭스 백테스트 (KOSPI200, 1500억 N/N + K슬롯 + LIST_EXIT)\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **매수**: 최근 N일 amount >= 1500억 N/N 만족 (무제한 후보)")
    lines.append("- **자본**: 진짜 K슬롯 모델 (cap=K, 자본 1/K 동적)")
    lines.append("- **매도 (LIST_EXIT)**: 다음날 N/N 깨지면 다다음날 시가 매도")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%\n")
    lines.append("---\n")

    # Calmar 매트릭스
    lines.append("## Calmar 매트릭스\n")
    header = "| N\\K | " + " | ".join(f"K={k}" for k in K_VALUES) + " |"
    sep = "|---:|" + "---:|" * len(K_VALUES)
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            cal = results[(n, k)]["calmar"]
            mark = " ⭐" if (n, k) == (best_calmar["n"], best_calmar["k"]) else ""
            row += f" {cal:.2f}{mark} |"
        lines.append(row)

    # CAGR 매트릭스
    lines.append("\n## CAGR 매트릭스 (%)\n")
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            c = results[(n, k)]["stats"].get("cagr", 0)
            row += f" {c:+.1f} |"
        lines.append(row)

    # MDD 매트릭스
    lines.append("\n## MDD 매트릭스 (%)\n")
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            m = results[(n, k)]["stats"].get("mdd", 0)
            row += f" {m:+.2f} |"
        lines.append(row)

    # 거래수 매트릭스
    lines.append("\n## 거래수 매트릭스\n")
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            t = results[(n, k)]["stats"].get("total", 0)
            row += f" {t} |"
        lines.append(row)

    # 미체결 매트릭스
    lines.append("\n## 미체결 매트릭스 (cap 때문에 매수 못한 신호)\n")
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            miss = results[(n, k)]["stats"].get("missed", 0)
            row += f" {miss} |"
        lines.append(row)

    # Top 10 조합
    lines.append("\n## Calmar Top 10 조합\n")
    lines.append("| 순위 | N | K | 거래수 | CAGR(%) | MDD(%) | Calmar | 자본 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    sorted_results = sorted(results.values(), key=lambda r: r["calmar"], reverse=True)[:10]
    for i, r in enumerate(sorted_results, 1):
        s = r["stats"]
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| {i} | {r['n']} | {r['k']} | {s['total']:,} | {cagr:+.2f} | {mdd:+.2f} | "
            f"{r['calmar']:.2f} | {s.get('final_equity', 1):.2f}x |"
        )

    # 패턴 분석: N=K 대각선 vs 오프-대각선
    lines.append("\n## 패턴 분석: N≈K 매칭 효과\n")
    lines.append("| N | K | Calmar | 비교 |")
    lines.append("|---:|---:|---:|---|")
    for n in N_VALUES:
        for k in K_VALUES:
            if n == k:
                lines.append(f"| **{n}** | **{k}** | **{results[(n,k)]['calmar']:.2f}** | N=K 대각선 |")
    lines.append("\n### 각 N에 대한 최적 K\n")
    lines.append("| N | 최적 K | Calmar | 2nd K | Calmar |")
    lines.append("|---:|---:|---:|---:|---:|")
    for n in N_VALUES:
        row_results = [(k, results[(n, k)]["calmar"]) for k in K_VALUES]
        row_results.sort(key=lambda x: x[1], reverse=True)
        best_k, best_cal = row_results[0]
        second_k, second_cal = row_results[1]
        lines.append(f"| {n} | **{best_k}** | {best_cal:.2f} | {second_k} | {second_cal:.2f} |")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
