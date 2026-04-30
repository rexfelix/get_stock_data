"""3000억 거래대금 N/N + K슬롯 N×K 매트릭스 백테스트.

1500억 버전(`backtest_n_k_matrix.py`)과 동일한 구조로 임계치만 3000억으로 상향.

매수: 3000억 N/N
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

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_n_k_matrix_3000.md"

THRESHOLD_WON = 300_000_000_000  # 3000억원
N_VALUES = [3, 5, 7, 10, 15]
K_VALUES = [3, 5, 7, 10, 15]


# 1500억 매트릭스 결과 (results/backtest_n_k_matrix.md 출처, 비교용 baseline)
# 변경 시 1500억 리포트와 동기화 필요
BASELINE_1500 = {
    (3, 3):  {"calmar": 23.65, "cagr": +89.58, "mdd": -3.79, "total": 33,  "missed": 30},
    (3, 5):  {"calmar":  4.12, "cagr": +75.50, "mdd": -18.32, "total": 169, "missed": 48},
    (3, 7):  {"calmar":  2.81, "cagr": +68.20, "mdd": -24.26, "total": 294, "missed": 31},
    (3, 10): {"calmar":  2.75, "cagr": +53.20, "mdd": -19.35, "total": 382, "missed": 17},
    (3, 15): {"calmar":  2.97, "cagr": +41.20, "mdd": -13.87, "total": 463, "missed": 2},
    (5, 3):  {"calmar": 32.33, "cagr": +86.78, "mdd": -2.68, "total": 27,  "missed": 24},
    (5, 5):  {"calmar":  5.81, "cagr": +82.90, "mdd": -14.28, "total": 108, "missed": 8},
    (5, 7):  {"calmar":  4.16, "cagr": +69.20, "mdd": -16.63, "total": 159, "missed": 3},
    (5, 10): {"calmar":  4.49, "cagr": +54.00, "mdd": -12.03, "total": 197, "missed": 0},
    (5, 15): {"calmar":  5.08, "cagr": +41.10, "mdd": -8.10,  "total": 235, "missed": 1},
    (7, 3):  {"calmar": 40.29, "cagr": +85.93, "mdd": -2.13, "total": 26,  "missed": 23},
    (7, 5):  {"calmar":  5.12, "cagr": +75.40, "mdd": -14.72, "total": 81,  "missed": 4},
    (7, 7):  {"calmar":  4.68, "cagr": +63.80, "mdd": -13.63, "total": 107, "missed": 2},
    (7, 10): {"calmar":  5.23, "cagr": +52.00, "mdd": -9.93,  "total": 129, "missed": 1},
    (7, 15): {"calmar":  5.77, "cagr": +38.70, "mdd": -6.71,  "total": 159, "missed": 1},
    (10, 3): {"calmar": 39.18, "cagr": +85.99, "mdd": -2.19, "total": 25,  "missed": 22},
    (10, 5): {"calmar":  5.29, "cagr": +73.80, "mdd": -13.95, "total": 62,  "missed": 3},
    (10, 7): {"calmar":  6.26, "cagr": +60.90, "mdd": -9.73,  "total": 75,  "missed": 3},
    (10, 10):{"calmar":  7.18, "cagr": +49.42, "mdd": -6.88,  "total": 92,  "missed": 8},
    (10, 15):{"calmar":  6.51, "cagr": +36.06, "mdd": -5.54,  "total": 105, "missed": 2},
    (15, 3): {"calmar": 41.88, "cagr": +105.71, "mdd": -2.52, "total": 17,  "missed": 0},
    (15, 5): {"calmar":  8.58, "cagr": +72.63, "mdd": -8.47,  "total": 39,  "missed": 1},
    (15, 7): {"calmar":  6.30, "cagr": +59.00, "mdd": -9.37,  "total": 48,  "missed": 4},
    (15, 10):{"calmar":  8.66, "cagr": +48.95, "mdd": -5.65,  "total": 57,  "missed": 3},
    (15, 15):{"calmar":  9.97, "cagr": +35.56, "mdd": -3.57,  "total": 60,  "missed": 0},
}


def main():
    print("=" * 60)
    print(f"N×K 매트릭스 백테스트 (3000억 N/N + K슬롯 + LIST_EXIT)")
    print("=" * 60)

    print("[1] 데이터 로드...")
    k200 = bt.load_kospi200_tickers()
    snapshot = bt.load_market_cap_snapshot()
    price_df = bt.load_price_data(k200["ticker"].tolist())
    daily_data = bt.build_daily_data(price_df, snapshot)
    panel = bt.build_daily_indicator_panel(daily_data, "amount")
    print(f"    {len(daily_data)}종목 시계열")

    # N별 신호 사전 계산 (3000억 임계)
    print(f"\n[2] N별 신호 생성 (임계 = 3000억)...")
    signals_by_n = {}
    for n in N_VALUES:
        signals = compute_5d_filter_signals(
            daily_data, threshold_won=THRESHOLD_WON, lookback=n, top_k=200,
        )
        n_per_day = [len(v) for v in signals.values()]
        signals_by_n[n] = signals
        avg = np.mean(n_per_day) if n_per_day else 0
        mx = max(n_per_day) if n_per_day else 0
        print(f"  N={n:2d}: {len(signals)}일치, 평균 {avg:.2f}, max {mx}")

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
    valid_results = [r for r in results.values() if r["stats"].get("total", 0) > 0]
    if valid_results:
        best_calmar = max(valid_results, key=lambda r: r["calmar"])
        best_cagr = max(valid_results, key=lambda r: r["stats"].get("cagr", 0))
        best_mdd = max(valid_results, key=lambda r: r["stats"].get("mdd", -999))

        print(f"\n  Calmar 최고: N={best_calmar['n']} K={best_calmar['k']} → "
              f"Calmar {best_calmar['calmar']:.2f}, "
              f"CAGR {best_calmar['stats']['cagr']:+.1f}%, MDD {best_calmar['stats']['mdd']:+.1f}%")
        print(f"  CAGR 최고: N={best_cagr['n']} K={best_cagr['k']} → "
              f"CAGR {best_cagr['stats']['cagr']:+.1f}%, "
              f"MDD {best_cagr['stats']['mdd']:+.1f}%, Calmar {best_cagr['calmar']:.2f}")
        print(f"  MDD 최저: N={best_mdd['n']} K={best_mdd['k']} → "
              f"MDD {best_mdd['stats']['mdd']:+.1f}%, "
              f"CAGR {best_mdd['stats']['cagr']:+.1f}%, Calmar {best_mdd['calmar']:.2f}")
    else:
        best_calmar = None
        print("  유효 결과 없음 (모든 조합 거래수 0)")

    # 리포트
    print(f"\n[5] 리포트 생성...")
    lines = ["# N×K 매트릭스 백테스트 (KOSPI200, 3000억 N/N + K슬롯 + LIST_EXIT)\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **매수**: 최근 N일 amount >= **3000억** N/N 만족 (무제한 후보)")
    lines.append("- **자본**: 진짜 K슬롯 모델 (cap=K, 자본 1/K 동적)")
    lines.append("- **매도 (LIST_EXIT)**: 다음날 N/N 깨지면 다다음날 시가 매도")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%\n")
    lines.append("---\n")

    # 신호 통계
    lines.append("## N별 신호 통계 (3000억 임계)\n")
    lines.append("| N | 신호 발생 일수 | 평균 신호/일 | 최대 신호/일 |")
    lines.append("|---:|---:|---:|---:|")
    for n in N_VALUES:
        sig = signals_by_n[n]
        n_per_day = [len(v) for v in sig.values()]
        avg = np.mean(n_per_day) if n_per_day else 0
        mx = max(n_per_day) if n_per_day else 0
        lines.append(f"| **N={n}** | {len(sig)} | {avg:.2f} | {mx} |")

    # Calmar 매트릭스
    lines.append("\n## Calmar 매트릭스\n")
    header = "| N\\K | " + " | ".join(f"K={k}" for k in K_VALUES) + " |"
    sep = "|---:|" + "---:|" * len(K_VALUES)
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            cal = results[(n, k)]["calmar"]
            mark = " ⭐" if (best_calmar is not None
                             and (n, k) == (best_calmar["n"], best_calmar["k"])) else ""
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
    sorted_results = sorted(
        [r for r in results.values() if r["stats"].get("total", 0) > 0],
        key=lambda r: r["calmar"], reverse=True,
    )[:10]
    for i, r in enumerate(sorted_results, 1):
        s = r["stats"]
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| {i} | {r['n']} | {r['k']} | {s['total']:,} | {cagr:+.2f} | {mdd:+.2f} | "
            f"{r['calmar']:.2f} | {s.get('final_equity', 1):.2f}x |"
        )

    # N=K 대각선
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

    # 1500억 vs 3000억 비교 표
    lines.append("\n---\n")
    lines.append("## 1500억 vs 3000억 비교 (동일 (N,K))\n")
    lines.append("Δ = 3000억 값 − 1500억 값 (양수 = 3000억 우위 / 음수 = 1500억 우위).\n")

    # Calmar Δ
    lines.append("### Calmar Δ\n")
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            new = results[(n, k)]["calmar"]
            base = BASELINE_1500.get((n, k), {}).get("calmar", 0)
            d = new - base
            arrow = "🟢" if d > 0 else ("🔴" if d < 0 else "⚪")
            row += f" {new:.2f} ({d:+.2f}){arrow} |"
        lines.append(row)

    # CAGR Δ
    lines.append("\n### CAGR Δ (%)\n")
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            new = results[(n, k)]["stats"].get("cagr", 0)
            base = BASELINE_1500.get((n, k), {}).get("cagr", 0)
            d = new - base
            row += f" {new:+.1f} ({d:+.1f}) |"
        lines.append(row)

    # MDD Δ
    lines.append("\n### MDD Δ (%)\n")
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            new = results[(n, k)]["stats"].get("mdd", 0)
            base = BASELINE_1500.get((n, k), {}).get("mdd", 0)
            d = new - base  # MDD는 음수, d>0 이면 MDD 개선(덜 음수)
            row += f" {new:+.2f} ({d:+.2f}) |"
        lines.append(row)

    # 거래수 Δ
    lines.append("\n### 거래수 Δ\n")
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            new = results[(n, k)]["stats"].get("total", 0)
            base = BASELINE_1500.get((n, k), {}).get("total", 0)
            d = new - base
            row += f" {new} ({d:+d}) |"
        lines.append(row)

    # 미체결 Δ
    lines.append("\n### 미체결 Δ (3000억이 작을수록 cap 효율 ↑)\n")
    lines.append(header)
    lines.append(sep)
    for n in N_VALUES:
        row = f"| **N={n}** |"
        for k in K_VALUES:
            new = results[(n, k)]["stats"].get("missed", 0)
            base = BASELINE_1500.get((n, k), {}).get("missed", 0)
            d = new - base
            row += f" {new} ({d:+d}) |"
        lines.append(row)

    # 종합 고찰
    lines.append("\n---\n")
    lines.append("## 종합 고찰 (가설 검증)\n")

    # 가설 1: 신호 빈도 감소
    n5_avg_3000 = np.mean([len(v) for v in signals_by_n[5].values()]) if signals_by_n[5] else 0
    lines.append(f"- **가설 1 (신호 빈도 감소)**: N=5 기준 일별 평균 신호 수 = {n5_avg_3000:.2f} 종목 (3000억).")
    lines.append("  N=5 기준 1500억은 일별 평균 약 5종목 수준이었음 → **검증** (감소 확인 시 ✅).\n")

    # 가설 2: 미체결 감소
    miss_3000_total = sum(r["stats"].get("missed", 0) for r in results.values())
    miss_1500_total = sum(b.get("missed", 0) for b in BASELINE_1500.values())
    lines.append(f"- **가설 2 (미체결 감소)**: 전체 미체결 합계 1500억 {miss_1500_total} → 3000억 {miss_3000_total}.")
    if miss_3000_total < miss_1500_total:
        lines.append("  → ✅ 임계 상승으로 cap 압박이 줄어 효율 개선.\n")
    else:
        lines.append("  → 의외로 미체결이 줄지 않음 (신호 분포 변화 영향).\n")

    # 가설 3: N=15에서 신호 부족
    n15_total = sum(r["stats"].get("total", 0) for k_val in K_VALUES
                    for r in [results[(15, k_val)]])
    lines.append(f"- **가설 3 (N=15 신호 부족)**: N=15 전체 거래수 합 = {n15_total} 건 (3000억).")
    lines.append("  1500억 대비 큰 폭 감소 시 백테스트 의미 약화 위험.\n")

    # Calmar 우월 영역
    n_better_3000 = sum(
        1 for n in N_VALUES for k in K_VALUES
        if results[(n, k)]["calmar"] > BASELINE_1500.get((n, k), {}).get("calmar", 0)
    )
    lines.append(f"- **Calmar 우월 영역**: 25개 조합 중 {n_better_3000}/25 에서 3000억 임계가 1500억보다 Calmar 우위.")
    if n_better_3000 > 12:
        lines.append("  → 임계 상승이 전반적으로 위험조정수익 개선에 기여.")
    else:
        lines.append("  → 임계 상승이 위험조정수익을 일관되게 개선하지는 않음 (영역별 차이 큼).")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
