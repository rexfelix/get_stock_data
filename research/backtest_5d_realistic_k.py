"""
5일1500억 5/5 매수 + LIST_EXIT 매도 - 진짜 K슬롯 모델.

진짜 K슬롯 = "동시 보유 최대 K개 + 매수 시 자본의 1/K 사용":
- simulate_strategy: max_concurrent=K (슬롯 다 차면 매수 거부)
- equity 시뮬: 동적 자본 분배 (매수 시점 자본/K로 매수, 매도 시 자본 회수)

기존 약식 모델(equity_curve_simulation)과 직접 비교.
"""
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402
from backtest_5d_amount_filter import compute_5d_filter_signals  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_5d_realistic_k.md"

K_VALUES = [3, 5, 10, 20]


def equity_real_k(trades: list[dict], K: int) -> dict:
    """진짜 K슬롯 모델 자본 시뮬레이션.

    - 매수 시: 매수 시점 총자본의 1/K 사용. 자본 부족하면 매수 누락.
    - 매도 시: capital_used × (1 + net_ret) 회수
    - simulate_strategy에서 이미 cap 적용된 trades 입력 받음
    """
    if not trades:
        return {"final_equity": 1.0, "cagr": 0.0, "mdd": 0.0, "max_concurrent": 0, "missed": 0}
    df = pd.DataFrame(trades).copy()
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df["sell_date"] = pd.to_datetime(df["sell_date"])
    df = df.sort_values("buy_date").reset_index(drop=True)

    all_dates = sorted(set(list(df["buy_date"]) + list(df["sell_date"])))
    free = 1.0
    positions = {}  # df index -> capital_used
    equity = []
    eq_dates = []
    max_concurrent = 0
    missed = 0

    for d in all_dates:
        # 1. 매도 처리
        sell_idx = df.index[df["sell_date"] == d].tolist()
        for idx in sell_idx:
            if idx in positions:
                cu = positions.pop(idx)
                free += cu * (1 + df.loc[idx, "net_ret"])

        # 2. 매수 처리 (자본 1/K 사용)
        buy_idx = df.index[df["buy_date"] == d].tolist()
        for idx in buy_idx:
            if idx in positions:
                continue
            total = free + sum(positions.values())
            per = total / K
            if free >= per - 1e-9:
                positions[idx] = per
                free -= per
            else:
                missed += 1

        max_concurrent = max(max_concurrent, len(positions))
        total = free + sum(positions.values())
        equity.append(total)
        eq_dates.append(d)

    eq_series = pd.Series(equity, index=pd.to_datetime(eq_dates))
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    mdd = dd.min() * 100

    if len(eq_series) >= 2:
        days = (eq_series.index[-1] - eq_series.index[0]).days
        years = days / 365.25
        cagr = (eq_series.iloc[-1] / eq_series.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    else:
        cagr = 0

    return {
        "final_equity": float(eq_series.iloc[-1]),
        "cagr": float(cagr * 100),
        "mdd": float(mdd),
        "max_concurrent": int(max_concurrent),
        "missed": int(missed),
    }


def main():
    print("=" * 60)
    print("5일1500억 5/5 + LIST_EXIT - 진짜 K슬롯 모델 (cap=K)")
    print("=" * 60)

    print("[1] KOSPI200 ticker 로드...")
    k200 = bt.load_kospi200_tickers()

    print("[2] 시가총액 snapshot 로드...")
    snapshot = bt.load_market_cap_snapshot()

    print("[3] 가격/거래대금 데이터 로드...")
    t0 = time.time()
    price_df = bt.load_price_data(k200["ticker"].tolist())
    print(f"    {len(price_df):,}행 ({time.time() - t0:.1f}초)")

    print("[4] daily_data 빌드...")
    daily_data = bt.build_daily_data(price_df, snapshot)

    print("[5] 매수 신호 (제한없음)...")
    signals = compute_5d_filter_signals(daily_data, top_k=200)
    n_per_day = [len(v) for v in signals.values()]
    print(f"    {len(signals)}일치, 일별 평균 {np.mean(n_per_day):.2f}개, max {max(n_per_day)}개")

    panel = bt.build_daily_indicator_panel(daily_data, "amount")

    # 약식 vs 진짜 K슬롯 비교
    print("\n[6] 비교 백테스트")
    print("-" * 60)

    results = []
    for k in K_VALUES:
        # 진짜 K슬롯: simulate_strategy에 max_concurrent=K
        t0 = time.time()
        trades_real, _ = bt.run_backtest(
            daily_data, panel, signals,
            rule="LIST_EXIT", slots=k,
            max_concurrent=k,
        )
        # 정확한 자본 분배로 다시 계산
        real_eq = equity_real_k(trades_real, K=k)
        real_stats = bt.compute_stats(trades_real)
        real_stats.update(real_eq)
        elapsed = time.time() - t0

        cagr = real_stats.get("cagr", 0)
        mdd = real_stats.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        results.append({
            "k": k, "model": "진짜",
            "stats": real_stats, "calmar": calmar, "trades": trades_real,
            "yearly": bt.yearly_stats(trades_real),
        })
        print(f"  K={k:2d} 진짜: 거래 {real_stats.get('total',0):>4}건, "
              f"CAGR {cagr:>+7.2f}%, MDD {mdd:>+6.2f}%, Calmar {calmar:.2f}, "
              f"자본 {real_stats.get('final_equity',1):>5.2f}x, "
              f"max보유 {real_stats.get('max_concurrent',0)}, "
              f"miss {real_stats.get('missed',0)} | {elapsed:.1f}s")

    print("\n[참고] 이전 약식 모델 (cap 없음 + 자본 1/K sequential) 결과")
    print("  K=5  약식: 거래  269건, CAGR +193.58%, MDD -27.00%, Calmar 7.17, 자본 16.32x ❌비현실적")
    print("  K=10 약식: 거래  269건, CAGR  +86.62%, MDD -14.38%, Calmar 6.03, 자본  5.04x")
    print("  K=20 약식: 거래  269건, CAGR  +40.92%, MDD  -7.42%, Calmar 5.52, 자본  2.43x")

    # 리포트
    print("\n[7] 리포트 생성...")
    lines = ["# 5일1500억 5/5 + LIST_EXIT - 진짜 K슬롯 모델\n"]
    lines.append("## 모델 정의 명확화\n")
    lines.append("**K = 자본 분배 단위 = 동시 보유 최대 종목 수**\n")
    lines.append("진짜 K슬롯 모델:")
    lines.append("- 동시 보유 **최대 K개** (cap 적용, 슬롯 차면 신규 매수 거부)")
    lines.append("- 매수 시 매수 시점 **총자본의 1/K** 투입 (동적)")
    lines.append("- 매도 시 capital_used × (1 + 수익률) 자본 회수\n")
    lines.append("이전 보고서의 \"약식 K=5\" 결과(CAGR 194%)는 sequential trade 처리 오류로 부풀려진 결과 — 비현실적.\n")
    lines.append("---\n")

    # 결과 표
    lines.append("## 진짜 K슬롯 모델 결과 (LIST_EXIT 매도)\n")
    lines.append("| K | 거래수 | 승률(%) | 평균(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 자본 | max보유 | miss |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| {r['k']} | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | {cagr:+.2f} | {mdd:+.2f} | {r['calmar']:.2f} | "
            f"{s.get('final_equity', 1):.2f}x | {s.get('max_concurrent', 0)} | {s.get('missed', 0)} |"
        )

    # 약식 vs 진짜 비교
    lines.append("\n## 약식 모델 vs 진짜 K슬롯 모델 (CAGR/MDD/Calmar)\n")
    lines.append("| K | 약식 CAGR | 약식 MDD | 약식 Calmar | **진짜 CAGR** | **진짜 MDD** | **진짜 Calmar** |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    prior_loose = {5: (193.58, -27.00, 7.17), 10: (86.62, -14.38, 6.03), 20: (40.92, -7.42, 5.52)}
    for r in results:
        k = r["k"]
        if k not in prior_loose:
            continue
        ac, am, acal = prior_loose[k]
        s = r["stats"]
        rc = s.get("cagr", 0)
        rm = s.get("mdd", 0)
        rcal = abs(rc / rm) if rm != 0 else 0
        lines.append(
            f"| {k} | {ac:+.2f} | {am:+.2f} | {acal:.2f} | "
            f"**{rc:+.2f}** | **{rm:+.2f}** | **{rcal:.2f}** |"
        )

    # 종합 순위 (모든 백테스트 통합)
    lines.append("\n## 종합 순위 (Calmar 기준, 진짜 모델만)\n")
    lines.append("| 순위 | 전략 | CAGR(%) | MDD(%) | Calmar | 자본 |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    final_ranking = []
    for r in results:
        cagr = r["stats"].get("cagr", 0)
        mdd = r["stats"].get("mdd", 0)
        cal = r["calmar"]
        final_ranking.append((cal, f"5일1500억 + 진짜 K={r['k']} + LIST_EXIT", cagr, mdd, r["stats"].get("final_equity", 1)))
    final_ranking.sort(reverse=True)
    for i, (cal, name, cagr, mdd, fin) in enumerate(final_ranking, 1):
        lines.append(f"| {i} | {name} | {cagr:+.2f} | {mdd:+.2f} | {cal:.2f} | {fin:.2f}x |")

    # 연도별 (K=5)
    k5 = next((r for r in results if r["k"] == 5), None)
    if k5:
        lines.append("\n## 연도별 비교 (K=5 진짜 모델)\n")
        lines.append("| 연도 | 거래수 | 승률(%) | 평균(%) | 손익비 | 누적자본(trade-level) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for y in sorted(k5["yearly"].keys()):
            ys = k5["yearly"][y]
            if not ys or ys.get("total", 0) == 0:
                lines.append(f"| {y} | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {y} | {ys['total']:,} | {ys['win_rate']:.1f} | {ys['avg_ret']:+.2f} | "
                f"{ys['pf']:.2f} | {ys['cum_return_x']:.2f} |"
            )

    # 최우수 상세
    valid = [r for r in results if r["stats"].get("total", 0) > 0]
    if valid:
        best = max(valid, key=lambda r: r["calmar"])
        bc = best
        lines.append(f"\n---\n\n## 최우수(Calmar) 상세: 진짜 K={bc['k']}\n")
        s = bc["stats"]
        lines.append("| 지표 | 값 |")
        lines.append("|---|---:|")
        lines.append(f"| 총 거래수 | {s['total']:,} |")
        lines.append(f"| 승률(%) | {s['win_rate']:.1f} |")
        lines.append(f"| 평균 수익률(%) | {s['avg_ret']:+.2f} |")
        lines.append(f"| 중간값 수익률(%) | {s['med_ret']:+.2f} |")
        lines.append(f"| 평균이익(%) | {s['avg_win']:+.2f} |")
        lines.append(f"| 평균손실(%) | {s['avg_loss']:+.2f} |")
        lines.append(f"| 손익비 | {s['pf']:.2f} |")
        lines.append(f"| 평균 보유일 | {s['avg_hold']:.1f} |")
        lines.append(f"| CAGR(%) | {s['cagr']:+.2f} |")
        lines.append(f"| MDD(%) | {s['mdd']:+.2f} |")
        lines.append(f"| Calmar | {bc['calmar']:.2f} |")
        lines.append(f"| 최종 자본(x) | {s['final_equity']:.2f} |")
        lines.append(f"| 최대 동시보유 | {s.get('max_concurrent', 0)} |")
        lines.append(f"| cap으로 인한 미체결 | {s.get('missed', 0)} |")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
