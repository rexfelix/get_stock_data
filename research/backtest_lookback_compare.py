"""
1500억 N/N 조건 비교 (N=5/7/10).

매수: 최근 N일 amount >= 1500억 모두 N/N 만족 (무제한 매수)
자본: 진짜 K=5 슬롯 (cap=5 + 자본 1/5 동적)
매도: LIST_EXIT
대상: KOSPI200, 2023~현재
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

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_lookback_compare.md"

K_SLOTS = 5  # 자본 슬롯 (진짜 모델)
LOOKBACKS = [5, 7, 10]


def main():
    print("=" * 60)
    print("1500억 N/N 조건 비교 (N=5/7/10) - 진짜 K=5 슬롯")
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

    panel = bt.build_daily_indicator_panel(daily_data, "amount")

    print(f"\n[5] N/N 비교 백테스트 (K={K_SLOTS} 슬롯, LIST_EXIT)")
    print("-" * 60)

    results = []
    for n in LOOKBACKS:
        # 신호 생성
        signals = compute_5d_filter_signals(
            daily_data, lookback=n, top_k=200,  # 무제한
        )
        n_per_day = [len(v) for v in signals.values()]
        n_total = sum(n_per_day)
        n_unique = len(set(t for v in signals.values() for t in v))

        t0 = time.time()
        trades, _ = bt.run_backtest(
            daily_data, panel, signals,
            rule="LIST_EXIT", slots=K_SLOTS,
            max_concurrent=K_SLOTS,
        )
        eq = equity_real_k(trades, K=K_SLOTS)
        stats = bt.compute_stats(trades)
        stats.update(eq)
        elapsed = time.time() - t0

        cagr = stats.get("cagr", 0)
        mdd = stats.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        results.append({
            "n": n, "stats": stats, "calmar": calmar, "trades": trades,
            "yearly": bt.yearly_stats(trades),
            "n_signal_days": len(signals),
            "n_total_signal": n_total,
            "n_unique": n_unique,
            "n_per_day_avg": np.mean(n_per_day) if n_per_day else 0,
            "n_per_day_max": max(n_per_day) if n_per_day else 0,
        })

        print(f"  N={n:2d}/{n}: 신호 {len(signals)}일/{n_total}건/{n_unique}종목, "
              f"평균 {np.mean(n_per_day):.2f}, max {max(n_per_day)} | "
              f"거래 {stats.get('total',0):>4}건, "
              f"CAGR {cagr:>+7.2f}%, MDD {mdd:>+6.2f}%, Calmar {calmar:.2f}, "
              f"자본 {stats.get('final_equity',1):.2f}x, miss {stats.get('missed', 0)} | {elapsed:.1f}s")

    # 추가: K=10 슬롯도 비교
    print(f"\n[참고] K=10 슬롯 (방어형)")
    extra = []
    for n in LOOKBACKS:
        signals = compute_5d_filter_signals(daily_data, lookback=n, top_k=200)
        t0 = time.time()
        trades, _ = bt.run_backtest(
            daily_data, panel, signals,
            rule="LIST_EXIT", slots=10, max_concurrent=10,
        )
        eq = equity_real_k(trades, K=10)
        stats = bt.compute_stats(trades)
        stats.update(eq)
        elapsed = time.time() - t0
        cagr = stats.get("cagr", 0)
        mdd = stats.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        extra.append({
            "n": n, "k": 10, "stats": stats, "calmar": calmar, "trades": trades,
        })
        print(f"  N={n:2d}/{n} K=10: 거래 {stats.get('total',0):>4}건, "
              f"CAGR {cagr:>+7.2f}%, MDD {mdd:>+6.2f}%, Calmar {calmar:.2f}, "
              f"자본 {stats.get('final_equity',1):.2f}x, miss {stats.get('missed', 0)} | {elapsed:.1f}s")

    # 리포트
    print("\n[6] 리포트 생성...")
    lines = ["# 1500억 N/N lookback 비교 백테스트 (KOSPI200)\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **대상**: KOSPI200 199종목, 2023-01-01 ~ 현재")
    lines.append("- **매수 조건**: 최근 N일 거래대금 1,500억원 이상이 N일 모두")
    lines.append("- **자본 분배**: 진짜 K슬롯 모델 (cap + 자본 1/K 동적)")
    lines.append("- **매도 (LIST_EXIT)**: 다음날 N/N 조건 깨지면 → 다다음날 시가 매도")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%\n")
    lines.append("---\n")

    # 신호 통계 비교
    lines.append("## N/N 조건별 신호 통계\n")
    lines.append("| N | 신호 발생일 | 총 신호 | 고유 종목 | 일평균 | max |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['n']} | {r['n_signal_days']} | {r['n_total_signal']} | "
            f"{r['n_unique']} | {r['n_per_day_avg']:.2f} | {r['n_per_day_max']} |"
        )

    # 결과 표 (K=5)
    lines.append(f"\n## 진짜 K=5 슬롯 + LIST_EXIT 결과\n")
    lines.append("| N | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 자본 | 미체결 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| {r['n']} | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | {s['med_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | {cagr:+.2f} | {mdd:+.2f} | {r['calmar']:.2f} | "
            f"{s.get('final_equity', 1):.2f}x | {s.get('missed', 0)} |"
        )

    # K=10 보조
    lines.append(f"\n## 진짜 K=10 슬롯 + LIST_EXIT (방어형 참고)\n")
    lines.append("| N | 거래수 | 승률(%) | 평균(%) | 손익비 | CAGR(%) | MDD(%) | Calmar | 자본 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in extra:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| {r['n']} | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | "
            f"{s['pf']:.2f} | {cagr:+.2f} | {mdd:+.2f} | {r['calmar']:.2f} | {s.get('final_equity', 1):.2f}x |"
        )

    # 연도별 (모든 N, K=5)
    all_years = set()
    for r in results:
        all_years.update(r.get("yearly", {}).keys())
    for y in sorted(all_years):
        lines.append(f"\n## {y}년 비교 (K=5)\n")
        lines.append("| N | 거래수 | 승률(%) | 평균(%) | 손익비 | 누적자본 |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in results:
            ys = r.get("yearly", {}).get(y, {})
            if not ys or ys.get("total", 0) == 0:
                lines.append(f"| {r['n']} | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {r['n']} | {ys['total']:,} | {ys['win_rate']:.1f} | {ys['avg_ret']:+.2f} | "
                f"{ys['pf']:.2f} | {ys['cum_return_x']:.2f} |"
            )

    # 최우수 상세
    valid = [r for r in results if r["stats"].get("total", 0) > 0]
    if valid:
        best = max(valid, key=lambda r: r["calmar"])
        bc = best
        lines.append(f"\n---\n\n## 최우수(Calmar) 상세: N={bc['n']}/{bc['n']} + K=5\n")
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
        lines.append(f"| CAGR(%) | {s.get('cagr', 0):+.2f} |")
        lines.append(f"| MDD(%) | {s.get('mdd', 0):+.2f} |")
        lines.append(f"| Calmar | {bc['calmar']:.2f} |")
        lines.append(f"| 최종 자본(x) | {s.get('final_equity', 1):.2f} |")

        ticker_name = dict(zip(snapshot["ticker"], snapshot["name"]))
        tdf = pd.DataFrame(bc["trades"])
        if not tdf.empty:
            tdf["return_pct"] = tdf["net_ret"] * 100
            tdf["name"] = tdf["ticker"].map(ticker_name).fillna(tdf["ticker"])
            lines.append("\n### 수익률 상위 거래 Top 20\n")
            lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
            lines.append("|---|---|---|---:|---:|")
            for _, r in tdf.nlargest(20, "return_pct").iterrows():
                lines.append(
                    f"| {r['name']}({r['ticker']}) | "
                    f"{pd.to_datetime(r['buy_date']).strftime('%Y-%m-%d')} | "
                    f"{pd.to_datetime(r['sell_date']).strftime('%Y-%m-%d')} | "
                    f"{r['hold_days']} | {r['return_pct']:+.1f} |"
                )
            lines.append("\n### 수익률 하위 거래 Bottom 20\n")
            lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
            lines.append("|---|---|---|---:|---:|")
            for _, r in tdf.nsmallest(20, "return_pct").iterrows():
                lines.append(
                    f"| {r['name']}({r['ticker']}) | "
                    f"{pd.to_datetime(r['buy_date']).strftime('%Y-%m-%d')} | "
                    f"{pd.to_datetime(r['sell_date']).strftime('%Y-%m-%d')} | "
                    f"{r['hold_days']} | {r['return_pct']:+.1f} |"
                )

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
