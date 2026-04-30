"""
5일 1500억 필터 매수 + Top K (K=3 vs K=5) 비교 백테스트.

매수 조건:
- 최근 5일 amount >= 1500억 5/5 만족 종목
- amount 큰 순으로 최대 K종목 → T+1 시가 매수 (자본 1/K 슬롯)

매도 규칙: LIST_EXIT, MA_INIT_STOP MA20-7%, HOLD_5/10/20 (대표 5종)
대상: KOSPI200, 2023~현재
"""
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402
from backtest_5d_amount_filter import compute_5d_filter_signals  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_5d_amount_topk.md"

K_VALUES = [3, 5]
RULES = [
    ("LIST_EXIT", {}),
    ("MA_INIT_STOP", {"ma_period": 20, "stop_pct": -0.07}),
    ("HOLD_N", {"hold_n": 5}),
    ("HOLD_N", {"hold_n": 10}),
    ("HOLD_N", {"hold_n": 20}),
]


def rule_label(rule: str, kwargs: dict) -> str:
    if rule == "MA_INIT_STOP":
        return f"MA{kwargs['ma_period']}_STOP{int(kwargs['stop_pct']*100)}"
    if rule == "HOLD_N":
        return f"HOLD_{kwargs['hold_n']}"
    return rule


def main():
    print("=" * 60)
    print("5일 1500억 필터 + Top K 비교 백테스트 (K=3 vs K=5)")
    print("=" * 60)

    print("[1] KOSPI200 ticker 로드...")
    k200 = bt.load_kospi200_tickers()
    print(f"    {len(k200)}종목")

    print("[2] 시가총액 snapshot 로드...")
    snapshot = bt.load_market_cap_snapshot()

    print("[3] 가격/거래대금 데이터 로드...")
    t0 = time.time()
    price_df = bt.load_price_data(k200["ticker"].tolist())
    print(f"    {len(price_df):,}행 ({time.time() - t0:.1f}초)")

    print("[4] daily_data 빌드...")
    daily_data = bt.build_daily_data(price_df, snapshot)
    print(f"    {len(daily_data)}종목")

    panel = bt.build_daily_indicator_panel(daily_data, "amount")

    results = []
    for k in K_VALUES:
        print(f"\n[5] K={k} 신호 생성...")
        signals = compute_5d_filter_signals(daily_data, top_k=k)
        n_signal_days = len(signals)
        n_total_signal = sum(len(v) for v in signals.values())
        n_unique_tickers = len(set(t for v in signals.values() for t in v))
        print(f"    {n_signal_days}일치, 총 신호 {n_total_signal}건, 고유 종목 {n_unique_tickers}개")

        for rule, kwargs in RULES:
            t0 = time.time()
            trades, stats = bt.run_backtest(
                daily_data, panel, signals,
                rule=rule, slots=k, **kwargs,
            )
            elapsed = time.time() - t0
            yr = bt.yearly_stats(trades)
            label = rule_label(rule, kwargs)
            results.append({
                "k": k, "rule_label": label, "rule": rule, "kwargs": kwargs,
                "stats": stats, "yearly": yr, "trades": trades,
            })
            cagr = stats.get("cagr", 0)
            mdd = stats.get("mdd", 0)
            calmar = abs(cagr / mdd) if mdd != 0 else 0
            print(f"  {label:14s} | K={k}: {stats.get('total',0):>5}건, "
                  f"승률 {stats.get('win_rate',0):>5.1f}%, "
                  f"CAGR {cagr:>+8.2f}%, MDD {mdd:>+7.2f}%, "
                  f"Calmar {calmar:.2f}, 자본 {stats.get('final_equity',1):>6.2f}x | {elapsed:.1f}s")

    # 리포트
    print("\n[6] 리포트 생성...")
    lines = ["# 5일 1500억 필터 매수 + Top K 비교 백테스트 (KOSPI200)\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **대상**: KOSPI200 199종목, 2023-01-01 ~ 현재")
    lines.append("- **매수 조건**: 최근 5일 거래대금 1,500억원 이상이 5일 모두")
    lines.append("- **매수 슬롯 K**: 3 또는 5 (조건 만족 종목 중 amount 큰 순으로 K개)")
    lines.append("- **자본 분배**: 자본 1/K 균등")
    lines.append("- **매도 규칙**: LIST_EXIT, MA_INIT_STOP(MA20+손절-7%), HOLD_5/10/20")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%\n")
    lines.append("---\n")

    # 종합 비교 테이블 (K=3 vs K=5)
    lines.append("## K=3 vs K=5 비교 (전체 기간)\n")
    lines.append("| K | 매도 규칙 | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 누적자본 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        lines.append(
            f"| {r['k']} | {r['rule_label']} | "
            f"{s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | {s['med_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | "
            f"{cagr:+.2f} | {mdd:+.2f} | {calmar:.2f} | {s.get('final_equity', 1):.2f} |"
        )

    # 매도 규칙별 K=3 vs K=5 직접 비교
    lines.append("\n## 매도 규칙별 K=3 vs K=5 직접 비교\n")
    lines.append("| 매도 규칙 | K=3 CAGR | K=3 MDD | K=3 Calmar | K=5 CAGR | K=5 MDD | K=5 Calmar | Δ Calmar |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    rule_labels = sorted(set(r["rule_label"] for r in results))
    for label in rule_labels:
        k3 = next((r for r in results if r["k"] == 3 and r["rule_label"] == label), None)
        k5 = next((r for r in results if r["k"] == 5 and r["rule_label"] == label), None)
        if not k3 or not k5:
            continue
        c3, m3 = k3["stats"].get("cagr", 0), k3["stats"].get("mdd", 0)
        c5, m5 = k5["stats"].get("cagr", 0), k5["stats"].get("mdd", 0)
        cal3 = abs(c3 / m3) if m3 != 0 else 0
        cal5 = abs(c5 / m5) if m5 != 0 else 0
        delta = cal5 - cal3
        marker = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
        lines.append(
            f"| {label} | {c3:+.2f} | {m3:+.2f} | {cal3:.2f} | {c5:+.2f} | {m5:+.2f} | {cal5:.2f} | {marker} {delta:+.2f} |"
        )

    # 연도별 (K=5만)
    k5_results = [r for r in results if r["k"] == 5]
    all_years = set()
    for r in k5_results:
        all_years.update(r.get("yearly", {}).keys())
    for y in sorted(all_years):
        lines.append(f"\n## {y}년 비교 (K=5)\n")
        lines.append("| 매도 규칙 | 거래수 | 승률(%) | 평균(%) | 손익비 | 누적자본 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in k5_results:
            ys = r.get("yearly", {}).get(y, {})
            if not ys or ys.get("total", 0) == 0:
                lines.append(f"| {r['rule_label']} | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {r['rule_label']} | "
                f"{ys['total']:,} | {ys['win_rate']:.1f} | {ys['avg_ret']:+.2f} | "
                f"{ys['pf']:.2f} | {ys['cum_return_x']:.2f} |"
            )

    # 최우수 (Calmar)
    valid = [r for r in results if r["stats"].get("total", 0) > 0]
    if valid:
        for r in valid:
            cagr = r["stats"].get("cagr", 0)
            mdd = r["stats"].get("mdd", 0)
            r["calmar"] = abs(cagr / mdd) if mdd != 0 else 0
        best = max(valid, key=lambda r: r.get("calmar", 0))
        bc = best
        lines.append(f"\n---\n\n## 최우수(Calmar) 상세: K={bc['k']} + {bc['rule_label']}\n")
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

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
