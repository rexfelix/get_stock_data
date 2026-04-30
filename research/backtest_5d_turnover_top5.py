"""
5일1500억 5/5 + 거래대금/시가총액 Top K 매수 + LIST_EXIT 매도.

매수 조건:
1. 최근 5일 amount >= 1500억 5/5 만족 종목 (1차 필터)
2. 그 중 turnover (= amount / mcap) 큰 순으로 Top K (2차 정렬)

자본 분배: 진짜 K슬롯 모델 (cap=K + 자본 1/K 동적)
매도: LIST_EXIT (다음날 1500억5/5+turnover Top K에서 빠지면 매도)
대상: KOSPI200, 2023~현재
"""
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402
from backtest_5d_realistic_k import equity_real_k  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_5d_turnover_top5.md"

THRESHOLD_WON = 150_000_000_000
LOOKBACK = 5
K_VALUES = [3, 5, 10]  # 자본 슬롯 (cap + 자본 분배)


def compute_turnover_topk_signals(daily_data: dict[str, pd.DataFrame],
                                   threshold_won: float = THRESHOLD_WON,
                                   lookback: int = LOOKBACK,
                                   top_k: int = 5) -> dict[pd.Timestamp, list[str]]:
    """1500억 5/5 1차 필터 + turnover 큰 순 top_k 2차 정렬."""
    rows = []
    for ticker, df in daily_data.items():
        sub = df[["date", "amount", "turnover", "mcap"]].copy()
        sub["above"] = (sub["amount"] >= threshold_won).astype(int)
        sub["above_count"] = sub["above"].rolling(lookback, min_periods=lookback).sum()
        sub["ticker"] = ticker
        rows.append(sub)
    full = pd.concat(rows, ignore_index=True)
    full = full.dropna(subset=["amount", "turnover", "mcap"])
    cond = (
        (full["above_count"] >= lookback)
        & (full["amount"] > 0)
        & (full["turnover"] > 0)
        & (full["mcap"] > 0)
    )
    full = full[cond]

    signals = {}
    for d, g in full.groupby("date"):
        # turnover 큰 순으로 top_k
        top = g.sort_values("turnover", ascending=False).head(top_k)
        signals[d] = top["ticker"].tolist()
    return signals


def main():
    print("=" * 60)
    print("5일1500억 5/5 + turnover Top5 + LIST_EXIT (진짜 K슬롯)")
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

    # 신호 생성: 1500억 5/5 + turnover Top5
    print("[5] 매수 신호 (1500억5/5 + turnover Top5)...")
    signals = compute_turnover_topk_signals(daily_data, top_k=5)
    n_signal_days = len(signals)
    n_total_signal = sum(len(v) for v in signals.values())
    n_unique_tickers = len(set(t for v in signals.values() for t in v))
    n_per_day = [len(v) for v in signals.values()]
    print(f"    {n_signal_days}일치, 총 신호 {n_total_signal}건, 고유 종목 {n_unique_tickers}개")
    print(f"    일별 신호 수: 평균 {np.mean(n_per_day):.2f}, max {max(n_per_day)}")

    # 일부 시그널 샘플 출력 (어떤 종목이 잡히는지 확인)
    sample_dates = sorted(signals.keys())[::100]
    print("\n  [샘플] 매수 신호 종목:")
    ticker_name = dict(zip(snapshot["ticker"], snapshot["name"]))
    for d in sample_dates[:5]:
        names = [f"{ticker_name.get(t, t)}({t})" for t in signals[d][:3]]
        print(f"    {d.strftime('%Y-%m-%d')}: {', '.join(names)}")

    panel = bt.build_daily_indicator_panel(daily_data, "amount")

    # 진짜 K슬롯 모델 백테스트
    print(f"\n[6] 진짜 K슬롯 모델 백테스트 (LIST_EXIT)")
    print("-" * 60)

    results = []
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
        elapsed = time.time() - t0

        cagr = stats.get("cagr", 0)
        mdd = stats.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        results.append({
            "k": k, "stats": stats, "calmar": calmar, "trades": trades,
            "yearly": bt.yearly_stats(trades),
        })
        print(f"  K={k:2d}: 거래 {stats.get('total',0):>4}건, "
              f"승률 {stats.get('win_rate',0):>5.1f}%, "
              f"CAGR {cagr:>+7.2f}%, MDD {mdd:>+6.2f}%, "
              f"Calmar {calmar:.2f}, 자본 {stats.get('final_equity',1):>5.2f}x, "
              f"miss {stats.get('missed',0)} | {elapsed:.1f}s")

    # 추가: 매도 규칙 비교 (HOLD_N, MA_INIT_STOP)
    print("\n[참고] 매도 규칙 비교 (K=5)")
    extra = []
    for rule, kwargs, label in [
        ("HOLD_N", {"hold_n": 5}, "HOLD_5"),
        ("HOLD_N", {"hold_n": 10}, "HOLD_10"),
        ("HOLD_N", {"hold_n": 20}, "HOLD_20"),
        ("MA_INIT_STOP", {"ma_period": 20, "stop_pct": -0.07}, "MA20_STOP-7"),
    ]:
        t0 = time.time()
        trades, _ = bt.run_backtest(
            daily_data, panel, signals,
            rule=rule, slots=5, max_concurrent=5, **kwargs,
        )
        eq = equity_real_k(trades, K=5)
        stats = bt.compute_stats(trades)
        stats.update(eq)
        elapsed = time.time() - t0
        cagr = stats.get("cagr", 0)
        mdd = stats.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        extra.append({
            "label": label, "stats": stats, "calmar": calmar, "trades": trades,
            "yearly": bt.yearly_stats(trades),
        })
        print(f"  {label:14s}: 거래 {stats.get('total',0):>4}건, "
              f"승률 {stats.get('win_rate',0):>5.1f}%, "
              f"CAGR {cagr:>+7.2f}%, MDD {mdd:>+6.2f}%, Calmar {calmar:.2f}, "
              f"자본 {stats.get('final_equity',1):>5.2f}x | {elapsed:.1f}s")

    # 리포트
    print("\n[7] 리포트 생성...")
    lines = ["# 5일1500억 5/5 + turnover Top5 + LIST_EXIT (진짜 K슬롯)\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **대상**: KOSPI200 199종목, 2023-01-01 ~ 현재")
    lines.append("- **매수 조건**:")
    lines.append("  1. 최근 5일 거래대금 1,500억원 이상 5/5 (1차 필터)")
    lines.append("  2. 그 중 거래대금/시가총액 비율 큰 순으로 Top 5 (2차 정렬)")
    lines.append("- **매수 슬롯 K**: 3 / 5 / 10 비교 (자본 1/K 동적 분배)")
    lines.append("- **매도 (LIST_EXIT)**: 다음날 매수 조건(1차+2차)에서 빠지면 → 다다음날 시가 매도")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%")
    lines.append(f"- **신호 통계**: {n_signal_days}일치, 일평균 {np.mean(n_per_day):.2f}종목, 고유 {n_unique_tickers}개\n")
    lines.append("---\n")

    # K 비교
    lines.append("## 진짜 K슬롯 모델 (LIST_EXIT)\n")
    lines.append("| K | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 자본 | 미체결 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| {r['k']} | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | {s['med_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | {cagr:+.2f} | {mdd:+.2f} | {r['calmar']:.2f} | "
            f"{s.get('final_equity', 1):.2f}x | {s.get('missed', 0)} |"
        )

    # 매도 규칙 비교 (K=5)
    lines.append("\n## 매도 규칙 비교 (K=5)\n")
    lines.append("| 매도 규칙 | 거래수 | 승률(%) | 평균(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 자본 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    r5 = next((r for r in results if r["k"] == 5), None)
    if r5:
        s = r5["stats"]
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| **LIST_EXIT** ⭐ | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | {cagr:+.2f} | {mdd:+.2f} | {r5['calmar']:.2f} | {s.get('final_equity', 1):.2f}x |"
        )
    for r in extra:
        s = r["stats"]
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| {r['label']} | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | {cagr:+.2f} | {mdd:+.2f} | {r['calmar']:.2f} | {s.get('final_equity', 1):.2f}x |"
        )

    # 정렬 기준 비교 (1500억5/5 + amount Top vs turnover Top vs 무제한)
    lines.append("\n## 매수 종목 선정 방식 비교 (K=5, LIST_EXIT, 진짜 모델)\n")
    lines.append("| 매수 조건 | 거래수 | CAGR(%) | MDD(%) | Calmar | 자본 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append("| 5일1500억 + amount Top5 (이전, 약식) | 276 | +124.38 | -25.08 | 4.96 | 8.13x |")
    lines.append("| 5일1500억 무제한 + 진짜 K=5 | 108 | +84.34 | -14.28 | 5.90 | 4.79x |")
    if r5:
        s = r5["stats"]
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| **5일1500억 + turnover Top5 + 진짜 K=5** | {s['total']:,} | "
            f"{cagr:+.2f} | {mdd:+.2f} | {r5['calmar']:.2f} | {s.get('final_equity', 1):.2f}x |"
        )

    # 연도별 (K=5)
    if r5:
        lines.append("\n## 연도별 비교 (K=5, LIST_EXIT)\n")
        lines.append("| 연도 | 거래수 | 승률(%) | 평균(%) | 손익비 | 누적자본(trade-level) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for y in sorted(r5["yearly"].keys()):
            ys = r5["yearly"][y]
            if not ys or ys.get("total", 0) == 0:
                lines.append(f"| {y} | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {y} | {ys['total']:,} | {ys['win_rate']:.1f} | {ys['avg_ret']:+.2f} | "
                f"{ys['pf']:.2f} | {ys['cum_return_x']:.2f} |"
            )

    # 최우수 상세
    all_candidates = results + extra
    valid = [r for r in all_candidates if r["stats"].get("total", 0) > 0]
    if valid:
        best = max(valid, key=lambda r: r.get("calmar", 0))
        bc = best
        label = f"K={bc['k']} + LIST_EXIT" if "k" in bc else f"K=5 + {bc['label']}"
        lines.append(f"\n---\n\n## 최우수(Calmar) 상세: {label}\n")
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
