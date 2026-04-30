"""
5일 1500억 + MA5>MA20 + 이격도(5)>97 매수 조건 + LIST_EXIT 매도.

매수 조건 (모두 만족):
1. 최근 5일 amount >= 1500억 모두
2. MA5 > MA20 (정배열)
3. 종가 > MA5 × 0.97 (이격도(5) > 97)

매도: 위 조건 집합에서 빠지면 → 다음날 시가 매도 (LIST_EXIT 변형)

자본 분배: K=5, K=10 두 가지 비교
대상: KOSPI200, 2023~현재
"""
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_5d_strict_filter.md"

THRESHOLD_WON = 150_000_000_000
LOOKBACK = 5
DISPARITY_THRESHOLD = 97  # 이격도 % 기준 (close > MA5 × 0.97)
K_VALUES = [5, 10]


def compute_strict_signals(daily_data: dict[str, pd.DataFrame],
                           threshold_won: float = THRESHOLD_WON,
                           lookback: int = LOOKBACK,
                           disparity_threshold: float = DISPARITY_THRESHOLD,
                           cap_k: int = 100) -> dict[pd.Timestamp, list[str]]:
    """매수 조건 모두 만족 종목 리스트.

    1. 최근 lookback일 amount >= threshold_won 5/5
    2. MA5 > MA20
    3. close / MA5 × 100 > disparity_threshold
    """
    rows = []
    for ticker, df in daily_data.items():
        sub = df[["date", "amount", "close", "ma5", "ma20"]].copy()
        sub["above"] = (sub["amount"] >= threshold_won).astype(int)
        sub["above_count"] = sub["above"].rolling(lookback, min_periods=lookback).sum()
        sub["ticker"] = ticker
        rows.append(sub)
    full = pd.concat(rows, ignore_index=True)

    # MA5/MA20 NaN 제거
    full = full.dropna(subset=["ma5", "ma20", "close"])
    # 이격도(5) 계산
    full["disparity5"] = full["close"] / full["ma5"] * 100

    # 조건 적용
    cond = (
        (full["above_count"] >= lookback)
        & (full["ma5"] > full["ma20"])
        & (full["disparity5"] > disparity_threshold)
    )
    full = full[cond]

    signals = {}
    for d, g in full.groupby("date"):
        # cap_k 초과 시 amount 큰 순
        if len(g) > cap_k:
            g = g.sort_values("amount", ascending=False).head(cap_k)
        signals[d] = g["ticker"].tolist()
    return signals


def main():
    print("=" * 60)
    print("5일1500억 + MA5>MA20 + 이격도(5)>97 매수 + LIST_EXIT 매도")
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

    print("[5] 매수 신호 생성...")
    signals = compute_strict_signals(daily_data)
    n_signal_days = len(signals)
    n_total_signal = sum(len(v) for v in signals.values())
    n_unique_tickers = len(set(t for v in signals.values() for t in v))
    avg_signals_per_day = n_total_signal / n_signal_days if n_signal_days else 0
    print(f"    {n_signal_days}일치, 총 신호 {n_total_signal}건, 고유 종목 {n_unique_tickers}개")
    print(f"    일평균 신호 종목 수: {avg_signals_per_day:.2f}")

    panel = bt.build_daily_indicator_panel(daily_data, "amount")

    results = []
    for k in K_VALUES:
        t0 = time.time()
        trades, stats = bt.run_backtest(
            daily_data, panel, signals,
            rule="LIST_EXIT", slots=k,
        )
        elapsed = time.time() - t0
        yr = bt.yearly_stats(trades)
        cagr = stats.get("cagr", 0)
        mdd = stats.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        results.append({
            "k": k, "stats": stats, "yearly": yr, "trades": trades, "calmar": calmar,
        })
        print(f"  K={k:2d}: {stats.get('total',0):>5}건, "
              f"승률 {stats.get('win_rate',0):>5.1f}%, "
              f"CAGR {cagr:>+8.2f}%, MDD {mdd:>+7.2f}%, "
              f"Calmar {calmar:.2f}, 자본 {stats.get('final_equity',1):>6.2f}x | {elapsed:.1f}s")

    # 추가: 매도 규칙도 비교 (LIST_EXIT만 사용자가 지정했지만 비교용으로 HOLD/MA_INIT_STOP도)
    print("\n[참고] 매도 규칙 비교 (K=5)")
    extra = []
    for rule, kwargs, label in [
        ("HOLD_N", {"hold_n": 5}, "HOLD_5"),
        ("HOLD_N", {"hold_n": 10}, "HOLD_10"),
        ("HOLD_N", {"hold_n": 20}, "HOLD_20"),
        ("MA_INIT_STOP", {"ma_period": 20, "stop_pct": -0.07}, "MA20_STOP-7"),
    ]:
        t0 = time.time()
        trades, stats = bt.run_backtest(
            daily_data, panel, signals,
            rule=rule, slots=5, **kwargs,
        )
        elapsed = time.time() - t0
        yr = bt.yearly_stats(trades)
        cagr = stats.get("cagr", 0)
        mdd = stats.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        extra.append({
            "label": label, "stats": stats, "yearly": yr, "trades": trades, "calmar": calmar,
        })
        print(f"  {label:14s}: {stats.get('total',0):>5}건, "
              f"승률 {stats.get('win_rate',0):>5.1f}%, "
              f"CAGR {cagr:>+8.2f}%, MDD {mdd:>+7.2f}%, "
              f"Calmar {calmar:.2f}, 자본 {stats.get('final_equity',1):>6.2f}x | {elapsed:.1f}s")

    # 리포트
    print("\n[6] 리포트 생성...")
    lines = ["# 5일1500억 + 정배열 + 이격도 매수 조건 백테스트 (KOSPI200)\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **대상**: KOSPI200 199종목, 2023-01-01 ~ 현재")
    lines.append("- **매수 조건 (모두 만족)**:")
    lines.append("  1. 최근 5일 거래대금 1,500억원 이상 5/5")
    lines.append("  2. MA5 > MA20 (정배열)")
    lines.append("  3. 종가 / MA5 × 100 > 97 (이격도(5) > 97 = 종가가 MA5의 -3% 이내)")
    lines.append("- **매도 (LIST_EXIT)**: 위 매수 조건 중 하나라도 깨지면 → 다음날 시가 매도")
    lines.append("- **자본 분배**: K슬롯 균등 (K=5, K=10 비교)")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%")
    lines.append(f"- **신호 통계**: {n_signal_days}일치, 일평균 {avg_signals_per_day:.2f}종목, 고유 {n_unique_tickers}종목\n")
    lines.append("---\n")

    # K 비교 (LIST_EXIT)
    lines.append("## 매수 조건 + LIST_EXIT 매도 (K 비교)\n")
    lines.append("| K | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 누적자본 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| {r['k']} | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | {s['med_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | {cagr:+.2f} | {mdd:+.2f} | {r['calmar']:.2f} | {s.get('final_equity', 1):.2f} |"
        )

    # 매도 규칙 비교 (참고)
    lines.append("\n## 매도 규칙 비교 (K=5)\n")
    lines.append("| 매도 규칙 | 거래수 | 승률(%) | 평균(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 누적자본 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    # LIST_EXIT K=5 결과를 맨 위에
    r5 = next((r for r in results if r["k"] == 5), None)
    if r5:
        s = r5["stats"]
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| **LIST_EXIT** ⭐ | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | {cagr:+.2f} | {mdd:+.2f} | {r5['calmar']:.2f} | {s.get('final_equity', 1):.2f} |"
        )
    for r in extra:
        s = r["stats"]
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| {r['label']} | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | {cagr:+.2f} | {mdd:+.2f} | {r['calmar']:.2f} | {s.get('final_equity', 1):.2f} |"
        )

    # 직전 백테스트와 비교 (단순 5일1500억 vs 정배열+이격도 추가)
    lines.append("\n## 매수 조건 강화 효과 비교 (LIST_EXIT 매도 고정)\n")
    lines.append("| 매수 조건 | K | 거래수 | CAGR(%) | MDD(%) | Calmar | 자본 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append("| 5일 1500억 (단순) | 3 | 245 | +107.37 | -22.50 | 4.77 | 6.63x |")
    lines.append("| 5일 1500억 (단순) | 5 | 276 | +124.38 | -25.08 | 4.96 | 8.13x |")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| **5일 1500억 + MA5>MA20 + 이격도>97** | **{r['k']}** | {s['total']:,} | "
            f"{cagr:+.2f} | {mdd:+.2f} | {r['calmar']:.2f} | {s.get('final_equity', 1):.2f}x |"
        )

    # 연도별 (K=5 LIST_EXIT)
    if r5:
        lines.append("\n## 연도별 비교 (K=5, LIST_EXIT)\n")
        lines.append("| 연도 | 거래수 | 승률(%) | 평균(%) | 손익비 | 누적자본 |")
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

    # 최우수 상세 (Calmar 기준)
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
        lines.append(f"| CAGR(%) | {s['cagr']:+.2f} |")
        lines.append(f"| MDD(%) | {s['mdd']:+.2f} |")
        lines.append(f"| Calmar | {bc['calmar']:.2f} |")
        lines.append(f"| 최종 자본(x) | {s['final_equity']:.2f} |")

        tdf = pd.DataFrame(bc["trades"])
        if not tdf.empty:
            tdf["return_pct"] = tdf["net_ret"] * 100
            ticker_name = dict(zip(snapshot["ticker"], snapshot["name"]))
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
