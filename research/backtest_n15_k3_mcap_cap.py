"""(N=15, K=3) + 시총 캡 사전 제외 매트릭스 백테스트.

5개 cap 임계 (∞, 50조, 30조, 20조, 10조) 각각에서
1500억 15/15 신호 + 시총 캡 필터 + LIST_EXIT 매도 + K=3 cap 결과 비교.

시총 거대주(삼성전자 등) 사전 제외 시 (15,3) 운영 안정성 변화 정량 검증.
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402
from backtest_5d_amount_filter import compute_5d_filter_signals  # noqa: E402
from backtest_5d_realistic_k import equity_real_k  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_n15_k3_mcap_cap.md"

THRESHOLD_WON = 150_000_000_000  # 1500억
N = 15
K = 3

# 시총 캡 시나리오 (원 단위)
CAP_SCENARIOS = [
    ("∞ (제외 없음)", float("inf")),
    ("50조", 50_000_000_000_000),
    ("30조", 30_000_000_000_000),
    ("20조", 20_000_000_000_000),
    ("10조", 10_000_000_000_000),
]


# ──────────────────────────────────────────────
# 헬퍼 (단위 테스트 대상)
# ──────────────────────────────────────────────
def apply_mcap_cap_filter(signals: dict[pd.Timestamp, list[str]],
                          daily_data: dict[str, pd.DataFrame],
                          cap_won: float) -> dict[pd.Timestamp, list[str]]:
    """signals 의 각 (date, ticker) 에서 mcap ≤ cap_won 종목만 통과."""
    if not signals:
        return {}

    cache = {}
    for ticker, df in daily_data.items():
        if "date" in df.columns:
            cache[ticker] = df.set_index("date")
        else:
            cache[ticker] = df

    out = {}
    for d, tickers in signals.items():
        passed = []
        for t in tickers:
            df = cache.get(t)
            if df is None or d not in df.index:
                continue
            mcap = df.loc[d].get("mcap", np.nan)
            if pd.isna(mcap):
                continue
            if mcap <= cap_won:
                passed.append(t)
        if passed:
            out[d] = passed
    return out


# ──────────────────────────────────────────────
# 분석 헬퍼
# ──────────────────────────────────────────────
def trade_summary(trades: list[dict]) -> dict:
    if not trades:
        return {"total": 0, "win_rate": 0, "avg_ret": 0, "max_loss": 0,
                "max_gain": 0, "avg_hold": 0, "unique": 0,
                "samsung_count": 0, "samsung_pct": 0,
                "most_t": "-", "most_n": 0}
    df = pd.DataFrame(trades)
    counts = Counter(t["ticker"] for t in trades)
    most_t, most_n = max(counts.items(), key=lambda x: x[1])
    samsung_n = counts.get("005930", 0)
    return {
        "total": len(df),
        "win_rate": (df["net_ret"] > 0).sum() / len(df) * 100,
        "avg_ret": df["net_ret"].mean() * 100,
        "max_loss": df["net_ret"].min() * 100,
        "max_gain": df["net_ret"].max() * 100,
        "avg_hold": df["hold_days"].mean(),
        "unique": len(set(t["ticker"] for t in trades)),
        "samsung_count": samsung_n,
        "samsung_pct": samsung_n / len(df) * 100 if df.size else 0,
        "most_t": most_t, "most_n": most_n,
    }


def run_one(daily_data, panel, signals) -> tuple[list[dict], dict]:
    trades, _ = bt.run_backtest(
        daily_data, panel, signals,
        rule="LIST_EXIT", slots=K, max_concurrent=K,
    )
    eq = equity_real_k(trades, K=K)
    cagr = eq.get("cagr", 0)
    mdd = eq.get("mdd", 0)
    eq["calmar"] = abs(cagr / mdd) if mdd != 0 else 0
    eq["total"] = len(trades)
    return trades, eq


def main():
    print("=" * 60)
    print("(N=15, K=3) + 시총 캡 사전 제외 매트릭스")
    print("=" * 60)

    print("[1] 데이터 로드...")
    k200 = bt.load_kospi200_tickers()
    snapshot = bt.load_market_cap_snapshot()
    price_df = bt.load_price_data(k200["ticker"].tolist())
    daily_data = bt.build_daily_data(price_df, snapshot)
    panel = bt.build_daily_indicator_panel(daily_data, "amount")
    print(f"    {len(daily_data)} 종목")

    # 시총 분포 확인 (참고용)
    print("\n[1-b] 시총 분포 (오늘 기준):")
    today_mcaps = []
    for ticker, df in daily_data.items():
        if not df.empty:
            last_mcap = df["mcap"].iloc[-1]
            if not pd.isna(last_mcap):
                today_mcaps.append((ticker, last_mcap))
    today_mcaps.sort(key=lambda x: -x[1])
    print(f"    Top 5 시총: " + ", ".join(
        f"{t} ({m/1e12:.1f}조)" for t, m in today_mcaps[:5]
    ))

    print("\n[2] 1500억 N=15 신호 생성...")
    raw = compute_5d_filter_signals(
        daily_data, threshold_won=THRESHOLD_WON, lookback=N, top_k=200,
    )
    raw_per = [len(v) for v in raw.values()]
    print(f"    {len(raw)}일치, 평균 {np.mean(raw_per) if raw_per else 0:.2f} 종목/일")

    # 5 cap 시나리오
    results = {}
    for label, cap in CAP_SCENARIOS:
        print(f"\n[3] cap={label} 시나리오...")
        if cap == float("inf"):
            filtered = raw  # 베이스
        else:
            filtered = apply_mcap_cap_filter(raw, daily_data, cap)
        fil_per = [len(v) for v in filtered.values()]
        avg = np.mean(fil_per) if fil_per else 0
        red = (1 - sum(fil_per) / max(sum(raw_per), 1)) * 100 if cap != float("inf") else 0
        print(f"    필터 후 {len(filtered)}일치, 평균 {avg:.2f} 종목/일 (감소율 {red:.1f}%)")

        t0 = time.time()
        trades, stats = run_one(daily_data, panel, filtered)
        ts = trade_summary(trades)
        results[label] = {
            "cap": cap, "trades": trades, "stats": stats, "ts": ts,
            "n_signal_days": len(filtered), "avg_per_day": avg,
            "reduction_pct": red,
        }
        print(f"    거래수 {stats['total']}, CAGR {stats['cagr']:+.2f}%, "
              f"MDD {stats['mdd']:+.2f}%, Calmar {stats['calmar']:.2f}, "
              f"005930 매수 {ts['samsung_count']}회 ({time.time()-t0:.1f}s)")

    # 리포트
    print("\n[4] 리포트 생성...")
    lines = ["# (N=15, K=3) + 시총 캡 사전 제외 매트릭스\n"]

    lines.append("## 1. 매매 규칙\n")
    lines.append("- **N/N**: 최근 15일 거래대금 ≥ 1500억 인 날이 15일 모두")
    lines.append("- **시총 캡 필터**: 매수 후보 종목 중 mcap ≤ cap_won 만 유지")
    lines.append("- **매수**: 통과 종목 amount 상위 K=3 → 다음날 시가 매수")
    lines.append("- **매도**: LIST_EXIT (1500억 15/15 깨지면 다다음날 시가 매도)")
    lines.append("- **자본**: 진짜 K슬롯 (K=3)")
    lines.append("- **시총 정의**: close × snapshot shares_outstanding\n")

    # 시총 분포
    lines.append("## 2. 시총 Top 5 (오늘 기준)\n")
    lines.append("| # | ticker | 시총(조원) |")
    lines.append("| ---: | :---: | ---: |")
    for i, (t, m) in enumerate(today_mcaps[:5], 1):
        lines.append(f"| {i} | {t} | {m/1e12:.1f} |")

    # 핵심 비교
    lines.append("\n## 3. cap별 핵심 비교\n")
    lines.append("| cap 임계 | 신호 일수 | 평균 종목/일 | 감소율 | 거래수 | 005930 매수 | CAGR(%) | MDD(%) | Calmar | 자본(x) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, _ in CAP_SCENARIOS:
        r = results[label]
        s = r["stats"]
        ts = r["ts"]
        lines.append(
            f"| **{label}** | {r['n_signal_days']} | {r['avg_per_day']:.2f} | "
            f"{r['reduction_pct']:.1f}% | {s['total']} | "
            f"{ts['samsung_count']} ({ts['samsung_pct']:.1f}%) | "
            f"{s['cagr']:+.2f} | {s['mdd']:+.2f} | "
            f"**{s['calmar']:.2f}** | {s['final_equity']:.2f} |"
        )

    # 거래 품질
    lines.append("\n## 4. 거래 품질 비교\n")
    lines.append("| 지표 | " + " | ".join(f"**{lbl}**" for lbl, _ in CAP_SCENARIOS) + " |")
    lines.append("| --- | " + " | ".join([":---:"] * len(CAP_SCENARIOS)) + " |")
    metric_rows = [
        ("거래수", lambda ts: f"{ts['total']}"),
        ("승률(%)", lambda ts: f"{ts['win_rate']:.1f}"),
        ("평균 수익률(%)", lambda ts: f"{ts['avg_ret']:+.2f}"),
        ("최대 단일 손실(%)", lambda ts: f"{ts['max_loss']:+.2f}"),
        ("최대 단일 이익(%)", lambda ts: f"{ts['max_gain']:+.2f}"),
        ("평균 보유일", lambda ts: f"{ts['avg_hold']:.1f}"),
        ("고유 종목 수", lambda ts: f"{ts['unique']}"),
        ("최다 거래 종목", lambda ts: f"{ts['most_t']} ({ts['most_n']}회)"),
        ("005930 매수 횟수", lambda ts: f"{ts['samsung_count']}"),
    ]
    for label, fmt in metric_rows:
        cells = " | ".join(fmt(results[lbl]["ts"]) for lbl, _ in CAP_SCENARIOS)
        lines.append(f"| {label} | {cells} |")

    # 거래 상세 (각 cap)
    for label, _ in CAP_SCENARIOS:
        r = results[label]
        if not r["trades"]:
            continue
        lines.append(f"\n## 5. cap={label} 거래 상세\n")
        lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
        lines.append("| :---: | :---: | :---: | ---: | ---: |")
        for t in sorted(r["trades"], key=lambda x: pd.Timestamp(x["buy_date"])):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            mark = " ⭐" if t["ticker"] == "005930" else ""
            lines.append(
                f"| {t['ticker']}{mark} | {bd} | {sd} | "
                f"{t['hold_days']} | {t['net_ret']*100:+.2f} |"
            )

    # 종합 결론
    lines.append("\n---\n\n## 6. 종합 결론\n")

    # cap별 Calmar 정리
    lines.append("### 6.1 cap별 Calmar 변화\n")
    base_cal = results["∞ (제외 없음)"]["stats"]["calmar"]
    lines.append(f"- 베이스 (∞): **{base_cal:.2f}**")
    for label, _ in CAP_SCENARIOS:
        if label == "∞ (제외 없음)":
            continue
        c = results[label]["stats"]["calmar"]
        delta = c - base_cal
        delta_pct = (c - base_cal) / base_cal * 100 if base_cal != 0 else 0
        sign = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
        lines.append(f"- cap={label}: {c:.2f} (Δ {delta:+.2f}, {delta_pct:+.1f}%) {sign}")

    # 005930 매수 추적
    lines.append("\n### 6.2 005930 매수 빈도 변화\n")
    for label, _ in CAP_SCENARIOS:
        ts = results[label]["ts"]
        lines.append(f"- cap={label}: {ts['samsung_count']}회 ({ts['samsung_pct']:.1f}%)")

    # 최적 cap 선정
    valid_results = [(lbl, r) for lbl, r in results.items() if r["stats"]["total"] >= 5]
    if valid_results:
        best_lbl, best_r = max(valid_results, key=lambda x: x[1]["stats"]["calmar"])
        lines.append(f"\n### 6.3 최적 cap 임계\n")
        lines.append(f"- **{best_lbl}** — Calmar {best_r['stats']['calmar']:.2f} "
                     f"(CAGR {best_r['stats']['cagr']:+.2f}%, MDD {best_r['stats']['mdd']:+.2f}%, "
                     f"거래수 {best_r['stats']['total']})")

    # Trade-off 분석
    lines.append("\n### 6.4 Trade-off 분석\n")
    lines.append("- **수익성 vs 안정성**: cap을 낮출수록 거대주 제외 → 다양화↑, 단 큰 winner 잃을 위험")
    lines.append(f"- 005930 매수 변화 추적이 핵심: 베이스 1회 → cap별 변화")

    # 결론
    lines.append("\n### 6.5 채택 권고\n")
    if best_r["stats"]["calmar"] > base_cal * 0.95:
        lines.append(f"- **{best_lbl}** 채택 가능 — 베이스 대비 Calmar 손실 5% 이내")
    elif best_r["stats"]["calmar"] > base_cal * 0.5:
        lines.append(f"- **베이스 (∞) 유지 권고** — 모든 cap 시나리오가 베이스보다 열위")
    else:
        lines.append(f"- **베이스 (∞) 명백히 우월** — cap 제외는 수익성 큰 손해")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
