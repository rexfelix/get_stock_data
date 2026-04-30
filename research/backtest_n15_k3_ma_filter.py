"""1500억 N=15 + MA 정배열 필터 + K=3 백테스트.

매수 조건:
- 최근 15일 거래대금 ≥ 1500억 인 날이 15일 모두 (N/N 필터)
- T일 종가 > T일 MA5 AND T일 MA5 > T일 MA20 (정배열 필터)
- 위 만족 종목 중 amount 상위 K=3 → 다음날 시가 매수

매도 규칙: LIST_EXIT vs MA5 이탈 비교
자본: 진짜 K슬롯 모델 (K=3)

베이스라인: (15,3) LIST_EXIT 필터 없음 — Calmar 41.88
"""
import os
import time
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402
from backtest_5d_amount_filter import compute_5d_filter_signals  # noqa: E402
from backtest_5d_realistic_k import equity_real_k  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_n15_k3_ma_filter.md"

THRESHOLD_WON = 150_000_000_000  # 1500억
N = 15
K = 3


# ──────────────────────────────────────────────
# 헬퍼 (단위 테스트 대상)
# ──────────────────────────────────────────────
def apply_ma_filter(signals: dict[pd.Timestamp, list[str]],
                    daily_data: dict[str, pd.DataFrame]) -> dict[pd.Timestamp, list[str]]:
    """signals 의 각 (date, ticker) 에 대해 close>MA5 AND MA5>MA20 만족 여부로 필터.

    - daily_data[ticker] 는 date 컬럼과 close/ma5/ma20 컬럼을 가진 DataFrame
    - 입력 signals 의 종목 순서를 보존 (amount 내림차순 유지)
    - MA NaN 또는 데이터 미존재 시 불통
    """
    if not signals:
        return {}

    # ticker별 date-인덱스 lookup 캐시
    cache = {}
    for ticker, df in daily_data.items():
        if "date" in df.columns:
            cache[ticker] = df.set_index("date")
        else:
            cache[ticker] = df  # 이미 인덱스가 date

    out = {}
    for d, tickers in signals.items():
        passed = []
        for t in tickers:
            df = cache.get(t)
            if df is None or d not in df.index:
                continue
            row = df.loc[d]
            close = row.get("close", np.nan)
            ma5 = row.get("ma5", np.nan)
            ma20 = row.get("ma20", np.nan)
            if pd.isna(close) or pd.isna(ma5) or pd.isna(ma20):
                continue
            if close > ma5 and ma5 > ma20:
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
                "max_gain": 0, "avg_hold": 0, "unique": 0, "most_t": "", "most_n": 0}
    df = pd.DataFrame(trades)
    counts = Counter(t["ticker"] for t in trades)
    most_t, most_n = max(counts.items(), key=lambda x: x[1])
    return {
        "total": len(df),
        "win_rate": (df["net_ret"] > 0).sum() / len(df) * 100,
        "avg_ret": df["net_ret"].mean() * 100,
        "max_loss": df["net_ret"].min() * 100,
        "max_gain": df["net_ret"].max() * 100,
        "avg_hold": df["hold_days"].mean(),
        "unique": len(set(t["ticker"] for t in trades)),
        "most_t": most_t,
        "most_n": most_n,
    }


def run_one(daily_data, panel, signals, rule: str) -> tuple[list[dict], dict]:
    trades, _ = bt.run_backtest(
        daily_data, panel, signals,
        rule=rule, slots=K, max_concurrent=K,
    )
    eq = equity_real_k(trades, K=K)
    cagr = eq.get("cagr", 0)
    mdd = eq.get("mdd", 0)
    eq["calmar"] = abs(cagr / mdd) if mdd != 0 else 0
    eq["total"] = len(trades)
    return trades, eq


def main():
    print("=" * 60)
    print("(N=15, K=3) + MA 정배열 필터 비교")
    print("=" * 60)

    print("[1] 데이터 로드...")
    k200 = bt.load_kospi200_tickers()
    snapshot = bt.load_market_cap_snapshot()
    price_df = bt.load_price_data(k200["ticker"].tolist())
    daily_data = bt.build_daily_data(price_df, snapshot)
    panel = bt.build_daily_indicator_panel(daily_data, "amount")
    print(f"    {len(daily_data)} 종목")

    print("\n[2] 1500억 N=15 신호 생성...")
    raw_signals = compute_5d_filter_signals(
        daily_data, threshold_won=THRESHOLD_WON, lookback=N, top_k=200,
    )
    raw_per_day = [len(v) for v in raw_signals.values()]
    print(f"    {len(raw_signals)}일치, 평균 {np.mean(raw_per_day):.2f} 종목/일")

    print("\n[3] MA 정배열 필터 적용...")
    filtered_signals = apply_ma_filter(raw_signals, daily_data)
    fil_per_day = [len(v) for v in filtered_signals.values()]
    print(f"    필터 후 {len(filtered_signals)}일치, 평균 {np.mean(fil_per_day) if fil_per_day else 0:.2f} 종목/일")
    if raw_per_day and fil_per_day:
        reduction = (1 - sum(fil_per_day) / sum(raw_per_day)) * 100
        print(f"    감소율: {reduction:.1f}%")

    print("\n[4] 베이스 (필터 없음, LIST_EXIT)...")
    t0 = time.time()
    trades_base, stats_base = run_one(daily_data, panel, raw_signals, "LIST_EXIT")
    print(f"    거래수 {stats_base['total']}, CAGR {stats_base['cagr']:+.2f}%, "
          f"MDD {stats_base['mdd']:+.2f}%, Calmar {stats_base['calmar']:.2f} ({time.time()-t0:.1f}s)")

    print("\n[5] 변형 A (필터 + LIST_EXIT)...")
    t0 = time.time()
    trades_a, stats_a = run_one(daily_data, panel, filtered_signals, "LIST_EXIT")
    print(f"    거래수 {stats_a['total']}, CAGR {stats_a['cagr']:+.2f}%, "
          f"MDD {stats_a['mdd']:+.2f}%, Calmar {stats_a['calmar']:.2f} ({time.time()-t0:.1f}s)")

    print("\n[6] 변형 B (필터 + MA5 이탈)...")
    t0 = time.time()
    trades_b, stats_b = run_one(daily_data, panel, filtered_signals, "MA5")
    print(f"    거래수 {stats_b['total']}, CAGR {stats_b['cagr']:+.2f}%, "
          f"MDD {stats_b['mdd']:+.2f}%, Calmar {stats_b['calmar']:.2f} ({time.time()-t0:.1f}s)")

    # 추가: 베이스 변형 (필터 없음, MA5 이탈) 도 비교 가치
    print("\n[7] 베이스2 (필터 없음, MA5 이탈) 참고...")
    t0 = time.time()
    trades_b0, stats_b0 = run_one(daily_data, panel, raw_signals, "MA5")
    print(f"    거래수 {stats_b0['total']}, CAGR {stats_b0['cagr']:+.2f}%, "
          f"MDD {stats_b0['mdd']:+.2f}%, Calmar {stats_b0['calmar']:.2f} ({time.time()-t0:.1f}s)")

    # 리포트
    print("\n[8] 리포트 생성...")
    lines = ["# (N=15, K=3) + MA 정배열 필터 비교\n"]

    lines.append("## 1. 매매 규칙\n")
    lines.append("- **N/N**: 최근 15일 거래대금 ≥ 1500억 인 날이 15일 모두")
    lines.append("- **MA 필터**: T일 종가 > T일 MA5 AND T일 MA5 > T일 MA20")
    lines.append("- **매수**: 위 통과 종목 중 amount 상위 K=3 → 다음날 시가 매수")
    lines.append("- **자본**: 진짜 K슬롯 (K=3)")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%\n")

    lines.append("## 2. 신호 빈도\n")
    lines.append("| 단계 | 신호 일수 | 평균 종목/일 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 1500억 15/15 (필터 전) | {len(raw_signals)} | {np.mean(raw_per_day) if raw_per_day else 0:.2f} |")
    lines.append(f"| + MA 정배열 (필터 후) | {len(filtered_signals)} | {np.mean(fil_per_day) if fil_per_day else 0:.2f} |")
    if raw_per_day and fil_per_day:
        lines.append(f"\n→ 신호 종목 수 **{(1 - sum(fil_per_day) / sum(raw_per_day)) * 100:.1f}% 감소**")

    lines.append("\n## 3. 4가지 시나리오 핵심 비교\n")
    lines.append("| 시나리오 | 필터 | 매도 | 거래수 | CAGR(%) | MDD(%) | Calmar | 자본(x) |")
    lines.append("|---|:---:|:---:|---:|---:|---:|---:|---:|")
    rows = [
        ("**베이스** (기존 (15,3))", "X", "LIST_EXIT", stats_base),
        ("**변형 A**", "MA정배열", "LIST_EXIT", stats_a),
        ("**변형 B**", "MA정배열", "MA5 이탈", stats_b),
        ("참고", "X", "MA5 이탈", stats_b0),
    ]
    for label, f, r, s in rows:
        lines.append(
            f"| {label} | {f} | {r} | {s['total']} | "
            f"{s['cagr']:+.2f} | {s['mdd']:+.2f} | "
            f"{s['calmar']:.2f} | {s['final_equity']:.2f} |"
        )

    lines.append("\n## 4. 거래 품질 비교\n")
    lines.append("| 지표 | 베이스 | 변형 A | 변형 B | 참고 |")
    lines.append("|---|:---:|:---:|:---:|:---:|")
    sb = trade_summary(trades_base)
    sa = trade_summary(trades_a)
    sB = trade_summary(trades_b)
    sB0 = trade_summary(trades_b0)
    metric_rows = [
        ("거래수", lambda s: f"{s['total']}"),
        ("승률(%)", lambda s: f"{s['win_rate']:.1f}"),
        ("평균 수익률(%)", lambda s: f"{s['avg_ret']:+.2f}"),
        ("최대 단일 손실(%)", lambda s: f"{s['max_loss']:+.2f}"),
        ("최대 단일 이익(%)", lambda s: f"{s['max_gain']:+.2f}"),
        ("평균 보유일", lambda s: f"{s['avg_hold']:.1f}"),
        ("고유 종목 수", lambda s: f"{s['unique']}"),
        ("최다 거래 종목", lambda s: f"{s['most_t']} ({s['most_n']}회)" if s['most_t'] else "-"),
    ]
    for label, fmt in metric_rows:
        lines.append(f"| {label} | {fmt(sb)} | {fmt(sa)} | {fmt(sB)} | {fmt(sB0)} |")

    # 변형 A 거래 상세 (만약 거래 있다면)
    if trades_a:
        lines.append("\n## 5. 변형 A 거래 상세 (필터 + LIST_EXIT)\n")
        lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
        lines.append("|:---:|:---:|:---:|---:|---:|")
        for t in sorted(trades_a, key=lambda x: pd.Timestamp(x["buy_date"])):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            lines.append(f"| {t['ticker']} | {bd} | {sd} | {t['hold_days']} | {t['net_ret']*100:+.2f} |")

    if trades_b:
        lines.append("\n## 6. 변형 B 거래 상세 (필터 + MA5 이탈)\n")
        lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
        lines.append("|:---:|:---:|:---:|---:|---:|")
        for t in sorted(trades_b, key=lambda x: pd.Timestamp(x["buy_date"])):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            lines.append(f"| {t['ticker']} | {bd} | {sd} | {t['hold_days']} | {t['net_ret']*100:+.2f} |")

    # 종합 결론
    lines.append("\n---\n\n## 7. 종합 결론\n")

    base_cal = stats_base["calmar"]
    a_cal = stats_a["calmar"]
    b_cal = stats_b["calmar"]

    lines.append("### 7.1 MA 필터 효과 판정\n")
    lines.append(f"- 베이스 Calmar: **{base_cal:.2f}**")
    lines.append(f"- 변형 A (필터+LIST_EXIT) Calmar: **{a_cal:.2f}** "
                 f"({'개선' if a_cal > base_cal else '악화'} {abs(a_cal-base_cal):.2f})")
    lines.append(f"- 변형 B (필터+MA5 이탈) Calmar: **{b_cal:.2f}** "
                 f"({'개선' if b_cal > base_cal else '악화'} {abs(b_cal-base_cal):.2f})")

    if max(a_cal, b_cal) > base_cal:
        winner = "변형 A" if a_cal > b_cal else "변형 B"
        lines.append(f"\n→ **{winner} 가 베이스보다 우월** — MA 필터가 효과적임")
    else:
        lines.append("\n→ **MA 필터는 (15,3) 에 도움이 안 됨** — 두 변형 모두 베이스 Calmar 미만")

    lines.append("\n### 7.2 매도 규칙 비교\n")
    if a_cal > b_cal:
        lines.append(f"- LIST_EXIT(A) Calmar {a_cal:.2f} > MA5 이탈(B) {b_cal:.2f} → **LIST_EXIT 우위**")
    else:
        lines.append(f"- MA5 이탈(B) Calmar {b_cal:.2f} > LIST_EXIT(A) {a_cal:.2f} → **MA5 이탈 우위**")

    lines.append("\n### 7.3 거래수·표본 충분성\n")
    lines.append(f"- 변형 A 거래수: {stats_a['total']}건")
    lines.append(f"- 변형 B 거래수: {stats_b['total']}건")
    lines.append(f"- 베이스 거래수: {stats_base['total']}건")
    if stats_a['total'] < 10 or stats_b['total'] < 10:
        lines.append("- ⚠️ 변형의 거래수가 10건 미만 → 통계적 무의미 가능성")

    lines.append("\n### 7.4 memory feedback 과의 연결\n")
    lines.append("- `feedback_simple_is_better`: 5/5 베이스에서 1500억+정배열+이격도 추가 시 Calmar 4.96→1.05 폭락 사례")
    lines.append("- 본 검증은 (15,3) 베이스에서 정배열만 추가 (이격도 미포함)")
    if max(a_cal, b_cal) > base_cal:
        lines.append("- 결과: feedback과 다르게 **MA 필터가 (15,3) 에는 도움** → 베이스 강도(N=15)가 매도+매수 충돌 완화")
    else:
        lines.append("- 결과: feedback과 일관되게 **MA 필터가 도움 안 됨** → 매수+매도 둘 다 엄격하면 whipsaw")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
