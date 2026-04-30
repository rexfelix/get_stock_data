"""1500억 N=15 + MA20 5일 연속 상승 필터 + K=3 백테스트.

매수 조건:
- 1500억 15/15 만족
- T일 기준 MA20[T-4 ~ T] 5개 값이 strictly increasing (5일 연속 상승)
- 통과 종목 중 amount 상위 K=3 → 다음날 시가 매수

매도 규칙: LIST_EXIT 단독
자본: 진짜 K슬롯 (K=3)

베이스라인: (15,3) LIST_EXIT — Calmar 41.88
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

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_n15_k3_ma20_uptrend.md"

THRESHOLD_WON = 150_000_000_000
N = 15
K = 3


# ──────────────────────────────────────────────
# 헬퍼 (단위 테스트 대상)
# ──────────────────────────────────────────────
def is_ma20_uptrend_5d(values) -> bool:
    """5개 값이 strictly increasing 이고 NaN 없을 때 True."""
    arr = list(values)
    if len(arr) != 5:
        return False
    if any(pd.isna(v) for v in arr):
        return False
    for i in range(1, 5):
        if arr[i] <= arr[i - 1]:
            return False
    return True


def apply_ma20_uptrend_filter(signals: dict[pd.Timestamp, list[str]],
                              daily_data: dict[str, pd.DataFrame]
                              ) -> dict[pd.Timestamp, list[str]]:
    """signals 의 각 (date, ticker) 에서 MA20[T-4 ~ T] 5일 단조 증가 만족 종목만 반환."""
    if not signals:
        return {}

    # ticker → date-인덱스 DataFrame 캐시
    cache = {}
    for ticker, df in daily_data.items():
        if "date" in df.columns:
            indexed = df.set_index("date").sort_index()
        else:
            indexed = df.sort_index()
        cache[ticker] = indexed

    out = {}
    for d, tickers in signals.items():
        passed = []
        for t in tickers:
            df = cache.get(t)
            if df is None or d not in df.index:
                continue
            try:
                pos = df.index.get_loc(d)
            except KeyError:
                continue
            if pos < 4:
                continue  # 5일 데이터 부족
            window = df["ma20"].iloc[pos - 4:pos + 1].tolist()
            if is_ma20_uptrend_5d(window):
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
                "max_gain": 0, "avg_hold": 0, "unique": 0, "most_t": "-", "most_n": 0}
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
    print("(N=15, K=3) + MA20 5일 연속 상승 필터")
    print("=" * 60)

    print("[1] 데이터 로드...")
    k200 = bt.load_kospi200_tickers()
    snapshot = bt.load_market_cap_snapshot()
    price_df = bt.load_price_data(k200["ticker"].tolist())
    daily_data = bt.build_daily_data(price_df, snapshot)
    panel = bt.build_daily_indicator_panel(daily_data, "amount")
    print(f"    {len(daily_data)} 종목")

    print("\n[2] 1500억 N=15 신호 생성...")
    raw = compute_5d_filter_signals(
        daily_data, threshold_won=THRESHOLD_WON, lookback=N, top_k=200,
    )
    raw_per = [len(v) for v in raw.values()]
    print(f"    {len(raw)}일치, 평균 {np.mean(raw_per) if raw_per else 0:.2f} 종목/일")

    print("\n[3] MA20 5일 연속 상승 필터 적용...")
    fil = apply_ma20_uptrend_filter(raw, daily_data)
    fil_per = [len(v) for v in fil.values()]
    print(f"    필터 후 {len(fil)}일치, 평균 {np.mean(fil_per) if fil_per else 0:.2f} 종목/일")
    if raw_per and fil_per:
        red = (1 - sum(fil_per) / sum(raw_per)) * 100
        print(f"    감소율: {red:.1f}%")

    print("\n[4] 베이스 (필터 없음, LIST_EXIT)...")
    t0 = time.time()
    trades_base, stats_base = run_one(daily_data, panel, raw)
    print(f"    거래수 {stats_base['total']}, CAGR {stats_base['cagr']:+.2f}%, "
          f"MDD {stats_base['mdd']:+.2f}%, Calmar {stats_base['calmar']:.2f} ({time.time()-t0:.1f}s)")

    print("\n[5] 변형 (MA20 상승 필터 + LIST_EXIT)...")
    t0 = time.time()
    trades_var, stats_var = run_one(daily_data, panel, fil)
    print(f"    거래수 {stats_var['total']}, CAGR {stats_var['cagr']:+.2f}%, "
          f"MDD {stats_var['mdd']:+.2f}%, Calmar {stats_var['calmar']:.2f} ({time.time()-t0:.1f}s)")

    # 리포트
    print("\n[6] 리포트 생성...")
    lines = ["# (N=15, K=3) + MA20 5일 연속 상승 필터\n"]

    lines.append("## 1. 매매 규칙\n")
    lines.append("- **N/N**: 최근 15일 거래대금 ≥ 1500억 인 날이 15일 모두")
    lines.append("- **MA20 추세 필터**: T일 기준 MA20[T-4]<MA20[T-3]<MA20[T-2]<MA20[T-1]<MA20[T] (5일 연속 상승)")
    lines.append("- **매수**: 위 통과 종목 중 amount 상위 K=3 → 다음날 시가 매수")
    lines.append("- **매도**: LIST_EXIT (1500억 15/15 깨지면 다다음날 시가 매도)")
    lines.append("- **자본**: 진짜 K슬롯 (K=3)")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%\n")

    lines.append("## 2. 신호 빈도\n")
    lines.append("| 단계 | 신호 일수 | 평균 종목/일 |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| 1500억 15/15 (필터 전) | {len(raw)} | {np.mean(raw_per) if raw_per else 0:.2f} |")
    lines.append(f"| + MA20 5일 상승 (필터 후) | {len(fil)} | {np.mean(fil_per) if fil_per else 0:.2f} |")
    if raw_per and fil_per:
        lines.append(f"\n→ 신호 종목 수 **{(1 - sum(fil_per) / sum(raw_per)) * 100:.1f}% 감소**")

    lines.append("\n## 3. 베이스 vs 변형 핵심 비교\n")
    lines.append("| 시나리오 | 필터 | 거래수 | CAGR(%) | MDD(%) | Calmar | 자본(x) |")
    lines.append("| --- | :---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(f"| **베이스 (15,3)** | X | {stats_base['total']} | "
                 f"{stats_base['cagr']:+.2f} | {stats_base['mdd']:+.2f} | "
                 f"{stats_base['calmar']:.2f} | {stats_base['final_equity']:.2f} |")
    lines.append(f"| **변형** | MA20↑5일 | {stats_var['total']} | "
                 f"{stats_var['cagr']:+.2f} | {stats_var['mdd']:+.2f} | "
                 f"{stats_var['calmar']:.2f} | {stats_var['final_equity']:.2f} |")

    lines.append("\n## 4. 거래 품질 비교\n")
    sb = trade_summary(trades_base)
    sv = trade_summary(trades_var)
    lines.append("| 지표 | 베이스 | 변형 |")
    lines.append("| --- | :---: | :---: |")
    rows = [
        ("거래수", lambda s: f"{s['total']}"),
        ("승률(%)", lambda s: f"{s['win_rate']:.1f}"),
        ("평균 수익률(%)", lambda s: f"{s['avg_ret']:+.2f}"),
        ("최대 단일 손실(%)", lambda s: f"{s['max_loss']:+.2f}"),
        ("최대 단일 이익(%)", lambda s: f"{s['max_gain']:+.2f}"),
        ("평균 보유일", lambda s: f"{s['avg_hold']:.1f}"),
        ("고유 종목 수", lambda s: f"{s['unique']}"),
        ("최다 거래 종목", lambda s: f"{s['most_t']} ({s['most_n']}회)"),
    ]
    for label, fmt in rows:
        lines.append(f"| {label} | {fmt(sb)} | {fmt(sv)} |")

    # 변형 거래 상세
    if trades_var:
        lines.append("\n## 5. 변형 거래 상세\n")
        lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
        lines.append("| :---: | :---: | :---: | ---: | ---: |")
        for t in sorted(trades_var, key=lambda x: pd.Timestamp(x["buy_date"])):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            lines.append(f"| {t['ticker']} | {bd} | {sd} | {t['hold_days']} | {t['net_ret']*100:+.2f} |")

    # 베이스 거래 상세 (비교용)
    if trades_base:
        lines.append("\n## 6. 베이스 거래 상세 (비교용)\n")
        lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
        lines.append("| :---: | :---: | :---: | ---: | ---: |")
        for t in sorted(trades_base, key=lambda x: pd.Timestamp(x["buy_date"])):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            lines.append(f"| {t['ticker']} | {bd} | {sd} | {t['hold_days']} | {t['net_ret']*100:+.2f} |")

    # 결론
    lines.append("\n---\n\n## 7. 종합 결론\n")
    bcal = stats_base["calmar"]
    vcal = stats_var["calmar"]

    lines.append("### 7.1 MA20 추세 필터 효과\n")
    lines.append(f"- 베이스 Calmar: **{bcal:.2f}** (CAGR {stats_base['cagr']:+.2f}%, MDD {stats_base['mdd']:+.2f}%)")
    lines.append(f"- 변형 Calmar: **{vcal:.2f}** (CAGR {stats_var['cagr']:+.2f}%, MDD {stats_var['mdd']:+.2f}%)")
    delta = vcal - bcal
    if delta > 0:
        lines.append(f"\n→ **개선**: Calmar +{delta:.2f}")
    elif delta < 0:
        lines.append(f"\n→ **악화**: Calmar {delta:.2f} ({abs(delta)/bcal*100:.1f}% 감소)")
    else:
        lines.append("\n→ **동일**: 변화 없음")

    lines.append("\n### 7.2 거래수·표본\n")
    lines.append(f"- 베이스: {stats_base['total']} 건")
    lines.append(f"- 변형: {stats_var['total']} 건")
    if stats_var['total'] < 5:
        lines.append("- ⚠️ 변형 거래수가 5건 미만 → **통계적 무의미**")
    elif stats_var['total'] < 10:
        lines.append("- ⚠️ 변형 거래수가 10건 미만 → 통계적 한계")

    lines.append("\n### 7.3 정배열 필터(이전 사례) 와 비교\n")
    lines.append("| 시도 | 베이스 Calmar | 변형 Calmar | 변화 |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| 정배열(close>MA5 AND MA5>MA20) | 41.88 | 2.17 | **-95%** |")
    lines.append(f"| 본 검증: MA20 5일 상승 | {bcal:.2f} | {vcal:.2f} | "
                 f"{(vcal-bcal)/bcal*100:+.1f}% |")

    lines.append("\n### 7.4 결론\n")
    if vcal > bcal * 1.05:
        lines.append("- MA20 추세 필터가 **(15,3) 베이스 개선** — 채택 검토 가능")
    elif vcal > bcal * 0.95:
        lines.append("- MA20 추세 필터 **효과 미미** — (15,3) 베이스 그대로 유지 권장")
    elif vcal > bcal * 0.5:
        lines.append("- MA20 추세 필터 **명백히 악화** — 폐기")
    else:
        lines.append("- MA20 추세 필터 **폭락 (정배열 필터와 유사 패턴)** — feedback memory 강화 사례")

    if stats_var['total'] < stats_base['total']:
        lines.append(f"\n  거래수 감소: {stats_base['total']} → {stats_var['total']} "
                     f"(필터로 {stats_base['total']-stats_var['total']}건 제외)")
    elif stats_var['total'] > stats_base['total']:
        lines.append(f"\n  거래수 증가(예상치 못한 결과): {stats_base['total']} → {stats_var['total']}")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
