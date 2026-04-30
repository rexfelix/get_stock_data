"""1500억 (N,K) + 월봉 MA10 위 필터 매트릭스 백테스트.

매수 조건:
- 1500억 N/N 만족
- 추가 (변형): T일 close > 직전 완료 월의 월봉 MA10
  (월봉 close = 매월 마지막 거래일 종가, 10개월 단순 이평)
- 통과 종목 amount 상위 K → 다음날 시가 매수

매도: LIST_EXIT
자본: 진짜 K슬롯

12 시나리오: (15,3)/(5,5)/(10,10) × (2024~2026 실 amount, 2019~2023 추정) × (베이스, +월봉MA10)

목적: 시기 의존성(2019~2023 Calmar 0.29) 극복 가능성 검증.
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
from backtest_n15_k3_2019_2023 import load_price_data_estimated  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_n_k_monthly_ma10.md"

THRESHOLD_WON = 150_000_000_000

NK_COMBOS = [(15, 3), (5, 5), (10, 10)]
PERIODS = [
    ("2024~2026", "2023-01-01", "2026-12-31", "real"),  # 실 amount (기존 베이스와 동일)
    ("2019~2023", "2019-01-01", "2023-12-31", "estimated"),  # 추정 amount
]


# ──────────────────────────────────────────────
# 헬퍼 (단위 테스트 대상)
# ──────────────────────────────────────────────
def compute_monthly_ma10(daily_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """각 ticker df 에 monthly_ma10 컬럼 추가.

    - 매월 마지막 거래일 close 로 월봉 close 생성
    - 10개월 단순 이평 (min_periods=10)
    - 직전 완료 월의 MA10 을 그 다음 달 일별로 매핑 (shift 1)
    """
    out = {}
    for ticker, df in daily_data.items():
        df = df.copy().sort_values("date").reset_index(drop=True)
        df["ym"] = pd.to_datetime(df["date"]).dt.to_period("M")
        # 매월 마지막 거래일의 close
        month_close = df.groupby("ym")["close"].last()
        # 10개월 이평
        ma10 = month_close.rolling(10, min_periods=10).mean()
        # 직전 완료 월의 MA10 → 다음 달에 적용 (shift 1)
        ma10_lagged = ma10.shift(1)
        df["monthly_ma10"] = df["ym"].map(ma10_lagged)
        df = df.drop(columns=["ym"])
        out[ticker] = df
    return out


def apply_monthly_ma10_filter(signals: dict[pd.Timestamp, list[str]],
                              daily_data: dict[str, pd.DataFrame]
                              ) -> dict[pd.Timestamp, list[str]]:
    """signals 의 각 (date, ticker) 에서 close > monthly_ma10 만 통과."""
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
            row = df.loc[d]
            close = row.get("close", np.nan)
            ma10 = row.get("monthly_ma10", np.nan)
            if pd.isna(close) or pd.isna(ma10):
                continue
            if close > ma10:
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
                "samsung_count": 0, "most_t": "-", "most_n": 0}
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
        "samsung_count": counts.get("005930", 0),
        "most_t": most_t, "most_n": most_n,
    }


def run_one(daily_data, panel, signals, n: int, k: int) -> tuple[list[dict], dict]:
    trades, _ = bt.run_backtest(
        daily_data, panel, signals,
        rule="LIST_EXIT", slots=k, max_concurrent=k,
    )
    eq = equity_real_k(trades, K=k)
    cagr = eq.get("cagr", 0)
    mdd = eq.get("mdd", 0)
    eq["calmar"] = abs(cagr / mdd) if mdd != 0 else 0
    eq["total"] = len(trades)
    return trades, eq


def load_period_data(period_label: str, start: str, end: str, mode: str,
                     k200_tickers: list[str], snapshot: pd.DataFrame
                     ) -> tuple[dict, dict]:
    """기간별 daily_data + panel 빌드. mode: 'real' or 'estimated'."""
    if mode == "real":
        # 기존 load_price_data 는 stocks ⨝ stock_all amount
        bt.START_DATE = start  # monkey patch
        price_df = bt.load_price_data(k200_tickers, start_date=start)
        # end 필터
        price_df = price_df[price_df["date"] <= pd.Timestamp(end)]
    else:
        bt.START_DATE = start
        price_df = load_price_data_estimated(k200_tickers, start, end)

    daily = bt.build_daily_data(price_df, snapshot)
    daily = compute_monthly_ma10(daily)  # monthly_ma10 컬럼 추가
    panel = bt.build_daily_indicator_panel(daily, "amount")
    return daily, panel


def main():
    print("=" * 60)
    print("(N,K) + 월봉 MA10 위 필터 매트릭스 (12 시나리오)")
    print("=" * 60)

    print("[1] 공통 데이터 로드...")
    k200 = bt.load_kospi200_tickers()
    snapshot = bt.load_market_cap_snapshot()
    tickers = k200["ticker"].tolist()

    results = {}  # (period_label, n, k, variant) -> result

    for period_label, start, end, mode in PERIODS:
        print(f"\n[2] 기간 {period_label} ({mode}) 데이터 로드...")
        t0 = time.time()
        daily, panel = load_period_data(period_label, start, end, mode, tickers, snapshot)
        print(f"    {len(daily)}종목 / panel {len(panel)}일 ({time.time()-t0:.1f}s)")

        for n, k in NK_COMBOS:
            print(f"\n  [{period_label} N={n} K={k}]")

            raw_signals = compute_5d_filter_signals(
                daily, threshold_won=THRESHOLD_WON, lookback=n, top_k=200,
            )
            raw_avg = np.mean([len(v) for v in raw_signals.values()]) if raw_signals else 0
            print(f"    1500억 N={n} 신호: {len(raw_signals)}일치, 평균 {raw_avg:.2f}")

            # 베이스 (필터 없음)
            t0 = time.time()
            trades_b, stats_b = run_one(daily, panel, raw_signals, n, k)
            print(f"    베이스: 거래수 {stats_b['total']}, CAGR {stats_b['cagr']:+.2f}%, "
                  f"MDD {stats_b['mdd']:+.2f}%, Calmar {stats_b['calmar']:.2f} ({time.time()-t0:.1f}s)")
            results[(period_label, n, k, "base")] = {
                "trades": trades_b, "stats": stats_b, "ts": trade_summary(trades_b),
                "n_signal_days": len(raw_signals), "avg_per_day": raw_avg,
            }

            # 변형 (월봉 MA10 위)
            t0 = time.time()
            filtered = apply_monthly_ma10_filter(raw_signals, daily)
            fil_avg = np.mean([len(v) for v in filtered.values()]) if filtered else 0
            trades_v, stats_v = run_one(daily, panel, filtered, n, k)
            print(f"    +월봉MA10: 신호 {len(filtered)}일/{fil_avg:.2f}, "
                  f"거래수 {stats_v['total']}, CAGR {stats_v['cagr']:+.2f}%, "
                  f"MDD {stats_v['mdd']:+.2f}%, Calmar {stats_v['calmar']:.2f} ({time.time()-t0:.1f}s)")
            results[(period_label, n, k, "ma10")] = {
                "trades": trades_v, "stats": stats_v, "ts": trade_summary(trades_v),
                "n_signal_days": len(filtered), "avg_per_day": fil_avg,
            }

    # 리포트
    print("\n[3] 리포트 생성...")
    lines = ["# (N,K) + 월봉 MA10 위 필터 매트릭스 (12 시나리오)\n"]

    lines.append("## 1. 매매 규칙\n")
    lines.append("- **N/N**: 최근 N일 거래대금 ≥ 1500억 인 날이 N일 모두")
    lines.append("- **월봉 MA10 필터**: T일 close > 직전 완료 월의 월봉 MA10")
    lines.append("  (월봉 close = 매월 마지막 거래일 종가, 10개월 단순 이평)")
    lines.append("- **매수**: 통과 종목 amount 상위 K → 다음날 시가 매수")
    lines.append("- **매도**: LIST_EXIT (1500억 N/N 깨지면 매도)")
    lines.append("- **자본**: 진짜 K슬롯\n")

    lines.append("## 2. 데이터 한계\n")
    lines.append("- **2019~2023**: amount = close × volume **추정** (DB stock_all.amount 2023-09-25 이후만 가용)")
    lines.append("- **월봉 MA10 초기 9개월 미계산**: 2019-01~2019-10 신호 사용 불가")
    lines.append("- mcap = close × snapshot shares (시계열 한계)\n")

    lines.append("## 3. 12 시나리오 매트릭스\n")
    lines.append("| 기간 | (N,K) | 변형 | 신호일수 | 평균/일 | 거래수 | CAGR(%) | MDD(%) | **Calmar** | 005930 매수 |")
    lines.append("| --- | :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for period_label, _, _, _ in PERIODS:
        for n, k in NK_COMBOS:
            for variant_label, key in [("베이스", "base"), ("+월봉MA10", "ma10")]:
                r = results[(period_label, n, k, key)]
                s = r["stats"]
                ts = r["ts"]
                lines.append(
                    f"| {period_label} | ({n},{k}) | {variant_label} | "
                    f"{r['n_signal_days']} | {r['avg_per_day']:.2f} | "
                    f"{s['total']} | {s['cagr']:+.2f} | {s['mdd']:+.2f} | "
                    f"**{s['calmar']:.2f}** | {ts['samsung_count']} |"
                )

    # Calmar 비교 표 (베이스 vs +월봉MA10)
    lines.append("\n## 4. 베이스 vs +월봉MA10 Calmar Δ\n")
    lines.append("| 기간 | (N,K) | 베이스 Calmar | +월봉MA10 Calmar | Δ | 효과 |")
    lines.append("| --- | :---: | ---: | ---: | ---: | :---: |")
    for period_label, _, _, _ in PERIODS:
        for n, k in NK_COMBOS:
            base_cal = results[(period_label, n, k, "base")]["stats"]["calmar"]
            ma10_cal = results[(period_label, n, k, "ma10")]["stats"]["calmar"]
            delta = ma10_cal - base_cal
            sign = "🟢 개선" if delta > 0.5 else ("🔴 악화" if delta < -0.5 else "⚪ 미미")
            lines.append(
                f"| {period_label} | ({n},{k}) | {base_cal:.2f} | "
                f"{ma10_cal:.2f} | {delta:+.2f} | {sign} |"
            )

    # 거래 품질 비교 (2019~2023 / +월봉MA10 핵심)
    lines.append("\n## 5. 거래 품질 비교\n")
    for period_label, _, _, _ in PERIODS:
        lines.append(f"\n### {period_label}\n")
        lines.append("| (N,K) | 변형 | 거래수 | 승률(%) | 평균(%) | 최대손(%) | 평균보유일 |")
        lines.append("| :---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for n, k in NK_COMBOS:
            for variant_label, key in [("베이스", "base"), ("+월봉MA10", "ma10")]:
                ts = results[(period_label, n, k, key)]["ts"]
                lines.append(
                    f"| ({n},{k}) | {variant_label} | {ts['total']} | "
                    f"{ts['win_rate']:.1f} | {ts['avg_ret']:+.2f} | "
                    f"{ts['max_loss']:+.2f} | {ts['avg_hold']:.1f} |"
                )

    # 종합 결론
    lines.append("\n---\n\n## 6. 종합 결론\n")

    # 시기 의존성 극복 검증
    lines.append("### 6.1 시기 의존성 극복 여부 (2019~2023)\n")
    lines.append("| (N,K) | 베이스 Calmar | +월봉MA10 Calmar | 극복 여부 |")
    lines.append("| :---: | ---: | ---: | --- |")
    for n, k in NK_COMBOS:
        b = results[("2019~2023", n, k, "base")]["stats"]["calmar"]
        m = results[("2019~2023", n, k, "ma10")]["stats"]["calmar"]
        if m >= 5:
            verdict = "✅ 극복 (Calmar 5+)"
        elif m > b * 1.5:
            verdict = f"🟡 부분 개선 ({b:.2f} → {m:.2f})"
        else:
            verdict = f"❌ 미극복 ({b:.2f} → {m:.2f})"
        lines.append(f"| ({n},{k}) | {b:.2f} | {m:.2f} | {verdict} |")

    # 강세장 영향
    lines.append("\n### 6.2 강세장 영향 (2024~2026)\n")
    lines.append("| (N,K) | 베이스 Calmar | +월봉MA10 Calmar | 영향 |")
    lines.append("| :---: | ---: | ---: | --- |")
    for n, k in NK_COMBOS:
        b = results[("2024~2026", n, k, "base")]["stats"]["calmar"]
        m = results[("2024~2026", n, k, "ma10")]["stats"]["calmar"]
        delta_pct = (m - b) / b * 100 if b > 0 else 0
        if abs(delta_pct) < 10:
            verdict = "🟢 영향 미미"
        elif delta_pct > 0:
            verdict = f"🟢 개선 ({delta_pct:+.0f}%)"
        else:
            verdict = f"🔴 악화 ({delta_pct:+.0f}%)"
        lines.append(f"| ({n},{k}) | {b:.2f} | {m:.2f} | {verdict} |")

    # 추천 조합
    lines.append("\n### 6.3 추천 조합\n")
    # 두 기간 모두 합리적인 조합 찾기
    best_combo = None
    best_score = -np.inf
    for n, k in NK_COMBOS:
        for variant_label, key in [("베이스", "base"), ("+월봉MA10", "ma10")]:
            cal_24 = results[("2024~2026", n, k, key)]["stats"]["calmar"]
            cal_19 = results[("2019~2023", n, k, key)]["stats"]["calmar"]
            # 두 기간 모두 의미 있는 결과 (min Calmar 기준)
            score = min(cal_24, cal_19)
            if score > best_score:
                best_score = score
                best_combo = (n, k, variant_label, cal_24, cal_19)
    if best_combo and best_score >= 1.0:
        n, k, vlabel, c24, c19 = best_combo
        lines.append(f"- **(N={n}, K={k}) {vlabel}**: 두 기간 min Calmar **{best_score:.2f}**")
        lines.append(f"  2024~2026 Calmar {c24:.2f} / 2019~2023 Calmar {c19:.2f}")
    else:
        lines.append(f"- 두 기간 모두 Calmar ≥ 1 인 조합 없음 — 모든 시나리오에서 시기 의존성 잔존")

    # feedback 4회 검증?
    lines.append("\n### 6.4 feedback_simple_is_better 와 비교\n")
    # 강세장에서 +월봉MA10 의 평균 변화
    avg_24_change = np.mean([
        (results[("2024~2026", n, k, "ma10")]["stats"]["calmar"] -
         results[("2024~2026", n, k, "base")]["stats"]["calmar"]) /
        max(results[("2024~2026", n, k, "base")]["stats"]["calmar"], 0.01) * 100
        for n, k in NK_COMBOS
    ])
    if avg_24_change < -50:
        lines.append(f"- 2024~2026 평균 Calmar 변화: {avg_24_change:+.1f}% → **whipsaw 4회째 검증** (월봉도 일봉처럼 폭락)")
    elif avg_24_change < -20:
        lines.append(f"- 2024~2026 평균 Calmar 변화: {avg_24_change:+.1f}% → 부분 악화 (whipsaw 약하게 발생)")
    else:
        lines.append(f"- 2024~2026 평균 Calmar 변화: {avg_24_change:+.1f}% → **월봉 MA10 은 일봉 필터와 다른 결과** (가설 H1/H2 지지)")

    # 채택 결정
    lines.append("\n### 6.5 채택 결정 영향\n")
    if best_score >= 5:
        lines.append("- **+월봉MA10 도입 가능성** — 시기 의존성 극복 후보")
    elif best_score >= 1:
        lines.append("- **부분적 개선** — 추가 검증 필요 (예: KOSPI MA200 regime 필터)")
    else:
        lines.append("- **+월봉MA10 도 시기 의존성 극복 못함** — 매수 필터 추가 자체가 한계")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
