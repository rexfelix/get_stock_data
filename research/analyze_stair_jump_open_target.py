"""
계단뛰기 (16·17 제거) — '시초가 매수 + 4% 익절/종가 매도' intraday 전략

두 시나리오:
  A) 시그널 당일 t=0 시초가 매수  (look-ahead, 실전 불가능)
  B) 다음날 t=+1 시초가 매수      (실전 가능)

매도 룰:
  - high >= entry × (1 + tgt%) → entry × (1 + tgt%) 에 매도 (익절)
  - 그 외 → 그 봉 종가 매도

비용: 왕복 0.26% (수수료 0.03% + 매도세 0.23%)

타겟 % 그리드: 2, 3, 4, 5, 7, 10
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from backtest_stair_jump import calc_indicators, load_universe, load_all
from analyze_stair_jump_stats_no1617 import find_signals_no1617
from analyze_stair_jump_stats import load_kospi_regime


FEE_BUY = 0.00015
FEE_SELL = 0.00015
TAX_SELL = 0.0023
ROUND_TRIP = FEE_BUY + FEE_SELL + TAX_SELL  # 0.0026
COST_PCT = ROUND_TRIP * 100

DATA_START = "2019-01-01"
DATA_END = "2026-05-17"

TARGET_GRID = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]


def collect_open_target_events(ticker: str, name: str, df: pd.DataFrame) -> list[dict]:
    """시그널마다 두 시나리오(A: t=0, B: t=+1)의 OHLC 를 기록."""
    df = calc_indicators(df)
    signals = find_signals_no1617(df)
    if not signals:
        return []

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    dates = df.index

    out = []
    for i in signals:
        ev = {
            "ticker": ticker, "name": name,
            "signal_date": dates[i],
            # A) 당일 t=0
            "A_open": float(o[i]), "A_high": float(h[i]),
            "A_low": float(l[i]), "A_close": float(c[i]),
        }
        # B) 다음날 t=+1
        if i + 1 < len(df) and not np.isnan(o[i + 1]) and o[i + 1] > 0:
            ev["B_date"] = dates[i + 1]
            ev["B_open"] = float(o[i + 1])
            ev["B_high"] = float(h[i + 1])
            ev["B_low"] = float(l[i + 1])
            ev["B_close"] = float(c[i + 1])
        else:
            ev["B_date"] = pd.NaT
            ev["B_open"] = ev["B_high"] = ev["B_low"] = ev["B_close"] = np.nan
        out.append(ev)
    return out


def simulate(ev_df: pd.DataFrame, scenario: str, tgt_pct: float) -> dict:
    """
    scenario: 'A' (당일 t=0) or 'B' (다음날 t=+1)
    return dict with per-trade and equity-curve stats.
    """
    prefix = scenario
    sub = ev_df.dropna(subset=[f"{prefix}_open"]).copy()
    if sub.empty:
        return None

    entry = sub[f"{prefix}_open"].astype(float).values
    high = sub[f"{prefix}_high"].astype(float).values
    close = sub[f"{prefix}_close"].astype(float).values

    target_price = entry * (1 + tgt_pct / 100)
    hit_target = high >= target_price

    # 매도가
    sell_price = np.where(hit_target, target_price, close)
    gross_ret = (sell_price - entry) / entry * 100
    net_ret = gross_ret - COST_PCT

    sub["gross_ret"] = gross_ret
    sub["net_ret"] = net_ret
    sub["hit_target"] = hit_target

    # 거래일 = scenario A 면 signal_date, B 면 B_date
    if scenario == "A":
        sub["trade_date"] = pd.to_datetime(sub["signal_date"])
    else:
        sub["trade_date"] = pd.to_datetime(sub["B_date"])

    n = len(sub)
    n_target = int(hit_target.sum())
    win = int((net_ret > 0).sum())

    # 일별 동일가중 분산투자 시뮬
    daily = sub.groupby("trade_date").agg(
        n_trades=("net_ret", "size"),
        gross=("gross_ret", "mean"),
        net=("net_ret", "mean"),
    )
    equity = (1 + daily["net"] / 100).cumprod()
    if len(equity) == 0:
        cagr = mdd = total_ret = np.nan
        sharpe = np.nan
    else:
        span_days = (daily.index.max() - daily.index.min()).days + 1
        years = span_days / 365.25
        total_ret = (equity.iloc[-1] - 1) * 100
        cagr = (equity.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else np.nan
        peak = equity.cummax()
        dd = (equity / peak - 1) * 100
        mdd = dd.min()
        std = daily["net"].std()
        sharpe = (daily["net"].mean() / std * np.sqrt(252)) if std and std > 0 else np.nan

    return {
        "scenario": scenario, "tgt": tgt_pct,
        "n": n,
        "target_hit_rate": n_target / n * 100,
        "gross_mean": float(np.mean(gross_ret)),
        "gross_median": float(np.median(gross_ret)),
        "net_mean": float(np.mean(net_ret)),
        "net_median": float(np.median(net_ret)),
        "net_winrate": win / n * 100,
        "net_up_mean": float(np.mean(net_ret[net_ret > 0])) if (net_ret > 0).any() else np.nan,
        "net_dn_mean": float(np.mean(net_ret[net_ret < 0])) if (net_ret < 0).any() else np.nan,
        "n_days": len(daily),
        "avg_trades_per_day": daily["n_trades"].mean() if len(daily) else np.nan,
        "avg_daily_net": daily["net"].mean() if len(daily) else np.nan,
        "total_ret_pct": total_ret,
        "cagr_pct": cagr,
        "mdd_pct": mdd,
        "sharpe": sharpe,
    }


def fmt(x, p=2, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if sign:
        return f"{x:+.{p}f}%"
    return f"{x:.{p}f}%"


def make_report(ev_df: pd.DataFrame, regime_df: pd.DataFrame) -> str:
    # regime
    ev_df = ev_df.copy()
    ev_df["signal_date"] = pd.to_datetime(ev_df["signal_date"])
    reg_series = regime_df["regime"]
    unique_dates = pd.DatetimeIndex(sorted(set(ev_df["signal_date"].unique())))
    full = pd.concat([reg_series, pd.Series(index=unique_dates, dtype=object)])
    full = full[~full.index.duplicated(keep="first")].sort_index().ffill()
    ev_df["regime"] = ev_df["signal_date"].map(full).fillna("약세장")

    lines = []
    lines.append("# 계단뛰기 (16·17 제거) — 시초가 매수 + N% 익절/종가 매도\n")
    lines.append("## 시나리오\n")
    lines.append("- **A (look-ahead, 실전 불가)**: 시그널 당일 t=0 시초가 매수, 같은 봉 안에서 익절 또는 종가 매도")
    lines.append("- **B (실전 가능)**: 다음날 t=+1 시초가 매수, 같은 봉 안에서 익절 또는 종가 매도")
    lines.append(f"- 비용: 왕복 {COST_PCT:.2f}% (수수료 0.03% + 매도세 0.23%)")
    lines.append(f"- 타겟 그리드: {TARGET_GRID} %")
    lines.append("")
    lines.append("> ⚠️ **A 시나리오 주의**: 시그널 조건이 종가/MA5/MA120 으로 판정되므로 장 마감 후에야 확정.")
    lines.append("> 따라서 시그널 당일 시초가에는 그 종목이 시그널이 될지 알 수 없음 → **실전 불가능, 이론값**.")
    lines.append("")
    lines.append(f"- 데이터: {DATA_START} ~ {DATA_END}, 총 시그널 {len(ev_df):,}건")
    lines.append("")

    # ── 시나리오 A: 타겟 그리드 ──
    for scenario, title in [("A", "1. 시나리오 A — 당일 t=0 시초가 매수 (이론값, look-ahead)"),
                            ("B", "2. 시나리오 B — 다음날 t=+1 시초가 매수 (실전 가능)")]:
        lines.append(f"## {title}\n")
        lines.append("### 타겟 % 그리드 — 전체\n")
        lines.append("| 타겟 | 표본 | 익절 도달률 | Gross 평균 | **Net 평균** | Net 승률 | 일평균 Net | 누적 수익 | CAGR | MDD | Sharpe |")
        lines.append("|----:|-----:|----------:|----------:|-----------:|--------:|----------:|---------:|----:|----:|------:|")
        results = []
        for tgt in TARGET_GRID:
            r = simulate(ev_df, scenario, tgt)
            if r is None:
                continue
            results.append(r)
            lines.append(
                f"| {tgt:.0f}% | {r['n']:,} | {fmt(r['target_hit_rate'], 1, False)} | "
                f"{fmt(r['gross_mean'])} | **{fmt(r['net_mean'])}** | {fmt(r['net_winrate'], 1, False)} | "
                f"{fmt(r['avg_daily_net'], 3)} | {fmt(r['total_ret_pct'], 1)} | "
                f"{fmt(r['cagr_pct'], 1)} | {fmt(r['mdd_pct'], 1, False)} | {r['sharpe']:.2f} |"
            )
        lines.append("")

        # 4% 타겟 강세/약세 분리
        lines.append(f"### 타겟 4% 만 — 강세장/약세장 분리\n")
        lines.append("| 구분 | 표본 | 익절 도달률 | Gross 평균 | **Net 평균** | Net 승률 | Net 상승 평균 | Net 하락 평균 |")
        lines.append("|------|-----:|----------:|----------:|-----------:|--------:|------------:|------------:|")
        for sub_label, sub in [("전체", ev_df),
                                ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                                ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
            r = simulate(sub, scenario, 4.0)
            if r is None:
                lines.append(f"| {sub_label} | 0 | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {sub_label} | {r['n']:,} | {fmt(r['target_hit_rate'], 1, False)} | "
                f"{fmt(r['gross_mean'])} | **{fmt(r['net_mean'])}** | {fmt(r['net_winrate'], 1, False)} | "
                f"{fmt(r['net_up_mean'])} | {fmt(r['net_dn_mean'])} |"
            )
        lines.append("")

        # 4% 일별 시뮬 by regime
        lines.append(f"### 타겟 4% — 일별 동일가중 분산투자 (강세/약세)\n")
        lines.append("| 구분 | 거래일 | 평균 시그널/일 | 일평균 Net | 누적 수익 | CAGR | MDD | Sharpe |")
        lines.append("|------|-------:|--------------:|----------:|---------:|----:|----:|------:|")
        for sub_label, sub in [("전체", ev_df),
                                ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                                ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
            r = simulate(sub, scenario, 4.0)
            if r is None or r["n_days"] == 0:
                lines.append(f"| {sub_label} | 0 | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {sub_label} | {r['n_days']:,} | {r['avg_trades_per_day']:.1f} | "
                f"{fmt(r['avg_daily_net'], 3)} | {fmt(r['total_ret_pct'], 1)} | "
                f"{fmt(r['cagr_pct'], 1)} | {fmt(r['mdd_pct'], 1, False)} | {r['sharpe']:.2f} |"
            )
        lines.append("")

    # ── 최종 결론 ──
    lines.append("## 3. 결론\n")
    rA4 = simulate(ev_df, "A", 4.0)
    rB4 = simulate(ev_df, "B", 4.0)
    lines.append("### 타겟 4% 핵심 비교\n")
    lines.append("| 지표 | A (당일, look-ahead) | B (다음날, 실전) |")
    lines.append("|------|-------------------:|----------------:|")
    lines.append(f"| 거래당 Gross 평균 | {fmt(rA4['gross_mean'])} | {fmt(rB4['gross_mean'])} |")
    lines.append(f"| **거래당 Net 평균** | **{fmt(rA4['net_mean'])}** | **{fmt(rB4['net_mean'])}** |")
    lines.append(f"| Net 승률 | {fmt(rA4['net_winrate'], 1, False)} | {fmt(rB4['net_winrate'], 1, False)} |")
    lines.append(f"| 4% 익절 도달률 | {fmt(rA4['target_hit_rate'], 1, False)} | {fmt(rB4['target_hit_rate'], 1, False)} |")
    lines.append(f"| 일평균 Net | {fmt(rA4['avg_daily_net'], 3)} | {fmt(rB4['avg_daily_net'], 3)} |")
    lines.append(f"| **누적 수익 (7.4년)** | **{fmt(rA4['total_ret_pct'], 1)}** | **{fmt(rB4['total_ret_pct'], 1)}** |")
    lines.append(f"| CAGR | {fmt(rA4['cagr_pct'], 1)} | {fmt(rB4['cagr_pct'], 1)} |")
    lines.append(f"| MDD | {fmt(rA4['mdd_pct'], 1, False)} | {fmt(rB4['mdd_pct'], 1, False)} |")
    lines.append(f"| Sharpe | {rA4['sharpe']:.2f} | {rB4['sharpe']:.2f} |")
    lines.append("")

    judgement = "✅ 수익" if rA4["net_mean"] > 0 else "❌ 손실"
    judgement_b = "✅ 수익" if rB4["net_mean"] > 0 else "❌ 손실"
    lines.append("### 판단\n")
    lines.append(f"- **A (당일 시초가, look-ahead)**: {judgement} — 그러나 **실전 불가능** (시그널 미확정)")
    lines.append(f"- **B (다음날 시초가, 실전)**: {judgement_b}")
    lines.append("")
    lines.append("### 왜 A 와 B 의 결과가 갈리는가?\n")
    lines.append("- A: 시그널 당일 = 조건 4 (open<close) 로 **양봉 100%, 시초가→종가 평균 +4.22%**.")
    lines.append("  - 시초가에 사면 종가까지 평균 +4.22% (gross) 가 보장 → 4% 익절 도달률도 매우 높음")
    lines.append("- B: 다음날은 양봉률 42.4%, 평균 +0.23% (시초가→종가 평균은 더 낮음)")
    lines.append("  - 시초가→종가 자체가 음수 기댓값 → 4% 타겟 도달도 어려움")
    lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("계단뛰기 (16·17 제거) — 시초가 매수 + N% 타겟 분석")
    print("=" * 60)

    ticker_list, name_map = load_universe()
    all_data = load_all(ticker_list, DATA_START, DATA_END)

    print(f"[3/4] 시그널 수집 ({len(all_data)} 종목)...")
    events = []
    for k, ticker in enumerate(ticker_list, 1):
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        if len(df) < 130:
            continue
        nm = name_map.get(ticker, ticker)
        events.extend(collect_open_target_events(ticker, nm, df))
        if k % 500 == 0:
            print(f"   {k}/{len(ticker_list)}, 누적 {len(events):,}")

    ev_df = pd.DataFrame(events)
    print(f"        총 시그널: {len(ev_df):,}")

    print("[4/4] 시뮬레이션 & 리포트 생성...")
    regime_df = load_kospi_regime()
    report = make_report(ev_df, regime_df)

    out = os.path.join(os.path.dirname(__file__), "results",
                        "stair_jump_open_target.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트: {out}")


if __name__ == "__main__":
    main()
