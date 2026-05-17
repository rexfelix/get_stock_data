"""
계단뛰기 (원본 17조건) — 매도 파라미터 풀기간 그리드 최적화

목적: 2019~2026 전 기간에 수익이 나는 (Tgt, Min, SL/MA5) 조합이 있는가?
강세장/약세장 분리 통계로 robust 한 조합 탐색.

비교 모드:
  - TPSL  : Tgt × Min × 고정 SL%
  - MA5   : Tgt × Min × MA5 종가이탈 손절

비용: 왕복 0.26% (run_backtest 내장)
유니버스: 시총 ≥ 3000억, 종가 ≥ 2000원
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

from backtest_stair_jump import (
    calc_indicators, run_backtest,
    load_universe, load_all,
    MCAP_THRESHOLD, PRICE_FLOOR, INITIAL_CAPITAL,
)
from analyze_stair_jump_stats import load_kospi_regime


DATA_START = "2019-01-01"
DATA_END = "2026-05-17"

# 그리드
TGT_GRID = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
MIN_GRID = [1.0, 2.0, 3.0, 5.0]
SL_GRID = [2.0, 3.0, 4.0, 5.0]


def run_one_combo(tgt: float, mn: float, sl: float, exit_mode: str,
                  ticker_list, name_map, all_data) -> pd.DataFrame:
    """모든 종목에 대해 한 조합 백테스트 → 통합 trades DataFrame."""
    all_trades = []
    for ticker in ticker_list:
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        df_ind = calc_indicators(df)
        df_test = df_ind.loc[DATA_START:DATA_END]
        if len(df_test) < 5:
            continue
        nm = name_map.get(ticker, ticker)
        cap, trades, equity, n_sig = run_backtest(
            df_test, ticker, nm, tgt, mn, sl, exit_mode=exit_mode,
        )
        if trades:
            all_trades.extend(trades)
    return pd.DataFrame(all_trades)


def metrics_by_regime(trades_df: pd.DataFrame, regime_df: pd.DataFrame) -> dict:
    """전체/강세/약세 분리 메트릭."""
    if trades_df.empty:
        return {"전체": {}, "강세장": {}, "약세장": {}}

    t = trades_df.copy()
    t["buy_date"] = pd.to_datetime(t["buy_date"])
    reg_series = regime_df["regime"]
    unique_dates = pd.DatetimeIndex(sorted(set(t["buy_date"].unique())))
    full = pd.concat([reg_series, pd.Series(index=unique_dates, dtype=object)])
    full = full[~full.index.duplicated(keep="first")].sort_index().ffill()
    t["regime"] = t["buy_date"].map(full).fillna("약세장")

    out = {}
    for label, sub in [("전체", t),
                       ("강세장", t[t["regime"] == "강세장"]),
                       ("약세장", t[t["regime"] == "약세장"])]:
        if sub.empty:
            out[label] = {"n_trades": 0}
            continue
        wins = sub[sub["pnl"] > 0]
        losses = sub[sub["pnl"] <= 0]
        gp = wins["pnl"].sum()
        gl = abs(losses["pnl"].sum())
        out[label] = {
            "n_trades": len(sub),
            "win_rate": len(wins) / len(sub) * 100,
            "total_pnl": sub["pnl"].sum(),
            "avg_ret": sub["return_pct"].mean(),
            "median_ret": sub["return_pct"].median(),
            "pf": gp / gl if gl > 0 else float("inf"),
            "up_mean": wins["return_pct"].mean() if len(wins) else np.nan,
            "dn_mean": losses["return_pct"].mean() if len(losses) else np.nan,
            "hold_days": (pd.to_datetime(sub["sell_date"]) -
                          pd.to_datetime(sub["buy_date"])).dt.days.mean(),
        }
    return out


def fmt_pf(v):
    if v == float("inf"):
        return "∞"
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.2f}"


def fmt_pct(x, p=2, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{p}f}%" if sign else f"{x:.{p}f}%"


def fmt_pnl(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:>+13,.0f}"


def make_report(results: list[dict], elapsed: float) -> str:
    df = pd.DataFrame(results)
    lines = []
    lines.append("# 계단뛰기 (원본 17조건) — 매도 파라미터 풀기간 그리드 최적화\n")
    lines.append("## 분석 개요\n")
    lines.append(f"- **데이터 기간**: {DATA_START} ~ {DATA_END} (약 7.4 년)")
    lines.append(f"- **유니버스**: 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억, 종가 ≥ {PRICE_FLOOR:,}원")
    lines.append(f"- **종목당 초기자본**: {INITIAL_CAPITAL:,}원 (per-ticker independent backtest)")
    lines.append(f"- **비용**: 왕복 약 0.26% (수수료+세금)")
    lines.append(f"- **그리드**: TPSL = {len(TGT_GRID)*len(MIN_GRID)*len(SL_GRID)}조합 + MA5 = {len(TGT_GRID)*len(MIN_GRID)}조합 (Min<Tgt 필터)")
    lines.append(f"- **시그널**: 원본 17조건 그대로 (16·17 포함)")
    lines.append("")

    # ── 1. 전체 기간 (전체 regime) TOP ──
    lines.append("## 1. 전체 기간 (regime 합산) — 총손익 TOP 15\n")
    lines.append("| # | 모드 | Tgt | Min | SL | 거래 | 승률 | PF | 총손익 | 거래당% | 보유일 |")
    lines.append("|---|------|---:|---:|----:|-----:|-----:|----:|--------:|--------:|------:|")
    df_sort = df.sort_values("total_pnl", ascending=False).reset_index(drop=True)
    for i, r in df_sort.head(15).iterrows():
        sl_s = f"{r['sl']:.0f}%" if r["mode"] == "tpsl" else "MA5"
        lines.append(
            f"| {i+1} | {r['mode']} | {r['tgt']:.0f}% | {r['min']:.0f}% | {sl_s} "
            f"| {int(r['n_trades']):,} | {r['win_rate']:.1f}% | {fmt_pf(r['pf'])} "
            f"| {fmt_pnl(r['total_pnl'])} | {fmt_pct(r['avg_ret'])} | {r['hold_days']:.1f} |"
        )
    lines.append("")

    # ── 2. PF TOP ──
    df_pf = df[df["pf"] != float("inf")].sort_values("pf", ascending=False).reset_index(drop=True)
    lines.append("## 2. 전체 기간 — 손익비(PF) TOP 15\n")
    lines.append("| # | 모드 | Tgt | Min | SL | 거래 | 승률 | PF | 총손익 | 거래당% |")
    lines.append("|---|------|---:|---:|----:|-----:|-----:|----:|--------:|--------:|")
    for i, r in df_pf.head(15).iterrows():
        sl_s = f"{r['sl']:.0f}%" if r["mode"] == "tpsl" else "MA5"
        lines.append(
            f"| {i+1} | {r['mode']} | {r['tgt']:.0f}% | {r['min']:.0f}% | {sl_s} "
            f"| {int(r['n_trades']):,} | {r['win_rate']:.1f}% | {fmt_pf(r['pf'])} "
            f"| {fmt_pnl(r['total_pnl'])} | {fmt_pct(r['avg_ret'])} |"
        )
    lines.append("")

    # ── 3. 약세장에서 수익 나는 조합 ──
    lines.append("## 3. 약세장 단독 수익 가능 조합 (PnL > 0)\n")
    bear_pos = df[df["bear_total_pnl"] > 0].sort_values("bear_total_pnl", ascending=False)
    if bear_pos.empty:
        lines.append("- **❌ 약세장에서 총손익 플러스인 조합이 단 하나도 없음**")
        # 약세장 손실 가장 적은 조합 5개
        bear_least = df.sort_values("bear_total_pnl", ascending=False).head(5)
        lines.append("\n약세장 손실 가장 적은 TOP 5:\n")
        lines.append("| # | 모드 | Tgt | Min | SL | 약세거래 | 약세PF | 약세총손익 | 약세거래당% |")
        lines.append("|---|------|---:|---:|----:|--------:|------:|----------:|----------:|")
        for i, r in bear_least.iterrows():
            sl_s = f"{r['sl']:.0f}%" if r["mode"] == "tpsl" else "MA5"
            lines.append(
                f"| {i+1} | {r['mode']} | {r['tgt']:.0f}% | {r['min']:.0f}% | {sl_s} "
                f"| {int(r['bear_n_trades']):,} | {fmt_pf(r['bear_pf'])} "
                f"| {fmt_pnl(r['bear_total_pnl'])} | {fmt_pct(r['bear_avg_ret'])} |"
            )
    else:
        lines.append("| # | 모드 | Tgt | Min | SL | 약세거래 | 약세승률 | 약세PF | 약세총손익 | 약세거래당% |")
        lines.append("|---|------|---:|---:|----:|--------:|--------:|------:|----------:|----------:|")
        for i, r in bear_pos.head(15).iterrows():
            sl_s = f"{r['sl']:.0f}%" if r["mode"] == "tpsl" else "MA5"
            lines.append(
                f"| {i+1} | {r['mode']} | {r['tgt']:.0f}% | {r['min']:.0f}% | {sl_s} "
                f"| {int(r['bear_n_trades']):,} | {r['bear_win_rate']:.1f}% | {fmt_pf(r['bear_pf'])} "
                f"| {fmt_pnl(r['bear_total_pnl'])} | {fmt_pct(r['bear_avg_ret'])} |"
            )
    lines.append("")

    # ── 4. 강세장 + 약세장 모두 PnL > 0 인 조합 ──
    robust = df[(df["bull_total_pnl"] > 0) & (df["bear_total_pnl"] > 0)].sort_values("total_pnl", ascending=False)
    lines.append("## 4. 강세장 AND 약세장 둘 다 PnL > 0 인 robust 조합\n")
    if robust.empty:
        lines.append("- **❌ 양쪽 모두 플러스인 조합 없음**")
    else:
        lines.append("| # | 모드 | Tgt | Min | SL | 전체PnL | 강세PnL | 약세PnL | 강세PF | 약세PF |")
        lines.append("|---|------|---:|---:|----:|---------:|---------:|---------:|------:|------:|")
        for i, r in robust.head(15).iterrows():
            sl_s = f"{r['sl']:.0f}%" if r["mode"] == "tpsl" else "MA5"
            lines.append(
                f"| {i+1} | {r['mode']} | {r['tgt']:.0f}% | {r['min']:.0f}% | {sl_s} "
                f"| {fmt_pnl(r['total_pnl'])} | {fmt_pnl(r['bull_total_pnl'])} | {fmt_pnl(r['bear_total_pnl'])} "
                f"| {fmt_pf(r['bull_pf'])} | {fmt_pf(r['bear_pf'])} |"
            )
    lines.append("")

    # ── 5. 강세장만 plus, 약세장 minus 분포 ──
    lines.append("## 5. 그리드 전체 분포 요약\n")
    n_total = len(df)
    n_bull_pos = (df["bull_total_pnl"] > 0).sum()
    n_bear_pos = (df["bear_total_pnl"] > 0).sum()
    n_overall_pos = (df["total_pnl"] > 0).sum()
    n_both_pos = ((df["bull_total_pnl"] > 0) & (df["bear_total_pnl"] > 0)).sum()
    lines.append(f"- 총 그리드 조합: **{n_total}개**")
    lines.append(f"- 전체 기간 PnL > 0: **{n_overall_pos}개** ({n_overall_pos/n_total*100:.1f}%)")
    lines.append(f"- 강세장 PnL > 0: **{n_bull_pos}개** ({n_bull_pos/n_total*100:.1f}%)")
    lines.append(f"- 약세장 PnL > 0: **{n_bear_pos}개** ({n_bear_pos/n_total*100:.1f}%)")
    lines.append(f"- **양쪽 PnL > 0: {n_both_pos}개 ({n_both_pos/n_total*100:.1f}%)**")
    lines.append("")
    lines.append("| 메트릭 | 전체 평균 | 전체 중앙 | 강세 평균 | 약세 평균 |")
    lines.append("|--------|---------:|---------:|---------:|---------:|")
    lines.append(f"| 총손익 | {fmt_pnl(df['total_pnl'].mean())} | {fmt_pnl(df['total_pnl'].median())} | "
                 f"{fmt_pnl(df['bull_total_pnl'].mean())} | {fmt_pnl(df['bear_total_pnl'].mean())} |")
    lines.append(f"| 승률 | {df['win_rate'].mean():.1f}% | {df['win_rate'].median():.1f}% | "
                 f"{df['bull_win_rate'].mean():.1f}% | {df['bear_win_rate'].mean():.1f}% |")
    lines.append(f"| 거래당 % | {fmt_pct(df['avg_ret'].mean())} | {fmt_pct(df['avg_ret'].median())} | "
                 f"{fmt_pct(df['bull_avg_ret'].mean())} | {fmt_pct(df['bear_avg_ret'].mean())} |")
    pf_finite = df[df["pf"] != float("inf")]["pf"]
    lines.append(f"| PF | {pf_finite.mean():.2f} | {pf_finite.median():.2f} | "
                 f"{df['bull_pf'].replace(float('inf'), np.nan).mean():.2f} | "
                 f"{df['bear_pf'].replace(float('inf'), np.nan).mean():.2f} |")
    lines.append("")

    # ── 6. 결론 ──
    lines.append("## 6. 결론\n")
    if n_both_pos > 0:
        best_robust = robust.iloc[0]
        sl_s = f"{best_robust['sl']:.0f}%" if best_robust["mode"] == "tpsl" else "MA5"
        lines.append(f"### ✅ Robust 조합 발견\n")
        lines.append(f"- **{best_robust['mode']} / Tgt={best_robust['tgt']:.0f}% / Min={best_robust['min']:.0f}% / SL={sl_s}**")
        lines.append(f"- 전체 PnL: {fmt_pnl(best_robust['total_pnl'])}원 ({int(best_robust['n_trades']):,}거래, PF {fmt_pf(best_robust['pf'])})")
        lines.append(f"- 강세장 PnL: {fmt_pnl(best_robust['bull_total_pnl'])}원 (PF {fmt_pf(best_robust['bull_pf'])})")
        lines.append(f"- 약세장 PnL: {fmt_pnl(best_robust['bear_total_pnl'])}원 (PF {fmt_pf(best_robust['bear_pf'])})")
    else:
        lines.append(f"### ❌ Robust 조합 없음\n")
        lines.append(f"- 그리드 {n_total}개 조합 중 강세장+약세장 둘 다 수익인 조합 **0개**")
        lines.append(f"- 약세장 단독 수익 조합: {n_bear_pos}/{n_total} ({n_bear_pos/n_total*100:.1f}%)")
        lines.append(f"- 강세장 단독 수익 조합: {n_bull_pos}/{n_total} ({n_bull_pos/n_total*100:.1f}%)")
        lines.append(f"- → 매도 파라미터 최적화만으로는 약세장 손실 회피 불가능")
        lines.append(f"- → **계단뛰기 시그널 자체의 단기 음수 기댓값 (entry mean-reversion)** 이 본질적 원인")
    lines.append("")
    lines.append("## 실행 정보\n")
    lines.append(f"- 실행 시간: {elapsed:.2f}초 ({elapsed/60:.1f}분)")
    lines.append(f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def main():
    t0 = time.time()
    print("=" * 60)
    print("계단뛰기 풀기간 매도 최적화 (2019~2026)")
    print("=" * 60)

    ticker_list, name_map = load_universe()
    all_data = load_all(ticker_list, DATA_START, DATA_END)
    regime_df = load_kospi_regime()

    # 그리드 생성
    combos = []
    for t, m in product(TGT_GRID, MIN_GRID):
        if m >= t:
            continue
        for s in SL_GRID:
            combos.append(("tpsl", t, m, s))
        combos.append(("ma5", t, m, 0.0))
    print(f"\n[3/3] 그리드: {len(combos)}조합 백테스트 시작...")

    results = []
    for k, (mode, t, m, s) in enumerate(combos, 1):
        sub_t = time.time()
        trades = run_one_combo(t, m, s, mode, ticker_list, name_map, all_data)
        m_by = metrics_by_regime(trades, regime_df)
        row = {
            "mode": mode, "tgt": t, "min": m, "sl": s,
            "n_trades": m_by["전체"].get("n_trades", 0),
            "win_rate": m_by["전체"].get("win_rate", 0),
            "pf": m_by["전체"].get("pf", 0),
            "total_pnl": m_by["전체"].get("total_pnl", 0),
            "avg_ret": m_by["전체"].get("avg_ret", 0),
            "hold_days": m_by["전체"].get("hold_days", 0),
            "bull_n_trades": m_by["강세장"].get("n_trades", 0),
            "bull_win_rate": m_by["강세장"].get("win_rate", 0),
            "bull_pf": m_by["강세장"].get("pf", 0),
            "bull_total_pnl": m_by["강세장"].get("total_pnl", 0),
            "bull_avg_ret": m_by["강세장"].get("avg_ret", 0),
            "bear_n_trades": m_by["약세장"].get("n_trades", 0),
            "bear_win_rate": m_by["약세장"].get("win_rate", 0),
            "bear_pf": m_by["약세장"].get("pf", 0),
            "bear_total_pnl": m_by["약세장"].get("total_pnl", 0),
            "bear_avg_ret": m_by["약세장"].get("avg_ret", 0),
        }
        results.append(row)
        elapsed_s = time.time() - sub_t
        pf_s = fmt_pf(row["pf"])
        sl_s = f"S={s:.0f}" if mode == "tpsl" else "MA5"
        bull_p = "+" if row["bull_total_pnl"] > 0 else "-"
        bear_p = "+" if row["bear_total_pnl"] > 0 else "-"
        print(f"  [{k:>3}/{len(combos)}] {mode:>4} T={t:>4.0f} M={m:>3.0f} {sl_s:>4} | "
              f"거래 {row['n_trades']:>5,} 승률 {row['win_rate']:>5.1f}% PF {pf_s:>5} "
              f"PnL {row['total_pnl']:>+13,.0f} | bull {bull_p} bear {bear_p} ({elapsed_s:.1f}s)")

    elapsed = time.time() - t0
    report = make_report(results, elapsed)
    out = os.path.join(os.path.dirname(__file__), "results",
                        "stair_jump_optimize_full.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트: {out} ({elapsed:.2f}초, {elapsed/60:.1f}분)")

    csv_out = os.path.join(os.path.dirname(__file__), "results",
                            "stair_jump_optimize_full.csv")
    pd.DataFrame(results).to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"그리드 결과 CSV: {csv_out}")


if __name__ == "__main__":
    main()
