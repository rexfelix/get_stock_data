"""
계단뛰기 약세장(2022~2024) 검증 — MA5 손절 모드.

2025+ 우상향 시장에서 발견한 (T=20% / M=5% / MA5) 최적해의 시기 의존성 검증.
- 2022 (KOSPI 약 -25% 약세장)
- 2023 (회복 시작)
- 2024 (혼조)
- 2022~2024 통합

각 기간별로 MA5 그리드 (Tgt × Min) 재실행 + 2025-2026 결과와 종합 비교.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd

from backtest_stair_jump import (
    calc_indicators, run_backtest, _metrics,
    load_universe, load_all, INITIAL_CAPITAL,
    MCAP_THRESHOLD, PRICE_FLOOR,
)


PERIODS = [
    ("2022-01-01", "2022-12-31", "2022(약세)"),
    ("2023-01-01", "2023-12-31", "2023(회복)"),
    ("2024-01-01", "2024-12-31", "2024(혼조)"),
    ("2022-01-01", "2024-12-31", "2022~2024"),
    ("2025-01-01", "2026-05-17", "2025~26(강세, 참고)"),
]

DATA_START = "2022-01-01"
DATA_END = "2026-05-17"

# MA5 그리드
TGT_GRID = [5.0, 7.0, 10.0, 15.0, 20.0]
MIN_GRID = [1.0, 2.0, 3.0, 5.0]


def run_period_grid(start: str, end: str, ticker_list, name_map, all_data) -> list[dict]:
    """주어진 기간에 MA5 그리드 (Tgt × Min) 백테스트."""
    rows = []
    combos = [(t, m) for t in TGT_GRID for m in MIN_GRID if m < t]
    for t, mn in combos:
        summaries = []
        all_trades = []
        n_signals_total = 0
        for ticker in ticker_list:
            if ticker not in all_data:
                continue
            df = all_data[ticker]
            df_ind = calc_indicators(df)
            df_test = df_ind.loc[start:end]
            if len(df_test) < 5:
                continue
            nm = name_map.get(ticker, ticker)
            cap, trades, equity, n_sig = run_backtest(
                df_test, ticker, nm, t, mn, 0.0, exit_mode="ma5",
            )
            n_signals_total += n_sig
            if trades:
                t_df = pd.DataFrame(trades)
                wins = (t_df["pnl"] > 0).sum()
                summaries.append({
                    "ticker": ticker, "name": nm,
                    "n_signals": n_sig, "n_trades": len(trades),
                    "final_equity": equity[-1]["equity"],
                    "win_rate": wins / len(trades) * 100,
                    "total_pnl": t_df["pnl"].sum(),
                    "avg_return": t_df["return_pct"].mean(),
                })
                all_trades.append(t_df)
            else:
                summaries.append({
                    "ticker": ticker, "name": nm,
                    "n_signals": n_sig, "n_trades": 0,
                    "final_equity": INITIAL_CAPITAL,
                    "win_rate": 0, "total_pnl": 0, "avg_return": 0,
                })
        sum_df = pd.DataFrame(summaries)
        trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        m = _metrics(trades_df, sum_df)
        rows.append({
            "tgt": t, "min": mn,
            "n_signals": n_signals_total,
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "pf": m["pf"],
            "total_pnl": m["total_pnl"],
            "avg_ret": m["avg_ret"],
            "stock_avg_ret": m["stock_avg_ret"],
            "hold_days": m["hold_days"],
        })
    return rows


def make_report(results: dict, elapsed: float) -> str:
    """results = {label: rows}"""
    lines = []
    lines.append("# 계단뛰기 약세장(2022~2024) 검증 — MA5 손절 모드\n")
    lines.append("## 배경\n")
    lines.append("- 2025-01-01 ~ 2026-05-17 (강세장) 19조합 그리드 최적해 = **Tgt=20% / Min=5% / MA5** (PF 1.37, +197.8M원, 종목평균 +2.85%)")
    lines.append("- 본 리포트: 동일 그리드를 약세·회복·혼조 기간에 재실행하여 **시기 의존성** 확인")
    lines.append(f"- 유니버스: 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억, 종가 ≥ {PRICE_FLOOR:,}원")
    lines.append(f"- 초기자본 종목당 {INITIAL_CAPITAL:,}원")
    lines.append("")

    # ── 1. 각 기간의 최적해 (PF·총손익 기준) ──
    lines.append("## 1. 기간별 최적 파라미터 (총손익 기준)\n")
    lines.append("| 기간 | 시그널 | 거래 | 최적(T/M) | 승률 | PF | 총손익 | 거래당% | 종목평균% | 보유일 |")
    lines.append("|------|------:|-----:|----------|-----:|----:|--------:|--------:|----------:|------:|")
    for label, rows in results.items():
        df = pd.DataFrame(rows)
        if df.empty or df["n_trades"].sum() == 0:
            lines.append(f"| {label} | 0 | 0 | — | — | — | — | — | — | — |")
            continue
        best = df.sort_values("total_pnl", ascending=False).iloc[0]
        pf_s = f"{best['pf']:.2f}" if best["pf"] != float("inf") else "∞"
        lines.append(
            f"| {label} | {int(best['n_signals']):,} | {int(best['n_trades']):,} "
            f"| T={best['tgt']:.0f}/M={best['min']:.0f} | {best['win_rate']:.1f}% | {pf_s} "
            f"| {best['total_pnl']:,.0f} | {best['avg_ret']:.2f}% | {best['stock_avg_ret']:.2f}% | {best['hold_days']:.1f} |"
        )
    lines.append("")

    # ── 2. 2025+ 채택해(T=20/M=5)의 기간별 성과 추적 ──
    lines.append("## 2. 2025+ 채택해 (T=20% / M=5% / MA5) 기간별 성과\n")
    lines.append("| 기간 | 시그널 | 거래 | 승률 | PF | 총손익 | 거래당% | 종목평균% | 보유일 |")
    lines.append("|------|------:|-----:|-----:|----:|--------:|--------:|----------:|------:|")
    for label, rows in results.items():
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        match = df[(df["tgt"] == 20.0) & (df["min"] == 5.0)]
        if match.empty:
            lines.append(f"| {label} | — | — | — | — | — | — | — | — |")
            continue
        r = match.iloc[0]
        pf_s = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "∞"
        lines.append(
            f"| {label} | {int(r['n_signals']):,} | {int(r['n_trades']):,} "
            f"| {r['win_rate']:.1f}% | {pf_s} | {r['total_pnl']:,.0f} "
            f"| {r['avg_ret']:.2f}% | {r['stock_avg_ret']:.2f}% | {r['hold_days']:.1f} |"
        )
    lines.append("")

    # ── 3. 기간 × 그리드 전체 표 (PF) ──
    lines.append("## 3. 기간 × 그리드 PF 매트릭스\n")
    grid_combos = [(t, m) for t in TGT_GRID for m in MIN_GRID if m < t]
    header = "| Tgt / Min | " + " | ".join(label for label in results.keys()) + " |"
    sep = "|---|" + "|".join(":---:" for _ in results) + "|"
    lines.append(header)
    lines.append(sep)
    for t, mn in grid_combos:
        cells = []
        for label, rows in results.items():
            df = pd.DataFrame(rows)
            match = df[(df["tgt"] == t) & (df["min"] == mn)]
            if match.empty:
                cells.append("—")
            else:
                pf = match.iloc[0]["pf"]
                if pf == float("inf"):
                    cells.append("∞")
                else:
                    cells.append(f"{pf:.2f}")
        lines.append(f"| T={t:.0f}/M={mn:.0f} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── 4. 기간 × 그리드 총손익 매트릭스 (단위: 억원) ──
    lines.append("## 4. 기간 × 그리드 총손익 매트릭스 (단위: 억원)\n")
    lines.append(header)
    lines.append(sep)
    for t, mn in grid_combos:
        cells = []
        for label, rows in results.items():
            df = pd.DataFrame(rows)
            match = df[(df["tgt"] == t) & (df["min"] == mn)]
            if match.empty:
                cells.append("—")
            else:
                pnl_eok = match.iloc[0]["total_pnl"] / 1e8
                cells.append(f"{pnl_eok:+.2f}")
        lines.append(f"| T={t:.0f}/M={mn:.0f} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── 5. 종합 고찰 ──
    lines.append("## 5. 종합 고찰\n")

    # 각 기간에서 T=20/M=5 가 그 기간의 최적인지 검증
    consistency = []
    for label, rows in results.items():
        df = pd.DataFrame(rows)
        if df.empty or df["n_trades"].sum() == 0:
            continue
        best = df.sort_values("total_pnl", ascending=False).iloc[0]
        adopted = df[(df["tgt"] == 20.0) & (df["min"] == 5.0)]
        if adopted.empty:
            continue
        a = adopted.iloc[0]
        gap_pct = (a["total_pnl"] - best["total_pnl"]) / abs(best["total_pnl"]) * 100 if best["total_pnl"] != 0 else 0
        consistency.append((label, best["tgt"], best["min"], best["total_pnl"], a["total_pnl"], gap_pct, a["pf"]))

    lines.append("### 채택해 vs 기간별 최적 비교\n")
    lines.append("| 기간 | 그 기간 최적(T/M) | 최적 총손익 | 채택해(T=20/M=5) 총손익 | 차이 | 채택해 PF |")
    lines.append("|------|-------------------|------------:|------------------------:|-----:|---------:|")
    for label, bt, bm, bp, ap, gp, apf in consistency:
        pf_s = f"{apf:.2f}" if apf != float("inf") else "∞"
        lines.append(f"| {label} | T={bt:.0f}/M={bm:.0f} | {bp:,.0f} | {ap:,.0f} | {gp:+.1f}% | {pf_s} |")
    lines.append("")

    # 약세장(2022) PF/PnL 추출
    df_2022 = pd.DataFrame(results.get("2022(약세)", []))
    if not df_2022.empty and df_2022["n_trades"].sum() > 0:
        best_2022 = df_2022.sort_values("total_pnl", ascending=False).iloc[0]
        adopted_2022 = df_2022[(df_2022["tgt"] == 20.0) & (df_2022["min"] == 5.0)]
        positive_2022 = (df_2022["total_pnl"] > 0).sum()
        lines.append("### 약세장(2022) 특이사항\n")
        lines.append(f"- 그리드 {len(df_2022)}조합 중 **총손익 플러스 = {positive_2022}개** (모두 플러스 시 19/19)")
        lines.append(f"- 2022 최적: T={best_2022['tgt']:.0f}/M={best_2022['min']:.0f}, PF {best_2022['pf']:.2f}, 총손익 {best_2022['total_pnl']:,.0f}원")
        if not adopted_2022.empty:
            a = adopted_2022.iloc[0]
            apf = a['pf'] if a['pf'] != float('inf') else 999
            judgement = "✅ 약세장 방어 성공" if a["total_pnl"] > 0 else "❌ 약세장 손실"
            lines.append(f"- 채택해 (T=20/M=5): PF {apf:.2f}, 총손익 {a['total_pnl']:,.0f}원 → {judgement}")
        lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- 실행 시간: {elapsed:.2f}초")
    lines.append(f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    print("=" * 60)
    print("계단뛰기 약세장(2022~2024) 검증 — MA5 손절 모드")
    print("=" * 60)

    ticker_list, name_map = load_universe()
    # 데이터 한 번만 로드 (전 기간 커버)
    all_data = load_all(ticker_list, DATA_START, DATA_END)

    results = {}
    for sd, ed, label in PERIODS:
        print(f"\n[{label}] {sd} ~ {ed}")
        period_start = time.time()
        rows = run_period_grid(sd, ed, ticker_list, name_map, all_data)
        df = pd.DataFrame(rows)
        if not df.empty and df["n_trades"].sum() > 0:
            best = df.sort_values("total_pnl", ascending=False).iloc[0]
            pf_s = f"{best['pf']:.2f}" if best['pf'] != float('inf') else "∞"
            print(f"  최적 T={best['tgt']:.0f}/M={best['min']:.0f} → PF {pf_s}, "
                  f"PNL {best['total_pnl']:,.0f}, 거래 {int(best['n_trades']):,} "
                  f"({time.time()-period_start:.1f}s)")
            adopted = df[(df["tgt"] == 20.0) & (df["min"] == 5.0)]
            if not adopted.empty:
                a = adopted.iloc[0]
                apf = f"{a['pf']:.2f}" if a['pf'] != float('inf') else "∞"
                print(f"  채택해 T=20/M=5 → PF {apf}, PNL {a['total_pnl']:,.0f}, "
                      f"거래 {int(a['n_trades']):,}")
        else:
            print(f"  거래 없음 ({time.time()-period_start:.1f}s)")
        results[label] = rows

    elapsed = time.time() - start_time
    report = make_report(results, elapsed)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "backtest_stair_jump_yearly.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n{'='*60}")
    print(f"리포트: {out} ({elapsed:.2f}초)")


if __name__ == "__main__":
    main()
