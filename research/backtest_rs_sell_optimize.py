"""
RS==N 진입 시 최적 매도 조건 탐색
- 진입: RS==N (정확 일치) & KOSPI200
- 매도 조건 비교:
  A) Peak RS 대비 -1 하락 (기존)
  B) Peak RS 대비 -2 하락
  C) Peak RS 대비 -3 하락
  D) RS < 진입값 (RS < N)
  E) RS < N-3
  F) RS < N-5
  G) RS < 90 (절대 하한)
- RS==90~99 각 진입값 × 매도조건 × 3기간
"""

import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from backtest_rs90_kospi200 import (
    ENGINE, FEE_BUY, FEE_SELL, TAX_SELL,
    get_kospi200_tickers, load_stock_prices, load_index_prices,
    calc_rs_at_date,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

INITIAL_CAPITAL = 100_000_000

PERIODS = [
    ("2023", "2023-01-02", "2023-12-29"),
    ("2024", "2024-01-02", "2024-12-30"),
    ("2025~", "2025-01-02", datetime.today().strftime("%Y-%m-%d")),
]

# 매도 조건 정의
SELL_CONDITIONS = [
    ("peak-1", "Peak-1 하락"),
    ("peak-2", "Peak-2 하락"),
    ("peak-3", "Peak-3 하락"),
    ("below-N", "RS<진입값"),
    ("below-N3", "RS<진입값-3"),
    ("below-N5", "RS<진입값-5"),
    ("below-90", "RS<90 절대하한"),
]

ENTRY_THRESHOLDS = list(range(99, 89, -1))


def check_sell(mode, entry_rs, cur_rs, peak_rs):
    """매도 조건 체크. 매도 사유 반환, 매도 안 하면 None"""
    if mode == "peak-1":
        if cur_rs < entry_rs:
            return f"RS={cur_rs}<{entry_rs}"
        if cur_rs < peak_rs:
            return f"peak{peak_rs}→{cur_rs}"
    elif mode == "peak-2":
        if cur_rs < entry_rs:
            return f"RS={cur_rs}<{entry_rs}"
        if peak_rs - cur_rs >= 2:
            return f"peak{peak_rs}→{cur_rs}(-{peak_rs-cur_rs})"
    elif mode == "peak-3":
        if cur_rs < entry_rs:
            return f"RS={cur_rs}<{entry_rs}"
        if peak_rs - cur_rs >= 3:
            return f"peak{peak_rs}→{cur_rs}(-{peak_rs-cur_rs})"
    elif mode == "below-N":
        if cur_rs < entry_rs:
            return f"RS={cur_rs}<{entry_rs}"
    elif mode == "below-N3":
        floor = max(entry_rs - 3, 1)
        if cur_rs < floor:
            return f"RS={cur_rs}<{floor}"
    elif mode == "below-N5":
        floor = max(entry_rs - 5, 1)
        if cur_rs < floor:
            return f"RS={cur_rs}<{floor}"
    elif mode == "below-90":
        if cur_rs < 90:
            return f"RS={cur_rs}<90"
    return None


def run_backtest(entry_rs, sell_mode, kospi200, pivot, index_prices,
                 rebal_dates, name_map, end_date):
    cash = INITIAL_CAPITAL
    holdings = {}
    portfolio_values = []
    trade_log = []
    last_date = pivot.index[pivot.index <= end_date][-1]

    for rdate in rebal_dates:
        rs_series = calc_rs_at_date(pivot, index_prices, rdate)
        if rs_series.empty:
            port_val = cash
            for tk, h in holdings.items():
                if rdate in pivot.index and tk in pivot.columns:
                    port_val += h["shares"] * pivot.loc[rdate, tk]
            portfolio_values.append((rdate, port_val))
            continue

        # 매도 판단
        sell_list = []
        for tk in list(holdings.keys()):
            cur_rs = int(rs_series[tk]) if tk in rs_series.index else 0
            h = holdings[tk]

            if tk not in kospi200:
                sell_list.append((tk, "KOSPI200제외"))
                continue

            reason = check_sell(sell_mode, entry_rs, cur_rs, h["peak_rs"])
            if reason:
                sell_list.append((tk, reason))
            else:
                # 보유 유지: peak 갱신
                if cur_rs > h["peak_rs"]:
                    h["peak_rs"] = cur_rs

        for tk, reason in sell_list:
            h = holdings.pop(tk)
            sell_price = pivot.loc[rdate, tk] if (rdate in pivot.index and tk in pivot.columns) else h["buy_price"]
            proceeds = h["shares"] * sell_price
            fee = proceeds * (FEE_SELL + TAX_SELL)
            cash += proceeds - fee
            ret_pct = (sell_price / h["buy_price"] - 1) * 100
            pnl = (sell_price - h["buy_price"]) * h["shares"] - fee - h["shares"] * h["buy_price"] * FEE_BUY
            trade_log.append({
                "ticker": tk, "name": name_map.get(tk, ""),
                "buy_date": h["buy_date"], "sell_date": rdate,
                "return_pct": ret_pct, "pnl": pnl,
                "hold_days": (rdate - h["buy_date"]).days,
                "reason": reason, "peak_rs": h["peak_rs"],
            })

        port_val = cash
        for tk, h in holdings.items():
            if rdate in pivot.index and tk in pivot.columns:
                port_val += h["shares"] * pivot.loc[rdate, tk]

        # 매수: RS == entry_rs
        new_buys = []
        for tk in rs_series.index:
            if tk in kospi200 and int(rs_series[tk]) == entry_rs and tk not in holdings:
                new_buys.append(tk)

        total_positions = len(holdings) + len(new_buys)
        if total_positions > 0:
            target_per_stock = port_val / total_positions
            for tk in new_buys:
                if rdate not in pivot.index or tk not in pivot.columns:
                    continue
                price = pivot.loc[rdate, tk]
                if price <= 0 or np.isnan(price):
                    continue
                buy_amount = min(target_per_stock, cash)
                shares = int(buy_amount / (price * (1 + FEE_BUY)))
                if shares <= 0:
                    continue
                cash -= shares * price * (1 + FEE_BUY)
                holdings[tk] = {
                    "shares": shares, "buy_price": price, "buy_date": rdate,
                    "peak_rs": entry_rs,
                }

        port_val = cash
        for tk, h in holdings.items():
            if rdate in pivot.index and tk in pivot.columns:
                port_val += h["shares"] * pivot.loc[rdate, tk]
        portfolio_values.append((rdate, port_val))

    # 미청산 마감
    for tk in list(holdings.keys()):
        h = holdings.pop(tk)
        sell_price = pivot.loc[last_date, tk] if tk in pivot.columns else h["buy_price"]
        proceeds = h["shares"] * sell_price
        fee = proceeds * (FEE_SELL + TAX_SELL)
        cash += proceeds - fee
        ret_pct = (sell_price / h["buy_price"] - 1) * 100
        pnl = (sell_price - h["buy_price"]) * h["shares"] - fee - h["shares"] * h["buy_price"] * FEE_BUY
        trade_log.append({
            "ticker": tk, "name": name_map.get(tk, ""),
            "buy_date": h["buy_date"], "sell_date": last_date,
            "return_pct": ret_pct, "pnl": pnl,
            "hold_days": (last_date - h["buy_date"]).days,
            "reason": "기간종료", "peak_rs": h["peak_rs"],
        })

    final_value = cash
    trades_df = pd.DataFrame(trade_log)

    vals = [v for _, v in portfolio_values]
    peak_v = vals[0] if vals else INITIAL_CAPITAL
    mdd = 0
    for v in vals:
        if v > peak_v:
            peak_v = v
        dd = (v - peak_v) / peak_v * 100
        if dd < mdd:
            mdd = dd

    return {
        "entry_rs": entry_rs,
        "sell_mode": sell_mode,
        "final_value": final_value,
        "total_return": (final_value / INITIAL_CAPITAL - 1) * 100,
        "mdd": mdd,
        "n_trades": len(trades_df),
        "win_rate": (trades_df["pnl"] > 0).sum() / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        "avg_return": trades_df["return_pct"].mean() if len(trades_df) > 0 else 0,
        "avg_hold_days": trades_df["hold_days"].mean() if len(trades_df) > 0 else 0,
        "avg_peak_rs": trades_df["peak_rs"].mean() if len(trades_df) > 0 else entry_rs,
        "portfolio_values": portfolio_values,
    }


def main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    print("[1/3] 데이터 로딩...")
    kospi200 = get_kospi200_tickers()
    stock_df = load_stock_prices("2023-01-02", datetime.today().strftime("%Y-%m-%d"))
    name_map = stock_df.groupby("ticker")["name"].first().to_dict()
    stock_df = stock_df.drop_duplicates(subset=["date", "ticker"], keep="last")
    pivot = stock_df.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index().ffill()
    index_prices = load_index_prices("^KS11", "2023-01-02", datetime.today().strftime("%Y-%m-%d"))
    print(f"      {len(kospi200)} KOSPI200, {pivot.shape[1]}종목")

    # 모든 결과 저장: results[period][entry_rs][sell_mode]
    all_results = {}
    all_benchmarks = {}

    print("\n[2/3] 백테스트 실행...")

    for pname, start_date, end_date in PERIODS:
        print(f"\n{'='*70}")
        print(f"  {pname} ({start_date} ~ {end_date})")
        print(f"{'='*70}")

        period_dates = pivot.index[(pivot.index >= start_date) & (pivot.index <= end_date)]
        rebal_dates = []
        last_week = None
        for d in period_dates:
            key = (d.year, d.isocalendar()[1])
            if key != last_week:
                rebal_dates.append(d)
                last_week = key

        # RS 캐싱
        rs_cache = {}
        for rd in rebal_dates:
            rs_cache[rd] = calc_rs_at_date(pivot, index_prices, rd)

        import backtest_rs90_kospi200
        orig_calc = backtest_rs90_kospi200.calc_rs_at_date
        def make_cached(cache):
            def cached(p, ip, rd):
                return cache.get(rd, pd.Series(dtype=float))
            return cached
        backtest_rs90_kospi200.calc_rs_at_date = make_cached(rs_cache)

        idx_vals = index_prices.loc[(index_prices.index >= start_date) & (index_prices.index <= end_date)]
        bench_ret = (idx_vals.iloc[-1] / idx_vals.iloc[0] - 1) * 100 if len(idx_vals) > 0 else 0
        all_benchmarks[pname] = bench_ret

        all_results[pname] = {}

        for entry_rs in ENTRY_THRESHOLDS:
            all_results[pname][entry_rs] = {}
            results_row = []

            for sell_key, sell_label in SELL_CONDITIONS:
                # below-N3, below-N5는 entry_rs에 따라 floor가 달라짐
                # below-90은 entry < 90이면 의미 없음 → skip
                if sell_key == "below-90" and entry_rs <= 90:
                    continue
                if sell_key == "below-N5" and entry_rs - 5 < 1:
                    continue
                if sell_key == "below-N3" and entry_rs - 3 < 1:
                    continue

                r = run_backtest(entry_rs, sell_key, kospi200, pivot, index_prices,
                                 rebal_dates, name_map, end_date)
                all_results[pname][entry_rs][sell_key] = r
                results_row.append(r)

            # 요약 출력 (진입값별 한 줄씩)
            best = max(results_row, key=lambda x: x["total_return"]) if results_row else None
            if best:
                print(f"  RS=={entry_rs:2d} best: {best['sell_mode']:>10s} {best['total_return']:>+8.2f}% "
                      f"(MDD {best['mdd']:.1f}%, {best['n_trades']}건, 보유 {best['avg_hold_days']:.0f}일)")

        backtest_rs90_kospi200.calc_rs_at_date = orig_calc

    elapsed = time.time() - start_time

    # 리포트 생성
    print(f"\n[3/3] 리포트 & 차트...")

    lines = []
    lines.append("# RS==N 진입 × 매도조건 최적화 백테스트\n")
    lines.append("## 전략 설명\n")
    lines.append("- **매수**: RS가 정확히 N인 KOSPI200 종목")
    lines.append("- **매도 조건 비교**:")
    lines.append("  - peak-1: Peak RS 대비 1점 하락 시 매도")
    lines.append("  - peak-2: Peak RS 대비 2점 하락 시 매도")
    lines.append("  - peak-3: Peak RS 대비 3점 하락 시 매도")
    lines.append("  - below-N: RS < 진입값 시 매도")
    lines.append("  - below-N3: RS < 진입값-3 시 매도")
    lines.append("  - below-N5: RS < 진입값-5 시 매도")
    lines.append("  - below-90: RS < 90 시 매도\n")

    # 진입값별 × 매도조건별 × 기간별 종합 테이블
    for entry_rs in ENTRY_THRESHOLDS:
        lines.append(f"## RS=={entry_rs} 진입\n")

        # 매도조건별 × 기간별 수익률 테이블
        lines.append("| 매도조건 | 2023 | 2024 | 2025~ | 평균 | 2023 MDD | 2024 MDD | 2025~ MDD | 2023 보유일 | 2024 보유일 | 2025~ 보유일 |")
        lines.append("|---------|------|------|-------|------|---------|---------|----------|----------|----------|----------|")

        for sell_key, sell_label in SELL_CONDITIONS:
            row = f"| {sell_label} |"
            rets = []
            mdds = []
            holds = []
            valid = True
            for pname, _, _ in PERIODS:
                if entry_rs in all_results[pname] and sell_key in all_results[pname][entry_rs]:
                    r = all_results[pname][entry_rs][sell_key]
                    row += f" {r['total_return']:+.1f}% |"
                    rets.append(r["total_return"])
                    mdds.append(r["mdd"])
                    holds.append(r["avg_hold_days"])
                else:
                    row += " - |"
                    valid = False

            if valid and rets:
                row += f" {np.mean(rets):+.1f}% |"
                for m in mdds:
                    row += f" {m:.1f}% |"
                for h in holds:
                    row += f" {h:.0f}일 |"
            else:
                row += " - |" + " - |" * 6

            lines.append(row)
        lines.append("")

    # 최적 매도조건 요약 (진입값별 3기간 평균 수익률 기준 best)
    lines.append("## 진입값별 최적 매도조건 요약\n")
    lines.append("| 진입RS | 최적매도 | 평균수익률 | 2023 | 2024 | 2025~ | 평균MDD | 평균보유일 | 3기간양수? |")
    lines.append("|--------|---------|----------|------|------|-------|--------|----------|----------|")

    for entry_rs in ENTRY_THRESHOLDS:
        best_mode = None
        best_avg = -999
        best_data = None

        for sell_key, sell_label in SELL_CONDITIONS:
            rets = []
            valid = True
            for pname, _, _ in PERIODS:
                if entry_rs in all_results[pname] and sell_key in all_results[pname][entry_rs]:
                    rets.append(all_results[pname][entry_rs][sell_key]["total_return"])
                else:
                    valid = False
                    break

            if valid and rets:
                avg = np.mean(rets)
                if avg > best_avg:
                    best_avg = avg
                    best_mode = sell_label
                    best_data = {
                        "rets": rets,
                        "mdds": [all_results[pname][entry_rs][sell_key]["mdd"]
                                 for pname, _, _ in PERIODS],
                        "holds": [all_results[pname][entry_rs][sell_key]["avg_hold_days"]
                                  for pname, _, _ in PERIODS],
                    }

        if best_data:
            all_pos = all(r > all_benchmarks[pname] for r, (pname, _, _) in zip(best_data["rets"], PERIODS))
            lines.append(
                f"| RS=={entry_rs} | {best_mode} | {best_avg:+.1f}% | "
                f"{best_data['rets'][0]:+.1f}% | {best_data['rets'][1]:+.1f}% | {best_data['rets'][2]:+.1f}% | "
                f"{np.mean(best_data['mdds']):.1f}% | {np.mean(best_data['holds']):.0f}일 | "
                f"{'O' if all_pos else 'X'} |"
            )
    lines.append("")

    # KOSPI 벤치마크
    lines.append("| KOSPI | - | "
                 f"{np.mean(list(all_benchmarks.values())):+.1f}% | "
                 f"{all_benchmarks['2023']:+.1f}% | {all_benchmarks['2024']:+.1f}% | "
                 f"{all_benchmarks['2025~']:+.1f}% | - | - | - |")
    lines.append("")

    lines.append(f"\n---\n실행 시간: {elapsed:.1f}초")
    lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report = "\n".join(lines)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_rs_sell_optimize.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 차트: 진입값별 최적 매도조건의 수익률 비교
    chart_dir = os.path.join(base_dir, "charts_rs_sell_optimize")
    os.makedirs(chart_dir, exist_ok=True)

    # 히트맵: 진입RS × 매도조건 → 3기간 평균 수익률
    sell_keys = [k for k, _ in SELL_CONDITIONS]
    sell_labels = [l for _, l in SELL_CONDITIONS]
    data = np.full((len(ENTRY_THRESHOLDS), len(sell_keys)), np.nan)

    for i, entry_rs in enumerate(ENTRY_THRESHOLDS):
        for j, sell_key in enumerate(sell_keys):
            rets = []
            for pname, _, _ in PERIODS:
                if entry_rs in all_results[pname] and sell_key in all_results[pname][entry_rs]:
                    rets.append(all_results[pname][entry_rs][sell_key]["total_return"])
            if rets:
                data[i, j] = np.mean(rets)

    fig, ax = plt.subplots(figsize=(14, 8))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(sell_labels)))
    ax.set_xticklabels(sell_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(ENTRY_THRESHOLDS)))
    ax.set_yticklabels([f"RS=={t}" for t in ENTRY_THRESHOLDS])

    for i in range(len(ENTRY_THRESHOLDS)):
        for j in range(len(sell_keys)):
            if not np.isnan(data[i, j]):
                val = data[i, j]
                vrange = np.nanmax(data) - np.nanmin(data)
                color = "white" if vrange > 0 and (val - np.nanmin(data)) > vrange * 0.7 else "black"
                ax.text(j, i, f"{val:+.0f}%", ha="center", va="center", fontsize=8,
                        fontweight="bold", color=color)

    ax.set_title("진입RS × 매도조건 → 3기간 평균 수익률 (%)", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(chart_dir, "heatmap_avg_return.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 히트맵2: 평균 보유일
    data_hold = np.full((len(ENTRY_THRESHOLDS), len(sell_keys)), np.nan)
    for i, entry_rs in enumerate(ENTRY_THRESHOLDS):
        for j, sell_key in enumerate(sell_keys):
            holds = []
            for pname, _, _ in PERIODS:
                if entry_rs in all_results[pname] and sell_key in all_results[pname][entry_rs]:
                    holds.append(all_results[pname][entry_rs][sell_key]["avg_hold_days"])
            if holds:
                data_hold[i, j] = np.mean(holds)

    fig, ax = plt.subplots(figsize=(14, 8))
    masked = np.ma.masked_invalid(data_hold)
    im = ax.imshow(masked, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(sell_labels)))
    ax.set_xticklabels(sell_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(ENTRY_THRESHOLDS)))
    ax.set_yticklabels([f"RS=={t}" for t in ENTRY_THRESHOLDS])

    for i in range(len(ENTRY_THRESHOLDS)):
        for j in range(len(sell_keys)):
            if not np.isnan(data_hold[i, j]):
                ax.text(j, i, f"{data_hold[i,j]:.0f}일", ha="center", va="center",
                        fontsize=8, fontweight="bold")

    ax.set_title("진입RS × 매도조건 → 평균 보유일", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(chart_dir, "heatmap_hold_days.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"      차트: {chart_dir}/")

    # 최종 요약
    print(f"\n{'='*80}")
    print("매도조건 최적화 완료!")
    print(f"{'='*80}")

    print(f"\n진입RS  최적매도조건      평균수익률   2023      2024      2025~     평균보유일")
    print("-" * 85)

    for entry_rs in ENTRY_THRESHOLDS:
        best_mode = None
        best_avg = -999
        best_rets = []
        best_hold = 0

        for sell_key, sell_label in SELL_CONDITIONS:
            rets = []
            holds = []
            for pname, _, _ in PERIODS:
                if entry_rs in all_results[pname] and sell_key in all_results[pname][entry_rs]:
                    rets.append(all_results[pname][entry_rs][sell_key]["total_return"])
                    holds.append(all_results[pname][entry_rs][sell_key]["avg_hold_days"])
            if len(rets) == len(PERIODS):
                avg = np.mean(rets)
                if avg > best_avg:
                    best_avg = avg
                    best_mode = sell_label
                    best_rets = rets
                    best_hold = np.mean(holds)

        if best_rets:
            print(f"  RS=={entry_rs:2d}  {best_mode:<14s}  {best_avg:>+8.1f}%  "
                  f"{best_rets[0]:>+8.1f}%  {best_rets[1]:>+8.1f}%  {best_rets[2]:>+8.1f}%  "
                  f"{best_hold:>6.0f}일")

    print(f"\n리포트: {report_path}")
    print(f"실행 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
