"""
RS>=90~99 & KOSPI200 매일 리밸런싱 전략 백테스트
- 기존 매주 리밸런싱 vs 매일 리밸런싱 비교
- RS>=90~99 각 임계값 × 3개 기간(2023, 2024, 2025~)
- 매일 RS를 재계산하고 조건 충족/미충족 시 즉시 매수/매도
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

THRESHOLDS = list(range(99, 89, -1))  # 99, 98, ..., 90


def run_single_backtest(threshold, kospi200, pivot, index_prices,
                        eval_dates, name_map, start_date, end_date):
    """단일 임계값 + 단일 기간 백테스트 (매일 평가)"""
    cash = INITIAL_CAPITAL
    holdings = {}
    portfolio_values = []
    trade_log = []

    last_date = pivot.index[pivot.index <= end_date][-1]

    for rdate in eval_dates:
        rs_series = calc_rs_at_date(pivot, index_prices, rdate)
        if rs_series.empty:
            port_val = cash
            for tk, h in holdings.items():
                if rdate in pivot.index and tk in pivot.columns:
                    port_val += h["shares"] * pivot.loc[rdate, tk]
            portfolio_values.append((rdate, port_val))
            continue

        target_tickers = set()
        for tk in rs_series.index:
            if tk in kospi200 and rs_series[tk] >= threshold:
                target_tickers.add(tk)

        # 매도
        for tk in [t for t in list(holdings.keys()) if t not in target_tickers]:
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
            })

        # 포트 가치
        port_val = cash
        for tk, h in holdings.items():
            if rdate in pivot.index and tk in pivot.columns:
                port_val += h["shares"] * pivot.loc[rdate, tk]

        # 신규 매수
        new_buys = target_tickers - set(holdings.keys())
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
                holdings[tk] = {"shares": shares, "buy_price": price, "buy_date": rdate}

        port_val = cash
        for tk, h in holdings.items():
            if rdate in pivot.index and tk in pivot.columns:
                port_val += h["shares"] * pivot.loc[rdate, tk]
        portfolio_values.append((rdate, port_val))

    # 미청산 포지션 마감
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
        })

    final_value = cash
    trades_df = pd.DataFrame(trade_log)

    # MDD
    vals = [v for _, v in portfolio_values]
    peak = vals[0] if vals else INITIAL_CAPITAL
    mdd = 0
    for v in vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < mdd:
            mdd = dd

    return {
        "threshold": threshold,
        "final_value": final_value,
        "total_return": (final_value / INITIAL_CAPITAL - 1) * 100,
        "mdd": mdd,
        "n_trades": len(trades_df),
        "win_rate": (trades_df["pnl"] > 0).sum() / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        "avg_return": trades_df["return_pct"].mean() if len(trades_df) > 0 else 0,
        "med_return": trades_df["return_pct"].median() if len(trades_df) > 0 else 0,
        "avg_hold_days": trades_df["hold_days"].mean() if len(trades_df) > 0 else 0,
        "max_profit": trades_df["return_pct"].max() if len(trades_df) > 0 else 0,
        "max_loss": trades_df["return_pct"].min() if len(trades_df) > 0 else 0,
        "portfolio_values": portfolio_values,
        "trades_df": trades_df,
    }


def generate_charts(all_results, all_benchmarks, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    period_names = [p[0] for p in PERIODS]
    n_periods = len(period_names)

    cmap = plt.cm.Greens
    colors = [cmap(0.3 + 0.7 * i / (len(THRESHOLDS) - 1)) for i in range(len(THRESHOLDS))]

    # 1) 기간별 수익률 곡선
    fig, axes = plt.subplots(1, n_periods, figsize=(7 * n_periods, 7), sharey=False)
    if n_periods == 1:
        axes = [axes]

    for pi, pname in enumerate(period_names):
        ax = axes[pi]
        results = all_results[pname]
        bench_ret = all_benchmarks[pname]

        for i, r in enumerate(results):
            dates = [d for d, _ in r["portfolio_values"]]
            returns = [(v / INITIAL_CAPITAL - 1) * 100 for _, v in r["portfolio_values"]]
            ax.plot(dates, returns, color=colors[i], linewidth=1.5,
                    label=f"RS>={r['threshold']} ({r['total_return']:+.0f}%)", alpha=0.85)

        ax.axhline(bench_ret, color="red", linewidth=2, linestyle="--",
                   label=f"KOSPI ({bench_ret:+.1f}%)")
        ax.set_title(f"{pname} (매일 리밸런싱)", fontsize=14, fontweight="bold")
        ax.set_ylabel("수익률 (%)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.suptitle("RS>=90~99 매일 리밸런싱 수익률 곡선 - 연도별", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "return_curves_yearly.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) 총수익률 그룹 바 차트
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(THRESHOLDS))
    width = 0.25

    period_colors = ["#42a5f5", "#66bb6a", "#ef5350"]
    for pi, pname in enumerate(period_names):
        results = all_results[pname]
        rets = [r["total_return"] for r in results]
        bars = ax.bar(x + pi * width, rets, width, label=pname, color=period_colors[pi],
                      edgecolor="black", linewidth=0.5, alpha=0.8)
        for bar, ret in zip(bars, rets):
            ax.text(bar.get_x() + bar.get_width() / 2, ret + (2 if ret >= 0 else -5),
                    f"{ret:+.0f}%", ha="center", va="bottom" if ret >= 0 else "top",
                    fontsize=7, fontweight="bold")

    for pi, pname in enumerate(period_names):
        ax.axhline(all_benchmarks[pname], color=period_colors[pi], linewidth=1.5,
                   linestyle="--", alpha=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"RS>={t}" for t in THRESHOLDS], fontsize=10)
    ax.set_title("RS>=90~99 매일 리밸런싱 총수익률 - 연도별", fontsize=14)
    ax.set_ylabel("총수익률 (%)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "total_return_grouped.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) MDD 그룹 바 차트
    fig, ax = plt.subplots(figsize=(16, 6))
    for pi, pname in enumerate(period_names):
        results = all_results[pname]
        mdds = [r["mdd"] for r in results]
        ax.bar(x + pi * width, mdds, width, label=pname, color=period_colors[pi],
               edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"RS>={t}" for t in THRESHOLDS], fontsize=10)
    ax.set_title("RS>=90~99 매일 리밸런싱 MDD - 연도별", fontsize=14)
    ax.set_ylabel("MDD (%)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mdd_grouped.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4) 수익/MDD 비율 히트맵
    fig, ax = plt.subplots(figsize=(12, 5))
    data = []
    for pname in period_names:
        row = []
        for r in all_results[pname]:
            ratio = r["total_return"] / abs(r["mdd"]) if r["mdd"] != 0 else 0
            row.append(ratio)
        data.append(row)

    data_arr = np.array(data)
    im = ax.imshow(data_arr, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels([f"RS>={t}" for t in THRESHOLDS])
    ax.set_yticks(range(len(period_names)))
    ax.set_yticklabels(period_names)

    for i in range(len(period_names)):
        for j in range(len(THRESHOLDS)):
            val = data_arr[i, j]
            color = "white" if abs(val) > (data_arr.max() - data_arr.min()) * 0.6 + data_arr.min() else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)

    ax.set_title("수익/MDD 비율 히트맵 - 매일 리밸런싱 (높을수록 효율적)", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "return_mdd_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"      차트 저장: {output_dir}/")


def generate_report(all_results, all_benchmarks, elapsed):
    lines = []
    lines.append("# RS>=90~99 & KOSPI200 매일 리밸런싱 전략 백테스트\n")
    lines.append("## 가설: 매일 평가하면 더 빠른 진입/이탈로 수익이 개선되는가?\n")
    lines.append(f"- **전략**: 매일 RS Rating 재계산, RS>=임계값 & KOSPI200 종목만 동일가중 보유")
    lines.append(f"- **리밸런싱**: 매일 (매 영업일)")
    lines.append(f"- **초기자본**: {INITIAL_CAPITAL:,}원")
    lines.append(f"- **비교 기간**: 2023년, 2024년, 2025년~현재")
    lines.append(f"- **비교 임계값**: RS>=99~90 (10단계)\n")

    # 기간별 종합 비교
    for pname, start, end in PERIODS:
        results = all_results[pname]
        bench = all_benchmarks[pname]

        lines.append(f"## {pname} ({start} ~ {end})\n")
        lines.append(f"KOSPI 벤치마크: **{bench:+.2f}%**\n")
        lines.append("| 임계값 | 총수익률 | 초과수익 | MDD | 수익/MDD | 거래수 | 승률 | 평균수익률 | 평균보유일 |")
        lines.append("|--------|---------|---------|------|---------|--------|------|----------|----------|")

        for r in results:
            excess = r["total_return"] - bench
            ratio = r["total_return"] / abs(r["mdd"]) if r["mdd"] != 0 else 0
            lines.append(
                f"| RS>={r['threshold']} | {r['total_return']:+.2f}% | {excess:+.2f}%p | "
                f"{r['mdd']:.2f}% | {ratio:.2f} | {r['n_trades']} | {r['win_rate']:.1f}% | "
                f"{r['avg_return']:.2f}% | {r['avg_hold_days']:.1f}일 |"
            )
        lines.append("")

    # 크로스 기간 비교
    lines.append("## 임계값별 x 기간별 총수익률 크로스 비교\n")
    header = "| 임계값 |"
    sep = "|--------|"
    for pname, _, _ in PERIODS:
        header += f" {pname} |"
        sep += "---------|"
    header += " 평균 | 표준편차 |"
    sep += "------|---------|"
    lines.append(header)
    lines.append(sep)

    for ti, th in enumerate(THRESHOLDS):
        row = f"| RS>={th} |"
        rets = []
        for pname, _, _ in PERIODS:
            r = all_results[pname][ti]
            row += f" {r['total_return']:+.2f}% |"
            rets.append(r["total_return"])
        avg = np.mean(rets)
        std = np.std(rets)
        row += f" {avg:+.2f}% | {std:.2f}% |"
        lines.append(row)

    row = "| KOSPI |"
    bench_rets = []
    for pname, _, _ in PERIODS:
        row += f" {all_benchmarks[pname]:+.2f}% |"
        bench_rets.append(all_benchmarks[pname])
    row += f" {np.mean(bench_rets):+.2f}% | {np.std(bench_rets):.2f}% |"
    lines.append(row)
    lines.append("")

    # 초과수익 크로스 비교
    lines.append("## 임계값별 x 기간별 초과수익(vs KOSPI) 크로스 비교\n")
    header = "| 임계값 |"
    sep = "|--------|"
    for pname, _, _ in PERIODS:
        header += f" {pname} |"
        sep += "---------|"
    header += " 평균 | 3기간 모두 양수? |"
    sep += "------|----------------|"
    lines.append(header)
    lines.append(sep)

    for ti, th in enumerate(THRESHOLDS):
        row = f"| RS>={th} |"
        excesses = []
        for pname, _, _ in PERIODS:
            r = all_results[pname][ti]
            ex = r["total_return"] - all_benchmarks[pname]
            row += f" {ex:+.2f}%p |"
            excesses.append(ex)
        avg_ex = np.mean(excesses)
        all_positive = all(e > 0 for e in excesses)
        row += f" {avg_ex:+.2f}%p | {'O' if all_positive else 'X'} |"
        lines.append(row)
    lines.append("")

    # MDD 크로스 비교
    lines.append("## 임계값별 x 기간별 MDD 크로스 비교\n")
    header = "| 임계값 |"
    sep = "|--------|"
    for pname, _, _ in PERIODS:
        header += f" {pname} |"
        sep += "---------|"
    header += " 최악 MDD |"
    sep += "---------|"
    lines.append(header)
    lines.append(sep)

    for ti, th in enumerate(THRESHOLDS):
        row = f"| RS>={th} |"
        mdds = []
        for pname, _, _ in PERIODS:
            r = all_results[pname][ti]
            row += f" {r['mdd']:.2f}% |"
            mdds.append(r["mdd"])
        row += f" {min(mdds):.2f}% |"
        lines.append(row)
    lines.append("")

    # 수익/MDD 크로스 비교
    lines.append("## 임계값별 x 기간별 수익/MDD 비율 크로스 비교\n")
    header = "| 임계값 |"
    sep = "|--------|"
    for pname, _, _ in PERIODS:
        header += f" {pname} |"
        sep += "---------|"
    header += " 평균 |"
    sep += "------|"
    lines.append(header)
    lines.append(sep)

    for ti, th in enumerate(THRESHOLDS):
        row = f"| RS>={th} |"
        ratios = []
        for pname, _, _ in PERIODS:
            r = all_results[pname][ti]
            ratio = r["total_return"] / abs(r["mdd"]) if r["mdd"] != 0 else 0
            row += f" {ratio:.2f} |"
            ratios.append(ratio)
        row += f" {np.mean(ratios):.2f} |"
        lines.append(row)
    lines.append("")

    lines.append(f"---\n실행 시간: {elapsed:.1f}초")
    lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    print("[1/4] KOSPI200 종목 조회...")
    kospi200 = get_kospi200_tickers()
    print(f"      {len(kospi200)}종목")

    earliest_start = "2023-01-02"
    latest_end = datetime.today().strftime("%Y-%m-%d")

    print("[2/4] 전체 주가 데이터 로딩...")
    stock_df = load_stock_prices(earliest_start, latest_end)
    name_map = stock_df.groupby("ticker")["name"].first().to_dict()
    stock_df = stock_df.drop_duplicates(subset=["date", "ticker"], keep="last")
    pivot = stock_df.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index().ffill()

    index_prices = load_index_prices("^KS11", earliest_start, latest_end)
    print(f"      종목: {pivot.shape[1]}개, 기간: {pivot.index[0].date()} ~ {pivot.index[-1].date()}")

    all_results = {}
    all_benchmarks = {}

    print("\n[3/4] 기간별 x 임계값별 백테스트 (매일 리밸런싱, RS>=90~99)...")

    for pname, start_date, end_date in PERIODS:
        print(f"\n{'='*60}")
        print(f"  기간: {pname} ({start_date} ~ {end_date})")
        print(f"{'='*60}")

        # 매일 평가 (매주가 아닌 모든 영업일)
        eval_dates = list(pivot.index[(pivot.index >= start_date) & (pivot.index <= end_date)])

        # RS 사전 캐싱 (매일이므로 더 많은 계산)
        print(f"  RS Rating 사전 계산 ({len(eval_dates)}일)...")
        rs_cache = {}
        for i, rd in enumerate(eval_dates):
            rs_cache[rd] = calc_rs_at_date(pivot, index_prices, rd)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(eval_dates)}...")

        orig_calc = calc_rs_at_date

        def make_cached(cache):
            def cached(p, ip, rd):
                return cache.get(rd, pd.Series(dtype=float))
            return cached

        import backtest_rs90_kospi200
        backtest_rs90_kospi200.calc_rs_at_date = make_cached(rs_cache)

        # 벤치마크
        idx_vals = index_prices.loc[(index_prices.index >= start_date) & (index_prices.index <= end_date)]
        if len(idx_vals) > 0:
            bench_ret = (idx_vals.iloc[-1] / idx_vals.iloc[0] - 1) * 100
        else:
            bench_ret = 0
        all_benchmarks[pname] = bench_ret
        print(f"  KOSPI 벤치마크: {bench_ret:+.2f}%")

        period_results = []
        for th in THRESHOLDS:
            r = run_single_backtest(th, kospi200, pivot, index_prices,
                                    eval_dates, name_map, start_date, end_date)
            period_results.append(r)
            excess = r["total_return"] - bench_ret
            print(f"  RS>={th:2d}: {r['total_return']:>+8.2f}% (초과 {excess:>+7.2f}%p) "
                  f"MDD {r['mdd']:>7.2f}%  거래 {r['n_trades']:>3d}건  승률 {r['win_rate']:>5.1f}%")

        all_results[pname] = period_results
        backtest_rs90_kospi200.calc_rs_at_date = orig_calc

    elapsed = time.time() - start_time
    print(f"\n[4/4] 리포트 & 차트 생성...")

    chart_dir = os.path.join(base_dir, "charts_rs_daily_rebal")
    generate_charts(all_results, all_benchmarks, chart_dir)

    report = generate_report(all_results, all_benchmarks, elapsed)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_rs_daily_rebal.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 콘솔 최종 요약
    print(f"\n{'='*80}")
    print("RS>=90~99 매일 리밸런싱 백테스트 완료!")
    print(f"{'='*80}")

    print(f"\n{'':>8s}", end="")
    for pname, _, _ in PERIODS:
        print(f"  {pname:>12s}", end="")
    print(f"  {'평균':>10s}")
    print("-" * 60)

    for ti, th in enumerate(THRESHOLDS):
        print(f"  RS>={th:2d}", end="")
        rets = []
        for pname, _, _ in PERIODS:
            r = all_results[pname][ti]
            print(f"  {r['total_return']:>+10.2f}%", end="")
            rets.append(r["total_return"])
        print(f"  {np.mean(rets):>+8.2f}%")

    print(f"  KOSPI ", end="")
    for pname, _, _ in PERIODS:
        print(f"  {all_benchmarks[pname]:>+10.2f}%", end="")
    print(f"  {np.mean(list(all_benchmarks.values())):>+8.2f}%")

    print(f"\n리포트: {report_path}")
    print(f"실행 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
