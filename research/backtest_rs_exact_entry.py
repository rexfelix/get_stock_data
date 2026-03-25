"""
RS == N 정확 진입 + Peak RS 매도 전략 - 연도별 비교 백테스트
- 매수: RS가 정확히 N인 KOSPI200 종목만 매수
- 보유: RS 상승/유지 중이면 계속 보유 (peak RS 추적)
- 매도1: 보유 중 최고 RS 대비 하락 시 매도
- 매도2: RS < N 시 매도
- RS 90~99 × 2023/2024/2025~ = 30개 백테스트
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


def run_single_backtest(entry_rs, kospi200, pivot, index_prices,
                        rebal_dates, name_map, end_date):
    """RS == entry_rs 정확 진입 + peak RS 매도 백테스트"""
    cash = INITIAL_CAPITAL
    holdings = {}  # ticker → {shares, buy_price, buy_date, peak_rs, entry_rs_val}
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
            peak = h["peak_rs"]

            reason = None
            if cur_rs < entry_rs:
                # 매도2: RS < 진입값
                reason = f"RS={cur_rs}<{entry_rs}"
            elif tk not in kospi200:
                reason = "KOSPI200제외"
            elif cur_rs < peak:
                # 매도1: peak RS 대비 하락
                reason = f"peak{peak}→{cur_rs}"
            else:
                # 보유 유지: peak 갱신
                if cur_rs > peak:
                    h["peak_rs"] = cur_rs

            if reason:
                sell_list.append((tk, reason))

        # 매도 실행
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
                "buy_price": h["buy_price"], "sell_price": sell_price,
                "return_pct": ret_pct, "pnl": pnl,
                "hold_days": (rdate - h["buy_date"]).days,
                "reason": reason, "peak_rs": h["peak_rs"],
            })

        # 포트 가치 (매도 후)
        port_val = cash
        for tk, h in holdings.items():
            if rdate in pivot.index and tk in pivot.columns:
                port_val += h["shares"] * pivot.loc[rdate, tk]

        # 매수: RS == entry_rs 정확히 일치하는 KOSPI200 종목 (미보유)
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

        # 포트 가치 기록
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
            "buy_price": h["buy_price"], "sell_price": sell_price,
            "return_pct": ret_pct, "pnl": pnl,
            "hold_days": (last_date - h["buy_date"]).days,
            "reason": "기간종료", "peak_rs": h["peak_rs"],
        })

    final_value = cash
    trades_df = pd.DataFrame(trade_log)

    # MDD
    vals = [v for _, v in portfolio_values]
    peak_v = vals[0] if vals else INITIAL_CAPITAL
    mdd = 0
    for v in vals:
        if v > peak_v:
            peak_v = v
        dd = (v - peak_v) / peak_v * 100
        if dd < mdd:
            mdd = dd

    # 매도사유 분류: peak매도 vs RS<N매도 vs 기간종료
    reason_cats = {"peak_sell": 0, "rs_below": 0, "period_end": 0, "other": 0}
    reason_pnl = {"peak_sell": 0, "rs_below": 0, "period_end": 0, "other": 0}
    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            r = t["reason"]
            if r.startswith("peak"):
                cat = "peak_sell"
            elif f"<{entry_rs}" in r:
                cat = "rs_below"
            elif r == "기간종료":
                cat = "period_end"
            else:
                cat = "other"
            reason_cats[cat] += 1
            reason_pnl[cat] += t["pnl"]

    return {
        "threshold": entry_rs,
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
        "avg_peak_rs": trades_df["peak_rs"].mean() if len(trades_df) > 0 else entry_rs,
        "portfolio_values": portfolio_values,
        "trades_df": trades_df,
        "reason_cats": reason_cats,
        "reason_pnl": reason_pnl,
    }


def generate_charts(all_results, all_benchmarks, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    period_names = [p[0] for p in PERIODS]
    n_periods = len(period_names)

    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.7 * i / (len(THRESHOLDS) - 1)) for i in range(len(THRESHOLDS))]

    # 1) 기간별 수익률 곡선
    fig, axes = plt.subplots(1, n_periods, figsize=(7 * n_periods, 7), sharey=False)
    if n_periods == 1:
        axes = [axes]

    for pi, pname in enumerate(period_names):
        ax = axes[pi]
        results = all_results[pname]
        for i, r in enumerate(results):
            dates = [d for d, _ in r["portfolio_values"]]
            returns = [(v / INITIAL_CAPITAL - 1) * 100 for _, v in r["portfolio_values"]]
            ax.plot(dates, returns, color=colors[i], linewidth=1.5,
                    label=f"RS=={r['threshold']} ({r['total_return']:+.0f}%)", alpha=0.85)

        bench = all_benchmarks[pname]
        ax.axhline(bench, color="red", linewidth=2, linestyle="--",
                   label=f"KOSPI ({bench:+.1f}%)")
        ax.set_title(f"{pname}", fontsize=14, fontweight="bold")
        ax.set_ylabel("수익률 (%)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.suptitle("RS==N 정확진입 + Peak매도 전략 - 연도별 비교", fontsize=16, y=1.02)
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
    ax.set_xticklabels([f"RS=={t}" for t in THRESHOLDS], fontsize=10)
    ax.set_title("RS==N 정확진입 총수익률 - 연도별 비교", fontsize=14)
    ax.set_ylabel("총수익률 (%)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "total_return_grouped.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) 수익/MDD 히트맵
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
    ax.set_xticklabels([f"RS=={t}" for t in THRESHOLDS])
    ax.set_yticks(range(len(period_names)))
    ax.set_yticklabels(period_names)

    for i in range(len(period_names)):
        for j in range(len(THRESHOLDS)):
            val = data_arr[i, j]
            vrange = data_arr.max() - data_arr.min()
            color = "white" if vrange > 0 and abs(val - data_arr.min()) > vrange * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)

    ax.set_title("수익/MDD 비율 히트맵 (RS==N 정확진입)", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "return_mdd_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4) 평균 peak RS & 평균 보유일 차트
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 평균 보유일
    ax = axes[0]
    for pi, pname in enumerate(period_names):
        results = all_results[pname]
        holds = [r["avg_hold_days"] for r in results]
        ax.plot(THRESHOLDS, holds, "o-", color=period_colors[pi], linewidth=2,
                markersize=6, label=pname)
    ax.set_xlabel("진입 RS")
    ax.set_ylabel("평균 보유일")
    ax.set_title("진입 RS별 평균 보유일", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 평균 peak RS
    ax = axes[1]
    for pi, pname in enumerate(period_names):
        results = all_results[pname]
        peaks = [r["avg_peak_rs"] for r in results]
        ax.plot(THRESHOLDS, peaks, "s-", color=period_colors[pi], linewidth=2,
                markersize=6, label=pname)
    ax.plot(THRESHOLDS, THRESHOLDS, "k--", linewidth=1, alpha=0.5, label="진입RS=peakRS")
    ax.set_xlabel("진입 RS")
    ax.set_ylabel("평균 Peak RS")
    ax.set_title("진입 RS별 평균 Peak RS (어디까지 올랐나)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "holddays_peakrs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"      차트 저장: {output_dir}/")


def generate_report(all_results, all_benchmarks, elapsed):
    lines = []
    lines.append("# RS==N 정확진입 + Peak RS 매도 전략 - 연도별 비교 리포트\n")
    lines.append(f"- **매수**: RS가 정확히 N인 KOSPI200 종목만 매수")
    lines.append(f"- **보유**: RS 상승/유지 중 계속 보유 (peak RS 추적)")
    lines.append(f"- **매도1**: 보유 중 최고 RS 대비 하락 시 매도")
    lines.append(f"- **매도2**: RS < N(진입값) 시 매도")
    lines.append(f"- **리밸런싱**: 매주 (첫 영업일)")
    lines.append(f"- **초기자본**: {INITIAL_CAPITAL:,}원\n")

    # 기간별 종합
    for pname, start, end in PERIODS:
        results = all_results[pname]
        bench = all_benchmarks[pname]

        lines.append(f"## {pname} ({start} ~ {end})\n")
        lines.append(f"KOSPI: **{bench:+.2f}%**\n")
        lines.append("| 진입RS | 총수익률 | 초과수익 | MDD | 수익/MDD | 거래수 | 승률 | 평균수익률 | 평균보유일 | 평균peakRS | peak매도 | RS<N매도 |")
        lines.append("|--------|---------|---------|------|---------|--------|------|----------|----------|----------|---------|---------|")

        for r in results:
            excess = r["total_return"] - bench
            ratio = r["total_return"] / abs(r["mdd"]) if r["mdd"] != 0 else 0
            rc = r["reason_cats"]
            lines.append(
                f"| RS=={r['threshold']} | {r['total_return']:+.2f}% | {excess:+.2f}%p | "
                f"{r['mdd']:.2f}% | {ratio:.2f} | {r['n_trades']} | {r['win_rate']:.1f}% | "
                f"{r['avg_return']:.2f}% | {r['avg_hold_days']:.1f}일 | {r['avg_peak_rs']:.1f} | "
                f"{rc['peak_sell']} | {rc['rs_below']} |"
            )
        lines.append("")

    # 크로스 비교: 총수익률
    lines.append("## 임계값별 × 기간별 총수익률 크로스 비교\n")
    header = "| 진입RS |"
    sep = "|--------|"
    for pname, _, _ in PERIODS:
        header += f" {pname} |"
        sep += "---------|"
    header += " 평균 | 표준편차 |"
    sep += "------|---------|"
    lines.append(header)
    lines.append(sep)

    for ti, th in enumerate(THRESHOLDS):
        row = f"| RS=={th} |"
        rets = []
        for pname, _, _ in PERIODS:
            r = all_results[pname][ti]
            row += f" {r['total_return']:+.2f}% |"
            rets.append(r["total_return"])
        row += f" {np.mean(rets):+.2f}% | {np.std(rets):.2f}% |"
        lines.append(row)

    row = "| KOSPI |"
    for pname, _, _ in PERIODS:
        row += f" {all_benchmarks[pname]:+.2f}% |"
    row += f" {np.mean(list(all_benchmarks.values())):+.2f}% | - |"
    lines.append(row)
    lines.append("")

    # 초과수익 크로스
    lines.append("## 임계값별 × 기간별 초과수익 크로스 비교\n")
    header = "| 진입RS |"
    sep = "|--------|"
    for pname, _, _ in PERIODS:
        header += f" {pname} |"
        sep += "---------|"
    header += " 평균 | 3기간 양수? |"
    sep += "------|----------|"
    lines.append(header)
    lines.append(sep)

    for ti, th in enumerate(THRESHOLDS):
        row = f"| RS=={th} |"
        excesses = []
        for pname, _, _ in PERIODS:
            r = all_results[pname][ti]
            ex = r["total_return"] - all_benchmarks[pname]
            row += f" {ex:+.2f}%p |"
            excesses.append(ex)
        avg_ex = np.mean(excesses)
        all_pos = all(e > 0 for e in excesses)
        row += f" {avg_ex:+.2f}%p | {'O' if all_pos else 'X'} |"
        lines.append(row)
    lines.append("")

    # 평균 peak RS 크로스
    lines.append("## 진입RS별 평균 Peak RS (어디까지 올라갔나)\n")
    header = "| 진입RS |"
    sep = "|--------|"
    for pname, _, _ in PERIODS:
        header += f" {pname} |"
        sep += "---------|"
    header += " 평균 | 상승폭 |"
    sep += "------|--------|"
    lines.append(header)
    lines.append(sep)

    for ti, th in enumerate(THRESHOLDS):
        row = f"| RS=={th} |"
        peaks = []
        for pname, _, _ in PERIODS:
            r = all_results[pname][ti]
            row += f" {r['avg_peak_rs']:.1f} |"
            peaks.append(r["avg_peak_rs"])
        avg_peak = np.mean(peaks)
        row += f" {avg_peak:.1f} | +{avg_peak - th:.1f} |"
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

    print("[2/4] 전체 주가 데이터 로딩...")
    stock_df = load_stock_prices("2023-01-02", datetime.today().strftime("%Y-%m-%d"))
    name_map = stock_df.groupby("ticker")["name"].first().to_dict()
    stock_df = stock_df.drop_duplicates(subset=["date", "ticker"], keep="last")
    pivot = stock_df.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index().ffill()

    index_prices = load_index_prices("^KS11", "2023-01-02", datetime.today().strftime("%Y-%m-%d"))
    print(f"      종목: {pivot.shape[1]}개, 기간: {pivot.index[0].date()} ~ {pivot.index[-1].date()}")

    print("\n[3/4] 기간별 × 임계값별 백테스트...")
    all_results = {}
    all_benchmarks = {}

    for pname, start_date, end_date in PERIODS:
        print(f"\n{'='*60}")
        print(f"  {pname} ({start_date} ~ {end_date})")
        print(f"{'='*60}")

        # 리밸런싱 날짜
        period_dates = pivot.index[(pivot.index >= start_date) & (pivot.index <= end_date)]
        rebal_dates = []
        last_week = None
        for d in period_dates:
            key = (d.year, d.isocalendar()[1])
            if key != last_week:
                rebal_dates.append(d)
                last_week = key

        # RS 캐싱
        print(f"  RS 사전 계산 ({len(rebal_dates)}회)...")
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

        # 벤치마크
        idx_vals = index_prices.loc[(index_prices.index >= start_date) & (index_prices.index <= end_date)]
        bench_ret = (idx_vals.iloc[-1] / idx_vals.iloc[0] - 1) * 100 if len(idx_vals) > 0 else 0
        all_benchmarks[pname] = bench_ret
        print(f"  KOSPI: {bench_ret:+.2f}%")

        period_results = []
        for th in THRESHOLDS:
            r = run_single_backtest(th, kospi200, pivot, index_prices,
                                    rebal_dates, name_map, end_date)
            period_results.append(r)
            excess = r["total_return"] - bench_ret
            rc = r["reason_cats"]
            print(f"  RS=={th:2d}: {r['total_return']:>+8.2f}% (초과 {excess:>+7.2f}%p) "
                  f"MDD {r['mdd']:>7.2f}%  거래 {r['n_trades']:>3d}건  승률 {r['win_rate']:>5.1f}%  "
                  f"보유 {r['avg_hold_days']:>5.1f}일  peakRS {r['avg_peak_rs']:.1f}  "
                  f"[peak매도:{rc['peak_sell']} RS<N:{rc['rs_below']}]")

        all_results[pname] = period_results
        backtest_rs90_kospi200.calc_rs_at_date = orig_calc

    elapsed = time.time() - start_time

    print(f"\n[4/4] 리포트 & 차트...")
    chart_dir = os.path.join(base_dir, "charts_rs_exact_entry")
    generate_charts(all_results, all_benchmarks, chart_dir)

    report = generate_report(all_results, all_benchmarks, elapsed)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_rs_exact_entry.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 최종 요약
    print(f"\n{'='*80}")
    print("RS==N 정확진입 + Peak매도 - 연도별 비교 완료!")
    print(f"{'='*80}")

    print(f"\n{'':>8s}", end="")
    for pname, _, _ in PERIODS:
        print(f"  {pname:>12s}", end="")
    print(f"  {'평균':>10s}")
    print("-" * 60)

    for ti, th in enumerate(THRESHOLDS):
        print(f"  RS=={th:2d}", end="")
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
