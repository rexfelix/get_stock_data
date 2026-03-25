"""
RS>=94 & KOSPI200 + RS 모멘텀 매도 조건 비교 백테스트
- 기본 전략: RS>=94 유지 시 보유, RS<94 시 매도
- 모멘텀 전략: RS>=94이더라도 RS가 전주 대비 하락하면 매도
- 2023, 2024, 2025~ 3기간 비교
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

RS_THRESHOLD = 94


def run_backtest(mode, kospi200, pivot, index_prices, rebal_dates, name_map, end_date):
    """
    mode:
      "base"     → RS<94 시에만 매도
      "rs_down"  → RS가 전주 대비 하락하면 매도 (RS>=94여도)
    """
    cash = INITIAL_CAPITAL
    holdings = {}       # ticker → {shares, buy_price, buy_date, prev_rs, peak_rs}
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
            prev_rs = h["prev_rs"]

            reason = None
            if cur_rs < RS_THRESHOLD:
                reason = f"RS={cur_rs}<{RS_THRESHOLD}"
            elif tk not in kospi200:
                reason = "KOSPI200 제외"
            elif mode == "rs_down" and prev_rs is not None and cur_rs < prev_rs:
                reason = f"RS하락({prev_rs}→{cur_rs})"

            if reason:
                sell_list.append((tk, reason, cur_rs))
            else:
                # 보유 유지: RS 갱신
                h["prev_rs"] = cur_rs

        # 매도 실행
        for tk, reason, cur_rs in sell_list:
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
                "reason": reason,
            })

        # 포트 가치 (매도 후)
        port_val = cash
        for tk, h in holdings.items():
            if rdate in pivot.index and tk in pivot.columns:
                port_val += h["shares"] * pivot.loc[rdate, tk]

        # 매수: KOSPI200 & RS>=94인 신규 종목
        target_tickers = set()
        for tk in rs_series.index:
            if tk in kospi200 and rs_series[tk] >= RS_THRESHOLD:
                target_tickers.add(tk)

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
                cur_rs = int(rs_series[tk])
                holdings[tk] = {
                    "shares": shares, "buy_price": price, "buy_date": rdate,
                    "prev_rs": cur_rs, "peak_rs": cur_rs,
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
            "reason": "기간종료",
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

    # 매도사유 분포
    reason_stats = {}
    if not trades_df.empty:
        for reason, grp in trades_df.groupby("reason"):
            n = len(grp)
            wr = (grp["pnl"] > 0).sum() / n * 100
            avg_r = grp["return_pct"].mean()
            total_pnl = grp["pnl"].sum()
            reason_stats[reason] = {"n": n, "win_rate": wr, "avg_return": avg_r, "total_pnl": total_pnl}

    return {
        "mode": mode,
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
        "reason_stats": reason_stats,
    }


def generate_charts(all_results, all_benchmarks, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    n_periods = len(PERIODS)
    mode_colors = {"base": "#1565c0", "rs_down": "#e65100"}
    mode_labels = {"base": "기본 (RS<94만 매도)", "rs_down": "모멘텀 (RS하락 시 매도)"}

    # 1) 기간별 수익률 곡선
    fig, axes = plt.subplots(1, n_periods, figsize=(7 * n_periods, 6))
    if n_periods == 1:
        axes = [axes]

    for pi, (pname, _, _) in enumerate(PERIODS):
        ax = axes[pi]
        for mode in ["base", "rs_down"]:
            r = all_results[pname][mode]
            dates = [d for d, _ in r["portfolio_values"]]
            returns = [(v / INITIAL_CAPITAL - 1) * 100 for _, v in r["portfolio_values"]]
            ax.plot(dates, returns, color=mode_colors[mode], linewidth=2,
                    label=f"{mode_labels[mode]} ({r['total_return']:+.1f}%)")

        bench = all_benchmarks[pname]
        ax.axhline(bench, color="red", linewidth=1.5, linestyle="--",
                   label=f"KOSPI ({bench:+.1f}%)")
        ax.set_title(f"{pname}", fontsize=14, fontweight="bold")
        ax.set_ylabel("수익률 (%)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.suptitle("RS>=94 기본 vs RS하락매도 전략 비교", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "return_curves_compare.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) 기간별 × 전략별 총수익률 바 차트
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_periods)
    w = 0.3

    for mi, mode in enumerate(["base", "rs_down"]):
        rets = [all_results[pname][mode]["total_return"] for pname, _, _ in PERIODS]
        bars = ax.bar(x + mi * w, rets, w, label=mode_labels[mode],
                      color=mode_colors[mode], edgecolor="black", linewidth=0.5, alpha=0.8)
        for bar, ret in zip(bars, rets):
            ax.text(bar.get_x() + bar.get_width() / 2, ret + 2,
                    f"{ret:+.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 벤치마크
    bench_rets = [all_benchmarks[pname] for pname, _, _ in PERIODS]
    ax.bar(x + 2 * w, bench_rets, w, label="KOSPI", color="#ef5350",
           edgecolor="black", linewidth=0.5, alpha=0.6)
    for i, br in enumerate(bench_rets):
        ax.text(x[i] + 2 * w, br + 2, f"{br:+.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x + w)
    ax.set_xticklabels([p[0] for p in PERIODS], fontsize=12)
    ax.set_title("기간별 총수익률 비교", fontsize=14)
    ax.set_ylabel("총수익률 (%)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "total_return_bars.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) 거래수/승률/MDD 비교 표
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("거래수", "n_trades", None),
        ("승률 (%)", "win_rate", 50),
        ("MDD (%)", "mdd", None),
    ]
    for ai, (title, key, hline) in enumerate(metrics):
        ax = axes[ai]
        for mi, mode in enumerate(["base", "rs_down"]):
            vals = [all_results[pname][mode][key] for pname, _, _ in PERIODS]
            ax.bar(x + mi * w, vals, w, label=mode_labels[mode],
                   color=mode_colors[mode], edgecolor="black", linewidth=0.5, alpha=0.8)
        if hline is not None:
            ax.axhline(hline, color="gray", linewidth=1, linestyle="--")
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels([p[0] for p in PERIODS])
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("기본 vs RS하락매도: 거래수/승률/MDD 비교", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_compare.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"      차트 저장: {output_dir}/")


def generate_report(all_results, all_benchmarks, elapsed):
    lines = []
    lines.append("# RS>=94 기본 vs RS하락매도 전략 비교 리포트\n")
    lines.append(f"- **기본 전략**: RS<94 시에만 매도")
    lines.append(f"- **모멘텀 전략**: RS가 전주 대비 1점이라도 하락하면 매도 (RS>=94여도)")
    lines.append(f"- **공통**: KOSPI200 & RS>={RS_THRESHOLD}, 매주 리밸런싱, 동일가중")
    lines.append(f"- **초기자본**: {INITIAL_CAPITAL:,}원\n")

    # 종합 비교 테이블
    lines.append("## 종합 비교\n")
    lines.append("| 기간 | 전략 | 총수익률 | 초과수익 | MDD | 수익/MDD | 거래수 | 승률 | 평균수익률 | 중위수익률 | 평균보유일 |")
    lines.append("|------|------|---------|---------|------|---------|--------|------|----------|----------|----------|")

    for pname, _, _ in PERIODS:
        bench = all_benchmarks[pname]
        for mode in ["base", "rs_down"]:
            r = all_results[pname][mode]
            label = "기본" if mode == "base" else "RS하락매도"
            excess = r["total_return"] - bench
            ratio = r["total_return"] / abs(r["mdd"]) if r["mdd"] != 0 else 0
            lines.append(
                f"| {pname} | {label} | {r['total_return']:+.2f}% | {excess:+.2f}%p | "
                f"{r['mdd']:.2f}% | {ratio:.2f} | {r['n_trades']} | {r['win_rate']:.1f}% | "
                f"{r['avg_return']:.2f}% | {r['med_return']:.2f}% | {r['avg_hold_days']:.1f}일 |"
            )
        lines.append(f"| {pname} | KOSPI | {bench:+.2f}% | - | - | - | - | - | - | - | - |")
    lines.append("")

    # 기간별 차이
    lines.append("## 기간별 전략 차이 (기본 - RS하락매도)\n")
    lines.append("| 기간 | 기본 수익률 | RS하락매도 수익률 | 차이 | 기본 MDD | RS하락매도 MDD | 기본 거래수 | RS하락매도 거래수 |")
    lines.append("|------|-----------|----------------|------|---------|--------------|-----------|----------------|")
    for pname, _, _ in PERIODS:
        b = all_results[pname]["base"]
        m = all_results[pname]["rs_down"]
        diff = b["total_return"] - m["total_return"]
        lines.append(
            f"| {pname} | {b['total_return']:+.2f}% | {m['total_return']:+.2f}% | "
            f"{'기본' if diff > 0 else 'RS하락매도'} {abs(diff):.2f}%p 우위 | "
            f"{b['mdd']:.2f}% | {m['mdd']:.2f}% | {b['n_trades']} | {m['n_trades']} |"
        )
    lines.append("")

    # 매도사유 분석
    lines.append("## 매도사유별 분석 (RS하락매도 전략)\n")
    for pname, _, _ in PERIODS:
        r = all_results[pname]["rs_down"]
        lines.append(f"### {pname}\n")
        lines.append("| 매도사유 | 건수 | 승률 | 평균수익률 | 총손익 |")
        lines.append("|---------|------|------|----------|--------|")
        for reason, stats in sorted(r["reason_stats"].items(), key=lambda x: -x[1]["n"]):
            lines.append(f"| {reason} | {stats['n']} | {stats['win_rate']:.1f}% | "
                        f"{stats['avg_return']:.2f}% | {stats['total_pnl']:,.0f}원 |")
        lines.append("")

    lines.append(f"---\n실행 시간: {elapsed:.1f}초")
    lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    print("[1/3] 데이터 로딩...")
    kospi200 = get_kospi200_tickers()
    print(f"      KOSPI200: {len(kospi200)}종목")

    stock_df = load_stock_prices("2023-01-02", datetime.today().strftime("%Y-%m-%d"))
    name_map = stock_df.groupby("ticker")["name"].first().to_dict()
    stock_df = stock_df.drop_duplicates(subset=["date", "ticker"], keep="last")
    pivot = stock_df.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index().ffill()

    index_prices = load_index_prices("^KS11", "2023-01-02", datetime.today().strftime("%Y-%m-%d"))
    print(f"      종목: {pivot.shape[1]}개")

    print("\n[2/3] 기간별 백테스트...")
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

        # 벤치마크
        idx_vals = index_prices.loc[(index_prices.index >= start_date) & (index_prices.index <= end_date)]
        bench_ret = (idx_vals.iloc[-1] / idx_vals.iloc[0] - 1) * 100 if len(idx_vals) > 0 else 0
        all_benchmarks[pname] = bench_ret

        all_results[pname] = {}
        for mode, label in [("base", "기본"), ("rs_down", "RS하락매도")]:
            r = run_backtest(mode, kospi200, pivot, index_prices, rebal_dates, name_map, end_date)
            all_results[pname][mode] = r
            excess = r["total_return"] - bench_ret
            print(f"  [{label:>6s}] {r['total_return']:>+8.2f}% (초과 {excess:>+7.2f}%p) "
                  f"MDD {r['mdd']:>7.2f}%  거래 {r['n_trades']:>3d}건  승률 {r['win_rate']:>5.1f}%  "
                  f"보유 {r['avg_hold_days']:>5.1f}일")

        print(f"  [KOSPI ] {bench_ret:>+8.2f}%")

    # 리포트 & 차트
    elapsed = time.time() - start_time
    print(f"\n[3/3] 리포트 & 차트...")

    chart_dir = os.path.join(base_dir, "charts_rs_momentum")
    generate_charts(all_results, all_benchmarks, chart_dir)

    report = generate_report(all_results, all_benchmarks, elapsed)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_rs_momentum.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 최종 요약
    print(f"\n{'='*70}")
    print("RS>=94 기본 vs RS하락매도 비교 완료!")
    print(f"{'='*70}")
    print(f"\n{'기간':>6s}  {'기본':>10s}  {'RS하락매도':>10s}  {'KOSPI':>10s}  {'우위':>14s}")
    print("-" * 60)
    for pname, _, _ in PERIODS:
        b = all_results[pname]["base"]["total_return"]
        m = all_results[pname]["rs_down"]["total_return"]
        bench = all_benchmarks[pname]
        winner = "기본" if b > m else "RS하락매도"
        diff = abs(b - m)
        print(f"  {pname:>4s}  {b:>+9.2f}%  {m:>+9.2f}%  {bench:>+9.2f}%  {winner} +{diff:.2f}%p")

    print(f"\n리포트: {report_path}")
    print(f"실행 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
