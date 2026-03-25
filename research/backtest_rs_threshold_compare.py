"""
RS 임계값별 KOSPI200 조건부 보유 전략 비교 백테스트
- RS>=99, >=98, ..., >=90 각각 독립 백테스트
- 동일 데이터/기간에서 임계값만 변경하여 비교
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
    ENGINE, FEE_BUY, FEE_SELL, TAX_SELL, START_DATE, END_DATE,
    get_kospi200_tickers, load_stock_prices, load_index_prices,
    calc_rs_at_date,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

INITIAL_CAPITAL = 100_000_000


def run_single_backtest(threshold: int, kospi200: set, pivot: pd.DataFrame,
                        index_prices: pd.Series, rebal_dates: list,
                        name_map: dict) -> dict:
    """단일 RS 임계값으로 백테스트 실행, 결과 dict 반환"""
    cash = INITIAL_CAPITAL
    holdings = {}
    portfolio_values = []
    trade_log = []

    for rdate in rebal_dates:
        rs_series = calc_rs_at_date(pivot, index_prices, rdate)
        if rs_series.empty:
            port_val = cash
            for tk, h in holdings.items():
                if rdate in pivot.index and tk in pivot.columns:
                    port_val += h["shares"] * pivot.loc[rdate, tk]
            portfolio_values.append((rdate, port_val))
            continue

        # 보유 대상: KOSPI200 & RS >= threshold
        target_tickers = set()
        for tk in rs_series.index:
            if tk in kospi200 and rs_series[tk] >= threshold:
                target_tickers.add(tk)

        # 매도
        sell_tickers = [tk for tk in list(holdings.keys()) if tk not in target_tickers]
        for tk in sell_tickers:
            h = holdings.pop(tk)
            if rdate in pivot.index and tk in pivot.columns:
                sell_price = pivot.loc[rdate, tk]
            else:
                sell_price = h["buy_price"]

            proceeds = h["shares"] * sell_price
            fee = proceeds * (FEE_SELL + TAX_SELL)
            cash += proceeds - fee

            ret_pct = (sell_price / h["buy_price"] - 1) * 100
            pnl = (sell_price - h["buy_price"]) * h["shares"] - fee - h["shares"] * h["buy_price"] * FEE_BUY
            trade_log.append({
                "ticker": tk, "name": name_map.get(tk, ""),
                "buy_date": h["buy_date"], "sell_date": rdate,
                "buy_price": h["buy_price"], "sell_price": sell_price,
                "shares": h["shares"], "return_pct": ret_pct, "pnl": pnl,
                "hold_days": (rdate - h["buy_date"]).days,
            })

        # 포트 가치 (매도 후)
        port_val = cash
        for tk, h in holdings.items():
            if rdate in pivot.index and tk in pivot.columns:
                port_val += h["shares"] * pivot.loc[rdate, tk]

        # 신규 매수
        new_buys = target_tickers - set(holdings.keys())
        if new_buys or holdings:
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
                    cost = shares * price * (1 + FEE_BUY)
                    cash -= cost
                    holdings[tk] = {"shares": shares, "buy_price": price, "buy_date": rdate}

        # 포트 가치 기록
        port_val = cash
        for tk, h in holdings.items():
            if rdate in pivot.index and tk in pivot.columns:
                port_val += h["shares"] * pivot.loc[rdate, tk]
        portfolio_values.append((rdate, port_val))

    # 미청산 포지션 마감
    last_date = pivot.index[-1]
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
            "shares": h["shares"], "return_pct": ret_pct, "pnl": pnl,
            "hold_days": (last_date - h["buy_date"]).days,
        })

    final_value = cash
    trades_df = pd.DataFrame(trade_log)

    # MDD
    vals = [v for _, v in portfolio_values]
    peak = vals[0]
    mdd = 0
    for v in vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < mdd:
            mdd = dd

    # 평균 보유종목수
    # rebal_dates에서 각 시점 보유종목수를 역산하기 어려우므로 trades 기반으로 추정
    # portfolio_values에서 중간 시점의 투자 비중으로 추정
    avg_holdings = 0
    if trades_df.empty:
        avg_holdings = 0
    else:
        # 각 리밸런싱 시점에서 보유중인 종목 수 세기
        holding_counts = []
        active_trades = []
        for _, t in trades_df.iterrows():
            active_trades.append((t["buy_date"], t["sell_date"]))
        for rd in rebal_dates:
            cnt = sum(1 for bd, sd in active_trades if bd <= rd < sd)
            # 기간종료 매도 포함
            cnt += sum(1 for bd, sd in active_trades if bd <= rd and sd == last_date and rd <= sd)
            holding_counts.append(cnt)
        # 간단히 trades 기반 추정 대신 포트폴리오 값 기반
        avg_holdings = np.mean(holding_counts) if holding_counts else 0

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


def generate_comparison_charts(results: list, index_prices: pd.Series, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 색상 맵: 99(진한 파랑) → 90(연한 파랑)
    cmap = plt.cm.Blues
    n = len(results)
    colors = [cmap(0.3 + 0.7 * i / (n - 1)) for i in range(n)]

    # 1) 수익률 곡선 비교
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, r in enumerate(results):
        dates = [d for d, _ in r["portfolio_values"]]
        returns = [(v / INITIAL_CAPITAL - 1) * 100 for _, v in r["portfolio_values"]]
        ax.plot(dates, returns, color=colors[i], linewidth=1.8,
                label=f"RS>={r['threshold']} ({r['total_return']:+.1f}%)", alpha=0.85)

    # 벤치마크
    idx_start = index_prices.loc[index_prices.index >= START_DATE].iloc[0]
    bench_dates = index_prices.index[index_prices.index >= START_DATE]
    bench_returns = [(index_prices.loc[d] / idx_start - 1) * 100 for d in bench_dates]
    ax.plot(bench_dates, bench_returns, "r--", linewidth=2, alpha=0.6, label=f"KOSPI ({bench_returns[-1]:+.1f}%)")

    ax.set_title("RS 임계값별 수익률 곡선 비교 (KOSPI200, 매주 리밸런싱)", fontsize=14)
    ax.set_ylabel("수익률 (%)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "return_curves_compare.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) 총수익률 바 차트
    thresholds = [r["threshold"] for r in results]
    total_returns = [r["total_return"] for r in results]
    bench_ret = bench_returns[-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar([f"RS>={t}" for t in thresholds], total_returns, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(bench_ret, color="red", linewidth=2, linestyle="--", label=f"KOSPI {bench_ret:+.1f}%")
    for bar, ret in zip(bars, total_returns):
        ax.text(bar.get_x() + bar.get_width() / 2, ret + 2, f"{ret:+.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("RS 임계값별 총수익률 비교", fontsize=14)
    ax.set_ylabel("총수익률 (%)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "total_return_bars.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) MDD 비교
    mdds = [r["mdd"] for r in results]
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = ["#d32f2f" if m < -15 else "#ff9800" if m < -10 else "#4caf50" for m in mdds]
    bars = ax.bar([f"RS>={t}" for t in thresholds], mdds, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, m in zip(bars, mdds):
        ax.text(bar.get_x() + bar.get_width() / 2, m - 0.5, f"{m:.1f}%",
                ha="center", va="top", fontsize=9)
    ax.set_title("RS 임계값별 MDD 비교", fontsize=14)
    ax.set_ylabel("MDD (%)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mdd_bars.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4) 승률 & 평균수익률 듀얼 차트
    win_rates = [r["win_rate"] for r in results]
    avg_returns = [r["avg_return"] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(thresholds))
    w = 0.35
    bars1 = ax1.bar(x - w/2, win_rates, w, color="#42a5f5", label="승률 (%)", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("승률 (%)", color="#42a5f5")
    ax1.set_ylim(0, max(win_rates) * 1.3 if win_rates else 100)
    ax1.axhline(50, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, avg_returns, w, color="#66bb6a", label="평균수익률 (%)", edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("평균수익률 (%)", color="#66bb6a")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"RS>={t}" for t in thresholds])
    ax1.set_title("RS 임계값별 승률 & 평균수익률", fontsize=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "winrate_avgret.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5) 거래수 & 평균보유일
    n_trades = [r["n_trades"] for r in results]
    avg_holds = [r["avg_hold_days"] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(thresholds))
    bars1 = ax1.bar(x - w/2, n_trades, w, color="#ef5350", label="거래수", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("거래수", color="#ef5350")

    ax2 = ax1.twinx()
    ax2.plot(x, avg_holds, "o-", color="#ab47bc", linewidth=2, markersize=8, label="평균보유일")
    ax2.set_ylabel("평균보유일", color="#ab47bc")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"RS>={t}" for t in thresholds])
    ax1.set_title("RS 임계값별 거래수 & 평균보유일", fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trades_holddays.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"      차트 저장: {output_dir}/")


def generate_report(results: list, benchmark_return: float, elapsed: float) -> str:
    lines = []
    lines.append("# RS 임계값별 KOSPI200 조건부 보유 전략 비교 리포트\n")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **전략**: 매주 RS Rating 재계산, RS>=임계값 & KOSPI200 종목만 동일가중 보유")
    lines.append(f"- **리밸런싱**: 매주 (첫 영업일)")
    lines.append(f"- **초기자본**: {INITIAL_CAPITAL:,}원")
    lines.append(f"- **비교 임계값**: RS>=99, 98, 97, ..., 90 (10개)")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%\n")

    # 종합 비교 테이블
    lines.append("## 종합 비교\n")
    lines.append("| 임계값 | 총수익률 | 초과수익 | MDD | 거래수 | 승률 | 평균수익률 | 중위수익률 | 평균보유일 | 최대이익 | 최대손실 |")
    lines.append("|--------|---------|---------|-----|--------|------|----------|----------|----------|---------|---------|")

    for r in results:
        excess = r["total_return"] - benchmark_return
        lines.append(
            f"| **RS>={r['threshold']}** | {r['total_return']:+.2f}% | {excess:+.2f}%p | "
            f"{r['mdd']:.2f}% | {r['n_trades']} | {r['win_rate']:.1f}% | "
            f"{r['avg_return']:.2f}% | {r['med_return']:.2f}% | {r['avg_hold_days']:.1f}일 | "
            f"{r['max_profit']:.2f}% | {r['max_loss']:.2f}% |"
        )

    lines.append(f"| **KOSPI지수** | {benchmark_return:+.2f}% | - | - | - | - | - | - | - | - | - |")
    lines.append("")

    # 수익률 순위
    sorted_by_return = sorted(results, key=lambda x: x["total_return"], reverse=True)
    lines.append("## 총수익률 순위\n")
    for i, r in enumerate(sorted_by_return, 1):
        excess = r["total_return"] - benchmark_return
        lines.append(f"{i}. **RS>={r['threshold']}**: {r['total_return']:+.2f}% "
                     f"(초과 {excess:+.2f}%p, MDD {r['mdd']:.2f}%)")
    lines.append("")

    # 리스크 대비 수익 (수익률/MDD)
    lines.append("## 리스크 대비 수익 (총수익률 / |MDD|)\n")
    lines.append("| 임계값 | 총수익률 | MDD | 수익/리스크 비율 |")
    lines.append("|--------|---------|-----|----------------|")
    for r in sorted(results, key=lambda x: x["total_return"] / abs(x["mdd"]) if x["mdd"] != 0 else 0, reverse=True):
        ratio = r["total_return"] / abs(r["mdd"]) if r["mdd"] != 0 else float("inf")
        lines.append(f"| RS>={r['threshold']} | {r['total_return']:+.2f}% | {r['mdd']:.2f}% | **{ratio:.2f}** |")
    lines.append("")

    # 임계값별 Top 3 수익 종목
    lines.append("## 임계값별 수익 Top 3 종목\n")
    for r in results:
        df = r["trades_df"]
        if df.empty:
            continue
        top3 = df.nlargest(3, "return_pct")
        names = ", ".join(f"{row['name']}({row['return_pct']:+.1f}%)" for _, row in top3.iterrows())
        lines.append(f"- **RS>={r['threshold']}**: {names}")
    lines.append("")

    # 고찰
    lines.append("## 고찰\n")

    best = sorted_by_return[0]
    worst = sorted_by_return[-1]
    best_risk = max(results, key=lambda x: x["total_return"] / abs(x["mdd"]) if x["mdd"] != 0 else 0)

    lines.append(f"### 최고 수익률: RS>={best['threshold']} ({best['total_return']:+.2f}%)")
    lines.append(f"### 최저 수익률: RS>={worst['threshold']} ({worst['total_return']:+.2f}%)")
    lines.append(f"### 최고 수익/리스크: RS>={best_risk['threshold']}")
    lines.append("")

    lines.append("### 주요 발견\n")
    lines.append("1. **임계값이 높을수록** (RS>=99 쪽): 종목 수가 적어 집중 투자, 변동성 클 수 있음")
    lines.append("2. **임계값이 낮을수록** (RS>=90 쪽): 종목 수가 많아 분산 효과, 안정적이나 수익률 희석 가능")
    lines.append("3. **최적 임계값**: 수익률과 MDD의 균형을 고려하여 선택")
    lines.append("")

    lines.append(f"---\n실행 시간: {elapsed:.1f}초")
    lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 1) 공통 데이터 로딩
    print("[1/3] 공통 데이터 로딩...")
    kospi200 = get_kospi200_tickers()
    print(f"      KOSPI200: {len(kospi200)}종목")

    stock_df = load_stock_prices(START_DATE, END_DATE)
    name_map = stock_df.groupby("ticker")["name"].first().to_dict()
    stock_df = stock_df.drop_duplicates(subset=["date", "ticker"], keep="last")
    pivot = stock_df.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index().ffill()

    index_prices = load_index_prices("^KS11", START_DATE, END_DATE)
    print(f"      종목: {pivot.shape[1]}개, 기간: {pivot.index[0].date()} ~ {pivot.index[-1].date()}")

    # 리밸런싱 날짜
    all_dates = pivot.index[pivot.index >= START_DATE]
    rebal_dates = []
    last_week = None
    for d in all_dates:
        key = (d.year, d.isocalendar()[1])
        if key != last_week:
            rebal_dates.append(d)
            last_week = key
    print(f"      리밸런싱: {len(rebal_dates)}회")

    # RS를 매번 계산하지 않도록 캐싱
    print("\n[2/3] RS Rating 사전 계산 (캐싱)...")
    rs_cache = {}
    for i, rd in enumerate(rebal_dates):
        rs_cache[rd] = calc_rs_at_date(pivot, index_prices, rd)
        if (i + 1) % 10 == 0:
            print(f"      {i+1}/{len(rebal_dates)} 완료")
    print(f"      {len(rs_cache)}회분 RS 캐싱 완료")

    # calc_rs_at_date를 캐시 버전으로 교체
    original_calc = calc_rs_at_date

    def cached_calc(pivot, index_prices, rdate):
        return rs_cache.get(rdate, pd.Series(dtype=float))

    # 모듈 레벨 함수를 임시 교체
    import backtest_rs90_kospi200
    backtest_rs90_kospi200.calc_rs_at_date = cached_calc

    # 2) 임계값별 백테스트
    print("\n[3/3] 임계값별 백테스트 실행...")
    thresholds = list(range(99, 89, -1))  # 99, 98, ..., 90
    results = []

    for th in thresholds:
        print(f"\n  === RS>={th} ===")
        r = run_single_backtest(th, kospi200, pivot, index_prices, rebal_dates, name_map)
        results.append(r)

        n_trades = r["n_trades"]
        total_ret = r["total_return"]
        mdd = r["mdd"]
        wr = r["win_rate"]
        print(f"      수익률: {total_ret:+.2f}%, MDD: {mdd:.2f}%, "
              f"거래: {n_trades}건, 승률: {wr:.1f}%")

    # 함수 복원
    backtest_rs90_kospi200.calc_rs_at_date = original_calc

    # 벤치마크
    idx_start = index_prices.loc[index_prices.index >= START_DATE].iloc[0]
    benchmark_return = (index_prices.iloc[-1] / idx_start - 1) * 100

    # 리포트 & 차트
    elapsed = time.time() - start_time

    chart_dir = os.path.join(base_dir, "charts_rs_threshold_compare")
    generate_comparison_charts(results, index_prices, chart_dir)

    report = generate_report(results, benchmark_return, elapsed)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_rs_threshold_compare.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 콘솔 요약
    print(f"\n{'='*70}")
    print("RS 임계값별 비교 백테스트 완료!")
    print(f"{'='*70}")
    print(f"\n{'임계값':>8s} {'총수익률':>10s} {'초과수익':>10s} {'MDD':>8s} {'거래수':>6s} {'승률':>6s} {'평균수익률':>10s} {'평균보유일':>10s}")
    print("-" * 70)
    for r in results:
        excess = r["total_return"] - benchmark_return
        print(f"  RS>={r['threshold']:2d}  {r['total_return']:>+9.2f}%  {excess:>+9.2f}%p  "
              f"{r['mdd']:>7.2f}%  {r['n_trades']:>5d}  {r['win_rate']:>5.1f}%  "
              f"{r['avg_return']:>+9.2f}%  {r['avg_hold_days']:>8.1f}일")
    print(f"  KOSPI    {benchmark_return:>+9.2f}%")
    print(f"\n리포트: {report_path}")
    print(f"실행 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
