"""
2봉 연속 급락 매수 전략 - 매도 조건 최적화
- 매수: KOSPI200 대상, 시가대비 종가 -6% 이상 2봉 연속 → 종가 매수
- 매도 조건 조합:
  - 이평선 종가 이탈: 5MA, 10MA, 15MA, 20MA
  - 손절: 매입단가 -2%, -3%
"""

import os
import time
from datetime import datetime
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_crash import (
    ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    START_DATE, END_DATE,
    get_kospi200_tickers, load_all_data,
)
from backtest_consecutive_crash import CRASH_PCT, CONSEC_DAYS

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 최적화 파라미터
MA_PERIODS = [5, 10, 15, 20]
STOP_LOSSES = [-1, -2, -3]  # %


def find_signals(df: pd.DataFrame) -> list[int]:
    """2봉 연속 시가대비 종가 -6% 이상 급락한 날의 인덱스 반환"""
    intraday_ret = (df["close"] - df["open"]) / df["open"] * 100
    is_crash = intraday_ret <= CRASH_PCT

    signals = []
    for i in range(CONSEC_DAYS - 1, len(df)):
        if all(is_crash.iloc[i - j] for j in range(CONSEC_DAYS)):
            if i + 1 < len(df):
                signals.append(i)
    return signals


def run_backtest_ma(df: pd.DataFrame, ticker: str, name: str,
                    ma_period: int, stop_loss_pct: float):
    """이평선 이탈 매도 + 손절 백테스트"""
    # 이평선 계산
    df = df.copy()
    df["sma"] = df["close"].rolling(window=ma_period).mean()

    signals = find_signals(df)
    trades = []
    dates = df.index.tolist()

    occupied_until = -1  # 포지션 보유 중이면 매도일 인덱스

    for sig_idx in signals:
        if sig_idx <= occupied_until:
            continue  # 이미 포지션 보유 중

        buy_price = df.iloc[sig_idx]["close"]
        buy_date = dates[sig_idx]
        max_qty = int(INITIAL_CAPITAL / (buy_price * (1 + FEE_BUY)))
        if max_qty <= 0:
            continue

        # 매도 탐색: sig_idx+1부터
        sell_idx = None
        sell_reason = None
        above_ma = False

        for j in range(sig_idx + 1, len(df)):
            close = df.iloc[j]["close"]
            sma = df.iloc[j]["sma"]

            # 손절 체크
            loss_pct = (close - buy_price) / buy_price * 100
            if loss_pct <= stop_loss_pct:
                sell_idx = j
                sell_reason = f"손절({stop_loss_pct}%)"
                break

            # 이평선 이탈 체크 (이평선 위로 올라간 후 이탈해야 매도)
            if not pd.isna(sma):
                if not above_ma and close > sma:
                    above_ma = True
                if above_ma and close < sma:
                    sell_idx = j
                    sell_reason = f"{ma_period}MA이탈"
                    break

        # 매도 못 찾으면 마지막 봉에서 청산
        if sell_idx is None:
            sell_idx = len(df) - 1
            sell_reason = "미청산"

        occupied_until = sell_idx
        sell_price = df.iloc[sell_idx]["close"]
        sell_date = dates[sell_idx]

        buy_cost = max_qty * buy_price
        buy_fee = buy_cost * FEE_BUY
        sell_revenue = max_qty * sell_price
        sell_fee = sell_revenue * FEE_SELL
        sell_tax = sell_revenue * TAX_SELL
        net_sell = sell_revenue - sell_fee - sell_tax
        pnl = net_sell - buy_cost - buy_fee
        ret_pct = pnl / (buy_cost + buy_fee) * 100

        hold_days = (pd.Timestamp(sell_date) - pd.Timestamp(buy_date)).days

        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": buy_date, "buy_price": buy_price,
            "sell_date": sell_date, "sell_price": sell_price,
            "quantity": max_qty, "pnl": pnl,
            "return_pct": ret_pct, "reason": sell_reason,
            "hold_days": hold_days,
        })

    return pd.DataFrame(trades)


def summarize_trades(trades_df: pd.DataFrame) -> dict:
    """거래 결과 요약"""
    if trades_df.empty:
        return {
            "n_trades": 0, "win_rate": 0, "avg_return": 0,
            "total_pnl": 0, "profit_factor": 0, "avg_hold_days": 0,
            "median_return": 0, "max_drawdown_trade": 0,
        }

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0

    if "hold_days" in trades_df.columns:
        avg_hold = trades_df["hold_days"].mean()
    else:
        avg_hold = 1.0

    return {
        "n_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100,
        "avg_return": trades_df["return_pct"].mean(),
        "median_return": trades_df["return_pct"].median(),
        "total_pnl": trades_df["pnl"].sum(),
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_hold_days": avg_hold,
        "max_drawdown_trade": trades_df["return_pct"].min(),
        "avg_win": wins["return_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["return_pct"].mean() if len(losses) > 0 else 0,
    }


def generate_report(results: list[dict], all_combo_trades: dict, elapsed: float) -> str:
    lines = []
    lines.append("# 2봉 연속 급락 매수 - 매도 조건 최적화 리포트 (KOSPI200)\n")

    lines.append("## 전략 개요\n")
    lines.append(f"- **대상**: KOSPI200")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **매수조건**: 시가대비 종가 {CRASH_PCT}% 이하가 {CONSEC_DAYS}봉 연속 → 종가 매수")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append("")

    lines.append("### 테스트 조합\n")
    lines.append(f"- **이평선 이탈 매도**: {MA_PERIODS}")
    lines.append(f"- **손절**: {STOP_LOSSES}%")
    lines.append(f"- **총 {len(MA_PERIODS) * len(STOP_LOSSES)}가지 조합**")
    lines.append("")

    # 기준선: 다음날 종가 매도 (1일 보유)
    lines.append("### 기준선 (다음날 종가 매도)\n")
    if "baseline" in all_combo_trades:
        bl = summarize_trades(all_combo_trades["baseline"])
        lines.append(f"- 거래: {bl['n_trades']}건, 승률: {bl['win_rate']:.1f}%, "
                      f"평균수익률: {bl['avg_return']:.2f}%, 총손익: {bl['total_pnl']:,.0f}원")
    lines.append("")

    # 조합별 결과 테이블
    lines.append("## 조합별 성과 비교\n")
    lines.append("| MA | 손절 | 거래수 | 승률 | 평균수익률 | 중위수익률 | 손익비 | 총손익 | 평균보유일 | 최대손실 |")
    lines.append("|-----|------|--------|------|----------|----------|--------|--------|----------|---------|")

    # 총손익 기준 정렬
    results_sorted = sorted(results, key=lambda x: x["total_pnl"], reverse=True)
    for r in results_sorted:
        pf_str = f"{r['profit_factor']:.2f}" if r["profit_factor"] != float("inf") else "∞"
        lines.append(
            f"| {r['ma_period']}MA | {r['stop_loss']}% | {r['n_trades']} "
            f"| {r['win_rate']:.1f}% | {r['avg_return']:.2f}% | {r['median_return']:.2f}% "
            f"| {pf_str} | {r['total_pnl']:,.0f}원 "
            f"| {r['avg_hold_days']:.1f}일 | {r['max_drawdown_trade']:.2f}% |"
        )
    lines.append("")

    # 최적 조합 상세
    best = results_sorted[0]
    best_key = f"{best['ma_period']}MA_{best['stop_loss']}%"
    lines.append(f"## 최적 조합: {best['ma_period']}MA + 손절 {best['stop_loss']}%\n")
    lines.append(f"- **총 손익**: {best['total_pnl']:,.0f}원")
    lines.append(f"- **승률**: {best['win_rate']:.1f}%")
    lines.append(f"- **평균 수익률**: {best['avg_return']:.2f}%")
    lines.append(f"- **손익비**: {best['profit_factor']:.2f}")
    lines.append(f"- **평균 보유일**: {best['avg_hold_days']:.1f}일")
    lines.append("")

    # 최적 조합의 매도 사유별 분석
    if best_key in all_combo_trades and not all_combo_trades[best_key].empty:
        best_trades = all_combo_trades[best_key]
        lines.append("### 매도 사유별 분석\n")
        lines.append("| 사유 | 건수 | 승률 | 평균수익률 | 총손익 | 평균보유일 |")
        lines.append("|------|------|------|----------|--------|----------|")
        for reason, grp in best_trades.groupby("reason"):
            n = len(grp)
            wr = (grp["pnl"] > 0).sum() / n * 100
            ar = grp["return_pct"].mean()
            tp = grp["pnl"].sum()
            hd = grp["hold_days"].mean()
            lines.append(f"| {reason} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 | {hd:.1f}일 |")
        lines.append("")

        # 전체 매매 기록
        lines.append("### 전체 매매 기록\n")
        lines.append("| # | 종목 | 매수일 | 매수가 | 매도일 | 매도가 | 수익률 | 손익 | 보유일 | 사유 |")
        lines.append("|---|------|--------|--------|--------|--------|--------|------|--------|------|")
        sorted_trades = best_trades.sort_values("buy_date")
        for i, (_, t) in enumerate(sorted_trades.iterrows()):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            lines.append(
                f"| {i+1} | {t['name']} | {bd} | {t['buy_price']:,.0f} "
                f"| {sd} | {t['sell_price']:,.0f} | {t['return_pct']:.2f}% | {t['pnl']:,.0f} "
                f"| {t['hold_days']}일 | {t['reason']} |"
            )
        lines.append("")

    # 차트
    lines.append("## 차트\n")
    lines.append("### 조합별 총손익 비교")
    lines.append("![총손익비교](charts_optimize_consec/total_pnl_comparison.png)\n")
    lines.append("### 조합별 승률 vs 손익비")
    lines.append("![승률vs손익비](charts_optimize_consec/winrate_vs_pf.png)\n")
    lines.append("### 최적 조합 거래별 수익률 분포")
    lines.append("![수익률분포](charts_optimize_consec/best_return_distribution.png)\n")

    lines.append("## 실행 정보\n")
    lines.append(f"- **실행 시간**: {elapsed:.2f}초")
    lines.append(f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    return "\n".join(lines)


def generate_charts(results: list[dict], all_combo_trades: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 차트1: 조합별 총손익 비교
    fig, ax = plt.subplots(figsize=(14, 7))
    results_sorted = sorted(results, key=lambda x: x["total_pnl"], reverse=True)
    labels = [f"{r['ma_period']}MA\n손절{r['stop_loss']}%" for r in results_sorted]
    pnls = [r["total_pnl"] for r in results_sorted]
    colors = ["green" if p >= 0 else "red" for p in pnls]
    bars = ax.bar(range(len(labels)), pnls, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("조합별 총 손익 비교", fontsize=14)
    ax.set_ylabel("총 손익 (원)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    # 값 표시
    for bar, pnl, r in zip(bars, pnls, results_sorted):
        label_text = f"{pnl:,.0f}\n승률:{r['win_rate']:.0f}%"
        ax.text(bar.get_x() + bar.get_width()/2, pnl,
                label_text, ha="center",
                va="bottom" if pnl >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "total_pnl_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 차트2: 승률 vs 손익비 산점도
    fig, ax = plt.subplots(figsize=(10, 8))
    for r in results:
        pf = min(r["profit_factor"], 10)  # 무한대 방지
        size = max(r["n_trades"] * 5, 30)
        color = "green" if r["total_pnl"] >= 0 else "red"
        ax.scatter(r["win_rate"], pf, s=size, c=color, alpha=0.7, edgecolors="black")
        ax.annotate(f"{r['ma_period']}MA/{r['stop_loss']}%",
                    (r["win_rate"], pf), fontsize=9,
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("승률 (%)")
    ax.set_ylabel("손익비")
    ax.set_title("승률 vs 손익비 (크기=거래건수)", fontsize=14)
    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--", label="손익비=1")
    ax.axvline(50, color="gray", linewidth=0.8, linestyle="--", label="승률=50%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "winrate_vs_pf.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 차트3: 최적 조합 거래별 수익률 분포
    best = sorted(results, key=lambda x: x["total_pnl"], reverse=True)[0]
    best_key = f"{best['ma_period']}MA_{best['stop_loss']}%"
    if best_key in all_combo_trades and not all_combo_trades[best_key].empty:
        bt = all_combo_trades[best_key]
        fig, ax = plt.subplots(figsize=(14, 6))
        bins = np.arange(
            max(bt["return_pct"].min() - 1, -50),
            min(bt["return_pct"].max() + 1, 50), 1
        )
        n, bins_out, patches = ax.hist(bt["return_pct"], bins=bins, edgecolor="black", alpha=0.7)
        for patch, b in zip(patches, bins_out):
            if b + (bins_out[1] - bins_out[0]) / 2 >= 0:
                patch.set_facecolor("green")
            else:
                patch.set_facecolor("red")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        avg_ret = bt["return_pct"].mean()
        ax.axvline(avg_ret, color="blue", linewidth=1.5, label=f"평균: {avg_ret:.2f}%")
        ax.set_title(f"최적 조합 ({best_key}) 거래별 수익률 분포 ({len(bt)}건)", fontsize=14)
        ax.set_xlabel("수익률 (%)")
        ax.set_ylabel("거래 건수")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "best_return_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 1) KOSPI200 종목 조회
    print("[1/4] KOSPI200 종목 조회...")
    kospi200 = get_kospi200_tickers()
    with ENGINE.connect() as conn:
        db_rows = conn.execute(
            text("SELECT DISTINCT ticker FROM stocks")
        ).fetchall()
    db_tickers = set(r[0] for r in db_rows)
    ticker_list = [t["ticker"] for t in kospi200 if t["ticker"] in db_tickers]
    name_map = {t["ticker"]: t["name"] for t in kospi200 if t["ticker"] in db_tickers}
    print(f"      KOSPI200 대상: {len(ticker_list)}종목")

    # 2) 데이터 로딩
    print("[2/4] 데이터 로딩...")
    all_data = {}
    batch_data = load_all_data(ticker_list, START_DATE, END_DATE)
    all_data.update(batch_data)
    print(f"      {len(all_data)}종목 로딩 완료")

    # 3) 기준선: 다음날 종가 매도
    print("[3/4] 기준선 + 조합별 백테스트 실행...")
    from backtest_consecutive_crash import run_backtest as run_baseline
    baseline_trades = []
    for ticker in ticker_list:
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        df_test = df.loc[START_DATE:]
        if len(df_test) < 5:
            continue
        name = name_map.get(ticker, ticker)
        _, trades_df = run_baseline(df_test, ticker, name)
        if not trades_df.empty:
            baseline_trades.append(trades_df)
    baseline_all = pd.concat(baseline_trades, ignore_index=True) if baseline_trades else pd.DataFrame()

    # 4) 조합별 백테스트
    combos = list(product(MA_PERIODS, STOP_LOSSES))
    results = []
    all_combo_trades = {"baseline": baseline_all}

    for ma_period, stop_loss in combos:
        combo_label = f"{ma_period}MA_{stop_loss}%"
        combo_trades = []

        for ticker in ticker_list:
            if ticker not in all_data:
                continue
            df = all_data[ticker]
            df_test = df.loc[START_DATE:]
            if len(df_test) < 5:
                continue

            name = name_map.get(ticker, ticker)
            trades_df = run_backtest_ma(df_test, ticker, name, ma_period, stop_loss)
            if not trades_df.empty:
                combo_trades.append(trades_df)

        combo_all = pd.concat(combo_trades, ignore_index=True) if combo_trades else pd.DataFrame()
        all_combo_trades[combo_label] = combo_all

        summary = summarize_trades(combo_all)
        summary["ma_period"] = ma_period
        summary["stop_loss"] = stop_loss
        results.append(summary)

        pf_str = f"{summary['profit_factor']:.2f}" if summary["profit_factor"] != float("inf") else "∞"
        print(f"      {combo_label}: {summary['n_trades']}건, "
              f"승률 {summary['win_rate']:.1f}%, 평균 {summary['avg_return']:.2f}%, "
              f"PF {pf_str}, 총손익 {summary['total_pnl']:,.0f}원")

    # 5) 차트 & 리포트
    chart_dir = os.path.join(base_dir, "charts_optimize_consec")
    print("[4/4] 차트 & 리포트 생성...")
    generate_charts(results, all_combo_trades, chart_dir)

    elapsed = time.time() - start_time
    report = generate_report(results, all_combo_trades, elapsed)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "optimize_consecutive_crash.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*50}")
    print(f"매도 조건 최적화 완료! ({elapsed:.2f}초)")
    print(f"리포트: {report_path}")
    best = sorted(results, key=lambda x: x["total_pnl"], reverse=True)[0]
    print(f"최적 조합: {best['ma_period']}MA + 손절 {best['stop_loss']}%")
    print(f"  총손익: {best['total_pnl']:,.0f}원, 승률: {best['win_rate']:.1f}%, "
          f"평균수익률: {best['avg_return']:.2f}%, 손익비: {best['profit_factor']:.2f}")


if __name__ == "__main__":
    run_main()
