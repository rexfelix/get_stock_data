"""
급락주 매매 전략 - 이격도 임계값 최적화
- KOSPI200 대상
- 급락 이격도(crash_th): 85~95% 범위
- 급등 이격도(surge_th): 105~135% 범위
- 20일 이동평균 고정
"""

import os
import time
from datetime import datetime
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from backtest_crash import (
    DB_URL, ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    START_DATE, END_DATE,
    get_kospi200_tickers, load_all_data, calc_indicators,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ──────────────────────────────────────────────
# 파라미터화된 백테스트 (핵심 로직만, 경량화)
# ──────────────────────────────────────────────
def run_backtest_param(df: pd.DataFrame, crash_th: float, surge_th: float):
    """이격도 임계값을 파라미터로 받는 경량 백테스트. 종목 합산용 지표만 반환."""
    capital = INITIAL_CAPITAL
    position = None
    gijoongbong = None
    above_20ma = False
    trades_pnl = []
    peak_equity = INITIAL_CAPITAL

    dates = df.index.tolist()
    closes = df["close"].values
    sma20s = df["sma20"].values
    sma5s = df["sma5"].values
    disparities = df["disparity"].values
    is_bearish = df["is_bearish"].values
    vol_up = df["vol_up"].values
    opens = df["open"].values
    lows = df["low"].values

    for i in range(1, len(df)):
        close = closes[i]
        sma20 = sma20s[i]
        sma5 = sma5s[i]
        disp = disparities[i]

        if np.isnan(sma20) or np.isnan(sma5):
            continue

        if position is not None:
            sell_reason = None
            if not above_20ma and close > sma20:
                above_20ma = True
            if close < position[3]:  # buy_candle_low
                sell_reason = "손절"
            elif disp >= surge_th:
                sell_reason = "급등신호"
            elif above_20ma and close < sma5:
                sell_reason = "5MA이탈"

            if sell_reason:
                qty = position[1]
                revenue = qty * close
                net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
                capital += net
                buy_cost = position[0] * qty
                buy_fee = buy_cost * FEE_BUY
                pnl = net - buy_cost - buy_fee
                ret = pnl / (buy_cost + buy_fee) * 100
                trades_pnl.append((pnl, ret, sell_reason))
                position = None
                above_20ma = False
        else:
            bought = False
            if gijoongbong is not None:
                if close > gijoongbong:  # gijoongbong = open_price
                    max_qty = int(capital / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        cost = max_qty * close
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = (close, max_qty, dates[i], lows[i])  # (price, qty, date, low)
                        above_20ma = False
                        bought = True
                    gijoongbong = None
            if not bought and gijoongbong is not None:
                if close > sma20:
                    gijoongbong = None
            if not bought:
                if disp <= crash_th and is_bearish[i] and vol_up[i]:
                    gijoongbong = opens[i]

        # 자산 추적 (MDD용)
        eq = capital + (position[1] * close if position else 0)
        if eq > peak_equity:
            peak_equity = eq

    # 미청산 포지션
    if position is not None:
        last_close = closes[-1]
        qty = position[1]
        revenue = qty * last_close
        net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
        capital += net
        buy_cost = position[0] * qty
        buy_fee = buy_cost * FEE_BUY
        pnl = net - buy_cost - buy_fee
        ret = pnl / (buy_cost + buy_fee) * 100
        trades_pnl.append((pnl, ret, "미청산"))

    final_equity = capital
    total_ret = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    n_trades = len(trades_pnl)
    if n_trades > 0:
        wins = sum(1 for p, _, _ in trades_pnl if p > 0)
        win_rate = wins / n_trades * 100
        gross_profit = sum(p for p, _, _ in trades_pnl if p > 0)
        gross_loss = abs(sum(p for p, _, _ in trades_pnl if p <= 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_ret = sum(r for _, r, _ in trades_pnl) / n_trades
    else:
        win_rate = 0
        pf = 0
        avg_ret = 0

    return {
        "total_return": total_ret,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_return": avg_ret,
        "final_equity": final_equity,
    }


# ──────────────────────────────────────────────
# 최적화 실행
# ──────────────────────────────────────────────
def run_optimization():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 파라미터 그리드
    crash_range = list(range(85, 96))       # 85~95
    surge_range = list(range(105, 136, 5))  # 105, 110, 115, 120, 125, 130, 135
    total_combos = len(crash_range) * len(surge_range)
    print(f"파라미터 그리드: 급락 {crash_range} x 급등 {surge_range} = {total_combos}개 조합")

    # 데이터 준비
    print("[1/4] KOSPI200 데이터 로딩...")
    kospi200 = get_kospi200_tickers()
    with ENGINE.connect() as conn:
        db_tickers = set(
            r[0] for r in conn.execute(text("SELECT DISTINCT ticker FROM stocks")).fetchall()
        )
    valid = [t for t in kospi200 if t["ticker"] in db_tickers]
    ticker_list = [t["ticker"] for t in valid]
    name_map = {t["ticker"]: t["name"] for t in valid}

    all_data = load_all_data(ticker_list, START_DATE, END_DATE)
    # 지표 미리 계산
    prepped = {}
    for ticker in ticker_list:
        if ticker not in all_data:
            continue
        df = calc_indicators(all_data[ticker])
        df_test = df.loc[START_DATE:]
        if len(df_test) >= 30:
            prepped[ticker] = df_test
    print(f"      {len(prepped)}종목 준비 완료")

    # 그리드 서치
    print(f"[2/4] 최적화 실행 ({total_combos}개 조합 x {len(prepped)}종목)...")
    results = []
    for combo_idx, (crash_th, surge_th) in enumerate(product(crash_range, surge_range)):
        combo_summaries = []
        all_pnl = []
        for ticker, df_test in prepped.items():
            res = run_backtest_param(df_test, crash_th, surge_th)
            combo_summaries.append(res)
            all_pnl.append(res["total_return"])

        traded = [s for s in combo_summaries if s["n_trades"] > 0]
        n_traded = len(traded)
        total_trades = sum(s["n_trades"] for s in traded)

        if n_traded > 0:
            avg_ret = np.mean([s["total_return"] for s in traded])
            med_ret = np.median([s["total_return"] for s in traded])
            avg_wr = np.mean([s["win_rate"] for s in traded])
            avg_pf = np.mean([s["profit_factor"] for s in traded if s["profit_factor"] != float("inf")])
            pos_stocks = sum(1 for s in traded if s["total_return"] > 0)
        else:
            avg_ret = med_ret = avg_wr = avg_pf = 0
            pos_stocks = 0

        results.append({
            "crash_th": crash_th,
            "surge_th": surge_th,
            "n_traded": n_traded,
            "total_trades": total_trades,
            "avg_return": avg_ret,
            "med_return": med_ret,
            "avg_win_rate": avg_wr,
            "avg_profit_factor": avg_pf,
            "pos_stocks": pos_stocks,
            "neg_stocks": n_traded - pos_stocks,
        })

        if (combo_idx + 1) % 11 == 0:
            print(f"      {combo_idx+1}/{total_combos} 완료... "
                  f"(현재: 급락≤{crash_th}, 급등≥{surge_th} → 평균 {avg_ret:.2f}%)")

    results_df = pd.DataFrame(results)
    elapsed = time.time() - start_time
    print(f"      최적화 완료! ({elapsed:.1f}초)")

    # 차트 & 리포트
    print("[3/4] 차트 생성...")
    chart_dir = os.path.join(base_dir, "charts_optimize")
    generate_optimize_charts(results_df, chart_dir)

    print("[4/4] 리포트 생성...")
    report = generate_optimize_report(results_df, elapsed, len(prepped))
    report_path = os.path.join(base_dir, "results", "backtest_optimize.md")
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"최적화 완료! ({elapsed:.1f}초)")
    print(f"리포트: {report_path}")

    # 최적 파라미터
    best = results_df.loc[results_df["avg_return"].idxmax()]
    print(f"\n최적 파라미터:")
    print(f"  급락 이격도 ≤ {best['crash_th']:.0f}%")
    print(f"  급등 이격도 ≥ {best['surge_th']:.0f}%")
    print(f"  종목 평균 수익률: {best['avg_return']:.2f}%")
    print(f"  거래 종목: {best['n_traded']:.0f}, 총 거래: {best['total_trades']:.0f}")
    print(f"  승률: {best['avg_win_rate']:.1f}%, 손익비: {best['avg_profit_factor']:.2f}")


# ──────────────────────────────────────────────
# 최적화 차트
# ──────────────────────────────────────────────
def generate_optimize_charts(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    crash_vals = sorted(results_df["crash_th"].unique())
    surge_vals = sorted(results_df["surge_th"].unique())

    # ── 히트맵 1: 평균 수익률 ──
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot = results_df.pivot(index="crash_th", columns="surge_th", values="avg_return")
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   extent=[surge_vals[0]-2.5, surge_vals[-1]+2.5,
                           crash_vals[-1]+0.5, crash_vals[0]-0.5])
    # 셀 값 표시
    for i, crash in enumerate(crash_vals):
        for j, surge in enumerate(surge_vals):
            val = pivot.loc[crash, surge]
            color = "white" if abs(val) > pivot.values.max() * 0.7 else "black"
            ax.text(surge, crash, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)
    ax.set_xlabel("급등 이격도 (%)", fontsize=12)
    ax.set_ylabel("급락 이격도 (%)", fontsize=12)
    ax.set_title("파라미터별 종목 평균 수익률 (%)", fontsize=14)
    ax.set_xticks(surge_vals)
    ax.set_yticks(crash_vals)
    plt.colorbar(im, label="평균 수익률 (%)")
    # 최적점 표시
    best_idx = results_df["avg_return"].idxmax()
    best = results_df.loc[best_idx]
    ax.plot(best["surge_th"], best["crash_th"], "r*", markersize=20, markeredgecolor="black")
    fig.savefig(os.path.join(output_dir, "heatmap_return.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 히트맵 2: 승률 ──
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_wr = results_df.pivot(index="crash_th", columns="surge_th", values="avg_win_rate")
    im = ax.imshow(pivot_wr.values, cmap="RdYlGn", aspect="auto",
                   extent=[surge_vals[0]-2.5, surge_vals[-1]+2.5,
                           crash_vals[-1]+0.5, crash_vals[0]-0.5])
    for i, crash in enumerate(crash_vals):
        for j, surge in enumerate(surge_vals):
            val = pivot_wr.loc[crash, surge]
            color = "white" if val > pivot_wr.values.max() * 0.8 else "black"
            ax.text(surge, crash, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)
    ax.set_xlabel("급등 이격도 (%)", fontsize=12)
    ax.set_ylabel("급락 이격도 (%)", fontsize=12)
    ax.set_title("파라미터별 평균 승률 (%)", fontsize=14)
    ax.set_xticks(surge_vals)
    ax.set_yticks(crash_vals)
    plt.colorbar(im, label="승률 (%)")
    fig.savefig(os.path.join(output_dir, "heatmap_winrate.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 히트맵 3: 총 거래 수 ──
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_tr = results_df.pivot(index="crash_th", columns="surge_th", values="total_trades")
    im = ax.imshow(pivot_tr.values, cmap="Blues", aspect="auto",
                   extent=[surge_vals[0]-2.5, surge_vals[-1]+2.5,
                           crash_vals[-1]+0.5, crash_vals[0]-0.5])
    for i, crash in enumerate(crash_vals):
        for j, surge in enumerate(surge_vals):
            val = pivot_tr.loc[crash, surge]
            color = "white" if val > pivot_tr.values.max() * 0.7 else "black"
            ax.text(surge, crash, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)
    ax.set_xlabel("급등 이격도 (%)", fontsize=12)
    ax.set_ylabel("급락 이격도 (%)", fontsize=12)
    ax.set_title("파라미터별 총 거래 건수", fontsize=14)
    ax.set_xticks(surge_vals)
    ax.set_yticks(crash_vals)
    plt.colorbar(im, label="거래 수")
    fig.savefig(os.path.join(output_dir, "heatmap_trades.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 히트맵 4: 손익비 ──
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_pf = results_df.pivot(index="crash_th", columns="surge_th", values="avg_profit_factor")
    # clip for display
    pf_clipped = pivot_pf.clip(upper=5)
    im = ax.imshow(pf_clipped.values, cmap="RdYlGn", aspect="auto",
                   extent=[surge_vals[0]-2.5, surge_vals[-1]+2.5,
                           crash_vals[-1]+0.5, crash_vals[0]-0.5])
    for i, crash in enumerate(crash_vals):
        for j, surge in enumerate(surge_vals):
            val = pivot_pf.loc[crash, surge]
            disp_val = f"{val:.2f}" if val < 10 else f"{val:.0f}"
            color = "white" if val > pf_clipped.values.max() * 0.7 else "black"
            ax.text(surge, crash, disp_val, ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)
    ax.set_xlabel("급등 이격도 (%)", fontsize=12)
    ax.set_ylabel("급락 이격도 (%)", fontsize=12)
    ax.set_title("파라미터별 평균 손익비", fontsize=14)
    ax.set_xticks(surge_vals)
    ax.set_yticks(crash_vals)
    plt.colorbar(im, label="손익비")
    fig.savefig(os.path.join(output_dir, "heatmap_pf.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 차트5: 급락 임계값별 수익률 (급등 고정 라인 차트) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for surge in surge_vals:
        subset = results_df[results_df["surge_th"] == surge]
        ax.plot(subset["crash_th"], subset["avg_return"], marker="o", label=f"급등≥{surge}%", linewidth=1.5)
    ax.set_xlabel("급락 이격도 임계값 (%)")
    ax.set_ylabel("종목 평균 수익률 (%)")
    ax.set_title("급락 임계값별 수익률 (급등 조건별)", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.savefig(os.path.join(output_dir, "crash_sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 차트6: 급등 임계값별 수익률 (급락 고정 라인 차트) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for crash in crash_vals:
        subset = results_df[results_df["crash_th"] == crash]
        ax.plot(subset["surge_th"], subset["avg_return"], marker="s", label=f"급락≤{crash}%", linewidth=1.5)
    ax.set_xlabel("급등 이격도 임계값 (%)")
    ax.set_ylabel("종목 평균 수익률 (%)")
    ax.set_title("급등 임계값별 수익률 (급락 조건별)", fontsize=14)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.savefig(os.path.join(output_dir, "surge_sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# 리포트 생성
# ──────────────────────────────────────────────
def generate_optimize_report(results_df, elapsed, n_stocks):
    lines = []
    lines.append("# 급락주 매매 전략 - 이격도 임계값 최적화 리포트\n")

    lines.append("## 최적화 개요\n")
    lines.append(f"- **대상**: KOSPI200 ({n_stocks}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **고정 파라미터**: 20일 이동평균, 5일 이동평균(매도)")
    lines.append(f"- **탐색 범위**:")
    lines.append(f"  - 급락 이격도: {sorted(results_df['crash_th'].unique())[0]}% ~ {sorted(results_df['crash_th'].unique())[-1]}%")
    lines.append(f"  - 급등 이격도: {sorted(results_df['surge_th'].unique())[0]}% ~ {sorted(results_df['surge_th'].unique())[-1]}%")
    lines.append(f"- **총 조합 수**: {len(results_df)}")
    lines.append("")

    # 최적 파라미터
    best_idx = results_df["avg_return"].idxmax()
    best = results_df.loc[best_idx]

    # 손익비 최적
    valid_pf = results_df[results_df["total_trades"] >= 50]
    best_pf = valid_pf.loc[valid_pf["avg_profit_factor"].idxmax()] if not valid_pf.empty else best

    # 승률 최적
    best_wr = results_df.loc[results_df["avg_win_rate"].idxmax()]

    lines.append("## 최적 파라미터\n")
    lines.append("### 수익률 기준 최적\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| **급락 이격도** | **≤ {best['crash_th']:.0f}%** |")
    lines.append(f"| **급등 이격도** | **≥ {best['surge_th']:.0f}%** |")
    lines.append(f"| 종목 평균 수익률 | {best['avg_return']:.2f}% |")
    lines.append(f"| 종목 중위 수익률 | {best['med_return']:.2f}% |")
    lines.append(f"| 거래 종목 수 | {best['n_traded']:.0f} |")
    lines.append(f"| 총 거래 건수 | {best['total_trades']:.0f} |")
    lines.append(f"| 평균 승률 | {best['avg_win_rate']:.1f}% |")
    lines.append(f"| 평균 손익비 | {best['avg_profit_factor']:.2f} |")
    lines.append(f"| 수익/손실 종목 | {best['pos_stocks']:.0f} / {best['neg_stocks']:.0f} |")
    lines.append("")

    lines.append("### 손익비 기준 최적 (거래 50건 이상)\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| **급락 이격도** | **≤ {best_pf['crash_th']:.0f}%** |")
    lines.append(f"| **급등 이격도** | **≥ {best_pf['surge_th']:.0f}%** |")
    lines.append(f"| 종목 평균 수익률 | {best_pf['avg_return']:.2f}% |")
    lines.append(f"| 평균 승률 | {best_pf['avg_win_rate']:.1f}% |")
    lines.append(f"| 평균 손익비 | {best_pf['avg_profit_factor']:.2f} |")
    lines.append(f"| 총 거래 건수 | {best_pf['total_trades']:.0f} |")
    lines.append("")

    # 기존 파라미터와 비교
    baseline = results_df[(results_df["crash_th"] == 90) & (results_df["surge_th"] == 120)]
    if not baseline.empty:
        bl = baseline.iloc[0]
        lines.append("## 기존 파라미터 vs 최적 파라미터 비교\n")
        lines.append("| 지표 | 기존 (90/120) | 최적 수익률 | 최적 손익비 |")
        lines.append("|------|-------------|-----------|-----------|")
        lines.append(f"| 급락/급등 | ≤90% / ≥120% | ≤{best['crash_th']:.0f}% / ≥{best['surge_th']:.0f}% | ≤{best_pf['crash_th']:.0f}% / ≥{best_pf['surge_th']:.0f}% |")
        lines.append(f"| 평균 수익률 | {bl['avg_return']:.2f}% | {best['avg_return']:.2f}% | {best_pf['avg_return']:.2f}% |")
        lines.append(f"| 승률 | {bl['avg_win_rate']:.1f}% | {best['avg_win_rate']:.1f}% | {best_pf['avg_win_rate']:.1f}% |")
        lines.append(f"| 손익비 | {bl['avg_profit_factor']:.2f} | {best['avg_profit_factor']:.2f} | {best_pf['avg_profit_factor']:.2f} |")
        lines.append(f"| 거래 수 | {bl['total_trades']:.0f} | {best['total_trades']:.0f} | {best_pf['total_trades']:.0f} |")
        lines.append(f"| 거래 종목 | {bl['n_traded']:.0f} | {best['n_traded']:.0f} | {best_pf['n_traded']:.0f} |")
        lines.append("")

    # 차트
    lines.append("## 히트맵\n")
    lines.append("### 수익률 히트맵")
    lines.append("![수익률](charts_optimize/heatmap_return.png)\n")
    lines.append("### 승률 히트맵")
    lines.append("![승률](charts_optimize/heatmap_winrate.png)\n")
    lines.append("### 거래 건수 히트맵")
    lines.append("![거래수](charts_optimize/heatmap_trades.png)\n")
    lines.append("### 손익비 히트맵")
    lines.append("![손익비](charts_optimize/heatmap_pf.png)\n")

    lines.append("## 민감도 분석\n")
    lines.append("### 급락 임계값 민감도")
    lines.append("![급락민감도](charts_optimize/crash_sensitivity.png)\n")
    lines.append("### 급등 임계값 민감도")
    lines.append("![급등민감도](charts_optimize/surge_sensitivity.png)\n")

    # 전체 결과 테이블 (수익률 기준 정렬)
    lines.append("## 전체 결과 (수익률 순)\n")
    lines.append("| 순위 | 급락(≤) | 급등(≥) | 평균수익률 | 중위수익률 | 승률 | 손익비 | 거래수 | 거래종목 |")
    lines.append("|------|---------|---------|----------|----------|------|--------|--------|---------|")
    sorted_df = results_df.sort_values("avg_return", ascending=False)
    for rank, (_, r) in enumerate(sorted_df.iterrows()):
        pf_str = f"{r['avg_profit_factor']:.2f}" if r['avg_profit_factor'] < 100 else "∞"
        marker = " **" if (r["crash_th"] == best["crash_th"] and r["surge_th"] == best["surge_th"]) else ""
        lines.append(
            f"| {rank+1} | {r['crash_th']:.0f}% | {r['surge_th']:.0f}% "
            f"| {r['avg_return']:.2f}%{marker} | {r['med_return']:.2f}% "
            f"| {r['avg_win_rate']:.1f}% | {pf_str} | {r['total_trades']:.0f} | {r['n_traded']:.0f} |"
        )
    lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- **실행 시간**: {elapsed:.1f}초")
    lines.append(f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **총 백테스트 횟수**: {len(results_df)} x {n_stocks} = {len(results_df)*n_stocks:,}")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    run_optimization()
