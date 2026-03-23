"""
급등 신호 최적화
- 급락 신호: 이격도(20, 95) 고정
- 급등 신호: 이격도(MA기간, 임계값) 최적화
- 매도 조건(5MA이탈, 손절)은 그대로 유지
- KOSPI200 대상
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
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from backtest_crash import (
    DB_URL, ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    START_DATE, END_DATE,
    get_kospi200_tickers, load_all_data,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

CRASH_TH = 95  # 고정


# ──────────────────────────────────────────────
# 지표 계산 (급등용 MA를 여러 기간 미리 계산)
# ──────────────────────────────────────────────
def calc_indicators_multi(df: pd.DataFrame, surge_ma_list: list[int]) -> pd.DataFrame:
    df = df.copy()
    df["sma20"] = df["close"].rolling(window=20).mean()
    df["sma5"] = df["close"].rolling(window=5).mean()
    df["disparity_20"] = df["close"] / df["sma20"] * 100
    df["is_bearish"] = df["close"] < df["open"]
    df["vol_up"] = df["volume"] > df["volume"].shift(1)
    # 급등용 이격도를 여러 MA 기간으로 미리 계산
    for ma in surge_ma_list:
        col = f"sma_{ma}"
        df[col] = df["close"].rolling(window=ma).mean()
        df[f"disp_{ma}"] = df["close"] / df[col] * 100
    return df


# ──────────────────────────────────────────────
# 파라미터화된 백테스트
# ──────────────────────────────────────────────
def run_backtest_surge(df: pd.DataFrame, surge_ma: int, surge_th: float):
    capital = INITIAL_CAPITAL
    position = None
    gijoongbong = None
    above_20ma = False

    trade_pnls = []  # (pnl, ret, reason)

    closes = df["close"].values
    sma20s = df["sma20"].values
    sma5s = df["sma5"].values
    disp20 = df["disparity_20"].values  # 급락 판단용 (20일 고정)
    surge_disp = df[f"disp_{surge_ma}"].values  # 급등 판단용
    is_bearish = df["is_bearish"].values
    vol_up = df["vol_up"].values
    opens = df["open"].values
    lows = df["low"].values

    for i in range(1, len(df)):
        c = closes[i]
        s20 = sma20s[i]
        s5 = sma5s[i]
        d20 = disp20[i]
        d_surge = surge_disp[i]

        if np.isnan(s20) or np.isnan(s5) or np.isnan(d_surge):
            continue

        if position is not None:
            sell_reason = None
            if not above_20ma and c > s20:
                above_20ma = True
            # 1) 손절: 매수봉 저가 이탈
            if c < position[3]:
                sell_reason = "손절"
            # 2) 급등 신호: surge_ma 기간 이격도 >= surge_th
            elif d_surge >= surge_th:
                sell_reason = "급등신호"
            # 3) 5MA 이탈
            elif above_20ma and c < s5:
                sell_reason = "5MA이탈"

            if sell_reason:
                qty = position[1]
                rev = qty * c
                net = rev - rev * FEE_SELL - rev * TAX_SELL
                capital += net
                cost = position[0] * qty
                fee = cost * FEE_BUY
                pnl = net - cost - fee
                ret = pnl / (cost + fee) * 100
                trade_pnls.append((pnl, ret, sell_reason))
                position = None
                above_20ma = False
        else:
            bought = False
            if gijoongbong is not None:
                if c > gijoongbong:
                    mq = int(capital / (c * (1 + FEE_BUY)))
                    if mq > 0:
                        cost = mq * c
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = (c, mq, i, lows[i])
                        above_20ma = False
                        bought = True
                    gijoongbong = None
            if not bought and gijoongbong is not None:
                if c > s20:
                    gijoongbong = None
            if not bought:
                # 급락 신호: 이격도(20, 95) 고정
                if d20 <= CRASH_TH and is_bearish[i] and vol_up[i]:
                    gijoongbong = opens[i]

    # 미청산
    if position is not None:
        c = closes[-1]
        qty = position[1]
        rev = qty * c
        net = rev - rev * FEE_SELL - rev * TAX_SELL
        capital += net
        cost = position[0] * qty
        fee = cost * FEE_BUY
        pnl = net - cost - fee
        ret = pnl / (cost + fee) * 100
        trade_pnls.append((pnl, ret, "미청산"))

    final_eq = capital
    total_ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n = len(trade_pnls)

    if n > 0:
        wins = [p for p, _, _ in trade_pnls if p > 0]
        losses = [p for p, _, _ in trade_pnls if p <= 0]
        win_rate = len(wins) / n * 100
        gp = sum(wins) if wins else 0
        gl = abs(sum(losses)) if losses else 0
        pf = gp / gl if gl > 0 else float("inf")
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        avg_ret = np.mean([r for _, r, _ in trade_pnls])

        # 사유별 집계
        reason_counts = {}
        for _, _, reason in trade_pnls:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    else:
        win_rate = pf = avg_win = avg_loss = avg_ret = 0
        reason_counts = {}

    return {
        "total_return": total_ret,
        "n_trades": n,
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_return": avg_ret,
        "reason_counts": reason_counts,
    }


# ──────────────────────────────────────────────
# 최적화 실행
# ──────────────────────────────────────────────
def run_optimization():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 파라미터 그리드
    surge_ma_range = [5, 10, 15, 20, 25, 30, 40, 50, 60]
    surge_th_range = list(range(105, 141, 5))  # 105 ~ 140
    total_combos = len(surge_ma_range) * len(surge_th_range)
    print(f"급등 신호 최적화")
    print(f"  급락 고정: 이격도(20, {CRASH_TH})")
    print(f"  급등 탐색: MA기간 {surge_ma_range} x 임계값 {surge_th_range}")
    print(f"  총 조합: {total_combos}")

    # 데이터 준비
    print("\n[1/4] KOSPI200 데이터 로딩...")
    kospi200 = get_kospi200_tickers()
    with ENGINE.connect() as conn:
        db_tickers = set(
            r[0] for r in conn.execute(text("SELECT DISTINCT ticker FROM stocks")).fetchall()
        )
    valid = [t for t in kospi200 if t["ticker"] in db_tickers]
    ticker_list = [t["ticker"] for t in valid]
    all_data = load_all_data(ticker_list, START_DATE, END_DATE)

    # 지표 미리 계산 (모든 surge MA 포함)
    prepped = {}
    for ticker in ticker_list:
        if ticker not in all_data:
            continue
        df = calc_indicators_multi(all_data[ticker], surge_ma_range)
        df_test = df.loc[START_DATE:]
        if len(df_test) >= 30:
            prepped[ticker] = df_test
    print(f"      {len(prepped)}종목 준비 완료")

    # 그리드 서치
    print(f"\n[2/4] 최적화 실행 ({total_combos}개 조합 x {len(prepped)}종목)...")
    results = []
    for combo_idx, (surge_ma, surge_th) in enumerate(product(surge_ma_range, surge_th_range)):
        summaries = []
        for ticker, df_test in prepped.items():
            # 해당 MA 컬럼이 충분한 데이터를 가지는지 확인
            if f"disp_{surge_ma}" not in df_test.columns:
                continue
            res = run_backtest_surge(df_test, surge_ma, surge_th)
            summaries.append(res)

        traded = [s for s in summaries if s["n_trades"] > 0]
        n_traded = len(traded)
        total_trades = sum(s["n_trades"] for s in traded)

        if n_traded > 0:
            avg_ret = np.mean([s["total_return"] for s in traded])
            med_ret = np.median([s["total_return"] for s in traded])
            avg_wr = np.mean([s["win_rate"] for s in traded])
            pf_vals = [s["profit_factor"] for s in traded if s["profit_factor"] != float("inf")]
            avg_pf = np.mean(pf_vals) if pf_vals else 0
            avg_win = np.mean([s["avg_win"] for s in traded if s["avg_win"] > 0])
            avg_loss = np.mean([s["avg_loss"] for s in traded if s["avg_loss"] > 0])
            pos_stocks = sum(1 for s in traded if s["total_return"] > 0)

            # 기대수익
            overall_wr = avg_wr / 100
            ev = overall_wr * avg_win - (1 - overall_wr) * avg_loss

            # 급등신호 발동 비율
            surge_fired = sum(
                s["reason_counts"].get("급등신호", 0) for s in traded
            )
        else:
            avg_ret = med_ret = avg_wr = avg_pf = avg_win = avg_loss = ev = 0
            pos_stocks = surge_fired = 0

        results.append({
            "surge_ma": surge_ma,
            "surge_th": surge_th,
            "n_traded": n_traded,
            "total_trades": total_trades,
            "avg_return": avg_ret,
            "med_return": med_ret,
            "avg_win_rate": avg_wr,
            "avg_profit_factor": avg_pf,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expected_value": ev,
            "pos_stocks": pos_stocks,
            "neg_stocks": n_traded - pos_stocks,
            "surge_fired": surge_fired,
        })

        if (combo_idx + 1) % len(surge_th_range) == 0:
            print(f"      MA={surge_ma} 완료 ({combo_idx+1}/{total_combos})")

    results_df = pd.DataFrame(results)
    elapsed = time.time() - start_time
    print(f"      최적화 완료! ({elapsed:.1f}초)")

    # 차트 & 리포트
    print("\n[3/4] 차트 생성...")
    chart_dir = os.path.join(base_dir, "charts_optimize_surge")
    generate_charts(results_df, chart_dir, surge_ma_range, surge_th_range)

    print("[4/4] 리포트 생성...")
    report = generate_report(results_df, elapsed, len(prepped), surge_ma_range, surge_th_range)
    report_path = os.path.join(base_dir, "results", "backtest_optimize_surge.md")
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"최적화 완료! ({elapsed:.1f}초)")
    print(f"리포트: {report_path}")

    best = results_df.loc[results_df["avg_return"].idxmax()]
    best_ev = results_df.loc[results_df["expected_value"].idxmax()]
    print(f"\n[수익률 최적] 급등 이격도({best['surge_ma']:.0f}일, {best['surge_th']:.0f}%)")
    print(f"  평균 수익률: {best['avg_return']:.2f}%, 승률: {best['avg_win_rate']:.1f}%, 손익비: {best['avg_profit_factor']:.2f}")
    print(f"\n[기대수익 최적] 급등 이격도({best_ev['surge_ma']:.0f}일, {best_ev['surge_th']:.0f}%)")
    print(f"  기대수익: {best_ev['expected_value']:,.0f}원/거래, 승률: {best_ev['avg_win_rate']:.1f}%")


# ──────────────────────────────────────────────
# 차트
# ──────────────────────────────────────────────
def generate_charts(results_df, output_dir, surge_ma_range, surge_th_range):
    os.makedirs(output_dir, exist_ok=True)

    # ── 히트맵 1: 수익률 ──
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot = results_df.pivot(index="surge_ma", columns="surge_th", values="avg_return")
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   extent=[surge_th_range[0]-2.5, surge_th_range[-1]+2.5,
                           surge_ma_range[-1]+2, surge_ma_range[0]-2])
    for i, ma in enumerate(surge_ma_range):
        for j, th in enumerate(surge_th_range):
            val = pivot.loc[ma, th]
            color = "white" if abs(val) > abs(pivot.values).max() * 0.6 else "black"
            ax.text(th, ma, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)
    ax.set_xlabel("급등 이격도 임계값 (%)", fontsize=12)
    ax.set_ylabel("급등 이동평균 기간 (일)", fontsize=12)
    ax.set_title(f"급등 신호 파라미터별 종목 평균 수익률 (%) [급락: 이격도(20,{CRASH_TH}) 고정]", fontsize=13)
    ax.set_xticks(surge_th_range)
    ax.set_yticks(surge_ma_range)
    plt.colorbar(im, label="평균 수익률 (%)")
    best_idx = results_df["avg_return"].idxmax()
    best = results_df.loc[best_idx]
    ax.plot(best["surge_th"], best["surge_ma"], "r*", markersize=20, markeredgecolor="black")
    fig.savefig(os.path.join(output_dir, "heatmap_return.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 히트맵 2: 기대수익 ──
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_ev = results_df.pivot(index="surge_ma", columns="surge_th", values="expected_value")
    im = ax.imshow(pivot_ev.values, cmap="RdYlGn", aspect="auto",
                   extent=[surge_th_range[0]-2.5, surge_th_range[-1]+2.5,
                           surge_ma_range[-1]+2, surge_ma_range[0]-2])
    for i, ma in enumerate(surge_ma_range):
        for j, th in enumerate(surge_th_range):
            val = pivot_ev.loc[ma, th]
            disp = f"{val/10000:.1f}" if abs(val) >= 10000 else f"{val:,.0f}"
            color = "white" if abs(val) > abs(pivot_ev.values).max() * 0.6 else "black"
            ax.text(th, ma, disp, ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)
    ax.set_xlabel("급등 이격도 임계값 (%)", fontsize=12)
    ax.set_ylabel("급등 이동평균 기간 (일)", fontsize=12)
    ax.set_title(f"급등 신호 파라미터별 거래당 기대수익 (만원)", fontsize=13)
    ax.set_xticks(surge_th_range)
    ax.set_yticks(surge_ma_range)
    plt.colorbar(im, label="기대수익 (원)")
    best_ev_idx = results_df["expected_value"].idxmax()
    best_ev = results_df.loc[best_ev_idx]
    ax.plot(best_ev["surge_th"], best_ev["surge_ma"], "r*", markersize=20, markeredgecolor="black")
    fig.savefig(os.path.join(output_dir, "heatmap_ev.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 히트맵 3: 급등신호 발동 횟수 ──
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_sf = results_df.pivot(index="surge_ma", columns="surge_th", values="surge_fired")
    im = ax.imshow(pivot_sf.values, cmap="YlOrRd", aspect="auto",
                   extent=[surge_th_range[0]-2.5, surge_th_range[-1]+2.5,
                           surge_ma_range[-1]+2, surge_ma_range[0]-2])
    for i, ma in enumerate(surge_ma_range):
        for j, th in enumerate(surge_th_range):
            val = pivot_sf.loc[ma, th]
            color = "white" if val > pivot_sf.values.max() * 0.6 else "black"
            ax.text(th, ma, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)
    ax.set_xlabel("급등 이격도 임계값 (%)", fontsize=12)
    ax.set_ylabel("급등 이동평균 기간 (일)", fontsize=12)
    ax.set_title("급등신호 매도 발동 횟수", fontsize=13)
    ax.set_xticks(surge_th_range)
    ax.set_yticks(surge_ma_range)
    plt.colorbar(im, label="발동 횟수")
    fig.savefig(os.path.join(output_dir, "heatmap_surge_fired.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 라인 차트: MA기간별 수익률 ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for ma in surge_ma_range:
        sub = results_df[results_df["surge_ma"] == ma]
        ax.plot(sub["surge_th"], sub["avg_return"], marker="o", label=f"MA{ma}", linewidth=1.5)
    ax.set_xlabel("급등 이격도 임계값 (%)")
    ax.set_ylabel("종목 평균 수익률 (%)")
    ax.set_title("급등 MA기간별 수익률 민감도", fontsize=14)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.savefig(os.path.join(output_dir, "sensitivity_ma.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 라인 차트: 임계값별 수익률 ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for th in surge_th_range:
        sub = results_df[results_df["surge_th"] == th]
        ax.plot(sub["surge_ma"], sub["avg_return"], marker="s", label=f"≥{th}%", linewidth=1.5)
    ax.set_xlabel("급등 이동평균 기간 (일)")
    ax.set_ylabel("종목 평균 수익률 (%)")
    ax.set_title("급등 임계값별 수익률 민감도", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.savefig(os.path.join(output_dir, "sensitivity_th.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# 리포트
# ──────────────────────────────────────────────
def generate_report(results_df, elapsed, n_stocks, surge_ma_range, surge_th_range):
    lines = []
    lines.append("# 급등 신호 최적화 리포트\n")

    lines.append("## 최적화 개요\n")
    lines.append(f"- **대상**: KOSPI200 ({n_stocks}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **고정 조건**:")
    lines.append(f"  - 급락 신호: 이격도(20일, ≤{CRASH_TH}%)")
    lines.append(f"  - 매도: 5MA 종가 이탈, 매수봉 저가 손절")
    lines.append(f"- **탐색 파라미터**:")
    lines.append(f"  - 급등 이동평균 기간: {surge_ma_range}")
    lines.append(f"  - 급등 이격도 임계값: {surge_th_range[0]}% ~ {surge_th_range[-1]}%")
    lines.append(f"- **총 조합 수**: {len(results_df)}")
    lines.append("")

    # 최적 파라미터
    best = results_df.loc[results_df["avg_return"].idxmax()]
    best_ev = results_df.loc[results_df["expected_value"].idxmax()]

    # 기존 (20, 120)
    baseline = results_df[(results_df["surge_ma"] == 20) & (results_df["surge_th"] == 120)]
    # 이전 최적 (20, 125)
    prev_best = results_df[(results_df["surge_ma"] == 20) & (results_df["surge_th"] == 125)]

    lines.append("## 최적 파라미터\n")
    lines.append("### 수익률 기준\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| **급등 이동평균** | **{best['surge_ma']:.0f}일** |")
    lines.append(f"| **급등 이격도** | **≥ {best['surge_th']:.0f}%** |")
    lines.append(f"| 종목 평균 수익률 | {best['avg_return']:.2f}% |")
    lines.append(f"| 종목 중위 수익률 | {best['med_return']:.2f}% |")
    lines.append(f"| 평균 승률 | {best['avg_win_rate']:.1f}% |")
    lines.append(f"| 평균 손익비 | {best['avg_profit_factor']:.2f} |")
    lines.append(f"| 거래당 기대수익 | {best['expected_value']:,.0f}원 |")
    lines.append(f"| 총 거래 수 | {best['total_trades']:.0f} |")
    lines.append(f"| 급등신호 발동 | {best['surge_fired']:.0f}회 |")
    lines.append("")

    lines.append("### 기대수익 기준\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| **급등 이동평균** | **{best_ev['surge_ma']:.0f}일** |")
    lines.append(f"| **급등 이격도** | **≥ {best_ev['surge_th']:.0f}%** |")
    lines.append(f"| 종목 평균 수익률 | {best_ev['avg_return']:.2f}% |")
    lines.append(f"| 평균 승률 | {best_ev['avg_win_rate']:.1f}% |")
    lines.append(f"| 거래당 기대수익 | {best_ev['expected_value']:,.0f}원 |")
    lines.append(f"| 평균수익(승) | {best_ev['avg_win']:,.0f}원 |")
    lines.append(f"| 평균손실(패) | {best_ev['avg_loss']:,.0f}원 |")
    lines.append(f"| 총 거래 수 | {best_ev['total_trades']:.0f} |")
    lines.append("")

    # 비교 테이블
    lines.append("## 기존 vs 이전최적 vs 최적 비교\n")
    rows_to_compare = [
        ("기존", baseline),
        ("이전최적 (20,125)", prev_best),
        ("수익률 최적", pd.DataFrame([best])),
        ("기대수익 최적", pd.DataFrame([best_ev])),
    ]
    lines.append("| 지표 | " + " | ".join(name for name, _ in rows_to_compare) + " |")
    lines.append("|------|" + "|".join("---" for _ in rows_to_compare) + "|")

    def _val(df, col, fmt=".2f"):
        if df is not None and not df.empty:
            v = df.iloc[0][col]
            return f"{v:{fmt}}"
        return "N/A"

    lines.append("| 급등 조건 | " + " | ".join(
        f"({_val(df,'surge_ma','.0f')}일, ≥{_val(df,'surge_th','.0f')}%)" if df is not None and not df.empty else "N/A"
        for _, df in rows_to_compare
    ) + " |")
    for label, col, fmt in [
        ("평균 수익률", "avg_return", ".2f"),
        ("승률", "avg_win_rate", ".1f"),
        ("손익비", "avg_profit_factor", ".2f"),
        ("기대수익(원)", "expected_value", ",.0f"),
        ("거래 수", "total_trades", ",.0f"),
        ("급등발동", "surge_fired", ",.0f"),
    ]:
        vals = []
        for _, df in rows_to_compare:
            vals.append(_val(df, col, fmt) if df is not None and not df.empty else "N/A")
        unit = "%" if "률" in label or "승률" in label else ""
        lines.append(f"| {label} | " + " | ".join(f"{v}{unit}" for v in vals) + " |")
    lines.append("")

    # 차트
    lines.append("## 히트맵\n")
    lines.append("### 수익률")
    lines.append("![수익률](charts_optimize_surge/heatmap_return.png)\n")
    lines.append("### 거래당 기대수익")
    lines.append("![기대수익](charts_optimize_surge/heatmap_ev.png)\n")
    lines.append("### 급등신호 발동 횟수")
    lines.append("![급등발동](charts_optimize_surge/heatmap_surge_fired.png)\n")

    lines.append("## 민감도 분석\n")
    lines.append("### MA기간별 수익률")
    lines.append("![MA민감도](charts_optimize_surge/sensitivity_ma.png)\n")
    lines.append("### 임계값별 수익률")
    lines.append("![임계값민감도](charts_optimize_surge/sensitivity_th.png)\n")

    # 전체 결과
    lines.append("## 전체 결과 (수익률 순)\n")
    lines.append("| 순위 | MA기간 | 급등(≥) | 평균수익률 | 승률 | 손익비 | 기대수익 | 거래수 | 급등발동 |")
    lines.append("|------|--------|---------|----------|------|--------|---------|--------|---------|")
    sorted_df = results_df.sort_values("avg_return", ascending=False)
    for rank, (_, r) in enumerate(sorted_df.iterrows()):
        pf_str = f"{r['avg_profit_factor']:.2f}" if r['avg_profit_factor'] < 100 else "∞"
        lines.append(
            f"| {rank+1} | {r['surge_ma']:.0f}일 | {r['surge_th']:.0f}% "
            f"| {r['avg_return']:.2f}% | {r['avg_win_rate']:.1f}% | {pf_str} "
            f"| {r['expected_value']:,.0f}원 | {r['total_trades']:.0f} | {r['surge_fired']:.0f} |"
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
