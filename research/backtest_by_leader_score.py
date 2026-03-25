"""
Leader Score 밴드별 매매 규칙 백테스트
- leaders 테이블 전체 종목 대상, leader_score를 90+/80s/70s/<70으로 그룹 분석
- 전략1: V6 급락주 매매 (이격도15일 92/125, 기준봉+양봉+갭하락)
- 전략2: 2봉 연속 급락 V1 (2봉 연속 -6%)
- 전략3: 2봉 연속 급락 V2 (V1 OR 총낙폭 -12%)
"""

import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text as sql_text

from backtest_crash import (
    ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    START_DATE, END_DATE, load_all_data,
)
from backtest_simple_buy import calc_indicators_15
from backtest_by_grade import run_v6, run_consec, grade_summary, format_pf

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BANDS = ["90+", "80s", "70s", "<70"]


def get_ticker_leader_scores() -> dict[str, dict]:
    query = sql_text("""
        SELECT DISTINCT ON (ticker) ticker, name, leader_score
        FROM leaders
        ORDER BY ticker, snapshot_date DESC
    """)
    with ENGINE.connect() as conn:
        rows = conn.execute(query).fetchall()

    result = {}
    for r in rows:
        score = float(r[2])
        if score >= 90:
            band = "90+"
        elif score >= 80:
            band = "80s"
        elif score >= 70:
            band = "70s"
        else:
            band = "<70"
        result[r[0]] = {"name": r[1], "leader_score": score, "band": band}
    return result


def generate_report(strategy_results: dict, band_counts: dict, elapsed: float) -> str:
    lines = []
    lines.append("# Leader Score 밴드별 매매 규칙 백테스트 리포트\n")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **대상**: leaders 테이블 전체 종목 (leader_score 밴드별 그룹)")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append("")

    lines.append("## Leader Score 밴드 분포\n")
    lines.append("| 밴드 | 종목수 |")
    lines.append("|------|--------|")
    for b in BANDS:
        lines.append(f"| {b} | {band_counts.get(b, 0)} |")
    lines.append("")

    for strat_name, band_trades in strategy_results.items():
        lines.append(f"## {strat_name}\n")

        lines.append("### Leader Score 밴드별 성과 비교\n")
        lines.append("| 밴드 | 종목수 | 거래수 | 승률 | 손익비 | 평균수익률 | 중위수익률 | 총손익 | 거래당기대수익 | 평균보유일 |")
        lines.append("|------|--------|--------|------|--------|----------|----------|--------|------------|----------|")

        for b in BANDS + ["전체"]:
            if b not in band_trades:
                continue
            s = grade_summary(band_trades[b])
            if s["n"] == 0:
                lines.append(f"| **{b}** | {band_counts.get(b, '-')} | 0 | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| **{b}** | {s['tickers']} | {s['n']} | {s['win_rate']:.1f}% | "
                f"{format_pf(s['pf'])} | {s['avg_ret']:.2f}% | {s['med_ret']:.2f}% | "
                f"{s['total_pnl']:,.0f}원 | {s['expect']:,.0f}원 | {s['hold']:.1f}일 |"
            )
        lines.append("")

        # 매도 사유별
        lines.append("### 밴드별 매도 사유\n")
        for b in BANDS:
            if b not in band_trades or band_trades[b].empty:
                continue
            tdf = band_trades[b]
            lines.append(f"#### Leader {b}\n")
            lines.append("| 사유 | 건수 | 승률 | 평균수익률 | 총손익 |")
            lines.append("|------|------|------|----------|--------|")
            for reason, grp in tdf.groupby("reason"):
                n = len(grp)
                wr = (grp["pnl"] > 0).sum() / n * 100
                ar = grp["return_pct"].mean()
                tp = grp["pnl"].sum()
                lines.append(f"| {reason} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 |")
            lines.append("")

        # V6 매수 유형별
        if "buy_type" in band_trades.get("전체", pd.DataFrame()).columns:
            lines.append("### 밴드별 매수 유형\n")
            for b in BANDS:
                if b not in band_trades or band_trades[b].empty:
                    continue
                tdf = band_trades[b]
                lines.append(f"#### Leader {b}\n")
                lines.append("| 유형 | 건수 | 승률 | 평균수익률 | 총손익 |")
                lines.append("|------|------|------|----------|--------|")
                for bt, grp in tdf.groupby("buy_type"):
                    n = len(grp)
                    wr = (grp["pnl"] > 0).sum() / n * 100
                    ar = grp["return_pct"].mean()
                    tp = grp["pnl"].sum()
                    lines.append(f"| {bt} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 |")
                lines.append("")

    lines.append(f"---\n실행 시간: {elapsed:.2f}초")
    lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def generate_charts(strategy_results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    colors = {"90+": "#4CAF50", "80s": "#8BC34A", "70s": "#FFC107", "<70": "#f44336"}

    for strat_name, band_trades in strategy_results.items():
        safe_name = strat_name.replace(" ", "_").replace("/", "_")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        pnls = [band_trades[b]["pnl"].sum() if b in band_trades and not band_trades[b].empty else 0 for b in BANDS]
        bar_colors = [colors[b] for b in BANDS]
        axes[0].bar(BANDS, pnls, color=bar_colors, alpha=0.8, edgecolor="black")
        axes[0].set_title(f"{strat_name}\n총손익", fontsize=12)
        axes[0].set_ylabel("원")
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].grid(True, alpha=0.3, axis="y")
        for i, (b, p) in enumerate(zip(BANDS, pnls)):
            n = len(band_trades[b]) if b in band_trades and not band_trades[b].empty else 0
            axes[0].text(i, p, f"{p/10000:,.0f}만\n({n}건)", ha="center",
                        va="bottom" if p >= 0 else "top", fontsize=9)

        wrs = []
        for b in BANDS:
            if b in band_trades and not band_trades[b].empty:
                t = band_trades[b]
                wrs.append((t["pnl"] > 0).sum() / len(t) * 100)
            else:
                wrs.append(0)
        axes[1].bar(BANDS, wrs, color=bar_colors, alpha=0.8, edgecolor="black")
        axes[1].set_title(f"{strat_name}\n승률", fontsize=12)
        axes[1].set_ylabel("%")
        axes[1].set_ylim(0, max(wrs) * 1.2 if max(wrs) > 0 else 100)
        axes[1].axhline(50, color="gray", linewidth=0.8, linestyle="--")
        axes[1].grid(True, alpha=0.3, axis="y")
        for i, w in enumerate(wrs):
            axes[1].text(i, w + 0.5, f"{w:.1f}%", ha="center", fontsize=10)

        expects = [band_trades[b]["pnl"].mean() if b in band_trades and not band_trades[b].empty else 0 for b in BANDS]
        exp_colors = ["green" if e >= 0 else "red" for e in expects]
        axes[2].bar(BANDS, expects, color=exp_colors, alpha=0.7, edgecolor="black")
        axes[2].set_title(f"{strat_name}\n거래당 기대수익", fontsize=12)
        axes[2].set_ylabel("원")
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].grid(True, alpha=0.3, axis="y")
        for i, e in enumerate(expects):
            axes[2].text(i, e, f"{e:,.0f}", ha="center",
                        va="bottom" if e >= 0 else "top", fontsize=9)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 통합 비교
    fig, ax = plt.subplots(figsize=(12, 6))
    strat_names = list(strategy_results.keys())
    x = np.arange(len(BANDS))
    width = 0.25
    for si, sn in enumerate(strat_names):
        bt = strategy_results[sn]
        avg_rets = [bt[b]["return_pct"].mean() if b in bt and not bt[b].empty else 0 for b in BANDS]
        ax.bar(x + si * width, avg_rets, width, label=sn, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(BANDS, fontsize=11)
    ax.set_title("전략별 × Leader Score 밴드별 평균 수익률 비교", fontsize=14)
    ax.set_ylabel("평균 수익률 (%)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "all_strategies_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    print("[1/4] leaders 점수 매핑 조회...")
    ticker_info = get_ticker_leader_scores()
    band_counts = {}
    for info in ticker_info.values():
        b = info["band"]
        band_counts[b] = band_counts.get(b, 0) + 1
    print(f"      전체: {len(ticker_info)}종목")
    for b in BANDS:
        print(f"      Leader {b}: {band_counts.get(b, 0)}종목")

    print("\n[2/4] 데이터 로딩...")
    all_tickers = list(ticker_info.keys())
    all_data = load_all_data(all_tickers, START_DATE, END_DATE)
    available = set(all_data.keys())
    print(f"      {len(available)}종목 데이터 로딩 완료")

    band_tickers = {b: [] for b in BANDS}
    for ticker, info in ticker_info.items():
        if ticker in available:
            band_tickers[info["band"]].append(ticker)

    print("\n[3/4] 백테스트 실행...")
    strategy_results = {}

    for strat_name, strat_func in [
        ("V6 급락주 매매", "v6"),
        ("2봉연속급락 V1", "v1"),
        ("2봉연속급락 V2", "v2"),
    ]:
        print(f"\n  === {strat_name} ===")
        band_trades = {}

        for band in BANDS:
            b_trades = []
            for ticker in band_tickers[band]:
                df = all_data[ticker]
                df_test = df.loc[START_DATE:]
                if len(df_test) < 20:
                    continue
                name = ticker_info[ticker]["name"]

                if strat_func == "v6":
                    trades = run_v6(calc_indicators_15(df_test), ticker, name)
                elif strat_func == "v1":
                    trades = run_consec(df_test, ticker, name, use_total_drop=False)
                else:
                    trades = run_consec(df_test, ticker, name, use_total_drop=True)

                if not trades.empty:
                    b_trades.append(trades)

            band_trades[band] = pd.concat(b_trades, ignore_index=True) if b_trades else pd.DataFrame()
            n = len(band_trades[band])
            if n > 0:
                wr = (band_trades[band]["pnl"] > 0).sum() / n * 100
                pnl = band_trades[band]["pnl"].sum()
                print(f"      Leader {band}: {n}건, 승률 {wr:.1f}%, 총손익 {pnl:,.0f}원")

        all_dfs = [band_trades[b] for b in BANDS if not band_trades[b].empty]
        band_trades["전체"] = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        strategy_results[strat_name] = band_trades

    print("\n[4/4] 리포트 & 차트 생성...")
    chart_dir = os.path.join(base_dir, "charts_by_leader_score")
    generate_charts(strategy_results, chart_dir)

    elapsed = time.time() - start_time
    report = generate_report(strategy_results, band_counts, elapsed)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_by_leader_score.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*70}")
    print("Leader Score 밴드별 백테스트 완료!")
    print(f"{'='*70}")
    for strat_name, band_trades in strategy_results.items():
        print(f"\n  [{strat_name}]")
        for b in BANDS:
            t = band_trades.get(b, pd.DataFrame())
            if t.empty:
                continue
            w = (t["pnl"] > 0).sum()
            n = len(t)
            print(f"    Leader {b:>3s}: {n:>4d}건, 승률 {w/n*100:>5.1f}%, "
                  f"평균수익률 {t['return_pct'].mean():>6.2f}%, "
                  f"총손익 {t['pnl'].sum():>13,.0f}원")

    print(f"\n리포트: {report_path}")
    print(f"실행 시간: {elapsed:.2f}초")


if __name__ == "__main__":
    run_main()
