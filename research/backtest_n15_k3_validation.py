"""(N=15, K=3) 실전 운영성 검증 백테스트.

1500억 매트릭스 Calmar 1위 (N=15, K=3) 조합에 대한 실전 운영 리스크 검증:
- 거래 분포 (시간/종목)
- 종목 집중도
- 무거래 구간 (No-Trade Gap)
- Stress Test (한 거래 -50% 또는 전손 시나리오)
- (N=5, K=5) 와 직접 비교
"""
import os
import time
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402
from backtest_5d_amount_filter import compute_5d_filter_signals  # noqa: E402
from backtest_5d_realistic_k import equity_real_k  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_n15_k3_validation.md"

THRESHOLD_WON = 150_000_000_000  # 1500억


# ──────────────────────────────────────────────
# 헬퍼 함수 (단위 테스트 대상)
# ──────────────────────────────────────────────
def compute_no_trade_gap(trades: list[dict]) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """trades 의 buy_date 정렬 후 인접 차이의 최대 (일 단위, gap 시작/끝 일자)."""
    if not trades:
        return 0, None, None
    if len(trades) == 1:
        return 0, None, None
    dates = sorted(pd.Timestamp(t["buy_date"]) for t in trades)
    max_gap = 0
    gap_start = gap_end = None
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if gap > max_gap:
            max_gap = gap
            gap_start = dates[i - 1]
            gap_end = dates[i]
    return max_gap, gap_start, gap_end


def stress_test_one_trade_loss(trades: list[dict], idx: int,
                               forced_net_ret: float) -> list[dict]:
    """trades 복사본에서 idx 거래의 net_ret 만 forced_net_ret 으로 대체. 원본 불변."""
    new = deepcopy(trades)
    if 0 <= idx < len(new):
        new[idx]["net_ret"] = forced_net_ret
    return new


def count_ticker_trades(trades: list[dict]) -> dict[str, int]:
    return dict(Counter(t["ticker"] for t in trades))


def unique_ticker_count(trades: list[dict]) -> int:
    return len(set(t["ticker"] for t in trades))


def most_repeated_ticker(trades: list[dict]) -> tuple[str, int]:
    counts = count_ticker_trades(trades)
    if not counts:
        return ("", 0)
    return max(counts.items(), key=lambda x: x[1])


# ──────────────────────────────────────────────
# 분석 함수
# ──────────────────────────────────────────────
def yearly_monthly_distribution(trades: list[dict]) -> tuple[dict, dict]:
    """연도별/월별 거래수 dict 반환."""
    if not trades:
        return {}, {}
    df = pd.DataFrame(trades)
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df["year"] = df["buy_date"].dt.year
    df["ym"] = df["buy_date"].dt.to_period("M").astype(str)
    yearly = df.groupby("year").size().to_dict()
    monthly = df.groupby("ym").size().to_dict()
    return yearly, monthly


def trade_pnl_summary(trades: list[dict]) -> dict:
    """거래별 손익 통계."""
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    return {
        "total": len(df),
        "win_rate": (df["net_ret"] > 0).sum() / len(df) * 100,
        "avg_ret": df["net_ret"].mean() * 100,
        "median_ret": df["net_ret"].median() * 100,
        "max_gain": df["net_ret"].max() * 100,
        "max_loss": df["net_ret"].min() * 100,
        "avg_hold": df["hold_days"].mean(),
        "max_hold": df["hold_days"].max(),
        "min_hold": df["hold_days"].min(),
    }


def stress_test_scenario_A(trades: list[dict], K: int,
                           forced_net_ret: float = -0.50) -> list[dict]:
    """시나리오 A: 17건 각각 한 번씩 강제 손실 → 각 케이스 stats 산출."""
    results = []
    for i, t in enumerate(trades):
        stressed = stress_test_one_trade_loss(trades, i, forced_net_ret)
        stats = equity_real_k(stressed, K=K)
        cagr = stats.get("cagr", 0)
        mdd = stats.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        results.append({
            "idx": i, "ticker": t["ticker"],
            "buy_date": pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d"),
            "orig_net_ret": t["net_ret"],
            "forced_net_ret": forced_net_ret,
            "cagr": cagr, "mdd": mdd, "calmar": calmar,
            "final_equity": stats.get("final_equity", 1.0),
        })
    return results


# ──────────────────────────────────────────────
# 백테스트 실행 (단일 N/K)
# ──────────────────────────────────────────────
def run_single_nk(daily_data, panel, n: int, k: int) -> tuple[list[dict], dict]:
    signals = compute_5d_filter_signals(
        daily_data, threshold_won=THRESHOLD_WON, lookback=n, top_k=200,
    )
    trades, _ = bt.run_backtest(
        daily_data, panel, signals,
        rule="LIST_EXIT", slots=k, max_concurrent=k,
    )
    eq = equity_real_k(trades, K=k)
    cagr = eq.get("cagr", 0)
    mdd = eq.get("mdd", 0)
    calmar = abs(cagr / mdd) if mdd != 0 else 0
    eq["calmar"] = calmar
    eq["total"] = len(trades)
    return trades, eq


# ──────────────────────────────────────────────
# 리포트 작성
# ──────────────────────────────────────────────
def _section_distribution(label: str, trades: list[dict]) -> list[str]:
    lines = [f"\n### {label} — 거래 분포\n"]
    yearly, monthly = yearly_monthly_distribution(trades)
    if yearly:
        lines.append("**연도별 거래수**:\n")
        lines.append("| 연도 | 거래수 |")
        lines.append("|---:|---:|")
        for y in sorted(yearly.keys()):
            lines.append(f"| {y} | {yearly[y]} |")
        lines.append("")
    if monthly:
        lines.append("**월별 거래수 (거래 발생 월만)**:\n")
        lines.append("| 월 | 거래수 |")
        lines.append("|---|---:|")
        for ym in sorted(monthly.keys()):
            lines.append(f"| {ym} | {monthly[ym]} |")
        lines.append("")
    gap, gap_s, gap_e = compute_no_trade_gap(trades)
    if gap > 0:
        lines.append(f"**최장 무거래 구간**: {gap}일 ({gap_s.strftime('%Y-%m-%d')} → {gap_e.strftime('%Y-%m-%d')})\n")
    else:
        lines.append("**최장 무거래 구간**: 0일 (단일 거래 또는 무거래)\n")
    return lines


def _section_concentration(label: str, trades: list[dict]) -> list[str]:
    lines = [f"\n### {label} — 종목 집중도\n"]
    counts = count_ticker_trades(trades)
    unique = unique_ticker_count(trades)
    most_t, most_n = most_repeated_ticker(trades)
    lines.append(f"- 고유 종목 수: **{unique}**")
    lines.append(f"- 총 거래수: {len(trades)}")
    lines.append(f"- 가장 많이 거래된 종목: **{most_t}** ({most_n}회)")
    if counts:
        lines.append(f"\n**종목별 거래 횟수 (Top 10)**:\n")
        lines.append("| 순위 | 종목 | 거래수 | 비중(%) |")
        lines.append("|---:|:---:|---:|---:|")
        for i, (t, c) in enumerate(sorted(counts.items(), key=lambda x: -x[1])[:10], 1):
            lines.append(f"| {i} | {t} | {c} | {c/len(trades)*100:.1f} |")
    return lines


def _section_pnl(label: str, trades: list[dict]) -> list[str]:
    lines = [f"\n### {label} — 거래 손익 분포\n"]
    pnl = trade_pnl_summary(trades)
    if not pnl:
        lines.append("(거래 없음)")
        return lines
    lines.append("| 지표 | 값 |")
    lines.append("|---|---:|")
    lines.append(f"| 총 거래수 | {pnl['total']} |")
    lines.append(f"| 승률 | {pnl['win_rate']:.1f}% |")
    lines.append(f"| 평균 수익률 | {pnl['avg_ret']:+.2f}% |")
    lines.append(f"| 중위 수익률 | {pnl['median_ret']:+.2f}% |")
    lines.append(f"| 최대 단일 이익 | {pnl['max_gain']:+.2f}% |")
    lines.append(f"| 최대 단일 손실 | {pnl['max_loss']:+.2f}% |")
    lines.append(f"| 평균 보유일 | {pnl['avg_hold']:.1f}일 |")
    lines.append(f"| 최대/최소 보유일 | {pnl['max_hold']} / {pnl['min_hold']}일 |")
    return lines


def _section_stress(label: str, trades: list[dict], K: int,
                    base_stats: dict) -> list[str]:
    lines = [f"\n### {label} — Stress Test (한 거래 강제 손실 시뮬레이션)\n"]
    lines.append(f"기준 (베이스라인): CAGR {base_stats['cagr']:+.2f}% / MDD {base_stats['mdd']:+.2f}% / Calmar {base_stats['calmar']:.2f}")
    lines.append("")
    for forced in [-0.50, -1.00]:
        scenario_label = f"-{int(abs(forced)*100)}% 직격 (전손)" if forced == -1.00 else f"-{int(abs(forced)*100)}% 직격"
        results = stress_test_scenario_A(trades, K=K, forced_net_ret=forced)
        if not results:
            continue
        cagrs = [r["cagr"] for r in results]
        mdds = [r["mdd"] for r in results]
        calmars = [r["calmar"] for r in results]
        worst = min(results, key=lambda r: r["calmar"])
        best = max(results, key=lambda r: r["calmar"])
        lines.append(f"\n**시나리오: 17건 중 1건이 {scenario_label}**\n")
        lines.append("| 지표 | 평균 | worst-case | best-case | 베이스 대비 평균 차이 |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(f"| CAGR(%) | {np.mean(cagrs):+.2f} | {min(cagrs):+.2f} | {max(cagrs):+.2f} | {np.mean(cagrs)-base_stats['cagr']:+.2f} |")
        lines.append(f"| MDD(%) | {np.mean(mdds):+.2f} | {min(mdds):+.2f} | {max(mdds):+.2f} | {np.mean(mdds)-base_stats['mdd']:+.2f} |")
        lines.append(f"| Calmar | {np.mean(calmars):.2f} | {min(calmars):.2f} | {max(calmars):.2f} | {np.mean(calmars)-base_stats['calmar']:+.2f} |")
        lines.append(f"\nworst-case 거래: {worst['ticker']} ({worst['buy_date']}, 원래 net_ret {worst['orig_net_ret']*100:+.1f}%)")
        lines.append(f"best-case 거래: {best['ticker']} ({best['buy_date']}, 원래 net_ret {best['orig_net_ret']*100:+.1f}%)")
        lines.append("")
    return lines


def main():
    print("=" * 60)
    print("(N=15, K=3) 실전 운영성 검증 + (N=5, K=5) 비교")
    print("=" * 60)

    print("[1] 데이터 로드...")
    k200 = bt.load_kospi200_tickers()
    snapshot = bt.load_market_cap_snapshot()
    price_df = bt.load_price_data(k200["ticker"].tolist())
    daily_data = bt.build_daily_data(price_df, snapshot)
    panel = bt.build_daily_indicator_panel(daily_data, "amount")
    print(f"    {len(daily_data)} 종목")

    print("\n[2] (15, 3) 백테스트...")
    t0 = time.time()
    trades_15_3, stats_15_3 = run_single_nk(daily_data, panel, 15, 3)
    print(f"    거래수 {stats_15_3['total']}, CAGR {stats_15_3['cagr']:+.2f}%, "
          f"MDD {stats_15_3['mdd']:+.2f}%, Calmar {stats_15_3['calmar']:.2f} ({time.time()-t0:.1f}s)")

    print("\n[3] (5, 5) 백테스트 (비교용)...")
    t0 = time.time()
    trades_5_5, stats_5_5 = run_single_nk(daily_data, panel, 5, 5)
    print(f"    거래수 {stats_5_5['total']}, CAGR {stats_5_5['cagr']:+.2f}%, "
          f"MDD {stats_5_5['mdd']:+.2f}%, Calmar {stats_5_5['calmar']:.2f} ({time.time()-t0:.1f}s)")

    print("\n[4] 분석 + 리포트 생성...")
    lines = ["# (N=15, K=3) 실전 운영성 검증 리포트\n"]
    lines.append("## 1. 검증 목적\n")
    lines.append("1500억 N×K 매트릭스에서 Calmar 1위인 (N=15, K=3) 조합이 백테스트 수치만 좋고 "
                 "실전 운영에는 부적합한지(거래 부족·집중 리스크) 정량 검증한다.\n")

    lines.append("## 2. 베이스라인 성과\n")
    lines.append("| 조합 | 거래수 | CAGR(%) | MDD(%) | Calmar | 자본(x) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| **(N=15, K=3)** | {stats_15_3['total']} | {stats_15_3['cagr']:+.2f} "
                 f"| {stats_15_3['mdd']:+.2f} | {stats_15_3['calmar']:.2f} | {stats_15_3['final_equity']:.2f} |")
    lines.append(f"| (N=5, K=5) | {stats_5_5['total']} | {stats_5_5['cagr']:+.2f} "
                 f"| {stats_5_5['mdd']:+.2f} | {stats_5_5['calmar']:.2f} | {stats_5_5['final_equity']:.2f} |")

    lines.append("\n---\n")
    lines.append("## 3. (N=15, K=3) 분석\n")
    lines.extend(_section_distribution("(15,3)", trades_15_3))
    lines.extend(_section_concentration("(15,3)", trades_15_3))
    lines.extend(_section_pnl("(15,3)", trades_15_3))
    lines.extend(_section_stress("(15,3)", trades_15_3, K=3, base_stats=stats_15_3))

    lines.append("\n---\n")
    lines.append("## 4. (N=5, K=5) 분석 (비교)\n")
    lines.extend(_section_distribution("(5,5)", trades_5_5))
    lines.extend(_section_concentration("(5,5)", trades_5_5))
    lines.extend(_section_pnl("(5,5)", trades_5_5))
    lines.extend(_section_stress("(5,5)", trades_5_5, K=5, base_stats=stats_5_5))

    # 직접 비교 표
    lines.append("\n---\n")
    lines.append("## 5. (15,3) vs (5,5) 핵심 비교\n")

    pnl_153 = trade_pnl_summary(trades_15_3)
    pnl_55 = trade_pnl_summary(trades_5_5)
    gap_153, _, _ = compute_no_trade_gap(trades_15_3)
    gap_55, _, _ = compute_no_trade_gap(trades_5_5)
    unique_153 = unique_ticker_count(trades_15_3)
    unique_55 = unique_ticker_count(trades_5_5)
    most_t_153, most_n_153 = most_repeated_ticker(trades_15_3)
    most_t_55, most_n_55 = most_repeated_ticker(trades_5_5)

    lines.append("| 지표 | (15,3) | (5,5) | 누가 더 |")
    lines.append("|---|:---:|:---:|:---:|")
    lines.append(f"| CAGR(%) | {stats_15_3['cagr']:+.2f} | {stats_5_5['cagr']:+.2f} | "
                 f"{'(15,3)' if stats_15_3['cagr'] > stats_5_5['cagr'] else '(5,5)'} |")
    lines.append(f"| MDD(%) | {stats_15_3['mdd']:+.2f} | {stats_5_5['mdd']:+.2f} | "
                 f"{'(15,3)' if stats_15_3['mdd'] > stats_5_5['mdd'] else '(5,5)'} 안전 |")
    lines.append(f"| Calmar | {stats_15_3['calmar']:.2f} | {stats_5_5['calmar']:.2f} | "
                 f"{'(15,3)' if stats_15_3['calmar'] > stats_5_5['calmar'] else '(5,5)'} |")
    lines.append(f"| 거래수 | {stats_15_3['total']} | {stats_5_5['total']} | "
                 f"{'(5,5) 풍부' if stats_5_5['total'] > stats_15_3['total'] else '(15,3)'} |")
    lines.append(f"| 고유 종목 수 | {unique_153} | {unique_55} | "
                 f"{'(5,5) 분산' if unique_55 > unique_153 else '(15,3)'} |")
    lines.append(f"| 최다 거래 종목 비중 | {most_n_153/max(stats_15_3['total'],1)*100:.1f}% ({most_t_153}) "
                 f"| {most_n_55/max(stats_5_5['total'],1)*100:.1f}% ({most_t_55}) | "
                 f"{'(5,5) 분산' if (most_n_55/max(stats_5_5['total'],1)) < (most_n_153/max(stats_15_3['total'],1)) else '(15,3)'} |")
    lines.append(f"| 최대 단일 손실 | {pnl_153.get('max_loss', 0):+.2f}% | {pnl_55.get('max_loss', 0):+.2f}% | "
                 f"{'(15,3) 안전' if pnl_153.get('max_loss', 0) > pnl_55.get('max_loss', 0) else '(5,5) 안전'} |")
    lines.append(f"| 최장 무거래 구간 | {gap_153}일 | {gap_55}일 | "
                 f"{'(5,5) 짧음' if gap_55 < gap_153 else '(15,3)'} |")
    lines.append(f"| 종목당 비중 (1/K) | 33.3% | 20.0% | (5,5) 분산 |")

    lines.append("\n---\n")
    lines.append("## 6. 종합 결론\n")

    # Stress test 평균 계산 (-50% 시나리오)
    sa_153 = stress_test_scenario_A(trades_15_3, K=3, forced_net_ret=-0.50)
    sa_55 = stress_test_scenario_A(trades_5_5, K=5, forced_net_ret=-0.50)
    avg_calmar_153 = np.mean([r["calmar"] for r in sa_153]) if sa_153 else 0
    avg_calmar_55 = np.mean([r["calmar"] for r in sa_55]) if sa_55 else 0
    sb_153 = stress_test_scenario_A(trades_15_3, K=3, forced_net_ret=-1.00)
    sb_55 = stress_test_scenario_A(trades_5_5, K=5, forced_net_ret=-1.00)
    avg_calmar_b153 = np.mean([r["calmar"] for r in sb_153]) if sb_153 else 0
    avg_calmar_b55 = np.mean([r["calmar"] for r in sb_55]) if sb_55 else 0

    lines.append("### 6.1 베이스라인\n")
    lines.append(f"- (15,3): Calmar **{stats_15_3['calmar']:.2f}** (1500억 매트릭스 1위)")
    lines.append(f"- (5,5):  Calmar **{stats_5_5['calmar']:.2f}**")
    lines.append(f"- 단순 수치 → **(15,3)이 약 {stats_15_3['calmar']/max(stats_5_5['calmar'],0.01):.1f}배 우월**")

    lines.append("\n### 6.2 거래 분포·집중도\n")
    lines.append(f"- 거래수: (15,3) {stats_15_3['total']}건 vs (5,5) {stats_5_5['total']}건 → **(5,5)가 {stats_5_5['total']/max(stats_15_3['total'],1):.1f}배 풍부**")
    lines.append(f"- 고유 종목: (15,3) {unique_153}개 vs (5,5) {unique_55}개")
    lines.append(f"- 최장 무거래 구간: (15,3) {gap_153}일 vs (5,5) {gap_55}일")
    lines.append(f"- 최다 거래 종목 비중: (15,3) {most_n_153/max(stats_15_3['total'],1)*100:.1f}% vs (5,5) {most_n_55/max(stats_5_5['total'],1)*100:.1f}%")

    lines.append("\n### 6.3 Stress Test\n")
    lines.append(f"- **-50% 직격 평균 Calmar**: (15,3) {avg_calmar_153:.2f} vs (5,5) {avg_calmar_55:.2f}")
    lines.append(f"- **전손 직격 평균 Calmar**: (15,3) {avg_calmar_b153:.2f} vs (5,5) {avg_calmar_b55:.2f}")

    lines.append("\n### 6.4 채택 가능 여부\n")
    if avg_calmar_b153 > stats_5_5["calmar"]:
        lines.append("- (15,3)은 **전손 시나리오 평균 Calmar조차 (5,5) 베이스라인을 상회** → 수치적으로 (15,3) 채택 정당화 가능")
    else:
        lines.append("- (15,3)은 **전손 시나리오 평균 Calmar가 (5,5) 베이스라인 이하** → 단일 종목 사고 1건이 우위를 무너뜨릴 수 있음")
    if gap_153 > 90:
        lines.append(f"- 무거래 구간 {gap_153}일 → 약 {gap_153/30:.1f}개월 동안 신규 진입 신호 없음, 운영 심리적 부담")
    if unique_153 < 10:
        lines.append(f"- 고유 종목 {unique_153}개 → 사실상 소수 종목 베팅, 분산 효과 제한적")
    if stats_15_3["total"] < 30:
        lines.append(f"- 거래수 {stats_15_3['total']}건 → 통계적 표본 부족, 백테스트 결과의 통계적 신뢰도 한계")

    lines.append("")
    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
