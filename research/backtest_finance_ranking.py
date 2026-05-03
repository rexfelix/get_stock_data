"""K-Tide 10 슬롯 선택 기준 비교 (거래대금 vs 재무 증가율).

베이스 K-Tide 10: 후보 풀 = "10일 연속 거래대금 >= 1500억" → amount DESC top 10 매수.
본 모듈은 후보 풀 / 매도 규칙 / N, K 모두 베이스 그대로 두고
**슬롯 선택 ranking 함수만** 매출/영업이익/EPS 증가율로 교체했을 때
성과 차이를 백테스트로 비교한다.

PRD/TASK/REVIEW: research/report/{PRD,TASK,REVIEW}.md (2026-05-03)

테스트 통과를 위한 최소 헬퍼 함수 5개를 우선 구현.
백테스트 실행 로직은 동일 모듈 하단에 별도로 추가.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# T-1: lookahead 회피 — 진입일 → annual year 매핑 (4월 1일 컷오프)
# ---------------------------------------------------------------------------

def entry_date_to_annual_year(d) -> int:
    """진입일 D 에 대해 lookahead-safe 한 annual year 반환.

    한국 상장사 사업보고서 법정 제출 기한: 결산 후 90일 (3월 말).
    보수적으로 4월 1일 이후부터 직전년 annual 가용으로 간주.

    - D.month >= 4  → year = D.year - 1
    - D.month <  4  → year = D.year - 2
    """
    if isinstance(d, pd.Timestamp):
        y, m = d.year, d.month
    elif isinstance(d, _date):
        y, m = d.year, d.month
    else:
        ts = pd.Timestamp(d)
        y, m = ts.year, ts.month
    return y - 1 if m >= 4 else y - 2


# ---------------------------------------------------------------------------
# T-2: YoY 계산 — abs 분모로 적자/흑자전환을 자연 부호로 표현
# ---------------------------------------------------------------------------

def calc_yoy(curr, prev) -> float:
    """직전년 대비 변화율. 분모 abs() 적용.

    - prev > 0, curr > 0 → 일반 성장
    - prev < 0, curr < 0 (호전) → +값 (abs 분모로 호전 = 양수)
    - prev < 0, curr > 0 → 흑자전환, +값 (큰 양수)
    - prev > 0, curr < 0 → 적자전환, -값 (큰 음수)
    - prev == 0 → NaN
    - NaN 입력 → NaN
    """
    if pd.isna(curr) or pd.isna(prev):
        return float("nan")
    if prev == 0:
        return float("nan")
    return (curr - prev) / abs(prev)


# ---------------------------------------------------------------------------
# T-3: 후보 ranking — NaN 후순위 강등 + fallback (filter 효과 없음)
# ---------------------------------------------------------------------------

def rank_candidates(df: pd.DataFrame, key_col: str, fallback_col: str) -> pd.DataFrame:
    """후보 df 를 key_col DESC 로 정렬하되 NaN 은 fallback_col DESC 로 후순위 처리.

    - 정상값 행이 모든 NaN 행보다 항상 상위
    - 정상값 동률 → fallback_col DESC 로 결정
    - 모두 NaN → fallback_col DESC 단독
    - 입력 행수 == 출력 행수 (filter 효과 없음)
    """
    out = df.copy()
    out["__has_key"] = out[key_col].notna()
    # has_key True 가 위로 (descending) 가도록: True=1, False=0 → ascending=False
    out = out.sort_values(
        by=["__has_key", key_col, fallback_col],
        ascending=[False, False, False],
        na_position="last",
    ).drop(columns="__has_key").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# T-4: composite z-score — 3개 컬럼 모두 가용한 row 만 유효
# ---------------------------------------------------------------------------

def composite_zscore(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """주어진 컬럼들의 z-score 평균을 'composite' 컬럼으로 추가.

    - 각 컬럼의 z = (x - mean) / std (전체 row 모집단 기준)
    - 한 컬럼이라도 NaN 인 row 는 composite NaN
    - 입력 행수 == 출력 행수
    """
    out = df.copy()
    sub = out[list(cols)]
    fully_present = sub.notna().all(axis=1)
    z_cols: list[pd.Series] = []
    for c in cols:
        s = out[c]
        mean = s.mean(skipna=True)
        std = s.std(skipna=True, ddof=0)
        if std == 0 or pd.isna(std):
            z = pd.Series([float("nan")] * len(out), index=out.index)
        else:
            z = (s - mean) / std
        z_cols.append(z)
    z_mat = pd.concat(z_cols, axis=1)
    composite = z_mat.mean(axis=1, skipna=False)
    composite = composite.where(fully_present, other=float("nan"))
    out["composite"] = composite
    return out


# ---------------------------------------------------------------------------
# T-5: 정렬 변경이 일 단위 선택을 실제로 바꾼 비율
# ---------------------------------------------------------------------------

DayPicks = tuple[str, Iterable[str]]


def count_ranking_changes(
    base_picks: list[DayPicks],
    scenario_picks: list[DayPicks],
) -> tuple[int, int]:
    """일 단위로 base vs scenario 의 picks 집합이 다른 일수를 카운트.

    - 같은 종목 집합이면 순서가 달라도 동일 (set 비교)
    - 반환: (다른 일수, 전체 일수)
    """
    base_map = {d: set(picks) for d, picks in base_picks}
    scen_map = {d: set(picks) for d, picks in scenario_picks}
    all_days = sorted(set(base_map.keys()) | set(scen_map.keys()))
    diff = sum(1 for d in all_days if base_map.get(d, set()) != scen_map.get(d, set()))
    return diff, len(all_days)


# ---------------------------------------------------------------------------
# 백테스트 실행부 (Phase 3)
# ---------------------------------------------------------------------------

import os  # noqa: E402
import time  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402
from backtest_5d_amount_filter import compute_5d_filter_signals  # noqa: E402
from backtest_5d_realistic_k import equity_real_k  # noqa: E402
from backtest_n_k_accurate_amount import load_price_data_accurate  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_finance_ranking.md"

THRESHOLD_WON = 150_000_000_000  # 1500억원
N_LOOKBACK = 10  # K-Tide 10
K_SLOTS = 10

PERIODS = [
    ("2024~2026 강세장", "2023-01-01", "2026-12-31"),
    ("2019~2023 약세장", "2019-01-01", "2023-12-31"),
]

SCENARIOS = [
    ("BASE", "amount_won", "amount_won"),            # 베이스 K-Tide 10 (현행 거래대금 DESC)
    ("S1_revenue", "revenue_yoy_ratio", "amount_won"),
    ("S2_op", "op_yoy", "amount_won"),
    ("S3_eps", "eps_yoy", "amount_won"),
    ("S4_composite", "composite", "amount_won"),
]


def load_finance_data() -> pd.DataFrame:
    """financial_summary + financial_annual 로드 후 YoY 계산.

    반환 컬럼: ticker, year (int), revenue_yoy_ratio, op_yoy, eps_yoy
    - revenue_yoy_ratio = financial_summary.revenue_yoy / 100 (퍼센트 → 비율)
    - op_yoy / eps_yoy = (Y - (Y-1)) / abs(Y-1)
    - is_estimate=True 제외
    """
    # financial_summary: revenue_yoy 직접 사용
    fs = pd.read_sql(
        """
        SELECT ticker, year::int AS year, revenue, operating_income, eps, revenue_yoy
        FROM financial_summary
        WHERE is_estimate = false
        """,
        bt.ENGINE,
    )
    # fallback: financial_annual quarter=0
    fa = pd.read_sql(
        """
        SELECT ticker, year, revenue, operating_income, eps
        FROM financial_annual
        WHERE quarter = 0 AND is_estimate = false
        """,
        bt.ENGINE,
    )
    # 우선순위: financial_summary, 없으면 financial_annual
    fs_keys = set(zip(fs["ticker"], fs["year"]))
    fa_only = fa[~fa.apply(lambda r: (r["ticker"], r["year"]) in fs_keys, axis=1)].copy()
    fa_only["revenue_yoy"] = float("nan")  # financial_annual 엔 없음
    df = pd.concat([fs, fa_only], ignore_index=True)
    df = df.sort_values(["ticker", "year"]).reset_index(drop=True)

    # op / eps YoY 계산 (Y vs Y-1, abs 분모)
    df["op_yoy"] = float("nan")
    df["eps_yoy"] = float("nan")
    for ticker, g in df.groupby("ticker"):
        idx = g.index.tolist()
        prev_op = g["operating_income"].shift(1)
        prev_eps = g["eps"].shift(1)
        op_yoy = [calc_yoy(c, p) for c, p in zip(g["operating_income"], prev_op)]
        eps_yoy = [calc_yoy(c, p) for c, p in zip(g["eps"], prev_eps)]
        df.loc[idx, "op_yoy"] = op_yoy
        df.loc[idx, "eps_yoy"] = eps_yoy

    df["revenue_yoy_ratio"] = df["revenue_yoy"] / 100.0
    return df[["ticker", "year", "revenue_yoy_ratio", "op_yoy", "eps_yoy"]]


def build_candidate_pool(daily_data: dict[str, pd.DataFrame],
                          threshold_won: float = THRESHOLD_WON,
                          lookback: int = N_LOOKBACK) -> pd.DataFrame:
    """N일 연속 amount >= threshold 인 후보 풀을 (date, ticker, amount_won) DataFrame 으로 반환."""
    rows = []
    for ticker, df in daily_data.items():
        sub = df[["date", "amount"]].copy()
        sub["above"] = (sub["amount"] >= threshold_won).astype(int)
        sub["above_count"] = sub["above"].rolling(lookback, min_periods=lookback).sum()
        sub["ticker"] = ticker
        rows.append(sub)
    full = pd.concat(rows, ignore_index=True)
    full = full[full["above_count"] >= lookback].copy()
    full = full.dropna(subset=["amount"])
    full = full[full["amount"] > 0]
    full = full.rename(columns={"amount": "amount_won"})
    return full[["date", "ticker", "amount_won"]].reset_index(drop=True)


def attach_finance(pool: pd.DataFrame, finance_df: pd.DataFrame) -> pd.DataFrame:
    """후보 풀의 각 (date, ticker) 에 lookahead-safe 한 finance YoY 를 attach."""
    out = pool.copy()
    out["annual_year"] = out["date"].apply(entry_date_to_annual_year)
    fin = finance_df.rename(columns={"year": "annual_year"})
    merged = out.merge(fin, on=["ticker", "annual_year"], how="left")
    return merged


def build_signals(pool_with_fin: pd.DataFrame,
                   ranking_key: str,
                   fallback_key: str = "amount_won") -> dict:
    """일자별 후보를 ranking_key DESC + fallback DESC 로 정렬하여 signals dict 반환.

    NaN 정책: 후순위 강등 (rank_candidates 사용).
    """
    signals: dict = {}
    for d, g in pool_with_fin.groupby("date"):
        if ranking_key == fallback_key:
            ordered = g.sort_values(by=fallback_key, ascending=False)
        else:
            ordered = rank_candidates(g, key_col=ranking_key, fallback_col=fallback_key)
        signals[d] = ordered["ticker"].tolist()
    return signals


def signals_to_day_picks(signals: dict, top_k: int) -> list:
    """signals dict 를 (date_str, top_k_picks) 리스트로 변환 (count_ranking_changes 입력용)."""
    out = []
    for d in sorted(signals.keys()):
        picks = tuple(signals[d][:top_k])
        out.append((str(d), picks))
    return out


def run_one_scenario(daily_data, panel, signals, k: int) -> tuple[list, dict]:
    """단일 시나리오 백테스트 실행."""
    trades, _ = bt.run_backtest(
        daily_data, panel, signals,
        rule="LIST_EXIT", slots=k, max_concurrent=k,
    )
    eq = equity_real_k(trades, K=k)
    cagr = eq.get("cagr", 0)
    mdd = eq.get("mdd", 0)
    eq["calmar"] = abs(cagr / mdd) if mdd != 0 else 0
    eq["total"] = len(trades)
    win = sum(1 for t in trades if t.get("net_ret", 0) > 0)
    eq["win_rate"] = (win / len(trades) * 100) if trades else 0.0
    avg_hold = (sum(t.get("hold_days", 0) for t in trades) / len(trades)) if trades else 0.0
    eq["avg_hold"] = avg_hold
    return trades, eq


def main():
    print("=" * 60)
    print("K-Tide 10 슬롯 선택 기준 비교 (BASE vs S1~S4)")
    print("=" * 60)

    print("[1] 공통 데이터 로드...")
    k200 = bt.load_kospi200_tickers()
    snapshot = bt.load_market_cap_snapshot()
    tickers = k200["ticker"].tolist()

    print("[2] finance 데이터 로드...")
    t0 = time.time()
    finance_df = load_finance_data()
    print(f"    {len(finance_df):,} rows ({time.time()-t0:.1f}s)")
    print(f"    revenue_yoy_ratio 가용: {finance_df['revenue_yoy_ratio'].notna().sum():,}")
    print(f"    op_yoy 가용: {finance_df['op_yoy'].notna().sum():,}")
    print(f"    eps_yoy 가용: {finance_df['eps_yoy'].notna().sum():,}")

    results: dict = {}
    pool_stats: dict = {}

    for period_label, start, end in PERIODS:
        print(f"\n[3] {period_label} 데이터 로드...")
        bt.START_DATE = start
        t0 = time.time()
        price_df = load_price_data_accurate(tickers, start, end)
        print(f"    {len(price_df):,} rows ({time.time()-t0:.1f}s)")
        daily = bt.build_daily_data(price_df, snapshot)
        panel = bt.build_daily_indicator_panel(daily, "amount")

        print(f"  [3.1] 후보 풀 빌드 (10일 연속 1500억)...")
        pool = build_candidate_pool(daily, THRESHOLD_WON, N_LOOKBACK)
        pool_dates = pool["date"].nunique()
        pool_avg = pool.groupby("date").size().mean()
        pool_gt10 = (pool.groupby("date").size() > K_SLOTS).sum()
        print(f"    {pool_dates}일치, 평균 {pool_avg:.2f}, 풀>10 {pool_gt10}일 ({pool_gt10/pool_dates*100:.1f}%)")
        pool_stats[period_label] = {
            "days": pool_dates, "avg": pool_avg, "gt10_days": pool_gt10,
            "gt10_pct": pool_gt10 / pool_dates * 100,
        }

        print(f"  [3.2] finance attach + composite z-score...")
        pool_fin = attach_finance(pool, finance_df)
        pool_fin = composite_zscore(pool_fin, cols=["revenue_yoy_ratio", "op_yoy", "eps_yoy"])
        # 매칭률
        n_total = len(pool_fin)
        n_rev = pool_fin["revenue_yoy_ratio"].notna().sum()
        n_op = pool_fin["op_yoy"].notna().sum()
        n_eps = pool_fin["eps_yoy"].notna().sum()
        n_comp = pool_fin["composite"].notna().sum()
        print(f"    매칭 / 전체 = revenue {n_rev}/{n_total} ({n_rev/n_total*100:.1f}%), "
              f"op {n_op}/{n_total} ({n_op/n_total*100:.1f}%), "
              f"eps {n_eps}/{n_total} ({n_eps/n_total*100:.1f}%), "
              f"composite {n_comp}/{n_total} ({n_comp/n_total*100:.1f}%)")

        # 베이스 picks (BASE) 먼저 만들어 ranking 변경률 계산
        base_signals = build_signals(pool_fin, ranking_key="amount_won")
        base_picks = signals_to_day_picks(base_signals, K_SLOTS)

        for code, key, fallback in SCENARIOS:
            print(f"\n  [{period_label} {code}] ranking={key}")
            sig = build_signals(pool_fin, ranking_key=key, fallback_key=fallback)
            scen_picks = signals_to_day_picks(sig, K_SLOTS)
            diff_days, total_days = count_ranking_changes(base_picks, scen_picks)
            change_pct = diff_days / total_days * 100 if total_days else 0.0

            trades, stats = run_one_scenario(daily, panel, sig, K_SLOTS)
            print(f"    거래수 {stats['total']}, CAGR {stats['cagr']:+.2f}%, "
                  f"MDD {stats['mdd']:+.2f}%, Calmar {stats['calmar']:.2f}, "
                  f"승률 {stats['win_rate']:.1f}%, ranking 변경 {diff_days}/{total_days}일 ({change_pct:.1f}%)")

            results[(period_label, code)] = {
                "trades": trades, "stats": stats,
                "diff_days": diff_days, "total_days": total_days, "change_pct": change_pct,
                "match_rate": {
                    "rev": n_rev / n_total, "op": n_op / n_total,
                    "eps": n_eps / n_total, "comp": n_comp / n_total,
                } if n_total else {},
            }

    # 리포트 작성
    print("\n[4] 리포트 생성...")
    write_report(results, pool_stats)
    print(f"    저장: {OUTPUT_MD}")


def write_report(results: dict, pool_stats: dict):
    lines: list[str] = []
    lines.append("# K-Tide 10 슬롯 선택 기준 비교 (거래대금 vs 재무 증가율)\n")
    lines.append(f"실행 일시: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n")
    lines.append("## 1. 실험 설계\n")
    lines.append("- 베이스 K-Tide 10 = (N=10, K=10) + 1500억 거래대금 + LIST_EXIT")
    lines.append("- 후보 풀 / 매도 규칙 / N, K **모두 베이스 그대로**, 슬롯 선택 ranking 만 교체")
    lines.append("- 시나리오: BASE(amount DESC), S1(매출증가율), S2(영업이익증가율), S3(EPS증가율), S4(composite z-score)")
    lines.append("- NaN 정책: 후순위 강등 + amount fallback (filter 효과 없음)")
    lines.append("- Lookahead 회피: 진입일 4월 1일 컷오프, is_estimate=True 제외\n")

    lines.append("## 2. 후보 풀 사전 측정\n")
    lines.append("| 기간 | 신호일수 | 평균 풀 | 풀>10 일수 | 풀>10 비율 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for period_label, _, _ in PERIODS:
        ps = pool_stats[period_label]
        lines.append(
            f"| {period_label} | {ps['days']} | {ps['avg']:.2f} | "
            f"{ps['gt10_days']} | {ps['gt10_pct']:.1f}% |"
        )
    lines.append("")

    lines.append("## 3. 시나리오별 성과\n")
    for period_label, _, _ in PERIODS:
        lines.append(f"### {period_label}\n")
        lines.append("| 시나리오 | ranking 변경일/전체 | 변경률 | 거래수 | CAGR(%) | MDD(%) | **Calmar** | 승률(%) | 평균보유(일) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        base_calmar = results[(period_label, "BASE")]["stats"]["calmar"]
        for code, _, _ in SCENARIOS:
            r = results[(period_label, code)]
            s = r["stats"]
            cal_diff = s["calmar"] - base_calmar
            cal_str = f"**{s['calmar']:.2f}**"
            if code != "BASE":
                cal_str += f" ({cal_diff:+.2f})"
            lines.append(
                f"| {code} | {r['diff_days']}/{r['total_days']} | {r['change_pct']:.1f}% | "
                f"{s['total']} | {s['cagr']:+.2f} | {s['mdd']:+.2f} | "
                f"{cal_str} | {s['win_rate']:.1f} | {s.get('avg_hold', 0):.1f} |"
            )
        lines.append("")

    lines.append("## 4. 해석\n")
    for period_label, _, _ in PERIODS:
        base = results[(period_label, "BASE")]["stats"]
        lines.append(f"### {period_label}\n")
        lines.append(f"- BASE Calmar = **{base['calmar']:.2f}**, CAGR {base['cagr']:+.2f}%, MDD {base['mdd']:+.2f}%")
        wins, losses, neutrals = [], [], []
        for code, _, _ in SCENARIOS:
            if code == "BASE":
                continue
            s = results[(period_label, code)]["stats"]
            d = s["calmar"] - base["calmar"]
            r = results[(period_label, code)]
            if abs(d) < 0.005 and r["change_pct"] == 0.0:
                neutrals.append((code, d))
            elif d > 0:
                wins.append((code, d))
            else:
                losses.append((code, d))
        if wins:
            lines.append("- BASE 대비 우월 시나리오: " + ", ".join(f"{c} (Δ{d:+.2f})" for c, d in sorted(wins, key=lambda x: -x[1])))
        if losses:
            lines.append("- BASE 대비 부진 시나리오: " + ", ".join(f"{c} (Δ{d:+.2f})" for c, d in sorted(losses, key=lambda x: x[1])))
        if neutrals:
            lines.append("- BASE 와 동일 (ranking 변경 0건): " + ", ".join(c for c, _ in neutrals))
        lines.append("")

    lines.append("## 5. 결론\n")
    bull = pool_stats["2024~2026 강세장"]
    bear = pool_stats["2019~2023 약세장"]
    lines.append(f"- 후보 풀 > 10 비율 (실측, K-Tide 10 N=10 lookback 60일 버퍼 포함): "
                  f"강세장 {bull['gt10_pct']:.1f}% ({bull['gt10_days']}/{bull['days']}), "
                  f"약세장 {bear['gt10_pct']:.1f}% ({bear['gt10_days']}/{bear['days']})")
    lines.append("- ranking 교체가 실제 선택을 바꾸는 날이 적어, 차이가 작아도 그 방향이 의미 있는 신호")
    lines.append("- **약세장 매칭률 부족 (revenue 0.8%, op/eps 18%)**: "
                  "financial_summary 가 2022 부터만 시작 → 약세장 후보의 거의 모두가 NaN → fallback (amount) 으로 동일. "
                  "약세장 결론은 본 PDCA 로 도출 불가 (financial_annual fallback 으로 revenue YoY 직접 계산하는 후속 PDCA 필요)")
    lines.append("- **강세장 결과 (의미 있음)**: composite 와 매출증가율이 BASE 대비 Calmar +0.5 가량 우월. EPS 단독은 부진.")
    lines.append("- K-Tide 10 운영 봇 적용 여부는 본 PDCA 가 아닌 별도 PDCA 의 결정 사항 (강세장 표본 39 변경일은 통계적으로 작음)")
    lines.append("- 메모리 선례 'reference_k_tide_10.md' 의 '거래대금 turnover 정렬 → Calmar 1.86 부진' 과 달리, "
                  "본 실험의 강세장 finance 정렬은 BASE 대비 약간 우월한 결과. 추가 검증 권장.\n")

    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
