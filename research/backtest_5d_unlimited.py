"""
5일 1500억 5/5 매수 (종목수 제한 없음) + LIST_EXIT 매도.

자본 분배 모델 비교:
- K=5/10/20: 고정 슬롯, 자본 1/K 균등
- K=∞: 동적 분배 — 매수 시점 free_capital을 신규 매수 종목 수로 균등 분배, 매도 시 회수

대상: KOSPI200, 2023~현재
"""
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402
from backtest_5d_amount_filter import compute_5d_filter_signals  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_5d_unlimited.md"

K_FIXED = [5, 10, 20]


def equity_curve_dynamic(trades: list[dict]) -> dict:
    """K=∞ 동적 자본 분배 시뮬레이션.

    - 매일 매도 처리: 청산된 종목의 capital_used × (1 + net_ret) → free_capital 회수
    - 매일 매수 처리: free_capital을 신규 매수 종목 수로 균등 분배
    - mark-to-market 단순화: 매수 자본 그대로 보유 자본으로 인식 (변동 무시)
    - 매도 시점에만 자본 변동 반영
    """
    if not trades:
        return {"final_equity": 1.0, "cagr": 0.0, "mdd": 0.0, "max_concurrent": 0}
    df = pd.DataFrame(trades).copy()
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df["sell_date"] = pd.to_datetime(df["sell_date"])

    all_dates = sorted(set(list(df["buy_date"]) + list(df["sell_date"])))

    free = 1.0
    positions = {}  # idx -> capital_used
    equity = []
    eq_dates = []
    max_concurrent = 0

    for d in all_dates:
        # 1. 매도 처리: 오늘 sell_date인 trade
        sell_idx = df.index[df["sell_date"] == d].tolist()
        for idx in sell_idx:
            if idx in positions:
                cu = positions.pop(idx)
                free += cu * (1 + df.loc[idx, "net_ret"])

        # 2. 매수 처리: 오늘 buy_date인 trade (미보유)
        buy_idx = df.index[df["buy_date"] == d].tolist()
        new_idx = [idx for idx in buy_idx if idx not in positions]
        if new_idx and free > 1e-9:
            per_pos = free / len(new_idx)
            for idx in new_idx:
                positions[idx] = per_pos
                free -= per_pos

        max_concurrent = max(max_concurrent, len(positions))
        total = free + sum(positions.values())
        equity.append(total)
        eq_dates.append(d)

    eq_series = pd.Series(equity, index=pd.to_datetime(eq_dates))
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    mdd = dd.min() * 100

    if len(eq_series) >= 2:
        days = (eq_series.index[-1] - eq_series.index[0]).days
        years = days / 365.25
        cagr = (eq_series.iloc[-1] / eq_series.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    else:
        cagr = 0

    return {
        "final_equity": float(eq_series.iloc[-1]),
        "cagr": float(cagr * 100),
        "mdd": float(mdd),
        "max_concurrent": int(max_concurrent),
    }


def main():
    print("=" * 60)
    print("5일1500억 5/5 (제한없음) + LIST_EXIT 매도")
    print("=" * 60)

    print("[1] KOSPI200 ticker 로드...")
    k200 = bt.load_kospi200_tickers()

    print("[2] 시가총액 snapshot 로드...")
    snapshot = bt.load_market_cap_snapshot()

    print("[3] 가격/거래대금 데이터 로드...")
    t0 = time.time()
    price_df = bt.load_price_data(k200["ticker"].tolist())
    print(f"    {len(price_df):,}행 ({time.time() - t0:.1f}초)")

    print("[4] daily_data 빌드...")
    daily_data = bt.build_daily_data(price_df, snapshot)
    print(f"    {len(daily_data)}종목")

    # cap_k=200으로 사실상 무제한
    print("[5] 매수 신호 생성 (제한없음)...")
    signals = compute_5d_filter_signals(daily_data, top_k=200)
    n_signal_days = len(signals)
    n_total_signal = sum(len(v) for v in signals.values())
    n_unique_tickers = len(set(t for v in signals.values() for t in v))
    n_per_day = [len(v) for v in signals.values()]
    print(f"    {n_signal_days}일치, 총 신호 {n_total_signal}건, 고유 종목 {n_unique_tickers}개")
    print(f"    일별 신호 수: min={min(n_per_day)}, max={max(n_per_day)}, "
          f"mean={np.mean(n_per_day):.2f}, median={int(np.median(n_per_day))}")

    panel = bt.build_daily_indicator_panel(daily_data, "amount")

    # K=고정 + K=∞ 비교 (모두 LIST_EXIT)
    print("\n[6] 자본 분배 모델별 백테스트 (LIST_EXIT 매도)")

    results = []

    # K=고정
    for k in K_FIXED:
        t0 = time.time()
        trades, stats = bt.run_backtest(
            daily_data, panel, signals,
            rule="LIST_EXIT", slots=k,
        )
        elapsed = time.time() - t0
        cagr = stats.get("cagr", 0)
        mdd = stats.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        results.append({
            "model": f"K={k} 고정",
            "stats": stats, "calmar": calmar, "trades": trades,
            "yearly": bt.yearly_stats(trades),
        })
        print(f"  K={k:2d} 고정 : {stats.get('total',0):>5}건, "
              f"CAGR {cagr:>+8.2f}%, MDD {mdd:>+7.2f}%, "
              f"Calmar {calmar:.2f}, 자본 {stats.get('final_equity',1):>6.2f}x | {elapsed:.1f}s")

    # K=∞ (동적 분배)
    t0 = time.time()
    trades, stats = bt.run_backtest(
        daily_data, panel, signals,
        rule="LIST_EXIT", slots=999,  # 시뮬상 슬롯 매우 큼 (cap 안 걸림)
    )
    # equity_curve_simulation 결과 무시하고 동적 모델로 재계산
    dyn = equity_curve_dynamic(trades)
    stats_dyn = bt.compute_stats(trades)
    stats_dyn.update(dyn)
    elapsed = time.time() - t0
    cagr = stats_dyn.get("cagr", 0)
    mdd = stats_dyn.get("mdd", 0)
    calmar = abs(cagr / mdd) if mdd != 0 else 0
    results.append({
        "model": "K=∞ 동적",
        "stats": stats_dyn, "calmar": calmar, "trades": trades,
        "yearly": bt.yearly_stats(trades),
    })
    print(f"  K=∞ 동적 : {stats_dyn.get('total',0):>5}건, "
          f"CAGR {cagr:>+8.2f}%, MDD {mdd:>+7.2f}%, "
          f"Calmar {calmar:.2f}, 자본 {stats_dyn.get('final_equity',1):>6.2f}x, "
          f"동시보유 max {stats_dyn.get('max_concurrent',0)}개 | {elapsed:.1f}s")

    # 리포트
    print("\n[7] 리포트 생성...")
    lines = ["# 5일1500억 5/5 (제한없음) + LIST_EXIT 매도 백테스트\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **대상**: KOSPI200 199종목, 2023-01-01 ~ 현재")
    lines.append("- **매수 조건**: 최근 5일 거래대금 1,500억원 이상이 5일 모두")
    lines.append("- **매수 종목 수**: **제한 없음** (조건 만족 모든 종목 매수)")
    lines.append("- **매도 (LIST_EXIT)**: 다음날 매수 조건 깨지면 → 다다음날 시가 매도")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%\n")
    lines.append("### 자본 분배 모델 4가지\n")
    lines.append("- **K=5/10/20 고정**: 매번 자본의 1/K씩 매수 (과거 결과 비교용)")
    lines.append("- **K=∞ 동적** ⭐: 매수 시점 free_capital을 신규 매수 종목 수로 균등 분배. 매도 시 회수\n")
    lines.append(f"### 신호 통계\n")
    lines.append(f"- 신호 발생일: {n_signal_days}일")
    lines.append(f"- 일별 신호 수: 평균 {np.mean(n_per_day):.2f}, 중앙값 {int(np.median(n_per_day))}, max {max(n_per_day)}")
    lines.append(f"- 고유 종목: {n_unique_tickers}개\n")
    lines.append("---\n")

    # 종합 비교
    lines.append("## 자본 분배 모델별 결과\n")
    lines.append("| 모델 | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 누적자본 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| {r['model']} | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | {s['med_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | {cagr:+.2f} | {mdd:+.2f} | {r['calmar']:.2f} | {s.get('final_equity', 1):.2f} |"
        )

    # 직전 결과와 비교 (Top3, Top5와 본 회차 비교)
    lines.append("\n## 종목 수 제한 비교 (LIST_EXIT 매도)\n")
    lines.append("| 매수 제한 | 자본 분배 | 거래수 | CAGR(%) | MDD(%) | Calmar | 자본 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    lines.append("| Top3 | K=3 | 245 | +107.37 | -22.50 | 4.77 | 6.63x |")
    lines.append("| Top5 | K=5 | 276 | +124.38 | -25.08 | 4.96 | 8.13x |")
    for r in results:
        s = r["stats"]
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        lines.append(
            f"| **무제한** | {r['model']} | {s['total']:,} | "
            f"{cagr:+.2f} | {mdd:+.2f} | {r['calmar']:.2f} | {s.get('final_equity', 1):.2f}x |"
        )

    # 연도별 (K=∞ 동적)
    dyn_r = next((r for r in results if "동적" in r["model"]), None)
    if dyn_r:
        lines.append("\n## 연도별 비교 (K=∞ 동적)\n")
        lines.append("| 연도 | 거래수 | 승률(%) | 평균(%) | 손익비 | 누적자본(trade-level) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for y in sorted(dyn_r["yearly"].keys()):
            ys = dyn_r["yearly"][y]
            if not ys or ys.get("total", 0) == 0:
                lines.append(f"| {y} | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {y} | {ys['total']:,} | {ys['win_rate']:.1f} | {ys['avg_ret']:+.2f} | "
                f"{ys['pf']:.2f} | {ys['cum_return_x']:.2f} |"
            )

    # 최우수 (Calmar)
    valid = [r for r in results if r["stats"].get("total", 0) > 0]
    if valid:
        best = max(valid, key=lambda r: r["calmar"])
        bc = best
        lines.append(f"\n---\n\n## 최우수(Calmar) 상세: {bc['model']}\n")
        s = bc["stats"]
        lines.append("| 지표 | 값 |")
        lines.append("|---|---:|")
        lines.append(f"| 총 거래수 | {s['total']:,} |")
        lines.append(f"| 승률(%) | {s['win_rate']:.1f} |")
        lines.append(f"| 평균 수익률(%) | {s['avg_ret']:+.2f} |")
        lines.append(f"| 중간값 수익률(%) | {s['med_ret']:+.2f} |")
        lines.append(f"| 평균이익(%) | {s['avg_win']:+.2f} |")
        lines.append(f"| 평균손실(%) | {s['avg_loss']:+.2f} |")
        lines.append(f"| 손익비 | {s['pf']:.2f} |")
        lines.append(f"| 평균 보유일 | {s['avg_hold']:.1f} |")
        lines.append(f"| CAGR(%) | {s.get('cagr', 0):+.2f} |")
        lines.append(f"| MDD(%) | {s.get('mdd', 0):+.2f} |")
        lines.append(f"| Calmar | {bc['calmar']:.2f} |")
        lines.append(f"| 최종 자본(x) | {s.get('final_equity', 1):.2f} |")
        if "max_concurrent" in s:
            lines.append(f"| 최대 동시보유 종목 수 | {s['max_concurrent']} |")

        tdf = pd.DataFrame(bc["trades"])
        if not tdf.empty:
            tdf["return_pct"] = tdf["net_ret"] * 100
            ticker_name = dict(zip(snapshot["ticker"], snapshot["name"]))
            tdf["name"] = tdf["ticker"].map(ticker_name).fillna(tdf["ticker"])
            lines.append("\n### 수익률 상위 거래 Top 20\n")
            lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
            lines.append("|---|---|---|---:|---:|")
            for _, r in tdf.nlargest(20, "return_pct").iterrows():
                lines.append(
                    f"| {r['name']}({r['ticker']}) | "
                    f"{pd.to_datetime(r['buy_date']).strftime('%Y-%m-%d')} | "
                    f"{pd.to_datetime(r['sell_date']).strftime('%Y-%m-%d')} | "
                    f"{r['hold_days']} | {r['return_pct']:+.1f} |"
                )
            lines.append("\n### 수익률 하위 거래 Bottom 20\n")
            lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
            lines.append("|---|---|---|---:|---:|")
            for _, r in tdf.nsmallest(20, "return_pct").iterrows():
                lines.append(
                    f"| {r['name']}({r['ticker']}) | "
                    f"{pd.to_datetime(r['buy_date']).strftime('%Y-%m-%d')} | "
                    f"{pd.to_datetime(r['sell_date']).strftime('%Y-%m-%d')} | "
                    f"{r['hold_days']} | {r['return_pct']:+.1f} |"
                )

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
