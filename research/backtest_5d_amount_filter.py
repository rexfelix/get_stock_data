"""
"최근 5일 거래대금 1500억 이상이 5일 모두" 매수 조건 + MA_INIT_STOP 매도 백테스트.

매수 조건:
- T일 마감 시점에 최근 5일 (T-4 ~ T) amount >= 150,000,000,000원(1500억) 인 날이 5일 모두
- 만족 종목 중 amount 큰 순으로 최대 3종목 → T+1 시가 매수 (자본 슬롯 3개)

매도 규칙: MA_INIT_STOP (MA5/20 × stop_pct -3%/-5%/-7%) = 6 조합

대상: KOSPI200, 2023~현재
"""
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_5d_amount_filter.md"

THRESHOLD_WON = 150_000_000_000  # 1500억원
LOOKBACK = 5
TOP_K = 3

MA_PERIODS = [5, 20]
STOP_PCTS = [-0.03, -0.05, -0.07]


def compute_5d_filter_signals(daily_data: dict[str, pd.DataFrame],
                              threshold_won: float = THRESHOLD_WON,
                              lookback: int = LOOKBACK,
                              top_k: int = TOP_K) -> dict[pd.Timestamp, list[str]]:
    """일별 매수 신호 종목 리스트 생성.

    조건: 최근 lookback일 동안 amount >= threshold_won 인 날이 lookback일 모두.
    여러 종목이 만족하면 amount 큰 순으로 top_k개.
    """
    rows = []
    for ticker, df in daily_data.items():
        sub = df[["date", "amount"]].copy()
        sub["above"] = (sub["amount"] >= threshold_won).astype(int)
        sub["above_count"] = sub["above"].rolling(lookback, min_periods=lookback).sum()
        sub["ticker"] = ticker
        rows.append(sub)
    full = pd.concat(rows, ignore_index=True)
    full = full[full["above_count"] >= lookback]
    full = full.dropna(subset=["amount"])
    full = full[full["amount"] > 0]

    signals = {}
    for d, g in full.groupby("date"):
        # amount 큰 순으로 top_k
        top = g.sort_values("amount", ascending=False).head(top_k)
        signals[d] = top["ticker"].tolist()
    return signals


def main():
    print("=" * 60)
    print("5일 1500억 필터 매수 + MA_INIT_STOP 매도 백테스트")
    print("=" * 60)

    print("[1] KOSPI200 ticker 로드...")
    k200 = bt.load_kospi200_tickers()
    print(f"    {len(k200)}종목")

    print("[2] 시가총액 snapshot 로드...")
    snapshot = bt.load_market_cap_snapshot()
    print(f"    {len(snapshot)}종목")

    print("[3] 가격/거래대금 데이터 로드...")
    t0 = time.time()
    price_df = bt.load_price_data(k200["ticker"].tolist())
    print(f"    {len(price_df):,}행 ({time.time() - t0:.1f}초)")

    print("[4] daily_data 빌드...")
    daily_data = bt.build_daily_data(price_df, snapshot)
    print(f"    {len(daily_data)}종목")

    print("[5] 5일 1500억 필터 신호 생성...")
    signals = compute_5d_filter_signals(daily_data)
    n_signal_days = len(signals)
    n_total_signal = sum(len(v) for v in signals.values())
    n_unique_tickers = len(set(t for v in signals.values() for t in v))
    print(f"    {n_signal_days}일치, 총 신호 {n_total_signal}건, 고유 종목 {n_unique_tickers}개")

    # signals 자체를 simulate_strategy의 top3_per_day로 사용
    # 단, panel은 amount 기준으로 빌드 (CAGR 계산 dates 추출용)
    panel = bt.build_daily_indicator_panel(daily_data, "amount")

    results = []
    for ma_p in MA_PERIODS:
        for stop_pct in STOP_PCTS:
            t0 = time.time()
            trades, stats = bt.run_backtest(
                daily_data, panel, signals,
                rule="MA_INIT_STOP",
                ma_period=ma_p, stop_pct=stop_pct,
            )
            elapsed = time.time() - t0
            yr = bt.yearly_stats(trades)
            rule_label = f"MA{ma_p}_STOP{int(stop_pct*100)}"
            results.append({
                "ma_period": ma_p,
                "stop_pct": stop_pct,
                "rule_label": rule_label,
                "stats": stats,
                "yearly": yr,
                "trades": trades,
            })
            print(f"  {rule_label:14s}: {stats.get('total',0):>5}건, "
                  f"승률 {stats.get('win_rate',0):>5.1f}%, "
                  f"CAGR {stats.get('cagr',0):>+8.2f}%, "
                  f"MDD {stats.get('mdd',0):>+7.2f}%, "
                  f"자본 {stats.get('final_equity',1):>6.2f}x | {elapsed:.1f}s")

    # 추가: LIST_EXIT (조건 빠지면 매도) + HOLD_N도 같이 비교
    print("\n[참고] 추가 매도 규칙 비교 (5일 필터 매수 + 다른 매도)")
    extra_results = []
    for rule, hold_n in [("LIST_EXIT", None), ("HOLD_5", 5), ("HOLD_10", 10), ("HOLD_20", 20)]:
        t0 = time.time()
        if rule.startswith("HOLD_"):
            trades, stats = bt.run_backtest(daily_data, panel, signals, rule="HOLD_N", hold_n=hold_n)
            label = rule
        else:
            trades, stats = bt.run_backtest(daily_data, panel, signals, rule=rule)
            label = rule
        elapsed = time.time() - t0
        extra_results.append({
            "rule_label": label, "stats": stats, "trades": trades,
            "yearly": bt.yearly_stats(trades),
        })
        print(f"  {label:14s}: {stats.get('total',0):>5}건, "
              f"승률 {stats.get('win_rate',0):>5.1f}%, "
              f"CAGR {stats.get('cagr',0):>+8.2f}%, "
              f"MDD {stats.get('mdd',0):>+7.2f}%, "
              f"자본 {stats.get('final_equity',1):>6.2f}x | {elapsed:.1f}s")

    # 리포트
    print("\n[6] 리포트 생성...")
    lines = ["# 5일 1500억 필터 매수 + 다양한 매도 규칙 백테스트 (KOSPI200)\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **대상**: KOSPI200 199종목, 2023-01-01 ~ 현재")
    lines.append("- **매수 조건**: 최근 5일 (T-4~T) 거래대금 1,500억원 이상인 날이 5일 모두")
    lines.append("  - 조건 만족 종목 중 amount 큰 순으로 최대 3종목 → T+1 시가 매수")
    lines.append("- **매도 규칙 (주력 비교)**: MA_INIT_STOP")
    lines.append("  - 매수 후 종가가 MA(5/20) 위로 올라가면 crossed_ma=True")
    lines.append("  - crossed=False 상태에서 매수가 대비 stop_pct 이상 하락 → 즉시 손절")
    lines.append("  - crossed=True 상태에서 close < MA → MA 이탈 매도")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%")
    lines.append("- **자본 분배**: 3슬롯 균등")
    lines.append(f"- **신호 통계**: {n_signal_days}일에 신호, 고유 종목 {n_unique_tickers}개\n")
    lines.append("---\n")

    # MA_INIT_STOP 6 조합
    lines.append("## MA_INIT_STOP 6 조합 비교 (전체 기간)\n")
    lines.append("| MA | 손절선 | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 누적자본 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        lines.append(
            f"| {r['ma_period']} | {int(r['stop_pct']*100)}% | "
            f"{s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | {s['med_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | "
            f"{cagr:+.2f} | {mdd:+.2f} | {calmar:.2f} | {s.get('final_equity', 1):.2f} |"
        )

    # 추가 매도 규칙
    lines.append("\n## 추가 매도 규칙 비교 (참고)\n")
    lines.append("| 매도규칙 | 거래수 | 승률(%) | 평균(%) | 손익비 | CAGR(%) | MDD(%) | Calmar | 누적자본 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in extra_results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        lines.append(
            f"| {r['rule_label']} | "
            f"{s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | "
            f"{s['pf']:.2f} | {cagr:+.2f} | {mdd:+.2f} | {calmar:.2f} | {s.get('final_equity', 1):.2f} |"
        )

    # 직전 백테스트(amount Top3 + MA_INIT_STOP)와 비교 표
    lines.append("\n## 매수 조건 비교: amount Top3 vs 5일 1500억 필터\n")
    lines.append("| 매수 조건 | 매도 규칙 | CAGR(%) | MDD(%) | Calmar | 자본 | 거래수 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    # 기존 amount Top3 + MA_INIT_STOP 결과 (직전 보고서에서 인용)
    prior = [
        ("amount Top3", "MA20+손절-3%", 199.82, -65.32, 17.23, 351),
        ("amount Top3", "MA20+손절-5%", 176.75, -68.05, 14.00, 332),
        ("amount Top3", "MA20+손절-7%", 208.07, -67.07, 18.49, 318),
        ("amount Top3", "MA5+손절-7%", 65.88, -70.03, 3.71, 484),
    ]
    for buy_label, rule_label, cagr, mdd, fin, n in prior:
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        lines.append(f"| {buy_label} | {rule_label} | {cagr:+.2f} | {mdd:+.2f} | {calmar:.2f} | {fin:.2f}x | {n} |")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        rule = f"MA{r['ma_period']}+손절{int(r['stop_pct']*100)}%"
        lines.append(f"| **5일 1500억** | {rule} | {cagr:+.2f} | {mdd:+.2f} | {calmar:.2f} | {s.get('final_equity', 1):.2f}x | {s['total']} |")

    # 연도별 (MA_INIT_STOP만)
    all_years = set()
    for r in results:
        all_years.update(r.get("yearly", {}).keys())
    for y in sorted(all_years):
        lines.append(f"\n## {y}년 비교 (MA_INIT_STOP)\n")
        lines.append("| MA | 손절선 | 거래수 | 승률(%) | 평균(%) | 손익비 | 누적자본 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in results:
            ys = r.get("yearly", {}).get(y, {})
            if not ys or ys.get("total", 0) == 0:
                lines.append(f"| {r['ma_period']} | {int(r['stop_pct']*100)}% | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {r['ma_period']} | {int(r['stop_pct']*100)}% | "
                f"{ys['total']:,} | {ys['win_rate']:.1f} | {ys['avg_ret']:+.2f} | "
                f"{ys['pf']:.2f} | {ys['cum_return_x']:.2f} |"
            )

    # 최우수 상세
    valid = [r for r in results if r["stats"].get("total", 0) > 0]
    if valid:
        for r in valid:
            cagr = r["stats"].get("cagr", 0)
            mdd = r["stats"].get("mdd", 0)
            r["calmar"] = abs(cagr / mdd) if mdd != 0 else 0
        best_calmar = max(valid, key=lambda r: r.get("calmar", 0))
        bc = best_calmar
        lines.append(f"\n---\n\n## 최우수(Calmar) 상세: 5일 필터 + MA{bc['ma_period']} + 손절{int(bc['stop_pct']*100)}%\n")
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
        lines.append(f"| CAGR(%) | {s['cagr']:+.2f} |")
        lines.append(f"| MDD(%) | {s['mdd']:+.2f} |")
        lines.append(f"| Calmar | {bc['calmar']:.2f} |")
        lines.append(f"| 최종 자본(x) | {s['final_equity']:.2f} |")

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
