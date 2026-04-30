"""
Top 3 지표 + MA_INIT_STOP 매도 규칙 백테스트.

매도 규칙:
- 매수 후 종가가 MA(5 또는 20) 위로 한 번이라도 올라가면 crossed_ma=True
- crossed_ma=False 상태에서 매수가 대비 stop_pct 이상 하락 → 즉시 손절
- crossed_ma=True 상태에서 close < MA → MA 이탈 매도

파라미터: 지표(amount/mcap/turnover) × MA(5/20) × stop_pct(-3%/-5%/-7%) = 18 조합
대상: KOSPI200, 2023~현재
"""
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_top3_ma_init_stop.md"

INDICATORS = ["amount", "mcap", "turnover"]
MA_PERIODS = [5, 20]
STOP_PCTS = [-0.03, -0.05, -0.07]


def main():
    print("=" * 60)
    print("Top 3 지표 + MA_INIT_STOP 매도 규칙 백테스트")
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

    results = []
    for indicator in INDICATORS:
        print(f"\n[5] panel/top3: {indicator}")
        panel = bt.build_daily_indicator_panel(daily_data, indicator)
        top3 = bt.compute_top3_per_day(panel, indicator, n=3)

        for ma_p in MA_PERIODS:
            for stop_pct in STOP_PCTS:
                t0 = time.time()
                trades, stats = bt.run_backtest(
                    daily_data, panel, top3,
                    rule="MA_INIT_STOP",
                    ma_period=ma_p, stop_pct=stop_pct,
                )
                elapsed = time.time() - t0
                yr = bt.yearly_stats(trades)
                rule_label = f"MA{ma_p}_STOP{int(stop_pct*100)}"
                results.append({
                    "indicator": indicator,
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

    # 리포트
    print("\n[6] 리포트 생성...")
    lines = ["# Top 3 지표 + MA_INIT_STOP 매도 규칙 백테스트 (KOSPI200)\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **대상**: KOSPI200 199종목, 2023-01-01 ~ 현재")
    lines.append("- **매수**: 매일 마감 후 지표(amount/mcap/turnover) Top3 → 다음날 시가 매수")
    lines.append("- **매도 규칙 (MA_INIT_STOP)**:")
    lines.append("  1. 매수 후 종가가 MA(5/20) 위로 올라가면 `crossed_ma=True`")
    lines.append("  2. `crossed_ma=False` 상태에서 매수가 대비 `stop_pct` 이상 하락 → **즉시 손절** (다음날 시가)")
    lines.append("  3. `crossed_ma=True` 상태에서 close < MA → **MA 이탈 매도** (다음날 시가)")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%")
    lines.append("- **자본 분배**: 3슬롯 균등 (각 매수 자본의 1/3)")
    lines.append("- **파라미터**: MA(5/20) × stop_pct(-3%/-5%/-7%) × 지표 3종 = 18 조합\n")
    lines.append("---\n")
    lines.append("## 18 조합 비교 (전체 기간)\n")
    lines.append("| 지표 | MA | 손절선 | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 누적자본 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        lines.append(
            f"| {r['indicator']} | {r['ma_period']} | {int(r['stop_pct']*100)}% | "
            f"{s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | {s['med_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | "
            f"{cagr:+.2f} | {mdd:+.2f} | {calmar:.2f} | {s.get('final_equity', 1):.2f} |"
        )

    # 연도별
    all_years = set()
    for r in results:
        all_years.update(r.get("yearly", {}).keys())
    for y in sorted(all_years):
        lines.append(f"\n## {y}년 비교\n")
        lines.append("| 지표 | MA | 손절선 | 거래수 | 승률(%) | 평균(%) | 손익비 | 누적자본 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in results:
            ys = r.get("yearly", {}).get(y, {})
            if not ys or ys.get("total", 0) == 0:
                lines.append(f"| {r['indicator']} | {r['ma_period']} | {int(r['stop_pct']*100)}% | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {r['indicator']} | {r['ma_period']} | {int(r['stop_pct']*100)}% | "
                f"{ys['total']:,} | {ys['win_rate']:.1f} | {ys['avg_ret']:+.2f} | "
                f"{ys['pf']:.2f} | {ys['cum_return_x']:.2f} |"
            )

    # 최우수 (Calmar 기준 — 위험 조정)
    valid = [r for r in results if r["stats"].get("total", 0) > 0]
    if valid:
        for r in valid:
            cagr = r["stats"].get("cagr", 0)
            mdd = r["stats"].get("mdd", 0)
            r["calmar"] = abs(cagr / mdd) if mdd != 0 else 0
        best_calmar = max(valid, key=lambda r: r.get("calmar", 0))
        best_cagr = max(valid, key=lambda r: r["stats"].get("cagr", -999))
        best_mdd = max(valid, key=lambda r: r["stats"].get("mdd", -999))
        lines.append("\n## 최우수 조합 요약\n")
        lines.append(
            f"- **Calmar 최고 (위험조정)**: {best_calmar['indicator']} + MA{best_calmar['ma_period']} + 손절{int(best_calmar['stop_pct']*100)}% "
            f"→ Calmar {best_calmar['calmar']:.2f}, CAGR {best_calmar['stats']['cagr']:+.2f}%, MDD {best_calmar['stats']['mdd']:+.2f}%"
        )
        lines.append(
            f"- **CAGR 최고**: {best_cagr['indicator']} + MA{best_cagr['ma_period']} + 손절{int(best_cagr['stop_pct']*100)}% "
            f"→ CAGR {best_cagr['stats']['cagr']:+.2f}%, MDD {best_cagr['stats']['mdd']:+.2f}%, 자본 {best_cagr['stats']['final_equity']:.2f}x"
        )
        lines.append(
            f"- **MDD 최저 (안정)**: {best_mdd['indicator']} + MA{best_mdd['ma_period']} + 손절{int(best_mdd['stop_pct']*100)}% "
            f"→ MDD {best_mdd['stats']['mdd']:+.2f}%, CAGR {best_mdd['stats']['cagr']:+.2f}%"
        )

        # Calmar 최고 상세
        bc = best_calmar
        lines.append(f"\n---\n\n## 최우수(Calmar) 상세: {bc['indicator']} + MA{bc['ma_period']} + 손절{int(bc['stop_pct']*100)}%\n")
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
