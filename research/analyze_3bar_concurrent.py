"""
3-bar 매매규칙 (KOSPI200 ∪ 시총≥3조) 동시 보유 종목수 분석

매도 전략별로 백테스트를 돌린 뒤, 날짜별 동시 보유 종목수를 계산해
최대/평균/분위수를 리포트한다.
"""

import os
import time
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from backtest_3bar import (
    ENGINE, EXIT_STRATEGIES, calc_indicators, run_backtest, get_mcap_tickers,
)
from backtest_crash import load_all_data, get_kospi200_tickers

END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = "2023-01-01"


def compute_concurrent(trades_df: pd.DataFrame) -> pd.Series:
    """각 trade (buy_date~sell_date)에서 동시 보유 종목수를 날짜별로 계산."""
    if trades_df.empty:
        return pd.Series(dtype=int)
    events = []
    for _, r in trades_df.iterrows():
        events.append((pd.Timestamp(r["buy_date"]), +1))
        # sell_date 당일은 매도일이므로 그 날은 보유(+1 유효), 다음 날부터 -1
        events.append((pd.Timestamp(r["sell_date"]) + pd.Timedelta(days=1), -1))
    ev_df = pd.DataFrame(events, columns=["date", "delta"])
    ev_df = ev_df.groupby("date")["delta"].sum().sort_index()
    concurrent = ev_df.cumsum()
    # 거래일 기준으로 resample
    daily = concurrent.asfreq("D", method="ffill").fillna(0)
    # 주말 제외 (영업일 기준)
    daily = daily[daily.index.weekday < 5]
    return daily.astype(int)


def main():
    t0 = time.time()
    # 유니버스 구성
    print("[1/3] 유니버스 구성 (KOSPI200 ∪ 시총≥3조)...")
    kospi200 = get_kospi200_tickers()
    with ENGINE.connect() as conn:
        db_rows = conn.execute(text("SELECT DISTINCT ticker FROM stocks")).fetchall()
    db_tickers = set(r[0] for r in db_rows)
    k200 = {t["ticker"]: t["name"] for t in kospi200 if t["ticker"] in db_tickers}
    mcap_map = get_mcap_tickers(3e12)
    with ENGINE.connect() as conn:
        name_rows = conn.execute(text("SELECT DISTINCT ticker, name FROM stocks")).fetchall()
    all_names = {r[0]: r[1] for r in name_rows}
    merged = sorted(set(k200) | set(mcap_map))
    name_map = {t: k200.get(t) or all_names.get(t, t) for t in merged}
    print(f"     {len(merged)}종목")

    # 데이터 로딩
    print(f"[2/3] 데이터 로딩...")
    all_data = {}
    for i in range(0, len(merged), 500):
        batch = merged[i:i+500]
        all_data.update(load_all_data(batch, START_DATE, END_DATE))
    ind_data = {}
    for t, df in all_data.items():
        df = df[~df.index.duplicated(keep="last")]
        ind_data[t] = calc_indicators(df)

    # 매도 전략별 백테스트 & 동시보유 계산
    print(f"[3/3] {len(EXIT_STRATEGIES)}개 매도전략 백테스트...")
    results = {}
    for ex in EXIT_STRATEGIES:
        print(f"  ▸ {ex}...", end=" ", flush=True)
        all_trades = []
        for ticker in merged:
            if ticker not in ind_data:
                continue
            df = ind_data[ticker]
            if len(df) < 20:
                continue
            nm = name_map.get(ticker, ticker)
            _, trades, _ = run_backtest(df, ticker, nm, ex, START_DATE)
            if not trades.empty:
                all_trades.append(trades)
        tdf = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        concurrent = compute_concurrent(tdf)
        results[ex] = {"trades_df": tdf, "concurrent": concurrent}
        print(f"거래 {len(tdf)}건, 최대 동시보유 {int(concurrent.max()) if len(concurrent) else 0}종목")

    # 리포트
    lines = []
    lines.append("# 3-bar 매매규칙 동시 보유 종목수 분석\n")
    lines.append(f"- **유니버스**: KOSPI200 ∪ 시총≥3조 ({len(merged)}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **가정**: 각 종목별 독립 백테스트 (전략이 각 종목에 동시 적용될 때의 보유수)\n")

    lines.append("## 매도전략별 동시 보유 종목수\n")
    lines.append("| 매도전략 | 거래수 | 최대 | 평균 | 중위 | 90% | 95% | 99% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ex in EXIT_STRATEGIES:
        c = results[ex]["concurrent"]
        n = len(results[ex]["trades_df"])
        if len(c) == 0:
            lines.append(f"| {ex} | 0 | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {ex} | {n}건 | **{int(c.max())}** | {c.mean():.1f} | {int(c.median())} "
            f"| {int(c.quantile(0.90))} | {int(c.quantile(0.95))} | {int(c.quantile(0.99))} |"
        )
    lines.append("")

    # 최대 동시보유 발생일
    lines.append("## 최대 동시보유 발생일 (매도전략별)\n")
    lines.append("| 매도전략 | 최대종목수 | 발생일 |")
    lines.append("|---|---|---|")
    for ex in EXIT_STRATEGIES:
        c = results[ex]["concurrent"]
        if len(c) == 0:
            continue
        peak = int(c.max())
        peak_dates = c[c == peak].index[:3]
        dstr = ", ".join(d.strftime("%Y-%m-%d") for d in peak_dates)
        lines.append(f"| {ex} | {peak}종목 | {dstr} |")
    lines.append("")

    # 분포 요약 (MA5이탈 기준)
    lines.append("## MA5이탈 매도 · 연도별 최대/평균 동시보유\n")
    c = results["MA5이탈"]["concurrent"]
    if len(c) > 0:
        lines.append("| 연도 | 최대 | 평균 | 중위 |")
        lines.append("|---|---|---|---|")
        for yr in [2023, 2024, 2025, 2026]:
            cy = c[c.index.year == yr]
            if len(cy) == 0:
                continue
            lines.append(f"| {yr} | {int(cy.max())} | {cy.mean():.1f} | {int(cy.median())} |")
    lines.append("")

    lines.append(f"## 실행 정보\n- **실행 시간**: {time.time()-t0:.1f}초")
    lines.append(f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    out = os.path.join(os.path.dirname(__file__), "results",
                       "analyze_3bar_concurrent.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n리포트: {out}")


if __name__ == "__main__":
    main()
