"""
시장 모드(상승/하락장) 기반 적응형 전략 백테스트.

전략:
- 매일 KOSPI close vs MA200 비교 → mode 결정
  - mode=UP (상승장): 공격형 N=5/K=5
  - mode=DOWN (하락장): 방어형 N=15/K=15

비교:
- 고정 공격: N=5, K=5 항상
- 고정 방어: N=15, K=15 항상
- 적응형: mode에 따라 변경
- KOSPI Buy&Hold (벤치마크)

기간: 2024-01-01 ~ 현재 (stock_all 거래대금 데이터 가용 기간)
"""
import os
import time

import numpy as np
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402
from backtest_5d_amount_filter import compute_5d_filter_signals  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_adaptive_market.md"

START_DATE = "2024-01-01"
# 백테스트 모듈의 시작일도 동일 적용 (2024-01-01부터 측정)
bt.START_DATE = START_DATE
UP_N, UP_K = 5, 5
DOWN_N, DOWN_K = 15, 15

FEE = bt.FEE_BUY + bt.FEE_SELL + bt.TAX_SELL


def load_kospi_mode(start_date: str = START_DATE) -> dict[pd.Timestamp, str]:
    """KOSPI MA200 기반 mode dict."""
    query = text("""
        SELECT date::date AS date, close
        FROM market_indices
        WHERE symbol='^KS11'
        ORDER BY date
    """)
    df = pd.read_sql(query, bt.ENGINE)
    df["date"] = pd.to_datetime(df["date"])
    df["ma200"] = df["close"].rolling(200, min_periods=200).mean()
    df["mode"] = np.where(df["close"] >= df["ma200"], "UP", "DOWN")
    df = df[df["date"] >= pd.Timestamp(start_date)].dropna(subset=["ma200"])
    return dict(zip(df["date"], df["mode"])), df


def simulate_adaptive(daily_data, signals_by_n, kospi_mode, dates,
                     up_n: int, up_k: int, down_n: int, down_k: int):
    """적응형 LIST_EXIT 시뮬레이션."""
    positions = {}
    trades = []
    ticker_idx = {t: df.set_index("date") for t, df in daily_data.items()}
    mode_log = []

    for i, d in enumerate(dates):
        today_mode = kospi_mode.get(d, "UP")
        today_n = up_n if today_mode == "UP" else down_n
        today_signals_set = set(signals_by_n.get(today_n, {}).get(d, []))

        sold_today = set()

        # 1. 매도 처리 (어제 발생한 신호)
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            if not pos["sell_signal"]:
                continue
            df = ticker_idx.get(ticker)
            if df is None or d not in df.index:
                continue
            row = df.loc[d]
            sell_price = row["open"]
            if pd.isna(sell_price) or sell_price <= 0:
                continue
            gross = (sell_price - pos["buy_price"]) / pos["buy_price"]
            net = gross - FEE
            trades.append({
                "ticker": ticker,
                "buy_date": pos["buy_date"],
                "buy_price": pos["buy_price"],
                "sell_date": d,
                "sell_price": sell_price,
                "hold_days": pos["hold_days"],
                "gross_ret": gross,
                "net_ret": net,
                "buy_mode": pos["buy_mode"],
                "buy_k": pos["buy_k"],
            })
            del positions[ticker]
            sold_today.add(ticker)

        # 2. 매수 처리: 어제 mode의 신호 + 어제 mode의 K
        if i > 0:
            prev_d = dates[i - 1]
            prev_mode = kospi_mode.get(prev_d, "UP")
            prev_n = up_n if prev_mode == "UP" else down_n
            prev_k = up_k if prev_mode == "UP" else down_k
            buy_signals = signals_by_n.get(prev_n, {}).get(prev_d, [])
            for ticker in buy_signals:
                if len(positions) >= prev_k:
                    break
                if ticker in positions or ticker in sold_today:
                    continue
                df = ticker_idx.get(ticker)
                if df is None or d not in df.index:
                    continue
                row = df.loc[d]
                buy_price = row["open"]
                if pd.isna(buy_price) or buy_price <= 0:
                    continue
                positions[ticker] = {
                    "buy_date": d,
                    "buy_price": float(buy_price),
                    "hold_days": 1,
                    "sell_signal": False,
                    "buy_mode": prev_mode,
                    "buy_k": prev_k,
                }

        # 3. 매도 신호 평가: 오늘 mode 기준
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            if pos["buy_date"] != d:
                pos["hold_days"] += 1
            if ticker not in today_signals_set:
                pos["sell_signal"] = True

        mode_log.append({"date": d, "mode": today_mode, "n_positions": len(positions)})

    return trades, mode_log


def equity_adaptive(trades: list[dict]) -> dict:
    """적응형 자본 시뮬: trade 별 buy_k에 따라 1/buy_k 분배."""
    if not trades:
        return {"final_equity": 1.0, "cagr": 0.0, "mdd": 0.0}
    df = pd.DataFrame(trades).copy()
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df["sell_date"] = pd.to_datetime(df["sell_date"])
    df = df.sort_values("buy_date").reset_index(drop=True)

    all_dates = sorted(set(list(df["buy_date"]) + list(df["sell_date"])))
    free = 1.0
    positions = {}
    equity = []
    eq_dates = []
    missed = 0

    for d in all_dates:
        sell_idx = df.index[df["sell_date"] == d].tolist()
        for idx in sell_idx:
            if idx in positions:
                cu = positions.pop(idx)
                free += cu * (1 + df.loc[idx, "net_ret"])

        buy_idx = df.index[df["buy_date"] == d].tolist()
        for idx in buy_idx:
            if idx in positions:
                continue
            K = int(df.loc[idx, "buy_k"])
            total = free + sum(positions.values())
            per = total / K
            if free >= per - 1e-9:
                positions[idx] = per
                free -= per
            else:
                missed += 1

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
        "missed": int(missed),
    }


def kospi_buy_hold(kospi_df: pd.DataFrame, start_date: str) -> dict:
    """KOSPI Buy & Hold 벤치마크."""
    df = kospi_df[kospi_df["date"] >= pd.Timestamp(start_date)].copy()
    if df.empty:
        return {}
    df = df.sort_values("date").reset_index(drop=True)
    initial = df["close"].iloc[0]
    df["equity"] = df["close"] / initial
    peak = df["equity"].cummax()
    dd = (df["equity"] - peak) / peak
    mdd = dd.min() * 100

    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    years = days / 365.25
    cagr = (df["equity"].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0

    return {
        "final_equity": float(df["equity"].iloc[-1]),
        "cagr": float(cagr * 100),
        "mdd": float(mdd),
    }


def main():
    print("=" * 60)
    print("적응형 시장 전략 백테스트 (상승장 공격 / 하락장 방어)")
    print("=" * 60)

    print("[1] KOSPI mode 데이터...")
    kospi_mode, kospi_df = load_kospi_mode(START_DATE)
    n_up = sum(1 for v in kospi_mode.values() if v == "UP")
    n_down = sum(1 for v in kospi_mode.values() if v == "DOWN")
    print(f"    {len(kospi_mode)}일치 mode 데이터")
    print(f"    UP {n_up}일 ({n_up/len(kospi_mode)*100:.1f}%), DOWN {n_down}일 ({n_down/len(kospi_mode)*100:.1f}%)")

    print("[2] 데이터 로드...")
    k200 = bt.load_kospi200_tickers()
    snapshot = bt.load_market_cap_snapshot()
    price_df = bt.load_price_data(k200["ticker"].tolist(), start_date=START_DATE)
    daily_data = bt.build_daily_data(price_df, snapshot)
    print(f"    {len(daily_data)}종목 시계열")

    panel = bt.build_daily_indicator_panel(daily_data, "amount")
    dates = sorted([d for d in panel.keys() if d >= pd.Timestamp(START_DATE)])
    print(f"    {len(dates)}일치 거래일")

    # 모든 N에 대한 신호 사전 계산
    print("\n[3] 신호 생성 (N=5, N=15)...")
    signals_by_n = {}
    for n in [UP_N, DOWN_N]:
        signals = compute_5d_filter_signals(daily_data, lookback=n, top_k=200)
        signals_by_n[n] = signals
        n_per_day = [len(v) for v in signals.values()]
        print(f"    N={n}: {len(signals)}일치, 평균 {np.mean(n_per_day):.2f}, max {max(n_per_day)}")

    # 백테스트 4종
    print("\n[4] 백테스트 실행")
    print("-" * 60)
    results = {}

    # (a) 고정 공격형
    t0 = time.time()
    trades_atk, _ = bt.run_backtest(
        daily_data, panel, signals_by_n[UP_N],
        rule="LIST_EXIT", slots=UP_K, max_concurrent=UP_K,
    )
    from backtest_5d_realistic_k import equity_real_k
    eq_atk = equity_real_k(trades_atk, K=UP_K)
    stats_atk = bt.compute_stats(trades_atk)
    stats_atk.update(eq_atk)
    results["고정 공격 (N=5,K=5)"] = (trades_atk, stats_atk)
    print(f"  고정 공격 (N=5, K=5):  {stats_atk['total']:>4}건, "
          f"CAGR {stats_atk['cagr']:>+7.2f}%, MDD {stats_atk['mdd']:>+6.2f}%, "
          f"자본 {stats_atk['final_equity']:>5.2f}x | {time.time()-t0:.1f}s")

    # (b) 고정 방어형
    t0 = time.time()
    trades_def, _ = bt.run_backtest(
        daily_data, panel, signals_by_n[DOWN_N],
        rule="LIST_EXIT", slots=DOWN_K, max_concurrent=DOWN_K,
    )
    eq_def = equity_real_k(trades_def, K=DOWN_K)
    stats_def = bt.compute_stats(trades_def)
    stats_def.update(eq_def)
    results["고정 방어 (N=15,K=15)"] = (trades_def, stats_def)
    print(f"  고정 방어 (N=15,K=15): {stats_def['total']:>4}건, "
          f"CAGR {stats_def['cagr']:>+7.2f}%, MDD {stats_def['mdd']:>+6.2f}%, "
          f"자본 {stats_def['final_equity']:>5.2f}x | {time.time()-t0:.1f}s")

    # (c) 적응형
    t0 = time.time()
    trades_adp, mode_log = simulate_adaptive(
        daily_data, signals_by_n, kospi_mode, dates,
        up_n=UP_N, up_k=UP_K, down_n=DOWN_N, down_k=DOWN_K,
    )
    eq_adp = equity_adaptive(trades_adp)
    stats_adp = bt.compute_stats(trades_adp)
    stats_adp.update(eq_adp)
    results["적응형"] = (trades_adp, stats_adp)
    print(f"  적응형:                 {stats_adp['total']:>4}건, "
          f"CAGR {stats_adp['cagr']:>+7.2f}%, MDD {stats_adp['mdd']:>+6.2f}%, "
          f"자본 {stats_adp['final_equity']:>5.2f}x | {time.time()-t0:.1f}s")

    # (d) KOSPI Buy & Hold
    bh = kospi_buy_hold(kospi_df, START_DATE)
    print(f"  KOSPI B&H:                       "
          f"CAGR {bh.get('cagr',0):>+7.2f}%, MDD {bh.get('mdd',0):>+6.2f}%, "
          f"자본 {bh.get('final_equity',1):>5.2f}x")

    # 적응형 mode별 통계
    if trades_adp:
        df_adp = pd.DataFrame(trades_adp)
        up_trades = df_adp[df_adp["buy_mode"] == "UP"]
        down_trades = df_adp[df_adp["buy_mode"] == "DOWN"]
        print(f"\n  [적응형 mode별 거래 통계]")
        print(f"    UP에서 매수:   {len(up_trades)}건, 평균 {up_trades['net_ret'].mean()*100:+.2f}%, "
              f"승률 {(up_trades['net_ret']>0).mean()*100:.1f}%")
        print(f"    DOWN에서 매수: {len(down_trades)}건, 평균 {down_trades['net_ret'].mean()*100:+.2f}%, "
              f"승률 {(down_trades['net_ret']>0).mean()*100:.1f}%")

    # 리포트
    print("\n[5] 리포트 생성...")
    lines = ["# 적응형 시장 전략 백테스트 (상승장 공격 / 하락장 방어)\n"]
    lines.append("## 매매 규칙\n")
    lines.append(f"- **대상**: KOSPI200 199종목, **{START_DATE} ~ 현재** (stock_all 거래대금 데이터 가용 기간)")
    lines.append("- **시장 모드 판정**: KOSPI close vs KOSPI MA200")
    lines.append("  - **UP (상승장)**: close ≥ MA200 → 공격형 (N=5, K=5)")
    lines.append("  - **DOWN (하락장)**: close < MA200 → 방어형 (N=15, K=15)")
    lines.append("- **매수**: 매수일의 mode에 따라 N/K 결정 (어제 mode + 어제 N의 신호 → 오늘 시가 매수, cap=K)")
    lines.append("- **매도 (LIST_EXIT)**: 오늘 mode의 N/N 조건 깨지면 → 다음날 시가 매도")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%\n")
    lines.append(f"### 기간 mode 분포\n")
    lines.append(f"- UP 일수: {n_up}일 ({n_up/len(kospi_mode)*100:.1f}%)")
    lines.append(f"- DOWN 일수: {n_down}일 ({n_down/len(kospi_mode)*100:.1f}%)\n")
    lines.append("⚠️ **데이터 한계**: stock_all 거래대금이 2023-09-25부터 가용. 2020~2022년 백테스트 불가능.\n")
    lines.append("---\n")

    # 비교 표
    lines.append("## 4가지 전략 비교\n")
    lines.append("| 전략 | 거래수 | 승률(%) | 평균(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | Calmar | 자본 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, (trades, stats) in results.items():
        s = stats
        cagr = s.get("cagr", 0)
        mdd = s.get("mdd", 0)
        cal = abs(cagr / mdd) if mdd != 0 else 0
        lines.append(
            f"| {name} | {s['total']:,} | {s['win_rate']:.1f} | {s['avg_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | {cagr:+.2f} | {mdd:+.2f} | {cal:.2f} | "
            f"{s.get('final_equity', 1):.2f}x |"
        )
    # KOSPI B&H
    bh_cagr = bh.get("cagr", 0)
    bh_mdd = bh.get("mdd", 0)
    bh_cal = abs(bh_cagr / bh_mdd) if bh_mdd != 0 else 0
    lines.append(
        f"| KOSPI Buy&Hold | - | - | - | - | - | "
        f"{bh_cagr:+.2f} | {bh_mdd:+.2f} | {bh_cal:.2f} | {bh.get('final_equity', 1):.2f}x |"
    )

    # 적응형 mode별 통계
    if trades_adp:
        lines.append("\n## 적응형 - mode별 거래 통계\n")
        lines.append("| 매수 시점 mode | 거래수 | 승률(%) | 평균수익(%) | 중간값(%) | 최대수익(%) | 최대손실(%) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for mode_name in ["UP", "DOWN"]:
            sub = df_adp[df_adp["buy_mode"] == mode_name]
            if len(sub) > 0:
                rets = sub["net_ret"] * 100
                lines.append(
                    f"| {mode_name} | {len(sub):,} | {(sub['net_ret']>0).mean()*100:.1f} | "
                    f"{rets.mean():+.2f} | {rets.median():+.2f} | "
                    f"{rets.max():+.2f} | {rets.min():+.2f} |"
                )

    # 연도별 비교
    lines.append("\n## 연도별 비교\n")
    lines.append("| 연도 | 고정 공격 거래수 | 공격 평균수익 | 고정 방어 거래수 | 방어 평균수익 | 적응형 거래수 | 적응형 평균수익 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    all_years = set()
    for name, (trades, _) in results.items():
        if trades:
            df_t = pd.DataFrame(trades)
            df_t["year"] = pd.to_datetime(df_t["buy_date"]).dt.year
            all_years.update(df_t["year"].unique())
    for y in sorted(all_years):
        row = [str(int(y))]
        for name in ["고정 공격 (N=5,K=5)", "고정 방어 (N=15,K=15)", "적응형"]:
            trades, _ = results[name]
            if trades:
                df_t = pd.DataFrame(trades)
                df_t["year"] = pd.to_datetime(df_t["buy_date"]).dt.year
                sub = df_t[df_t["year"] == y]
                if len(sub) > 0:
                    avg_ret = sub["net_ret"].mean() * 100
                    row.append(f"{len(sub):,}")
                    row.append(f"{avg_ret:+.2f}%")
                else:
                    row.append("0")
                    row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    # 모드 시계열 샘플 (월별 모드 분포)
    if mode_log:
        mode_df = pd.DataFrame(mode_log)
        mode_df["month"] = mode_df["date"].dt.to_period("M")
        monthly = mode_df.groupby("month").agg(
            up_days=("mode", lambda x: (x == "UP").sum()),
            down_days=("mode", lambda x: (x == "DOWN").sum()),
            avg_positions=("n_positions", "mean"),
        ).reset_index()
        lines.append("\n## 월별 모드 분포 + 적응형 평균 보유 종목 수\n")
        lines.append("| 월 | UP일 | DOWN일 | 평균 보유 종목 |")
        lines.append("|---|---:|---:|---:|")
        for _, r in monthly.iterrows():
            lines.append(f"| {r['month']} | {r['up_days']} | {r['down_days']} | {r['avg_positions']:.1f} |")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
