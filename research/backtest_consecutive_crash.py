"""
2봉 연속 시가대비 종가 -6% 이상 급락 백테스트
- 조건: 당일 시가 대비 종가가 -6% 이상 하락한 봉이 2일 연속 발생
- 매수: 2번째 급락봉 종가에 매수
- 매도: 다음 거래일 종가에 매도
- 대상: 전 종목
"""

import argparse
import os
import time
from datetime import datetime

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
    generate_summary_charts,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 파라미터
CRASH_PCT = -6.0  # 시가 대비 종가 하락률 기준 (%)
CONSEC_DAYS = 2   # 연속 급락일 수

# 거래대금 필터 파라미터 (amount 단위: 백만원)
AMT_AVG_TH = 100_000    # 20일 평균거래대금 1000억 = 100,000백만원
AMT_HIGH_TH = 150_000   # 1500억 = 150,000백만원
AMT_HIGH_DAYS = 10       # 20일 중 1500억 이상인 날 최소 일수


def load_amount_data(tickers: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """stock_all 테이블에서 거래대금(amount) 데이터 로딩"""
    placeholders = ",".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT date, ticker, amount
        FROM stock_all
        WHERE ticker IN ({placeholders})
          AND date >= '{start_date}'::date - interval '60 days'
          AND date <= '{end_date}'
        ORDER BY ticker, date ASC
    """
    df_all = pd.read_sql(query, ENGINE)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["amount"] = pd.to_numeric(df_all["amount"], errors="coerce")

    result = {}
    for ticker, group in df_all.groupby("ticker"):
        g = group.set_index("date").sort_index()
        result[ticker] = g
    return result


def find_signals(df: pd.DataFrame, amount_df: pd.DataFrame = None) -> list[int]:
    """2봉 연속 시가대비 종가 -6% 이상 급락한 날의 인덱스(정수) 반환
    amount_df가 주어지면 거래대금 필터도 적용"""
    df = df.copy()
    df["intraday_ret"] = (df["close"] - df["open"]) / df["open"] * 100
    df["is_crash"] = df["intraday_ret"] <= CRASH_PCT

    signals = []
    for i in range(CONSEC_DAYS - 1, len(df)):
        # 연속 CONSEC_DAYS일 모두 급락인지 확인
        if all(df["is_crash"].iloc[i - j] for j in range(CONSEC_DAYS)):
            # 다음날이 있어야 매도 가능
            if i + 1 < len(df):
                # 거래대금 필터
                if amount_df is not None:
                    current_date = df.index[i]
                    if not _check_amount_filter(amount_df, current_date):
                        continue
                signals.append(i)
    return signals


def _check_amount_filter(amount_df: pd.DataFrame, current_date) -> bool:
    """당일 제외 과거 20일 거래대금 필터 확인"""
    past = amount_df[amount_df.index < current_date].tail(20)
    if len(past) < 20:
        return False
    avg_amt = past["amount"].mean()
    high_days = (past["amount"] >= AMT_HIGH_TH).sum()
    return avg_amt >= AMT_AVG_TH and high_days >= AMT_HIGH_DAYS


def run_backtest(df: pd.DataFrame, ticker: str, name: str, amount_df: pd.DataFrame = None):
    signals = find_signals(df, amount_df)
    trades = []
    dates = df.index.tolist()

    for sig_idx in signals:
        buy_date = dates[sig_idx]
        sell_date = dates[sig_idx + 1]
        buy_price = df.iloc[sig_idx]["close"]
        sell_price = df.iloc[sig_idx + 1]["close"]

        # 비용 계산 (종목당 초기자본 기준)
        max_qty = int(INITIAL_CAPITAL / (buy_price * (1 + FEE_BUY)))
        if max_qty <= 0:
            continue

        buy_cost = max_qty * buy_price
        buy_fee = buy_cost * FEE_BUY
        sell_revenue = max_qty * sell_price
        sell_fee = sell_revenue * FEE_SELL
        sell_tax = sell_revenue * TAX_SELL
        net_sell = sell_revenue - sell_fee - sell_tax
        pnl = net_sell - buy_cost - buy_fee
        ret_pct = pnl / (buy_cost + buy_fee) * 100

        # 급락 상세 정보
        day1_ret = (df.iloc[sig_idx - 1]["close"] - df.iloc[sig_idx - 1]["open"]) / df.iloc[sig_idx - 1]["open"] * 100
        day2_ret = (df.iloc[sig_idx]["close"] - df.iloc[sig_idx]["open"]) / df.iloc[sig_idx]["open"] * 100

        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": buy_date, "buy_price": buy_price,
            "sell_date": sell_date, "sell_price": sell_price,
            "quantity": max_qty, "pnl": pnl,
            "return_pct": ret_pct, "reason": "1일보유",
            "day1_crash": day1_ret, "day2_crash": day2_ret,
        })

    trades_df = pd.DataFrame(trades)

    # 요약
    summary = {
        "ticker": ticker, "name": name,
        "n_trades": len(trades_df),
    }

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        summary["win_rate"] = len(wins) / len(trades_df) * 100
        summary["total_pnl"] = trades_df["pnl"].sum()
        summary["avg_return"] = trades_df["return_pct"].mean()
        summary["total_return"] = trades_df["pnl"].sum() / INITIAL_CAPITAL * 100
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        summary["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        summary["win_rate"] = 0
        summary["total_pnl"] = 0
        summary["avg_return"] = 0
        summary["total_return"] = 0
        summary["profit_factor"] = 0

    # 바이앤홀드 (기간 내)
    df_period = df.loc[START_DATE:]
    if len(df_period) > 1:
        summary["bnh_return"] = (df_period["close"].iloc[-1] / df_period["close"].iloc[0] - 1) * 100
    else:
        summary["bnh_return"] = 0

    summary["mdd"] = 0  # 1일 보유라 MDD 의미 없음

    return summary, trades_df


def generate_report(summaries_df, all_trades_df, elapsed, label, mode="all"):
    traded = summaries_df[summaries_df["n_trades"] > 0]

    lines = []
    lines.append(f"# 2봉 연속 급락 매수 백테스트 리포트 ({label})\n")

    lines.append("## 전략 개요\n")
    lines.append(f"- **대상**: {label} ({len(summaries_df)}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append("")

    lines.append("### 매매 규칙\n")
    lines.append("| 구분 | 조건 |")
    lines.append("|------|------|")
    lines.append(f"| 급락 기준 | 당일 시가 대비 종가 {CRASH_PCT}% 이하 |")
    lines.append(f"| 연속 조건 | {CONSEC_DAYS}봉 연속 급락 |")
    lines.append("| **매수** | **2번째 급락봉 종가에 매수** |")
    lines.append("| **매도** | **다음 거래일 종가에 매도 (1일 보유)** |")
    lines.append("")

    lines.append("## 통합 성과 요약\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 분석 종목 수 | {len(summaries_df)} |")
    lines.append(f"| 거래 발생 종목 | {len(traded)} ({len(traded)/len(summaries_df)*100:.1f}%) |")
    lines.append(f"| 총 거래 건수 | {len(all_trades_df) if not all_trades_df.empty else 0} |")

    if not all_trades_df.empty:
        all_wins = all_trades_df[all_trades_df["pnl"] > 0]
        all_losses = all_trades_df[all_trades_df["pnl"] <= 0]
        total_win_rate = len(all_wins) / len(all_trades_df) * 100
        gp = all_wins["pnl"].sum() if len(all_wins) > 0 else 0
        gl = abs(all_losses["pnl"].sum()) if len(all_losses) > 0 else 0
        pf = gp / gl if gl > 0 else float("inf")
        total_pnl = all_trades_df["pnl"].sum()
        avg_trade_ret = all_trades_df["return_pct"].mean()
        med_trade_ret = all_trades_df["return_pct"].median()

        lines.append(f"| **전체 승률** | **{total_win_rate:.1f}% ({len(all_wins)}승 / {len(all_losses)}패)** |")
        lines.append(f"| **전체 손익비** | **{pf:.2f}** |")
        lines.append(f"| **전체 총 손익** | **{total_pnl:,.0f}원** |")
        lines.append(f"| 거래당 평균 수익률 | {avg_trade_ret:.2f}% |")
        lines.append(f"| 거래당 중위 수익률 | {med_trade_ret:.2f}% |")
        lines.append(f"| 평균 수익 (승) | {all_wins['pnl'].mean():,.0f}원 |" if len(all_wins) > 0 else "| 평균 수익 (승) | N/A |")
        lines.append(f"| 평균 손실 (패) | {all_losses['pnl'].mean():,.0f}원 |" if len(all_losses) > 0 else "| 평균 손실 (패) | N/A |")

        # 급락 크기별 분석
        lines.append("")
        lines.append("## 급락 크기별 분석\n")
        lines.append("| 2일 합산 급락률 구간 | 건수 | 승률 | 평균 수익률 | 총 손익 |")
        lines.append("|---------------------|------|------|------------|---------|")
        tc = all_trades_df.copy()
        tc["total_crash"] = tc["day1_crash"] + tc["day2_crash"]
        bins = [(-999, -20), (-20, -16), (-16, -12), (-12, -8)]
        bin_labels = ["-20% 이하", "-20%~-16%", "-16%~-12%", "-12%~-8%"]
        for (lo, hi), bl in zip(bins, bin_labels):
            subset = tc[(tc["total_crash"] > lo) & (tc["total_crash"] <= hi)]
            if len(subset) > 0:
                wr = (subset["pnl"] > 0).sum() / len(subset) * 100
                ar = subset["return_pct"].mean()
                tp = subset["pnl"].sum()
                lines.append(f"| {bl} | {len(subset)} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 |")
            else:
                lines.append(f"| {bl} | 0 | - | - | - |")

        # 월별 성과
        lines.append("")
        lines.append("## 월별 성과\n")
        lines.append("| 월 | 거래수 | 승률 | 평균수익률 | 총손익 |")
        lines.append("|----|--------|------|----------|--------|")
        tc["month"] = pd.to_datetime(tc["buy_date"]).dt.to_period("M")
        for month, grp in tc.groupby("month"):
            n = len(grp)
            wr = (grp["pnl"] > 0).sum() / n * 100
            ar = grp["return_pct"].mean()
            tp = grp["pnl"].sum()
            lines.append(f"| {month} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 |")
        lines.append("")

    # 차트
    lines.append("## 차트\n")
    lines.append("### 거래별 수익률 분포")
    chart_prefix = f"charts_consec_crash_{mode}"
    lines.append(f"![수익률분포]({chart_prefix}/return_distribution.png)\n")
    lines.append("### 종목별 수익률 분포")
    lines.append(f"![종목별]({chart_prefix}/stock_return_distribution.png)\n")
    lines.append("### 급락 크기 vs 반등 수익률")
    lines.append(f"![급락vs반등]({chart_prefix}/crash_vs_bounce.png)\n")
    lines.append("### 월별 손익")
    lines.append(f"![월별손익]({chart_prefix}/monthly_pnl.png)\n")

    # TOP/BOTTOM 종목 (거래 많은 순)
    if not traded.empty:
        lines.append("## 거래 빈도 TOP 20 종목\n")
        lines.append("| # | 종목 | 코드 | 거래수 | 승률 | 평균수익률 | 총손익 |")
        lines.append("|---|------|------|--------|------|----------|--------|")
        for i, (_, r) in enumerate(traded.nlargest(20, "n_trades").iterrows()):
            lines.append(
                f"| {i+1} | {r['name']} | {r['ticker']} | {r['n_trades']} "
                f"| {r['win_rate']:.0f}% | {r['avg_return']:.2f}% | {r['total_pnl']:,.0f}원 |"
            )
        lines.append("")

    # 전체 매매 기록
    if not all_trades_df.empty:
        lines.append("## 전체 매매 기록\n")
        lines.append("| # | 종목 | 매수일 | 매수가 | 매도일 | 매도가 | 수익률 | 손익 | D1급락 | D2급락 |")
        lines.append("|---|------|--------|--------|--------|--------|--------|------|--------|--------|")
        sorted_trades = all_trades_df.sort_values("buy_date")
        for i, (_, t) in enumerate(sorted_trades.iterrows()):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            lines.append(
                f"| {i+1} | {t['name']} | {bd} | {t['buy_price']:,.0f} "
                f"| {sd} | {t['sell_price']:,.0f} | {t['return_pct']:.2f}% | {t['pnl']:,.0f} "
                f"| {t['day1_crash']:.1f}% | {t['day2_crash']:.1f}% |"
            )
        lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- **실행 시간**: {elapsed:.2f}초")
    lines.append(f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **분석 종목 수**: {len(summaries_df)}")
    lines.append("")

    return "\n".join(lines)


def generate_charts(all_trades_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if all_trades_df.empty:
        return

    # 차트1: 거래별 수익률 분포
    fig, ax = plt.subplots(figsize=(14, 6))
    bins = np.arange(
        max(all_trades_df["return_pct"].min() - 1, -30),
        min(all_trades_df["return_pct"].max() + 1, 30), 0.5
    )
    n, bins_out, patches = ax.hist(all_trades_df["return_pct"], bins=bins, edgecolor="black", alpha=0.7)
    for patch, b in zip(patches, bins_out):
        if b + (bins_out[1] - bins_out[0]) / 2 >= 0:
            patch.set_facecolor("green")
        else:
            patch.set_facecolor("red")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    avg_ret = all_trades_df["return_pct"].mean()
    ax.axvline(avg_ret, color="blue", linewidth=1.5, linestyle="-", label=f"평균: {avg_ret:.2f}%")
    ax.set_title(f"거래별 수익률 분포 (총 {len(all_trades_df)}건)", fontsize=14)
    ax.set_xlabel("수익률 (%)")
    ax.set_ylabel("거래 건수")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "return_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 차트2: 종목별 평균 수익률 분포
    fig, ax = plt.subplots(figsize=(14, 6))
    stock_avg = all_trades_df.groupby("name")["return_pct"].mean()
    bins2 = np.arange(stock_avg.min() - 1, stock_avg.max() + 1, 1)
    n2, bins2_out, patches2 = ax.hist(stock_avg, bins=bins2, edgecolor="black", alpha=0.7)
    for patch, b in zip(patches2, bins2_out):
        if b + (bins2_out[1] - bins2_out[0]) / 2 >= 0:
            patch.set_facecolor("green")
        else:
            patch.set_facecolor("red")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"종목별 평균 수익률 분포 ({len(stock_avg)}종목)", fontsize=14)
    ax.set_xlabel("평균 수익률 (%)")
    ax.set_ylabel("종목 수")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "stock_return_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 차트3: 급락 크기 vs 반등 수익률
    fig, ax = plt.subplots(figsize=(10, 8))
    tc = all_trades_df.copy()
    tc["total_crash"] = tc["day1_crash"] + tc["day2_crash"]
    colors = ["green" if r > 0 else "red" for r in tc["return_pct"]]
    ax.scatter(tc["total_crash"], tc["return_pct"], alpha=0.4, s=20, c=colors)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("2일 합산 급락률 (%)")
    ax.set_ylabel("반등 수익률 (%)")
    ax.set_title("급락 크기 vs 다음날 반등 수익률", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "crash_vs_bounce.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 차트4: 월별 손익
    fig, ax = plt.subplots(figsize=(14, 6))
    tc["month"] = pd.to_datetime(tc["buy_date"]).dt.to_period("M")
    monthly = tc.groupby("month").agg(
        total_pnl=("pnl", "sum"),
        n_trades=("pnl", "count"),
    )
    colors_bar = ["green" if p >= 0 else "red" for p in monthly["total_pnl"]]
    x = range(len(monthly))
    ax.bar(x, monthly["total_pnl"], color=colors_bar, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha="right")
    ax.set_title("월별 총 손익", fontsize=14)
    ax.set_ylabel("손익 (원)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    # 거래 건수 표시
    for i, (_, row) in enumerate(monthly.iterrows()):
        ax.text(i, row["total_pnl"], f"{row['n_trades']}건", ha="center",
                va="bottom" if row["total_pnl"] >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "monthly_pnl.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_main(mode: str = "all"):
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 1) 종목 선택
    if mode == "kospi200":
        print("[1/4] KOSPI200 종목 조회...")
        kospi200 = get_kospi200_tickers()
        with ENGINE.connect() as conn:
            db_rows = conn.execute(
                text("SELECT DISTINCT ticker FROM stocks")
            ).fetchall()
        db_tickers = set(r[0] for r in db_rows)
        ticker_list = [t["ticker"] for t in kospi200 if t["ticker"] in db_tickers]
        name_map = {t["ticker"]: t["name"] for t in kospi200 if t["ticker"] in db_tickers}
        label = "KOSPI200"
    elif mode == "bigvolume":
        print("[1/4] 전종목 조회 (거래대금 필터 적용)...")
        with ENGINE.connect() as conn:
            db_rows = conn.execute(
                text("SELECT DISTINCT ticker, name FROM stocks ORDER BY ticker")
            ).fetchall()
        ticker_list = [r[0] for r in db_rows]
        name_map = {r[0]: r[1] for r in db_rows}
        label = f"거래대금필터(평균{AMT_AVG_TH//100:.0f}억+{AMT_HIGH_TH//100:.0f}억×{AMT_HIGH_DAYS}일)"
    else:
        print("[1/4] 전종목 조회...")
        with ENGINE.connect() as conn:
            db_rows = conn.execute(
                text("SELECT DISTINCT ticker, name FROM stocks ORDER BY ticker")
            ).fetchall()
        ticker_list = [r[0] for r in db_rows]
        name_map = {r[0]: r[1] for r in db_rows}
        label = "전종목"
    print(f"      [{label}] 대상: {len(ticker_list)}종목")

    # 2) 데이터 로딩
    print("[2/4] 데이터 로딩...")
    all_data = {}
    batch_size = 500
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i:i + batch_size]
        batch_data = load_all_data(batch, START_DATE, END_DATE)
        all_data.update(batch_data)
        if len(ticker_list) > batch_size:
            print(f"      {min(i + batch_size, len(ticker_list))}/{len(ticker_list)} 로딩...")
    print(f"      {len(all_data)}종목 로딩 완료")

    # 2-1) 거래대금 데이터 로딩 (bigvolume 모드)
    all_amount = {}
    if mode == "bigvolume":
        print("      거래대금 데이터 로딩...")
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i + batch_size]
            batch_amt = load_amount_data(batch, START_DATE, END_DATE)
            all_amount.update(batch_amt)
            if len(ticker_list) > batch_size:
                print(f"      거래대금 {min(i + batch_size, len(ticker_list))}/{len(ticker_list)} 로딩...")
        print(f"      거래대금 {len(all_amount)}종목 로딩 완료")

    # 3) 백테스트 실행
    print("[3/4] 종목별 백테스트 실행...")
    summaries = []
    all_trades = []
    for idx, ticker in enumerate(ticker_list):
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        df_test = df.loc[START_DATE:]
        if len(df_test) < 5:
            continue

        name = name_map.get(ticker, ticker)
        amount_df = all_amount.get(ticker) if mode == "bigvolume" else None
        summary, trades_df = run_backtest(df_test, ticker, name, amount_df)
        summaries.append(summary)
        if not trades_df.empty:
            all_trades.append(trades_df)

        if (idx + 1) % 500 == 0:
            print(f"      {idx+1}/{len(ticker_list)} 완료...")

    summaries_df = pd.DataFrame(summaries)
    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    traded_count = (summaries_df["n_trades"] > 0).sum()
    print(f"      완료! 거래 발생: {traded_count}종목, 총 {len(all_trades_df)}건")

    # 4) 차트 & 리포트
    chart_dir = os.path.join(base_dir, f"charts_consec_crash_{mode}")
    print("[4/4] 차트 & 리포트 생성...")
    generate_charts(all_trades_df, chart_dir)

    elapsed = time.time() - start_time
    report = generate_report(summaries_df, all_trades_df, elapsed, label, mode)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, f"backtest_consecutive_crash_{mode}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*50}")
    print(f"[{label}] 2봉 연속 급락 매수 백테스트 완료! ({elapsed:.2f}초)")
    print(f"리포트: {report_path}")
    print(f"분석 종목: {len(summaries_df)}, 거래 발생: {traded_count}, 총 거래: {len(all_trades_df)}")
    if not all_trades_df.empty:
        w = (all_trades_df["pnl"] > 0).sum()
        avg = all_trades_df["return_pct"].mean()
        print(f"  승률: {w}/{len(all_trades_df)} = {w/len(all_trades_df)*100:.1f}%")
        print(f"  거래당 평균 수익률: {avg:.2f}%")
        print(f"  총 손익: {all_trades_df['pnl'].sum():,.0f}원")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2봉 연속 급락 매수 백테스트")
    parser.add_argument(
        "mode", nargs="?", default="all",
        choices=["kospi200", "bigvolume", "all"],
        help="kospi200: KOSPI200 / bigvolume: 거래대금필터 / all: 전종목",
    )
    args = parser.parse_args()
    run_main(args.mode)
