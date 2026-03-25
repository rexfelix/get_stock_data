"""
RS >= 90 && KOSPI200 조건부 보유 전략 백테스트
- 매주 리밸런싱: RS Rating 재계산 → RS>=90 && KOSPI200 종목만 보유
- 조건 미충족 시 매도, 신규 충족 시 매수
- 동일가중 포트폴리오
- 기간: 2025-01-01 ~ 현재
"""

import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = "postgresql://{}:{}@{}:{}/{}".format(
    os.getenv("DB_USER", "rexfelix"),
    os.getenv("DB_PASSWORD", ""),
    os.getenv("DB_HOST", "localhost"),
    os.getenv("DB_PORT", "5432"),
    os.getenv("DB_NAME", "stock_db"),
)
ENGINE = create_engine(DB_URL)

FEE_BUY = 0.00015
FEE_SELL = 0.00015
TAX_SELL = 0.0023

START_DATE = "2025-01-02"
END_DATE = datetime.today().strftime("%Y-%m-%d")

QUARTER_DAYS = 63  # 1분기 영업일

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def get_kospi200_tickers() -> set[str]:
    """네이버 금융에서 KOSPI200 종목 코드 set"""
    tickers = set()
    for page in range(1, 22):
        url = f"https://finance.naver.com/sise/entryJongmok.naver?&page={page}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.select('td a[href*="main.naver?code="]')
        for link in links:
            code = re.search(r"code=(\d+)", link["href"])
            if code:
                tickers.add(code.group(1))
    return tickers


def load_stock_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """stocks 테이블에서 전종목 OHLCV 로드"""
    # RS 계산에 1년 이전 데이터 필요
    query = text("""
        SELECT ticker, name, date, close
        FROM stocks
        WHERE date >= CAST(:start AS date) - interval '400 days'
          AND date <= CAST(:end AS date)
        ORDER BY date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn, params={"start": start_date, "end": end_date})
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_index_prices(symbol: str, start_date: str, end_date: str) -> pd.Series:
    """지수 종가 시리즈"""
    query = text("""
        SELECT date, close FROM market_indices
        WHERE symbol = :sym
          AND date >= CAST(:start AS date) - interval '400 days'
          AND date <= CAST(:end AS date)
        ORDER BY date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn, params={"sym": symbol, "start": start_date, "end": end_date})
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date", keep="last")
    return df.set_index("date")["close"]


def calc_rs_at_date(pivot: pd.DataFrame, index_prices: pd.Series,
                    calc_date: pd.Timestamp) -> pd.Series:
    """특정 날짜 기준 전종목 RS Rating 계산 (1~99)"""
    # calc_date 이하의 영업일만 사용
    valid_dates = pivot.index[pivot.index <= calc_date]
    if len(valid_dates) < QUARTER_DAYS * 4 + 1:
        return pd.Series(dtype=float)

    # 분기 경계일
    boundaries = []
    for i in range(5):
        idx = len(valid_dates) - 1 - (i * QUARTER_DAYS)
        if idx < 0:
            return pd.Series(dtype=float)
        boundaries.append(valid_dates[idx])

    # 종목 분기별 수익률
    bp = []
    for bd in boundaries:
        bp.append(pivot.loc[bd])

    quarter_returns = []
    for i in range(4):
        ret = (bp[i] - bp[i + 1]) / bp[i + 1].replace(0, np.nan)
        quarter_returns.append(ret)

    # 지수 분기별 수익률
    idx_prices = index_prices[~index_prices.index.duplicated(keep="last")]
    idx_bp = []
    for bd in boundaries:
        valid_idx = idx_prices.index[idx_prices.index <= bd]
        if len(valid_idx) == 0:
            return pd.Series(dtype=float)
        idx_bp.append(idx_prices.loc[valid_idx[-1]])

    idx_returns = []
    for i in range(4):
        if idx_bp[i + 1] == 0:
            idx_returns.append(0.0)
        else:
            idx_returns.append((idx_bp[i] - idx_bp[i + 1]) / idx_bp[i + 1])

    # RS raw
    rs_raw = (
        2 * (quarter_returns[0] - idx_returns[0])
        + (quarter_returns[1] - idx_returns[1])
        + (quarter_returns[2] - idx_returns[2])
        + (quarter_returns[3] - idx_returns[3])
    )
    rs_raw = rs_raw.dropna()

    if len(rs_raw) == 0:
        return pd.Series(dtype=float)

    # 퍼센타일 (1~99)
    arr = rs_raw.values
    n = len(arr)
    sorted_idx = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[sorted_idx] = np.arange(n)

    unique_vals, inverse = np.unique(arr, return_inverse=True)
    for vi in range(len(unique_vals)):
        mask = inverse == vi
        ranks[mask] = ranks[mask].mean()

    percentiles = ranks / (n - 1) if n > 1 else np.full(n, 0.5)
    rs_rating = np.clip(np.round(percentiles * 98 + 1), 1, 99).astype(int)

    return pd.Series(rs_rating, index=rs_raw.index)


def run_backtest():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 1) KOSPI200 종목 조회
    print("[1/4] KOSPI200 종목 조회...")
    kospi200 = get_kospi200_tickers()
    print(f"      {len(kospi200)}종목")

    # 2) 데이터 로딩
    print("[2/4] 주가 데이터 로딩...")
    stock_df = load_stock_prices(START_DATE, END_DATE)
    name_map = stock_df.groupby("ticker")["name"].first().to_dict()

    # 중복 제거 후 피벗
    stock_df = stock_df.drop_duplicates(subset=["date", "ticker"], keep="last")
    pivot = stock_df.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index().ffill()

    index_prices = load_index_prices("^KS11", START_DATE, END_DATE)
    print(f"      종목: {pivot.shape[1]}개, 기간: {pivot.index[0].date()} ~ {pivot.index[-1].date()}")

    # 3) 리밸런싱 날짜 (매주 월요일에 가장 가까운 영업일)
    all_dates = pivot.index[pivot.index >= START_DATE]
    # 매주 첫 영업일 추출
    rebal_dates = []
    last_week = None
    for d in all_dates:
        wk = d.isocalendar()[1]
        yr = d.year
        key = (yr, wk)
        if key != last_week:
            rebal_dates.append(d)
            last_week = key

    print(f"[3/4] 백테스트 실행... (리밸런싱 {len(rebal_dates)}회)")

    # 포트폴리오 추적
    initial_capital = 100_000_000  # 1억
    cash = initial_capital
    holdings = {}  # ticker → {shares, buy_price, buy_date}
    portfolio_values = []  # (date, value)
    trade_log = []  # 매매 기록
    composition_log = []  # 리밸런싱 시점 종목 구성

    for ri, rdate in enumerate(rebal_dates):
        # RS Rating 계산
        rs_series = calc_rs_at_date(pivot, index_prices, rdate)
        if rs_series.empty:
            # 포트폴리오 가치 기록
            port_val = cash
            for tk, h in holdings.items():
                if rdate in pivot.index and tk in pivot.columns:
                    port_val += h["shares"] * pivot.loc[rdate, tk]
            portfolio_values.append((rdate, port_val))
            continue

        # 이번 주 보유 대상: KOSPI200 & RS >= 90
        target_tickers = set()
        for tk in rs_series.index:
            if tk in kospi200 and rs_series[tk] >= 90:
                target_tickers.add(tk)

        # 매도: 조건 미충족 종목
        sell_tickers = [tk for tk in list(holdings.keys()) if tk not in target_tickers]
        for tk in sell_tickers:
            h = holdings.pop(tk)
            if rdate in pivot.index and tk in pivot.columns:
                sell_price = pivot.loc[rdate, tk]
            else:
                sell_price = h["buy_price"]  # fallback

            proceeds = h["shares"] * sell_price
            fee = proceeds * (FEE_SELL + TAX_SELL)
            cash += proceeds - fee

            ret_pct = (sell_price / h["buy_price"] - 1) * 100
            pnl = (sell_price - h["buy_price"]) * h["shares"] - fee - h["shares"] * h["buy_price"] * FEE_BUY

            rs_val = int(rs_series[tk]) if tk in rs_series.index else "N/A"
            trade_log.append({
                "ticker": tk,
                "name": name_map.get(tk, ""),
                "buy_date": h["buy_date"],
                "sell_date": rdate,
                "buy_price": h["buy_price"],
                "sell_price": sell_price,
                "shares": h["shares"],
                "return_pct": ret_pct,
                "pnl": pnl,
                "reason": f"RS={rs_val}, {'not KOSPI200' if tk not in kospi200 else 'RS<90'}",
                "hold_days": (rdate - h["buy_date"]).days,
            })

        # 현재 포트폴리오 가치 계산 (매도 후)
        port_val = cash
        for tk, h in holdings.items():
            if rdate in pivot.index and tk in pivot.columns:
                port_val += h["shares"] * pivot.loc[rdate, tk]

        # 신규 매수 대상
        new_buys = target_tickers - set(holdings.keys())

        if new_buys or holdings:
            total_positions = len(holdings) + len(new_buys)
            if total_positions > 0:
                target_per_stock = port_val / total_positions

                # 신규 매수
                for tk in new_buys:
                    if rdate not in pivot.index or tk not in pivot.columns:
                        continue
                    price = pivot.loc[rdate, tk]
                    if price <= 0 or np.isnan(price):
                        continue

                    buy_amount = min(target_per_stock, cash)
                    shares = int(buy_amount / (price * (1 + FEE_BUY)))
                    if shares <= 0:
                        continue

                    cost = shares * price * (1 + FEE_BUY)
                    cash -= cost
                    holdings[tk] = {
                        "shares": shares,
                        "buy_price": price,
                        "buy_date": rdate,
                    }

        # 포트폴리오 가치 기록 (매수 후)
        port_val = cash
        for tk, h in holdings.items():
            if rdate in pivot.index and tk in pivot.columns:
                port_val += h["shares"] * pivot.loc[rdate, tk]
        portfolio_values.append((rdate, port_val))

        # 구성 기록
        holding_names = [f"{name_map.get(tk, tk)}(RS{int(rs_series[tk]) if tk in rs_series.index else '?'})"
                         for tk in sorted(holdings.keys())]
        composition_log.append({
            "date": rdate,
            "n_holdings": len(holdings),
            "port_value": port_val,
            "return_pct": (port_val / initial_capital - 1) * 100,
            "tickers": holding_names,
        })

        if (ri + 1) % 10 == 0 or ri == len(rebal_dates) - 1:
            print(f"      [{ri+1}/{len(rebal_dates)}] {rdate.date()}: "
                  f"{len(holdings)}종목 보유, 포트 {port_val:,.0f}원 "
                  f"({(port_val/initial_capital-1)*100:+.2f}%)")

    # 미청산 포지션 마감 (마지막 날짜 기준)
    last_date = pivot.index[-1]
    for tk in list(holdings.keys()):
        h = holdings.pop(tk)
        if tk in pivot.columns:
            sell_price = pivot.loc[last_date, tk]
        else:
            sell_price = h["buy_price"]

        proceeds = h["shares"] * sell_price
        fee = proceeds * (FEE_SELL + TAX_SELL)
        cash += proceeds - fee

        ret_pct = (sell_price / h["buy_price"] - 1) * 100
        pnl = (sell_price - h["buy_price"]) * h["shares"] - fee - h["shares"] * h["buy_price"] * FEE_BUY
        trade_log.append({
            "ticker": tk,
            "name": name_map.get(tk, ""),
            "buy_date": h["buy_date"],
            "sell_date": last_date,
            "buy_price": h["buy_price"],
            "sell_price": sell_price,
            "shares": h["shares"],
            "return_pct": ret_pct,
            "pnl": pnl,
            "reason": "기간종료",
            "hold_days": (last_date - h["buy_date"]).days,
        })

    final_value = cash
    total_return = (final_value / initial_capital - 1) * 100

    # 벤치마크: KOSPI200 지수 수익률
    idx_start = index_prices.loc[index_prices.index >= START_DATE].iloc[0]
    idx_end = index_prices.iloc[-1]
    benchmark_return = (idx_end / idx_start - 1) * 100

    trades_df = pd.DataFrame(trade_log)
    elapsed = time.time() - start_time

    # 4) 리포트 & 차트 생성
    print("\n[4/4] 리포트 & 차트 생성...")
    report = generate_report(
        initial_capital, final_value, total_return, benchmark_return,
        trades_df, composition_log, portfolio_values, elapsed
    )

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_rs90_kospi200.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    generate_charts(portfolio_values, initial_capital, index_prices,
                    composition_log, trades_df, base_dir)

    print(f"\n{'='*60}")
    print(f"RS>=90 & KOSPI200 조건부 보유 전략 백테스트 완료")
    print(f"{'='*60}")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print(f"  초기자본: {initial_capital:,}원")
    print(f"  최종가치: {final_value:,.0f}원")
    print(f"  총수익률: {total_return:+.2f}%")
    print(f"  벤치마크(KOSPI): {benchmark_return:+.2f}%")
    print(f"  초과수익률: {total_return - benchmark_return:+.2f}%p")
    if not trades_df.empty:
        print(f"  총거래: {len(trades_df)}건")
        print(f"  승률: {(trades_df['pnl'] > 0).sum() / len(trades_df) * 100:.1f}%")
        print(f"  평균수익률: {trades_df['return_pct'].mean():.2f}%")
        print(f"  평균보유일: {trades_df['hold_days'].mean():.1f}일")
    print(f"  리포트: {report_path}")
    print(f"  실행시간: {elapsed:.1f}초")


def generate_report(initial_capital, final_value, total_return, benchmark_return,
                    trades_df, composition_log, portfolio_values, elapsed):
    lines = []
    lines.append("# RS>=90 & KOSPI200 조건부 보유 전략 백테스트\n")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **전략**: 매주 RS Rating 재계산, RS>=90 & KOSPI200 종목만 동일가중 보유")
    lines.append(f"- **리밸런싱**: 매주 (첫 영업일)")
    lines.append(f"- **초기자본**: {initial_capital:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%\n")

    lines.append("## 성과 요약\n")
    lines.append("| 항목 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 최종가치 | {final_value:,.0f}원 |")
    lines.append(f"| **총수익률** | **{total_return:+.2f}%** |")
    lines.append(f"| 벤치마크(KOSPI) | {benchmark_return:+.2f}% |")
    lines.append(f"| **초과수익률** | **{total_return - benchmark_return:+.2f}%p** |")

    if not trades_df.empty:
        n = len(trades_df)
        wins = (trades_df["pnl"] > 0).sum()
        lines.append(f"| 총거래 | {n}건 |")
        lines.append(f"| 승률 | {wins/n*100:.1f}% |")
        lines.append(f"| 평균수익률 | {trades_df['return_pct'].mean():.2f}% |")
        lines.append(f"| 중위수익률 | {trades_df['return_pct'].median():.2f}% |")
        lines.append(f"| 평균보유일 | {trades_df['hold_days'].mean():.1f}일 |")
        lines.append(f"| 최대이익 | {trades_df['return_pct'].max():.2f}% |")
        lines.append(f"| 최대손실 | {trades_df['return_pct'].min():.2f}% |")

    # MDD 계산
    if portfolio_values:
        vals = [v for _, v in portfolio_values]
        peak = vals[0]
        mdd = 0
        for v in vals:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < mdd:
                mdd = dd
        lines.append(f"| **MDD** | **{mdd:.2f}%** |")
    lines.append("")

    # 리밸런싱 히스토리
    lines.append("## 리밸런싱 히스토리\n")
    lines.append("| 날짜 | 보유종목수 | 포트가치 | 수익률 | 보유종목 |")
    lines.append("|------|----------|---------|--------|---------|")
    for c in composition_log:
        tickers_str = ", ".join(c["tickers"][:10])
        if len(c["tickers"]) > 10:
            tickers_str += f" 외 {len(c['tickers'])-10}종목"
        lines.append(f"| {c['date'].strftime('%Y-%m-%d')} | {c['n_holdings']} | "
                     f"{c['port_value']:,.0f}원 | {c['return_pct']:+.2f}% | {tickers_str} |")
    lines.append("")

    # 거래 내역 (상위 수익 / 하위 손실)
    if not trades_df.empty:
        lines.append("## 수익 상위 10건\n")
        lines.append("| 종목 | 매수일 | 매도일 | 수익률 | 보유일 | 사유 |")
        lines.append("|------|--------|--------|--------|--------|------|")
        top = trades_df.nlargest(10, "return_pct")
        for _, r in top.iterrows():
            lines.append(f"| {r['name']} | {r['buy_date'].strftime('%Y-%m-%d')} | "
                        f"{r['sell_date'].strftime('%Y-%m-%d')} | {r['return_pct']:+.2f}% | "
                        f"{r['hold_days']}일 | {r['reason']} |")
        lines.append("")

        lines.append("## 손실 상위 10건\n")
        lines.append("| 종목 | 매수일 | 매도일 | 수익률 | 보유일 | 사유 |")
        lines.append("|------|--------|--------|--------|--------|------|")
        bot = trades_df.nsmallest(10, "return_pct")
        for _, r in bot.iterrows():
            lines.append(f"| {r['name']} | {r['buy_date'].strftime('%Y-%m-%d')} | "
                        f"{r['sell_date'].strftime('%Y-%m-%d')} | {r['return_pct']:+.2f}% | "
                        f"{r['hold_days']}일 | {r['reason']} |")
        lines.append("")

    lines.append(f"---\n실행 시간: {elapsed:.1f}초")
    lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def generate_charts(portfolio_values, initial_capital, index_prices,
                    composition_log, trades_df, base_dir):
    chart_dir = os.path.join(base_dir, "charts_rs90_kospi200")
    os.makedirs(chart_dir, exist_ok=True)

    dates = [d for d, _ in portfolio_values]
    values = [v for _, v in portfolio_values]
    returns = [(v / initial_capital - 1) * 100 for v in values]

    # 벤치마크 수익률 시리즈
    idx_start_val = index_prices.loc[index_prices.index >= START_DATE].iloc[0]
    bench_dates = index_prices.index[index_prices.index >= START_DATE]
    bench_returns = [(index_prices.loc[d] / idx_start_val - 1) * 100 for d in bench_dates]

    # 1) 수익률 곡선 비교
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, returns, "b-", linewidth=2, label="RS90 & KOSPI200 전략")
    ax.plot(bench_dates, bench_returns, "r--", linewidth=1.5, alpha=0.7, label="KOSPI 지수")
    ax.fill_between(dates, returns, 0, alpha=0.1, color="blue")
    ax.set_title("RS>=90 & KOSPI200 조건부 보유 전략 vs KOSPI", fontsize=14)
    ax.set_ylabel("수익률 (%)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(chart_dir, "return_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) 보유종목수 추이
    if composition_log:
        fig, ax = plt.subplots(figsize=(14, 4))
        comp_dates = [c["date"] for c in composition_log]
        comp_n = [c["n_holdings"] for c in composition_log]
        ax.bar(comp_dates, comp_n, width=5, color="steelblue", alpha=0.7)
        ax.set_title("주간 보유 종목 수 추이", fontsize=13)
        ax.set_ylabel("종목수")
        ax.grid(True, alpha=0.3, axis="y")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(chart_dir, "holdings_count.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3) 개별 거래 수익률 분포
    if not trades_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(trades_df["return_pct"], bins=40, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
        avg = trades_df["return_pct"].mean()
        ax.axvline(avg, color="green", linewidth=1.5, linestyle="--", label=f"평균 {avg:.2f}%")
        ax.set_title("거래별 수익률 분포", fontsize=13)
        ax.set_xlabel("수익률 (%)")
        ax.set_ylabel("건수")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(chart_dir, "return_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"      차트 저장: {chart_dir}/")


if __name__ == "__main__":
    run_backtest()
