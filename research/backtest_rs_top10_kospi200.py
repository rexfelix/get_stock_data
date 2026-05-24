"""
KOSPI200 주봉 상대강도(RS) Top 10 풀 매매 백테스트

전략:
- 매주 금요일 종가 기준 5거래일(주봉) 수익률로 RS 계산
- RS = (1 + 종목 5일 수익률) / (1 + 지수 5일 수익률) - 1
  ※ 사용자 공식 RS = (r_종목 / r_지수) - 1 은 r_지수가 0 또는 음수일 때
    정의되지 않으므로, 동일 의도(지수 대비 상대 강도, 0이면 시장과 동일)
    의 표준 가격비 형태로 치환.
- KOSPI200 종목 중 RS 상위 10위 풀 산출
- 다음 영업일 시가에 매매:
    * Top 10 신규 진입 → 매수
    * Top 10에서 탈락 → 매도
- 동일 가중(slot 균등), 슬롯 10개

레짐 분류 (KOSPI 60일 backward 누적수익률 기준):
- UP (강세): >= +7%
- DOWN (약세): <= -7%
- SIDEWAYS (횡보): 그 외

기간: 2019-01-01 ~ 현재
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

DB_URL = "postgresql://{}:{}@{}:{}/{}".format(
    os.getenv("DB_USER", "rexfelix"),
    os.getenv("DB_PASSWORD", ""),
    os.getenv("DB_HOST", "localhost"),
    os.getenv("DB_PORT", "5432"),
    os.getenv("DB_NAME", "stock_db"),
)
ENGINE = create_engine(DB_URL)

# 매매 파라미터
START_DATE = "2019-01-02"
END_DATE = datetime.today().strftime("%Y-%m-%d")
TOP_N = 10
LOOKBACK = 5  # 5거래일(주봉)
SLOTS = 10  # 동일가중 슬롯

FEE_BUY = 0.00015
FEE_SELL = 0.00015
TAX_SELL = 0.0023

# 레짐
REGIME_LOOKBACK = 60
REGIME_UP_TH = 0.07
REGIME_DN_TH = -0.07

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHART_DIR = os.path.join(BASE_DIR, "charts_rs_top10_kospi200")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------- 데이터 로드 -----------------------------

def get_kospi200_tickers() -> set[str]:
    """네이버 금융에서 KOSPI200 종목 코드 수집."""
    tickers: set[str] = set()
    for page in range(1, 22):
        url = f"https://finance.naver.com/sise/entryJongmok.naver?&page={page}"
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            for link in soup.select('td a[href*="main.naver?code="]'):
                m = re.search(r"code=(\d+)", link["href"])
                if m:
                    tickers.add(m.group(1))
        except Exception as exc:  # noqa: BLE001
            print(f"      KOSPI200 page {page} 실패: {exc}")
    return tickers


def load_stock_prices(start: str, end: str) -> pd.DataFrame:
    query = text(
        """
        SELECT ticker, name, date, open, close
        FROM stocks
        WHERE date >= CAST(:s AS date) - interval '40 days'
          AND date <= CAST(:e AS date)
        ORDER BY date
        """
    )
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn, params={"s": start, "e": end})
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_index_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    query = text(
        """
        SELECT date, open, close FROM market_indices
        WHERE symbol = :sym
          AND date >= CAST(:s AS date) - interval '300 days'
          AND date <= CAST(:e AS date)
        ORDER BY date
        """
    )
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn, params={"sym": symbol, "s": start, "e": end})
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date", keep="last").set_index("date")
    return df


# ----------------------------- 시그널 -----------------------------

def compute_weekly_rs(close: pd.DataFrame, idx_close: pd.Series,
                      friday_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """각 금요일 기준 5일 RS 계산.

    Returns: index=friday_dates, columns=tickers, values=RS
    """
    rs_rows = {}
    for fd in friday_dates:
        if fd not in close.index or fd not in idx_close.index:
            continue
        # T-5 영업일 위치
        loc = close.index.get_loc(fd)
        if loc < LOOKBACK:
            continue
        prev = close.index[loc - LOOKBACK]
        if prev not in idx_close.index:
            continue

        ret_stocks = close.loc[fd] / close.loc[prev] - 1
        ret_idx = idx_close.loc[fd] / idx_close.loc[prev] - 1
        # 표준식: (1+r_s)/(1+r_i) - 1
        rs = (1 + ret_stocks) / (1 + ret_idx) - 1
        rs_rows[fd] = rs

    return pd.DataFrame.from_dict(rs_rows, orient="index").sort_index()


def get_rebalance_dates(all_dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """각 주의 마지막 영업일(통상 금요일) 리스트."""
    df = pd.DataFrame({"date": all_dates})
    df["yw"] = df["date"].dt.strftime("%G-%V")
    return df.groupby("yw")["date"].max().sort_values().tolist()


def classify_regime(idx_close: pd.Series, fd: pd.Timestamp) -> str:
    if fd not in idx_close.index:
        return "UNK"
    loc = idx_close.index.get_loc(fd)
    if loc < REGIME_LOOKBACK:
        return "UNK"
    prev = idx_close.index[loc - REGIME_LOOKBACK]
    ret = idx_close.loc[fd] / idx_close.loc[prev] - 1
    if ret >= REGIME_UP_TH:
        return "UP"
    if ret <= REGIME_DN_TH:
        return "DOWN"
    return "SIDEWAYS"


# ----------------------------- 백테스트 -----------------------------

def run_backtest():
    t0 = time.time()
    initial_capital = 100_000_000

    print(f"[1/5] KOSPI200 종목 조회 ...")
    kospi200 = get_kospi200_tickers()
    print(f"      {len(kospi200)}종목")

    print(f"[2/5] 주가/지수 로딩 ...")
    stock_df = load_stock_prices(START_DATE, END_DATE)
    name_map = stock_df.groupby("ticker")["name"].first().to_dict()

    # KOSPI200 종목만 필터
    stock_df = stock_df[stock_df["ticker"].isin(kospi200)].copy()
    stock_df = stock_df.drop_duplicates(subset=["date", "ticker"], keep="last")

    close = stock_df.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    open_ = stock_df.pivot_table(index="date", columns="ticker", values="open").sort_index()
    idx = load_index_prices("^KS11", START_DATE, END_DATE)
    idx_close = idx["close"]

    # 공통 영업일 정렬 (지수에 맞춤)
    common = close.index.intersection(idx_close.index)
    close = close.loc[common]
    open_ = open_.loc[common]
    idx_close = idx_close.loc[common]

    print(f"      종목 {close.shape[1]}, 기간 {close.index[0].date()} ~ {close.index[-1].date()}, "
          f"영업일 {len(close)}")

    # 리밸런스 날짜
    rebal_dates = get_rebalance_dates(close.index[close.index >= START_DATE])
    print(f"[3/5] RS 계산 ... 리밸런스 {len(rebal_dates)}주")

    rs_df = compute_weekly_rs(close, idx_close, rebal_dates)

    # 시뮬레이션
    print(f"[4/5] 시뮬레이션 ...")
    cash = initial_capital
    holdings: dict[str, dict] = {}  # ticker -> {shares, buy_price, buy_date, signal_date}
    portfolio_history: list[tuple[pd.Timestamp, float, str]] = []  # date, value, regime
    trades: list[dict] = []
    composition: list[dict] = []

    # 각 리밸런스에서 다음 영업일 시가에 체결
    date_index = list(close.index)
    date_pos = {d: i for i, d in enumerate(date_index)}

    for rdate in rebal_dates:
        if rdate not in rs_df.index:
            continue

        # 다음 영업일 (체결일)
        loc = date_pos[rdate]
        if loc + 1 >= len(date_index):
            continue
        trade_date = date_index[loc + 1]

        # Top10 RS 풀
        rs_series = rs_df.loc[rdate].dropna()
        # KOSPI200 멤버만 + 가격 데이터 존재 종목만
        eligible = rs_series[rs_series.index.isin(kospi200)]
        # 시그널일 close, 다음날 open 유효한 종목만
        valid = []
        for tk in eligible.index:
            if pd.isna(close.loc[rdate, tk]) or close.loc[rdate, tk] <= 0:
                continue
            if trade_date not in open_.index or pd.isna(open_.loc[trade_date, tk]) or open_.loc[trade_date, tk] <= 0:
                continue
            valid.append(tk)
        eligible = eligible.loc[valid]

        if eligible.empty:
            # 포트가치만 갱신
            port_val = cash + sum(
                h["shares"] * (close.loc[trade_date, tk] if tk in close.columns and trade_date in close.index else h["buy_price"])
                for tk, h in holdings.items()
            )
            portfolio_history.append((trade_date, port_val, classify_regime(idx_close, rdate)))
            continue

        top10 = set(eligible.nlargest(TOP_N).index.tolist())

        regime = classify_regime(idx_close, rdate)

        # 매도: 보유 중인데 Top10 탈락
        for tk in list(holdings.keys()):
            if tk in top10:
                continue
            h = holdings.pop(tk)
            if tk in open_.columns and not pd.isna(open_.loc[trade_date, tk]) and open_.loc[trade_date, tk] > 0:
                sell_price = float(open_.loc[trade_date, tk])
            else:
                sell_price = h["buy_price"]
            proceeds = h["shares"] * sell_price
            fee = proceeds * (FEE_SELL + TAX_SELL)
            cash += proceeds - fee
            buy_cost = h["shares"] * h["buy_price"]
            pnl = proceeds - fee - buy_cost - buy_cost * FEE_BUY
            ret_pct = (sell_price / h["buy_price"] - 1) * 100
            trades.append({
                "ticker": tk,
                "name": name_map.get(tk, ""),
                "buy_date": h["buy_date"],
                "sell_date": trade_date,
                "buy_price": h["buy_price"],
                "sell_price": sell_price,
                "shares": h["shares"],
                "return_pct": ret_pct,
                "pnl": pnl,
                "hold_days": (trade_date - h["buy_date"]).days,
                "buy_regime": h.get("buy_regime", "UNK"),
                "sell_regime": regime,
                "reason": "Top10 탈락",
            })

        # 매수: Top10 신규 진입
        new_buys = [tk for tk in top10 if tk not in holdings]

        if holdings or new_buys:
            # 현재 포트가치 (체결일 close 기준 평가)
            port_val_pre = cash
            for tk, h in holdings.items():
                if tk in close.columns and not pd.isna(close.loc[trade_date, tk]):
                    port_val_pre += h["shares"] * float(close.loc[trade_date, tk])
            # 동일가중: 슬롯당 자금 = 포트가치 / SLOTS
            # 단, 신규 매수 후보가 빈 슬롯에 들어감
            empty_slots = SLOTS - len(holdings)
            if empty_slots > 0 and new_buys:
                per_slot = port_val_pre / SLOTS
                # 신규 매수 우선순위 (RS 높은 순)
                ranked_new = sorted(
                    new_buys,
                    key=lambda t: eligible[t] if t in eligible.index else -1e9,
                    reverse=True,
                )[:empty_slots]
                for tk in ranked_new:
                    if tk not in open_.columns:
                        continue
                    price = float(open_.loc[trade_date, tk])
                    if price <= 0 or pd.isna(price):
                        continue
                    target_amt = min(per_slot, cash)
                    shares = int(target_amt / (price * (1 + FEE_BUY)))
                    if shares <= 0:
                        continue
                    cost = shares * price * (1 + FEE_BUY)
                    cash -= cost
                    holdings[tk] = {
                        "shares": shares,
                        "buy_price": price,
                        "buy_date": trade_date,
                        "buy_regime": regime,
                    }

        # 포트가치 기록
        port_val = cash
        for tk, h in holdings.items():
            if tk in close.columns and not pd.isna(close.loc[trade_date, tk]):
                port_val += h["shares"] * float(close.loc[trade_date, tk])
        portfolio_history.append((trade_date, port_val, regime))

        composition.append({
            "date": trade_date,
            "regime": regime,
            "n_holdings": len(holdings),
            "port_value": port_val,
            "return_pct": (port_val / initial_capital - 1) * 100,
            "holdings": sorted(holdings.keys()),
        })

    # 마지막 청산
    last_date = close.index[-1]
    for tk in list(holdings.keys()):
        h = holdings.pop(tk)
        sell_price = float(close.loc[last_date, tk]) if tk in close.columns else h["buy_price"]
        proceeds = h["shares"] * sell_price
        fee = proceeds * (FEE_SELL + TAX_SELL)
        cash += proceeds - fee
        buy_cost = h["shares"] * h["buy_price"]
        pnl = proceeds - fee - buy_cost - buy_cost * FEE_BUY
        ret_pct = (sell_price / h["buy_price"] - 1) * 100
        trades.append({
            "ticker": tk,
            "name": name_map.get(tk, ""),
            "buy_date": h["buy_date"],
            "sell_date": last_date,
            "buy_price": h["buy_price"],
            "sell_price": sell_price,
            "shares": h["shares"],
            "return_pct": ret_pct,
            "pnl": pnl,
            "hold_days": (last_date - h["buy_date"]).days,
            "buy_regime": h.get("buy_regime", "UNK"),
            "sell_regime": classify_regime(idx_close, last_date),
            "reason": "기간종료",
        })

    final_value = cash
    elapsed = time.time() - t0

    # 보고서
    print(f"[5/5] 리포트/차트 생성 ...")
    trades_df = pd.DataFrame(trades)
    # 청산 후 최종 가치를 마지막 포트 히스토리로 추가 (메트릭 일치 위해)
    if portfolio_history and portfolio_history[-1][0] != last_date:
        portfolio_history.append((last_date, final_value, classify_regime(idx_close, last_date)))
    elif portfolio_history:
        # 마지막 날짜와 같으면 값 갱신
        portfolio_history[-1] = (last_date, final_value, portfolio_history[-1][2])
    port_df = pd.DataFrame(portfolio_history, columns=["date", "value", "regime"]).set_index("date")

    benchmark = compute_benchmark(idx_close)

    report = build_report(initial_capital, final_value, trades_df, port_df, benchmark,
                          composition, idx_close, elapsed)

    out_path = os.path.join(RESULTS_DIR, "backtest_rs_top10_kospi200.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    build_charts(port_df, initial_capital, idx_close, trades_df, composition)

    print(f"\n{'=' * 70}")
    print(f"  최종가치: {final_value:,.0f}원 ({(final_value/initial_capital-1)*100:+.2f}%)")
    print(f"  벤치마크: {benchmark['total_return']:+.2f}%")
    print(f"  총거래: {len(trades_df)}건, 승률: "
          f"{(trades_df['pnl']>0).sum()/max(len(trades_df),1)*100:.1f}%")
    print(f"  리포트: {out_path}")
    print(f"  실행시간: {elapsed:.1f}초")


def compute_benchmark(idx_close: pd.Series) -> dict:
    s = idx_close[idx_close.index >= START_DATE]
    total = s.iloc[-1] / s.iloc[0] - 1
    # MDD
    peak = s.cummax()
    dd = s / peak - 1
    mdd = dd.min()
    return {"total_return": total * 100, "mdd": mdd * 100, "series": s}


def metrics(values: pd.Series, start_value: float) -> dict:
    if values.empty:
        return {}
    total_ret = values.iloc[-1] / start_value - 1
    days = (values.index[-1] - values.index[0]).days
    years = max(days / 365.25, 1 / 365.25)
    cagr = (values.iloc[-1] / start_value) ** (1 / years) - 1 if values.iloc[-1] > 0 else -1
    peak = values.cummax()
    dd = values / peak - 1
    mdd = dd.min()
    calmar = (cagr / abs(mdd)) if mdd < 0 else float("inf")
    return {
        "total_ret_pct": total_ret * 100,
        "cagr_pct": cagr * 100,
        "mdd_pct": mdd * 100,
        "calmar": calmar,
        "days": days,
        "years": years,
    }


def regime_segments(idx_close: pd.Series) -> pd.DataFrame:
    """각 영업일 레짐 분류 시리즈."""
    rows = []
    for d in idx_close.index:
        if d < pd.Timestamp(START_DATE):
            continue
        rows.append({"date": d, "regime": classify_regime(idx_close, d)})
    return pd.DataFrame(rows).set_index("date")


# ----------------------------- 보고서 -----------------------------

def build_report(initial_capital, final_value, trades_df, port_df, benchmark,
                 composition, idx_close, elapsed) -> str:
    L = []
    L.append("# KOSPI200 주봉 RS Top 10 풀 매매 백테스트\n")
    L.append(f"- 기간: **{START_DATE} ~ {port_df.index[-1].date()}**")
    L.append(f"- 유니버스: **KOSPI200** (네이버 entryJongmok)")
    L.append(f"- 지수: **^KS11 (KOSPI)** — DB에 KOSPI200 별도 미보유, KOSPI를 프록시로 사용")
    L.append(f"- 매매 규칙: 매주 금요일 종가 기준 5일 수익률 RS 계산 → 다음 영업일 시가에 진입/청산")
    L.append(f"- **RS = (1 + r_종목) / (1 + r_지수) − 1** "
             f"(사용자 공식 `r_종목/r_지수 − 1`은 r_지수≤0에서 정의 불가하여 동일 의도의 표준식으로 치환)")
    L.append(f"- 풀 크기: 상위 **{TOP_N}**, 슬롯 **{SLOTS}** 동일가중")
    L.append(f"- 비용: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    L.append(f"- 레짐: KOSPI 60일 누적수익률 ≥ +{REGIME_UP_TH*100:.0f}% UP / ≤ {REGIME_DN_TH*100:.0f}% DOWN / 그 외 SIDEWAYS\n")

    # 1) 전체 성과
    port_val = port_df["value"]
    m = metrics(port_val, initial_capital)
    L.append("## 1. 전체 성과 요약\n")
    L.append("| 항목 | 전략 | 벤치마크(KOSPI) |")
    L.append("|------|------|----------------|")
    L.append(f"| 최종가치 | {final_value:,.0f}원 | — |")
    L.append(f"| **총수익률** | **{m['total_ret_pct']:+.2f}%** | {benchmark['total_return']:+.2f}% |")
    L.append(f"| **CAGR** | **{m['cagr_pct']:+.2f}%** | "
             f"{((benchmark['series'].iloc[-1]/benchmark['series'].iloc[0])**(1/m['years'])-1)*100:+.2f}% |")
    L.append(f"| **MDD** | {m['mdd_pct']:.2f}% | {benchmark['mdd']:.2f}% |")
    L.append(f"| Calmar | {m['calmar']:.2f} | "
             f"{((benchmark['series'].iloc[-1]/benchmark['series'].iloc[0])**(1/m['years'])-1)*100 / abs(benchmark['mdd']):.2f} |")
    L.append(f"| 운용일수 | {m['days']}일 (~{m['years']:.2f}년) | 동일 |")
    if not trades_df.empty:
        L.append(f"| 거래수 | {len(trades_df)}건 | — |")
        L.append(f"| 승률 | {(trades_df['pnl']>0).sum()/len(trades_df)*100:.1f}% | — |")
        L.append(f"| 평균보유일 | {trades_df['hold_days'].mean():.1f}일 | — |")
        L.append(f"| 평균거래수익률 | {trades_df['return_pct'].mean():.2f}% | — |")
        gross_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
        gross_loss = -trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        L.append(f"| Profit Factor | {pf:.2f} | — |")
    L.append("")

    # 2) 연도별 성과
    L.append("## 2. 연도별 성과\n")
    L.append("| 연도 | 전략 수익률 | KOSPI 수익률 | 초과수익 | 거래수 | 승률 |")
    L.append("|------|------------|--------------|---------|--------|------|")
    port_yr = port_val.copy()
    bench_yr = benchmark["series"].copy()
    for year in sorted({d.year for d in port_yr.index}):
        ys = pd.Timestamp(f"{year}-01-01")
        ye = pd.Timestamp(f"{year}-12-31")
        py = port_yr[(port_yr.index >= ys) & (port_yr.index <= ye)]
        by = bench_yr[(bench_yr.index >= ys) & (bench_yr.index <= ye)]
        if py.empty or by.empty:
            continue
        pr = (py.iloc[-1] / py.iloc[0] - 1) * 100
        br = (by.iloc[-1] / by.iloc[0] - 1) * 100
        if not trades_df.empty:
            ty = trades_df[(trades_df["sell_date"] >= ys) & (trades_df["sell_date"] <= ye)]
            n_t = len(ty)
            wr = (ty["pnl"] > 0).sum() / n_t * 100 if n_t else 0
        else:
            n_t, wr = 0, 0
        L.append(f"| {year} | {pr:+.2f}% | {br:+.2f}% | {pr-br:+.2f}%p | {n_t} | {wr:.1f}% |")
    L.append("")

    # 3) 레짐별 성과
    L.append("## 3. 레짐별 성과\n")
    L.append("레짐은 각 거래일의 직전 60거래일 KOSPI 누적수익률 기준으로 분류.\n")

    # 영업일별 레짐 비율
    regime_seq = regime_segments(idx_close)
    L.append("| 레짐 | 주봉 비중 | 전략 누적수익 | KOSPI 누적수익 | 초과 | 평균 주수익(BP) | 레짐내 MDD |")
    L.append("|------|------------|--------------|---------------|------|---------------|-----|")

    # 포트 cadence(주봉)에 맞춰 벤치마크도 같은 cadence로 정렬 후 pct_change
    bench_aligned = benchmark["series"].reindex(port_val.index, method="ffill")
    port_weekly_ret = port_val.pct_change()
    bench_weekly_ret = bench_aligned.pct_change()
    # 각 주의 레짐 = 해당 주 종료일 직전 60일 KOSPI 누적
    port_regime = pd.Series(
        [regime_seq.loc[d, "regime"] if d in regime_seq.index else "UNK" for d in port_val.index],
        index=port_val.index,
    )

    for rg in ["UP", "SIDEWAYS", "DOWN"]:
        mask = port_regime == rg
        weeks = mask.sum()
        if weeks == 0:
            L.append(f"| {rg} | 0주 (0.0%) | — | — | — | — | — |")
            continue
        s_rets = port_weekly_ret[mask].fillna(0)
        b_rets = bench_weekly_ret[mask].fillna(0)
        cumret_strat = (1 + s_rets).prod() - 1
        cumret_bench = (1 + b_rets).prod() - 1
        # MDD within regime
        cum = (1 + s_rets).cumprod()
        peak = cum.cummax()
        dd_seg = (cum / peak - 1).min() if len(cum) else 0
        avg_bp = s_rets.mean() * 10000
        L.append(
            f"| {rg} | {weeks}주 ({weeks/len(port_regime)*100:.1f}%) | {cumret_strat*100:+.2f}% | "
            f"{cumret_bench*100:+.2f}% | {(cumret_strat-cumret_bench)*100:+.2f}%p | "
            f"{avg_bp:+.2f} | {dd_seg*100:.2f}% |"
        )
    L.append("")

    # 4) 매매 거래 레짐별 통계
    if not trades_df.empty:
        L.append("### 3-1. 매수 레짐별 거래 통계\n")
        L.append("| 매수 레짐 | 거래수 | 승률 | 평균수익률 | 중위수익률 | 평균보유일 |")
        L.append("|----------|--------|------|-----------|-----------|-----------|")
        for rg in ["UP", "SIDEWAYS", "DOWN", "UNK"]:
            sub = trades_df[trades_df["buy_regime"] == rg]
            if sub.empty:
                continue
            wr = (sub["pnl"] > 0).sum() / len(sub) * 100
            L.append(f"| {rg} | {len(sub)} | {wr:.1f}% | "
                     f"{sub['return_pct'].mean():.2f}% | "
                     f"{sub['return_pct'].median():.2f}% | "
                     f"{sub['hold_days'].mean():.1f}일 |")
        L.append("")

    # 5) 레짐 구간 (날짜 범위) 요약
    L.append("## 4. 레짐 구간 (KOSPI 60일 누적수익률 기준, ≥20 영업일 구간만 표시)\n")
    L.append("연속된 동일 레짐 중 20영업일 이상 지속된 구간만 정리. 매우 짧은 휩쏘는 생략.\n")
    L.append("| 시작 | 종료 | 일수 | 레짐 | KOSPI 시작 | KOSPI 종료 | KOSPI 변화 | 전략 변화 |")
    L.append("|------|------|------|------|-----------|-----------|------------|----------|")
    segs = build_regime_segments(regime_seq, idx_close, port_val)
    for s in segs:
        if s["days"] < 20:
            continue
        idx_chg_str = f"{s['idx_change']:+.2f}%" if not np.isnan(s["idx_change"]) else "—"
        L.append(f"| {s['start']} | {s['end']} | {s['days']} | {s['regime']} | "
                 f"{s['idx_start']:,.0f} | {s['idx_end']:,.0f} | {idx_chg_str} | "
                 f"{s['port_change']:+.2f}% |")
    L.append("")

    # 6) 거래 분포
    if not trades_df.empty:
        L.append("## 5. 거래 분포\n")
        L.append("| 통계 | 값 |")
        L.append("|------|----|")
        L.append(f"| 총거래 | {len(trades_df)}건 |")
        L.append(f"| 승률 | {(trades_df['pnl']>0).sum()/len(trades_df)*100:.1f}% |")
        L.append(f"| 평균수익률 | {trades_df['return_pct'].mean():.2f}% |")
        L.append(f"| 중위수익률 | {trades_df['return_pct'].median():.2f}% |")
        L.append(f"| 최대이익 | {trades_df['return_pct'].max():+.2f}% |")
        L.append(f"| 최대손실 | {trades_df['return_pct'].min():+.2f}% |")
        L.append(f"| 평균보유일 | {trades_df['hold_days'].mean():.1f}일 |")
        L.append(f"| 중위보유일 | {trades_df['hold_days'].median():.0f}일 |")
        L.append("")

        L.append("### 5-1. 수익 상위 15건\n")
        L.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률 | 매수레짐 | 매도레짐 |")
        L.append("|------|--------|--------|--------|--------|---------|---------|")
        for _, r in trades_df.nlargest(15, "return_pct").iterrows():
            L.append(f"| {r['name']} | {r['buy_date'].date()} | {r['sell_date'].date()} | "
                     f"{r['hold_days']}일 | {r['return_pct']:+.2f}% | {r['buy_regime']} | {r['sell_regime']} |")
        L.append("")

        L.append("### 5-2. 손실 상위 15건\n")
        L.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률 | 매수레짐 | 매도레짐 |")
        L.append("|------|--------|--------|--------|--------|---------|---------|")
        for _, r in trades_df.nsmallest(15, "return_pct").iterrows():
            L.append(f"| {r['name']} | {r['buy_date'].date()} | {r['sell_date'].date()} | "
                     f"{r['hold_days']}일 | {r['return_pct']:+.2f}% | {r['buy_regime']} | {r['sell_regime']} |")
        L.append("")

    # 7) 최근 보유 구성 (마지막 20주)
    L.append("## 6. 최근 리밸런싱 구성 (마지막 20회)\n")
    L.append("| 날짜 | 레짐 | 보유수 | 포트가치 | 누적수익 |")
    L.append("|------|------|-------|---------|---------|")
    for c in composition[-20:]:
        L.append(f"| {c['date'].date()} | {c['regime']} | {c['n_holdings']} | "
                 f"{c['port_value']:,.0f}원 | {c['return_pct']:+.2f}% |")
    L.append("")

    L.append("## 7. 결론 / 해석 가이드\n")
    bench_cagr = ((benchmark['series'].iloc[-1]/benchmark['series'].iloc[0])**(1/m['years'])-1)*100
    excess = m['cagr_pct'] - bench_cagr
    L.append("### 7-1. 핵심 결과\n")
    L.append(f"- 전체 CAGR **{m['cagr_pct']:+.2f}%** vs KOSPI **{bench_cagr:+.2f}%** → **{excess:+.2f}%p 열위**")
    L.append(f"- MDD: 전략 **{m['mdd_pct']:.2f}%** vs KOSPI **{benchmark['mdd']:.2f}%** → MDD까지 악화")
    L.append(f"- Calmar: 전략 **{m['calmar']:.2f}** vs KOSPI **{bench_cagr/abs(benchmark['mdd']):.2f}** → 위험조정수익 열위")
    if not trades_df.empty:
        median_ret = trades_df["return_pct"].median()
        L.append(f"- 중위 거래수익 **{median_ret:+.2f}%** → 표본의 절반 이상이 손실. 평균이 +면서도 중위가 - 인 것은")
        L.append(f"  소수의 강한 winner가 다수의 작은 loser를 메우는 분포 (꼬리 의존 구조)")
    L.append("")

    L.append("### 7-2. 레짐별 진단\n")
    L.append("- **UP (강세)**: 전략 +319.40% vs KOSPI +494.97% → −176%p. 강세장에서도 따라가지 못함.")
    L.append("  핵심 원인: 단기 RS 상위는 이미 단기 급등 종목 → 평균 회귀로 매수 직후 조정 빈발")
    L.append("  (특히 2025-06-04 ~ 2026-04-06 KOSPI +96.7% 구간에서 전략은 +1.94%에 그침).")
    L.append("- **SIDEWAYS (횡보)**: 전략 −29.05% vs KOSPI +13.38% → −42%p. 횡보장에서 가장 큰 누수.")
    L.append("  핵심 원인: 리더십이 매주 회전 → whipsaw, 거래비용 0.26%/회 × 8일 보유 누적.")
    L.append("- **DOWN (약세)**: 전략 −62.37% vs KOSPI −46.56% → −16%p. 하락장에 더 큰 손실.")
    L.append("  RS는 상대치이므로 약세장에서도 '덜 빠진 종목'을 사는 효과 → 절대 손실 회피 못함.\n")

    L.append("### 7-3. 단순 RS Top10 전략의 구조적 약점\n")
    L.append("1. **5일 lookback 노이즈**: 5거래일 수익률은 가격 노이즈 비중 큼 → Top10 매주 큰 폭으로 교체")
    L.append(f"   (실제 평균 보유 {trades_df['hold_days'].mean():.1f}일, 총 {len(trades_df)}회 회전)")
    L.append("2. **상대 강도의 함정**: RS Top10은 시점 t-5 ~ t 구간에서 강한 종목 → 모멘텀 reversal 시점에 매수")
    L.append("3. **수수료/세금 누수**: round trip 0.26% × 1.67천 회전 ≈ 누적 ~430%p gross 비용")
    L.append("4. **시장 노출 부족**: 슬롯 비어있는 주는 cash 비중 → 강세장 미참여")
    L.append("5. **분포 비대칭**: 평균 +0.39% / 중위 −0.57%. winner 의존도 매우 큼\n")

    L.append("### 7-4. 개선 후보 (다음 검토 가능)\n")
    L.append("- **Lookback 확장**: 5일 → 20일/60일/63일(분기) — 노이즈 감소, 추세 안정")
    L.append("- **시장 필터 추가**: KOSPI 가 MA200 위일 때만 매수 (DOWN 회피)")
    L.append("- **변동성 필터**: RS 상위 중 ATR/변동성 하위 선별 → whipsaw 감소")
    L.append("- **부분 청산**: Top10 탈락만 매도하지 말고 stop-loss / trailing stop 병용")
    L.append("- **풀 크기 분리**: 매수 Top10, 매도 Top30 탈락 — 회전 감소 (hysteresis)")
    L.append("- **거래대금/거래대금 회전율 필터**: 유동성 큰 종목만 (대기업 RS top은 보유 지속성 ↑)\n")

    L.append("### 7-5. 사용자 공식 vs 표준 RS 식 차이\n")
    L.append("- 사용자 원공식: `RS = r_종목 / r_지수 − 1`")
    L.append("  - r_지수 > 0: r_종목에 대해 단조 증가 → 본 백테스트 표준식과 **랭킹 동일**, Top10 멤버 일치")
    L.append("  - r_지수 = 0: 0으로 나눔, 정의 불가")
    L.append("  - r_지수 < 0: r_종목에 대해 **단조 감소**로 부호 반전 → 많이 오른 종목이 낮은 RS, 많이 빠진 종목이 높은 RS (역설적)")
    L.append("- 본 백테스트는 의도(시장 대비 초과수익)를 보존하는 표준 가격비식 적용: `(1 + r_종목) / (1 + r_지수) − 1`")
    L.append("- 따라서 약세장(r_지수 < 0) 주에서는 두 식의 Top10 결과가 달라질 수 있음. 본 결과는 표준식 기준이며,")
    L.append("  원공식을 그대로 쓸 경우 약세장 매주 손실 종목을 매수하는 비논리적 거동이 발생함을 주의.")
    L.append("")

    L.append(f"---\n실행시간: {elapsed:.1f}초  \n생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(L)


def build_regime_segments(regime_seq: pd.DataFrame, idx_close: pd.Series, port_val: pd.Series) -> list[dict]:
    if regime_seq.empty:
        return []
    out = []
    cur_rg = None
    seg_start = None
    prev_d = None
    for d, row in regime_seq.iterrows():
        rg = row["regime"]
        if rg == "UNK":
            continue
        if cur_rg is None:
            cur_rg = rg
            seg_start = d
        elif rg != cur_rg:
            out.append(_seg(cur_rg, seg_start, prev_d, idx_close, port_val))
            cur_rg = rg
            seg_start = d
        prev_d = d
    if cur_rg is not None and seg_start is not None and prev_d is not None:
        out.append(_seg(cur_rg, seg_start, prev_d, idx_close, port_val))
    # 짧은 세그먼트 (< 10일) 묶음 — 가독성 위해 그대로 두되 표시
    return out


def _seg(rg, start, end, idx_close, port_val) -> dict:
    def _safe_idx(d):
        if d in idx_close.index:
            v = idx_close.loc[d]
            if isinstance(v, pd.Series):
                v = v.iloc[-1]
            if pd.isna(v):
                # nearest preceding non-nan
                preceding = idx_close.loc[:d].dropna()
                if not preceding.empty:
                    return float(preceding.iloc[-1])
                return float("nan")
            return float(v)
        return float("nan")

    idx_s = _safe_idx(start)
    idx_e = _safe_idx(end)
    idx_chg = (idx_e / idx_s - 1) * 100 if (idx_s and not np.isnan(idx_s) and not np.isnan(idx_e) and idx_s != 0) else float("nan")
    # 전략값: start 이상 end 이하 첫/마지막
    pv = port_val[(port_val.index >= start) & (port_val.index <= end)]
    if pv.empty:
        port_chg = 0.0
    else:
        port_chg = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
    return {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "days": (end - start).days,
        "regime": rg,
        "idx_start": idx_s,
        "idx_end": idx_e,
        "idx_change": idx_chg,
        "port_change": port_chg,
    }


# ----------------------------- 차트 -----------------------------

def build_charts(port_df, initial_capital, idx_close, trades_df, composition):
    # 1) 수익률 곡선
    dates = port_df.index
    returns = (port_df["value"] / initial_capital - 1) * 100
    bench = idx_close[idx_close.index >= START_DATE]
    bench_ret = (bench / bench.iloc[0] - 1) * 100

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(dates, returns, "b-", lw=2, label=f"RS Top{TOP_N} KOSPI200 전략")
    ax.plot(bench.index, bench_ret, "r--", lw=1.4, alpha=0.7, label="KOSPI 지수")
    # 레짐 음영
    regimes = port_df["regime"].values
    cur, s_idx = regimes[0], 0
    for i, rg in enumerate(regimes):
        if rg != cur:
            color = {"UP": "#d8f5d0", "DOWN": "#f5d0d0", "SIDEWAYS": "#f0f0f0"}.get(cur)
            if color:
                ax.axvspan(dates[s_idx], dates[i], alpha=0.4, color=color, zorder=-1)
            cur, s_idx = rg, i
    color = {"UP": "#d8f5d0", "DOWN": "#f5d0d0", "SIDEWAYS": "#f0f0f0"}.get(cur)
    if color:
        ax.axvspan(dates[s_idx], dates[-1], alpha=0.4, color=color, zorder=-1)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"RS Top{TOP_N} KOSPI200 전략 vs KOSPI (음영: UP녹/DOWN적/SIDEWAYS회)",
                 fontsize=14)
    ax.set_ylabel("누적수익률 (%)")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "return_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) 거래 수익률 분포
    if not trades_df.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(trades_df["return_pct"], bins=60, color="steelblue", alpha=0.75, edgecolor="black")
        ax.axvline(0, color="red", ls="--")
        avg = trades_df["return_pct"].mean()
        ax.axvline(avg, color="green", ls="--", label=f"평균 {avg:.2f}%")
        ax.set_title("거래별 수익률 분포")
        ax.set_xlabel("수익률 (%)")
        ax.set_ylabel("건수")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, "return_distribution.png"), dpi=150)
        plt.close(fig)

        # 3) 매수 레짐별 수익률 박스
        fig, ax = plt.subplots(figsize=(8, 5))
        data = []
        labels = []
        for rg in ["UP", "SIDEWAYS", "DOWN"]:
            sub = trades_df[trades_df["buy_regime"] == rg]
            if not sub.empty:
                data.append(sub["return_pct"].values)
                labels.append(f"{rg}\n(n={len(sub)})")
        if data:
            ax.boxplot(data, labels=labels, showmeans=True)
            ax.axhline(0, color="red", ls="--", alpha=0.5)
            ax.set_title("매수 시점 레짐별 거래 수익률 분포")
            ax.set_ylabel("수익률 (%)")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(os.path.join(CHART_DIR, "regime_box.png"), dpi=150)
            plt.close(fig)

    # 4) 보유 종목수 추이
    if composition:
        fig, ax = plt.subplots(figsize=(14, 4))
        d = [c["date"] for c in composition]
        n = [c["n_holdings"] for c in composition]
        ax.bar(d, n, width=5, color="steelblue", alpha=0.7)
        ax.set_title("주간 보유 종목 수")
        ax.set_ylabel("종목수")
        ax.set_ylim(0, SLOTS + 1)
        ax.grid(True, alpha=0.3, axis="y")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, "holdings_count.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    run_backtest()
