"""
3-bar 매매규칙 백테스트

규칙 (results/3-bar_매매규칙.md):
- hma = 3봉 고가 이동평균, lma = 3봉 저가 이동평균
- 매수 시그널: (lma < 고가 < hma) AND (저가 < lma) AND 종가>MA20 AND MA20 상승추세
- 매수 체결: 시그널 봉 종가에 매수 (same-day close entry)
- 매도: 5가지 방식 비교
    (a) MA5 이탈 (기준봉 이후 MA5를 한 번 상향한 뒤 종가 MA5 이탈 시)
    (b) 익일 시가 매도 (1봉 홀딩)
    (c) 고정 -3% 손절 / +7% 익절
    (d) ATR(14) x 2 트레일링
    (e) 최근 3봉 저가 이탈 (Swing Low)

비교: 연도별(2023~현재) × 5개 매도 방식
"""

import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_crash import (
    ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    load_all_data,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

END_DATE = datetime.today().strftime("%Y-%m-%d")

EXIT_STRATEGIES = [
    "MA5이탈",
    "익일시가",
    "3%손절7%익절",
    "ATR2x",
    "3봉저가이탈",
]

# 거래대금 필터: 직전 N봉 중 M봉 이상 amount >= threshold
AMOUNT_THRESHOLD = 1500e8  # 1500억
AMOUNT_WINDOW = 10
AMOUNT_MIN_HITS = 5


def load_amount_data(tickers: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """stock_all에서 거래대금(amount) 로드."""
    placeholders = ",".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT ticker, date, amount
        FROM stock_all
        WHERE ticker IN ({placeholders})
          AND date >= '{start_date}'::date - interval '90 days'
          AND date <= '{end_date}'
        ORDER BY ticker, date ASC
    """
    df_all = pd.read_sql(query, ENGINE)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["amount"] = pd.to_numeric(df_all["amount"], errors="coerce") * 1e6  # 백만원→원
    result = {}
    for ticker, group in df_all.groupby("ticker"):
        g = group.set_index("date").sort_index()
        g = g[~g.index.duplicated(keep="last")]
        result[ticker] = g[["amount"]]
    return result


# ──────────────────────────────────────────────
# 지표 계산
# ──────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame, apply_amount_filter: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["hma3"] = df["high"].rolling(window=3).mean()
    df["lma3"] = df["low"].rolling(window=3).mean()
    df["sma5"] = df["close"].rolling(window=5).mean()
    df["sma20"] = df["close"].rolling(window=20).mean()
    df["ma20_up"] = df["sma20"] > df["sma20"].shift(1)

    # ATR(14)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(window=14).mean()

    # 거래대금 필터 (시그널 당일 포함 직전 10봉 중 N봉 >= 1500억)
    if apply_amount_filter and "amount" in df.columns:
        amt = df["amount"].fillna(0)
        hit = (amt >= AMOUNT_THRESHOLD).astype(int)
        df["amount_pass"] = hit.rolling(window=AMOUNT_WINDOW).sum() >= AMOUNT_MIN_HITS
    else:
        df["amount_pass"] = True
    return df


# ──────────────────────────────────────────────
# 3-bar 매수 시그널 탐지
# ──────────────────────────────────────────────
def find_signals(df: pd.DataFrame) -> list[int]:
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    hma = df["hma3"].values
    lma = df["lma3"].values
    ma20 = df["sma20"].values
    ma20_up = df["ma20_up"].values
    amt_pass = df["amount_pass"].values

    signals = []
    for i in range(len(df)):
        if np.isnan(hma[i]) or np.isnan(lma[i]) or np.isnan(ma20[i]):
            continue
        if not (lma[i] < high[i] < hma[i]):
            continue
        if not (low[i] < lma[i]):
            continue
        if not (close[i] > ma20[i]):
            continue
        if not ma20_up[i]:
            continue
        if not amt_pass[i]:
            continue
        signals.append(i)
    return signals


# ──────────────────────────────────────────────
# 매수 실행
# ──────────────────────────────────────────────
def do_buy(capital, price, date, entry_idx):
    max_qty = int(capital / (price * (1 + FEE_BUY)))
    if max_qty <= 0:
        return capital, None
    cost = max_qty * price
    fee = cost * FEE_BUY
    new_cap = capital - cost - fee
    pos = {"entry_price": price, "quantity": max_qty, "entry_date": date,
           "entry_idx": entry_idx,
           "ma5_crossed": False, "high_since_entry": price}
    return new_cap, pos


def do_sell(capital, position, price, date, reason, ticker, name):
    qty = position["quantity"]
    revenue = qty * price
    fee = revenue * FEE_SELL
    tax = revenue * TAX_SELL
    net = revenue - fee - tax
    capital += net
    buy_cost = position["entry_price"] * qty
    buy_fee = buy_cost * FEE_BUY
    pnl = net - buy_cost - buy_fee
    ret = pnl / (buy_cost + buy_fee) * 100
    trade = {
        "ticker": ticker, "name": name,
        "buy_date": position["entry_date"],
        "buy_price": position["entry_price"],
        "sell_date": date, "sell_price": price,
        "quantity": qty, "pnl": pnl,
        "return_pct": ret, "reason": reason,
    }
    return capital, trade


# ──────────────────────────────────────────────
# 백테스트 엔진 (매도 전략별)
# ──────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, ticker: str, name: str,
                 exit_strategy: str, start_date: str):
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    equity_curve = []
    dates = df.index.tolist()
    signals = set(find_signals(df))

    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    close = df["close"].values
    sma5 = df["sma5"].values
    atr = df["atr14"].values

    for i in range(len(df)):
        date = dates[i]
        c = close[i]

        # ── 매도 판정 (포지션 보유시) ──
        if position is not None:
            sell_price = None
            reason = None
            hold_days = i - position["entry_idx"]

            if exit_strategy == "MA5이탈":
                # MA5를 한 번 상향한 뒤 종가로 이탈
                if not np.isnan(sma5[i]):
                    if c > sma5[i]:
                        position["ma5_crossed"] = True
                    elif position["ma5_crossed"] and c < sma5[i]:
                        sell_price = c
                        reason = "MA5이탈"

            elif exit_strategy == "익일시가":
                if hold_days >= 1:
                    sell_price = open_[i]
                    reason = "익일시가"

            elif exit_strategy == "3%손절7%익절":
                ep = position["entry_price"]
                stop = ep * 0.97
                tp = ep * 1.07
                # 손절 우선 (갭하락 시 시가로 체결)
                if low[i] <= stop:
                    sell_price = min(open_[i], stop)
                    reason = "-3%손절"
                elif high[i] >= tp:
                    sell_price = open_[i] if open_[i] >= tp else tp
                    reason = "+7%익절"

            elif exit_strategy == "ATR2x":
                # 진입 이후 최고가 갱신 후 ATR*2 이탈시 매도
                position["high_since_entry"] = max(position["high_since_entry"], high[i])
                if not np.isnan(atr[i]):
                    stop = position["high_since_entry"] - 2 * atr[i]
                    if low[i] <= stop:
                        sell_price = min(open_[i], stop)
                        reason = "ATR2x이탈"

            elif exit_strategy == "3봉저가이탈":
                # 직전 3봉 최저가 하향 이탈시 매도
                if i >= 3 and i > position["entry_idx"]:
                    swing_low = min(low[i-3:i])
                    if low[i] <= swing_low:
                        sell_price = min(open_[i], swing_low)
                        reason = "3봉저가이탈"

            if sell_price is not None:
                capital, trade = do_sell(capital, position, sell_price, date,
                                         reason, ticker, name)
                trades.append(trade)
                position = None

        # ── 매수 판정 ──
        if position is None and i in signals:
            if df.index[i] >= pd.Timestamp(start_date):
                capital, pos = do_buy(capital, c, date, i)
                if pos is not None:
                    position = pos

        eq = capital + (position["quantity"] * c if position else 0)
        equity_curve.append({"date": date, "equity": eq})

    # 미청산 청산
    if position is not None:
        last = df.iloc[-1]
        capital, trade = do_sell(capital, position, last["close"],
                                 dates[-1], "미청산", ticker, name)
        trades.append(trade)

    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df = equity_df.set_index("date")
    trades_df = pd.DataFrame(trades)

    end_eq = equity_df["equity"].iloc[-1] if not equity_df.empty else INITIAL_CAPITAL
    total_ret = (end_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    bnh_ret = 0
    df_period = df.loc[start_date:]
    if len(df_period) > 0:
        bnh_ret = (df_period["close"].iloc[-1] / df_period["close"].iloc[0] - 1) * 100

    summary = {
        "ticker": ticker, "name": name,
        "total_return": total_ret,
        "bnh_return": bnh_ret,
        "n_trades": len(trades_df),
        "final_equity": end_eq,
    }

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        summary["win_rate"] = len(wins) / len(trades_df) * 100
        summary["total_pnl"] = trades_df["pnl"].sum()
        summary["avg_return"] = trades_df["return_pct"].mean()
        gp = wins["pnl"].sum() if len(wins) > 0 else 0
        gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        summary["profit_factor"] = gp / gl if gl > 0 else float("inf")
    else:
        summary["win_rate"] = 0
        summary["total_pnl"] = 0
        summary["avg_return"] = 0
        summary["profit_factor"] = 0

    if not equity_df.empty:
        roll_max = equity_df["equity"].cummax()
        dd = equity_df["equity"] / roll_max - 1
        summary["mdd"] = dd.min() * 100
    else:
        summary["mdd"] = 0

    return summary, trades_df, equity_df


# ──────────────────────────────────────────────
# 기간 × 매도전략 실행
# ──────────────────────────────────────────────
def run_period(all_data_indicators, ticker_list, name_map,
               start_date, end_date, label, exit_strategy):
    print(f"  [{label}] {exit_strategy}... ", end="", flush=True)
    t0 = time.time()
    summaries, all_trades = [], []
    for ticker in ticker_list:
        if ticker not in all_data_indicators:
            continue
        df = all_data_indicators[ticker]
        df_test = df.loc[:end_date]
        if len(df_test) < 20:
            continue
        nm = name_map.get(ticker, ticker)
        s, t, _ = run_backtest(df_test, ticker, nm, exit_strategy, start_date)
        summaries.append(s)
        if not t.empty:
            all_trades.append(t)

    sum_df = pd.DataFrame(summaries)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    traded = (sum_df["n_trades"] > 0).sum() if not sum_df.empty else 0
    print(f"거래 {traded}종목, {len(trades_df)}건 ({time.time()-t0:.1f}s)")
    return {"label": label, "exit": exit_strategy,
            "summaries": sum_df, "trades": trades_df}


# ──────────────────────────────────────────────
# 지표 추출
# ──────────────────────────────────────────────
def _extract_metrics(summaries_df, trades_df):
    traded = summaries_df[summaries_df["n_trades"] > 0] if not summaries_df.empty else pd.DataFrame()
    m = {"n_traded": len(traded)}
    if not traded.empty:
        m["avg_ret"] = traded["total_return"].mean()
        m["med_ret"] = traded["total_return"].median()
        m["mdd"] = traded["mdd"].mean()
    else:
        m["avg_ret"] = m["med_ret"] = m["mdd"] = 0
    m["n_trades"] = len(trades_df) if not trades_df.empty else 0
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        m["win_rate"] = len(wins) / len(trades_df) * 100
        gp = wins["pnl"].sum() if len(wins) > 0 else 0
        gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        m["pf"] = gp / gl if gl > 0 else float("inf")
        m["total_pnl"] = trades_df["pnl"].sum()
        m["avg_trade_ret"] = trades_df["return_pct"].mean()
        tc = trades_df.copy()
        tc["hd"] = (pd.to_datetime(tc["sell_date"]) - pd.to_datetime(tc["buy_date"])).dt.days
        m["hold_days"] = tc["hd"].mean()
    else:
        m["win_rate"] = m["pf"] = m["total_pnl"] = m["avg_trade_ret"] = m["hold_days"] = 0
    return m


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def get_mcap_tickers(threshold_won: float = 3e12) -> dict:
    """시가총액 추정 >= threshold 종목 {ticker: mcap} 반환.
    mcap ≈ 최근 종가 × (최근 연간 net_income×1e8 / eps)
    """
    query = text("""
        WITH fs AS (
            SELECT DISTINCT ON (ticker) ticker, net_income, eps
            FROM financial_summary
            WHERE is_estimate=false AND eps IS NOT NULL AND eps != 0
              AND net_income IS NOT NULL
            ORDER BY ticker, year DESC
        ),
        latest_close AS (
            SELECT DISTINCT ON (ticker) ticker, close
            FROM stocks WHERE close > 0
            ORDER BY ticker, date DESC
        )
        SELECT lc.ticker,
               (lc.close::numeric * fs.net_income::numeric * 1e8 / fs.eps) AS mcap
        FROM latest_close lc
        JOIN fs USING(ticker)
        WHERE (lc.close::numeric * fs.net_income::numeric * 1e8 / fs.eps) >= :thr
    """)
    with ENGINE.connect() as conn:
        rows = conn.execute(query, {"thr": threshold_won}).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def run_main(universe: str = "kospi200", amount_filter: bool = False):
    start_time = time.time()
    base_dir = os.path.dirname(__file__)
    uni_label = {
        "kospi200": "KOSPI200",
        "all": "전종목",
        "k200_or_3t": "KOSPI200 ∪ 시총≥3조",
    }[universe]
    if amount_filter:
        uni_label += f" + 거래대금필터({AMOUNT_WINDOW}봉중{AMOUNT_MIN_HITS}봉≥{AMOUNT_THRESHOLD/1e8:.0f}억)"

    # 종목 조회
    if universe in ("kospi200", "k200_or_3t"):
        from backtest_crash import get_kospi200_tickers
        print("[1/4] KOSPI200 종목 조회...")
        kospi200 = get_kospi200_tickers()
        with ENGINE.connect() as conn:
            db_rows = conn.execute(text("SELECT DISTINCT ticker FROM stocks")).fetchall()
        db_tickers = set(r[0] for r in db_rows)
        k200_tickers = {t["ticker"]: t["name"] for t in kospi200 if t["ticker"] in db_tickers}

        if universe == "kospi200":
            ticker_list = list(k200_tickers.keys())
            name_map = k200_tickers
        else:
            print("     시총 ≥3조 종목 계산...")
            mcap_map = get_mcap_tickers(3e12)
            # 이름 맵 병합: k200 + 시총≥3조
            with ENGINE.connect() as conn:
                name_rows = conn.execute(
                    text("SELECT DISTINCT ticker, name FROM stocks")
                ).fetchall()
            all_names = {r[0]: r[1] for r in name_rows}
            merged_tickers = set(k200_tickers.keys()) | set(mcap_map.keys())
            ticker_list = sorted(merged_tickers)
            name_map = {t: k200_tickers.get(t) or all_names.get(t, t) for t in ticker_list}
            print(f"     KOSPI200 {len(k200_tickers)}개 + 시총≥3조 {len(mcap_map)}개 → 합집합 {len(ticker_list)}개")
    else:
        print("[1/4] 전종목 조회...")
        with ENGINE.connect() as conn:
            db_rows = conn.execute(
                text("SELECT DISTINCT ticker, name FROM stocks ORDER BY ticker")
            ).fetchall()
        ticker_list = [r[0] for r in db_rows]
        name_map = {r[0]: r[1] for r in db_rows}
    print(f"     {len(ticker_list)}종목")

    # 데이터 로딩 (전체 기간 한 번에)
    full_start = "2023-01-01"
    full_end = END_DATE
    print(f"[2/4] 데이터 로딩 ({full_start} ~ {full_end})...")
    all_data = {}
    batch_size = 500
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i:i + batch_size]
        batch_data = load_all_data(batch, full_start, full_end)
        all_data.update(batch_data)
    print(f"     {len(all_data)}종목 로딩 완료")

    # 거래대금 로딩 & 병합
    if amount_filter:
        print("     거래대금 로딩 (stock_all)...")
        amt_data = {}
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i + batch_size]
            batch_amt = load_amount_data(batch, full_start, full_end)
            amt_data.update(batch_amt)

    print("[3/4] 지표 계산...")
    all_data_ind = {}
    for ticker, df in all_data.items():
        df = df[~df.index.duplicated(keep="last")]
        if amount_filter and ticker in amt_data:
            df = df.join(amt_data[ticker], how="left")
        all_data_ind[ticker] = calc_indicators(df, apply_amount_filter=amount_filter)

    periods = [
        ("2023-01-01", "2023-12-31", "2023"),
        ("2024-01-01", "2024-12-31", "2024"),
        ("2025-01-01", "2025-12-31", "2025"),
        ("2026-01-01", END_DATE, "2026"),
        ("2023-01-01", END_DATE, "전체"),
    ]

    print(f"[4/4] 백테스트 실행: {len(periods)}기간 × {len(EXIT_STRATEGIES)}전략")
    all_results = {}
    for sd, ed, yr in periods:
        all_results[yr] = {}
        for ex in EXIT_STRATEGIES:
            r = run_period(all_data_ind, ticker_list, name_map, sd, ed, yr, ex)
            all_results[yr][ex] = r

    elapsed = time.time() - start_time

    # ── 리포트 생성 ──
    lines = []
    lines.append(f"# 3-bar 매매규칙 백테스트 ({uni_label})\n")
    lines.append("## 전략 개요\n")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append(f"- **대상**: {uni_label} · **기간**: 2023-01-01 ~ {END_DATE}\n")

    lines.append("### 매수 규칙\n")
    lines.append("| 조건 | 설명 |")
    lines.append("|------|------|")
    lines.append("| 위치 | lma3 < 고가 < hma3 (당일 고가가 3봉 이평 저가~고가 사이) |")
    lines.append("| 꼬리 | 저가 < lma3 (3봉 저가이평 하향 찌름) |")
    lines.append("| 추세 | 종가 > MA20 AND MA20 상승 (ma20[0] > ma20[-1]) |")
    lines.append("| 체결 | 시그널 봉 종가 매수 |")
    lines.append("")

    lines.append("### 매도 전략 (5종)\n")
    lines.append("| 전략 | 설명 |")
    lines.append("|------|------|")
    lines.append("| MA5이탈 | MA5를 한 번 상향한 뒤 종가 MA5 이탈 시 매도 (원규칙) |")
    lines.append("| 익일시가 | 진입 익일 시가에 무조건 매도 (1봉 홀딩) |")
    lines.append("| 3%손절7%익절 | 고정 -3% 손절 / +7% 익절 (손절 우선) |")
    lines.append("| ATR2x | 진입 후 최고가 - ATR(14)×2 트레일링 스탑 |")
    lines.append("| 3봉저가이탈 | 직전 3봉 저가 하향 이탈 시 매도 |")
    lines.append("")

    year_labels = [p[2] for p in periods]

    # 매도전략 × 연도 매트릭스
    metric_rows = [
        ("거래 종목", "n_traded", "{:.0f}"),
        ("총 거래", "n_trades", "{:.0f}건"),
        ("종목 평균 수익률", "avg_ret", "{:.2f}%"),
        ("종목 중위 수익률", "med_ret", "{:.2f}%"),
        ("전체 승률", "win_rate", "{:.1f}%"),
        ("손익비", "pf", "{:.2f}"),
        ("거래당 평균 수익률", "avg_trade_ret", "{:.2f}%"),
        ("평균 보유일", "hold_days", "{:.1f}일"),
        ("종목 평균 MDD", "mdd", "{:.2f}%"),
        ("총 손익(원)", "total_pnl", "{:,.0f}"),
    ]

    for ex in EXIT_STRATEGIES:
        lines.append(f"## 매도전략 · {ex} — 연도별 성과\n")
        lines.append("| 지표 | " + " | ".join(year_labels) + " |")
        lines.append("|------" + "|-----" * len(year_labels) + "|")
        metrics = [_extract_metrics(all_results[yr][ex]["summaries"],
                                    all_results[yr][ex]["trades"]) for yr in year_labels]
        for row_label, key, fmt in metric_rows:
            cells = " | ".join(
                (fmt.format(m[key]) if not (isinstance(m[key], float) and np.isinf(m[key]))
                 else "∞") for m in metrics
            )
            lines.append(f"| {row_label} | {cells} |")
        lines.append("")

    # 전체 기간 매도전략 비교표
    lines.append("## 매도전략 종합 비교 (전체 2023~현재)\n")
    lines.append("| 매도전략 | 거래수 | 승률 | 손익비 | 거래당수익률 | 종목평균수익률 | 평균보유일 | MDD | 총손익 |")
    lines.append("|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for ex in EXIT_STRATEGIES:
        m = _extract_metrics(all_results["전체"][ex]["summaries"],
                              all_results["전체"][ex]["trades"])
        pf_str = "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}"
        lines.append(
            f"| {ex} | {m['n_trades']}건 | {m['win_rate']:.1f}% | {pf_str} "
            f"| {m['avg_trade_ret']:.2f}% | {m['avg_ret']:.2f}% | {m['hold_days']:.1f}일 "
            f"| {m['mdd']:.2f}% | {m['total_pnl']:,.0f}원 |"
        )
    lines.append("")

    # 최고 전략 TOP 5
    best_ex = max(EXIT_STRATEGIES,
                  key=lambda e: _extract_metrics(all_results["전체"][e]["summaries"],
                                                  all_results["전체"][e]["trades"])["total_pnl"])
    lines.append(f"## 최고 성과 전략 · {best_ex} — 종목 TOP 10 (전체 기간)\n")
    sdf = all_results["전체"][best_ex]["summaries"]
    traded = sdf[sdf["n_trades"] > 0] if not sdf.empty else pd.DataFrame()
    if not traded.empty:
        lines.append("| # | 종목 | 코드 | 수익률 | 거래수 | 승률 | MDD |")
        lines.append("|---|------|------|--------|--------|------|-----|")
        for j, (_, row) in enumerate(traded.nlargest(10, "total_return").iterrows()):
            lines.append(
                f"| {j+1} | {row['name']} | {row['ticker']} | {row['total_return']:.2f}% "
                f"| {row['n_trades']} | {row['win_rate']:.0f}% | {row['mdd']:.2f}% |"
            )
    lines.append("")

    # 연도 × 전략 수익률 매트릭스
    lines.append("## 연도 × 매도전략 수익률 매트릭스 (종목 평균)\n")
    lines.append("| 연도 | " + " | ".join(EXIT_STRATEGIES) + " |")
    lines.append("|------" + "|-----" * len(EXIT_STRATEGIES) + "|")
    for yr in year_labels:
        cells = []
        for ex in EXIT_STRATEGIES:
            m = _extract_metrics(all_results[yr][ex]["summaries"],
                                  all_results[yr][ex]["trades"])
            cells.append(f"{m['avg_ret']:.2f}%")
        lines.append(f"| {yr} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 종합 고찰\n")
    # 최고 전략
    best_pnl = _extract_metrics(all_results["전체"][best_ex]["summaries"],
                                 all_results["전체"][best_ex]["trades"])
    best_pf = "∞" if np.isinf(best_pnl["pf"]) else f"{best_pnl['pf']:.2f}"
    lines.append(f"- **최고 총손익 전략: {best_ex}** — {best_pnl['total_pnl']:,.0f}원, "
                 f"승률 {best_pnl['win_rate']:.1f}%, 손익비 {best_pf}")

    # 연도별 플러스 달성
    for ex in EXIT_STRATEGIES:
        plus_yrs = [yr for yr in ["2023", "2024", "2025", "2026"]
                    if _extract_metrics(all_results[yr][ex]["summaries"],
                                         all_results[yr][ex]["trades"])["total_pnl"] > 0]
        lines.append(f"- {ex}: 플러스 연도 {len(plus_yrs)}/4 ({','.join(plus_yrs) if plus_yrs else '없음'})")

    lines.append("")
    lines.append("## 실행 정보\n")
    lines.append(f"- **실행 시간**: {elapsed:.2f}초")
    lines.append(f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    report = "\n".join(lines)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    suffix_map = {"kospi200": "", "all": "_all", "k200_or_3t": "_k200_or_3t"}
    suffix = suffix_map[universe]
    if amount_filter:
        suffix += "_amt"
    report_path = os.path.join(results_dir, f"backtest_3bar{suffix}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*50}")
    print(f"3-bar 매매규칙 백테스트 완료! ({elapsed:.2f}초)")
    print(f"리포트: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="3-bar 매매규칙 백테스트")
    parser.add_argument("--all", action="store_true", help="전종목 대상")
    parser.add_argument("--k200-3t", action="store_true",
                        help="KOSPI200 ∪ 시총≥3조 (기본: KOSPI200)")
    parser.add_argument("--amt", action="store_true",
                        help=f"거래대금 필터 추가 ({AMOUNT_WINDOW}봉중{AMOUNT_MIN_HITS}봉≥{AMOUNT_THRESHOLD/1e8:.0f}억)")
    args = parser.parse_args()
    if args.all:
        uni = "all"
    elif args.k200_3t:
        uni = "k200_or_3t"
    else:
        uni = "kospi200"
    run_main(universe=uni, amount_filter=args.amt)
