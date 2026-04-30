"""
Top 3 지표 비교 백테스트 - KOSPI200 대상.

- 지표 3종: 거래대금(amount), 시가총액(mcap=close*shares), 거래회전율(turnover=amount/mcap)
- 매도 규칙 6종: LIST_EXIT, MA5, MA20, HOLD_5, HOLD_10, HOLD_20
- 18 조합 비교, 2023~현재
- 수수료: 매수 0.015% + 매도 0.015% + 세금 0.18%
"""
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

ENGINE = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

SNAPSHOT_CSV = "/Volumes/SSD/project/py/invest/data_center/research/report/top3_indicators/market_cap_snapshot.csv"
OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_top3_indicators.md"

FEE_BUY = 0.00015
FEE_SELL = 0.00015
TAX_SELL = 0.0018

START_DATE = "2023-01-01"
RULES = ["LIST_EXIT", "MA5", "MA20", "HOLD_5", "HOLD_10", "HOLD_20"]
INDICATORS = ["amount", "mcap", "turnover"]


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def compute_ma(prices: pd.Series, n: int) -> pd.Series:
    return prices.rolling(window=n, min_periods=n).mean()


def compute_top3_by_column(df: pd.DataFrame, col: str, n: int = 3) -> list[str]:
    """1일치 DataFrame에서 col 기준 상위 n종목 ticker 반환."""
    sorted_df = df.sort_values(col, ascending=False, kind="mergesort").head(n)
    return sorted_df["ticker"].tolist()


def apply_fees(gross_ret: float) -> float:
    return gross_ret - FEE_BUY - FEE_SELL - TAX_SELL


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_kospi200_tickers() -> pd.DataFrame:
    df = pd.read_sql("SELECT ticker, name FROM kospi200_members ORDER BY ticker", ENGINE)
    return df


def load_market_cap_snapshot() -> pd.DataFrame:
    df = pd.read_csv(SNAPSHOT_CSV, dtype={"ticker": str})
    df["ticker"] = df["ticker"].str.zfill(6)
    return df[["ticker", "name", "shares_outstanding"]]


def load_price_data(tickers: list[str], start_date: str = START_DATE) -> pd.DataFrame:
    """stocks + stock_all 결합. KOSPI200 종목 OHLCV + amount 시계열."""
    placeholders = ",".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT s.ticker, s.name, s.date,
               s.open, s.high, s.low, s.close, s.volume,
               sa.amount
        FROM stocks s
        LEFT JOIN stock_all sa ON s.ticker = sa.ticker AND s.date::date = sa.date
        WHERE s.ticker IN ({placeholders})
          AND s.date >= '{start_date}'::date - interval '60 days'
        ORDER BY s.ticker, s.date
    """
    df = pd.read_sql(query, ENGINE)
    df["date"] = pd.to_datetime(df["date"])
    # amount 단위: stock_all.amount는 백만원 단위(get_stock_all에서 그대로 저장됨) → 원으로 환산
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1_000_000
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_daily_data(price_df: pd.DataFrame, snapshot: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """ticker별 시계열 dict. mcap, turnover, MA5, MA20 컬럼 추가."""
    shares_map = dict(zip(snapshot["ticker"], snapshot["shares_outstanding"]))

    out = {}
    for ticker, g in price_df.groupby("ticker"):
        if ticker not in shares_map:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        shares = shares_map[ticker]
        g["shares"] = shares
        g["mcap"] = g["close"].astype(float) * shares
        g["turnover"] = g["amount"] / g["mcap"]
        g["ma5"] = compute_ma(g["close"].astype(float), 5)
        g["ma20"] = compute_ma(g["close"].astype(float), 20)
        out[ticker] = g
    return out


def build_daily_indicator_panel(
    daily_data: dict[str, pd.DataFrame], indicator: str
) -> dict[pd.Timestamp, pd.DataFrame]:
    """date → DataFrame(ticker, 지표값) panel."""
    rows = []
    for ticker, df in daily_data.items():
        sub = df[["date", indicator]].copy()
        sub["ticker"] = ticker
        rows.append(sub)
    full = pd.concat(rows, ignore_index=True)
    full = full.dropna(subset=[indicator])
    full = full[full[indicator] > 0]
    panel = {}
    for d, g in full.groupby("date"):
        panel[d] = g[["ticker", indicator]].reset_index(drop=True)
    return panel


def compute_top3_per_day(panel: dict, indicator: str, n: int = 3) -> dict[pd.Timestamp, list[str]]:
    return {d: compute_top3_by_column(g, indicator, n=n) for d, g in panel.items()}


# ---------------------------------------------------------------------------
# 시뮬레이션
# ---------------------------------------------------------------------------

def simulate_strategy(
    daily_data: dict[str, pd.DataFrame],
    top3_per_day: dict[pd.Timestamp, list[str]],
    dates: list[pd.Timestamp],
    rule: str,
    hold_n: Optional[int] = None,
    ma_period: Optional[int] = None,
    stop_pct: Optional[float] = None,
    max_concurrent: Optional[int] = None,
) -> list[dict]:
    """
    매수: T-1 마감 Top3 → T 시가 매수 (보유 중이 아닌 경우만)
    매도: T-1 마감 시점 매도 신호 → T 시가 매도

    매도 규칙:
    - LIST_EXIT: T일 마감 시 Top3에 없으면 → T+1 시가 매도
    - MA5/MA20: T일 종가 < MA → T+1 시가 매도
    - HOLD_N: 보유일수 N일 도달 → T+1 시가 매도
    - MA_INIT_STOP: 매수 후 종가가 MA 위로 한 번이라도 가면 crossed_ma=True
                    crossed=False 상태에서 매수가 대비 stop_pct 이상 하락 → 손절
                    crossed=True 상태에서 close < MA → MA 이탈 매도
                    (ma_period: 5 or 20, stop_pct: 음수, 예 -0.05)
    """
    positions = {}  # ticker -> {buy_date, buy_price, hold_days, sell_signal}
    trades = []

    # ticker → date → row 빠른 조회용
    ticker_idx = {}
    for t, df in daily_data.items():
        ticker_idx[t] = df.set_index("date")

    for i, d in enumerate(dates):
        sold_today = set()  # 같은 날 매도→재매수 방지

        # 1. 매도 처리: 어제 발생한 신호 → 오늘 시가 매도
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            if not pos["sell_signal"]:
                continue
            df = ticker_idx[ticker]
            if d not in df.index:
                continue
            row = df.loc[d]
            sell_price = row["open"]
            if pd.isna(sell_price) or sell_price <= 0:
                continue
            gross = (sell_price - pos["buy_price"]) / pos["buy_price"]
            net = apply_fees(gross)
            trades.append({
                "ticker": ticker,
                "buy_date": pos["buy_date"],
                "buy_price": pos["buy_price"],
                "sell_date": d,
                "sell_price": sell_price,
                "hold_days": pos["hold_days"],
                "gross_ret": gross,
                "net_ret": net,
                "rule": rule,
            })
            del positions[ticker]
            sold_today.add(ticker)

        # 2. 매수 처리: 어제 마감 Top3 → 오늘 시가 매수
        if i > 0:
            prev_d = dates[i - 1]
            for ticker in top3_per_day.get(prev_d, []):
                # 동시 보유 cap 적용
                if max_concurrent is not None and len(positions) >= max_concurrent:
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
                    "crossed_ma": False,
                }

        # 3. 매도 신호 결정 (오늘 마감 기준)
        today_top3 = top3_per_day.get(d, [])
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            df = ticker_idx[ticker]
            if d not in df.index:
                continue
            row = df.loc[d]
            close = row["close"]
            ma5 = row.get("ma5", np.nan)
            ma20 = row.get("ma20", np.nan)

            # 매수일 당일은 hold_days=1, 다음날 +1
            if pos["buy_date"] != d:
                pos["hold_days"] += 1

            signal = False
            if rule == "LIST_EXIT":
                if ticker not in today_top3:
                    signal = True
            elif rule == "MA5":
                if not pd.isna(ma5) and not pd.isna(close) and close < ma5:
                    signal = True
            elif rule == "MA20":
                if not pd.isna(ma20) and not pd.isna(close) and close < ma20:
                    signal = True
            elif rule == "HOLD_N":
                if hold_n is not None and pos["hold_days"] >= hold_n:
                    signal = True
            elif rule == "MA_INIT_STOP":
                ma_val = ma5 if ma_period == 5 else ma20
                if not pd.isna(ma_val) and not pd.isna(close):
                    # 1) MA 위로 갔는지 플래그 갱신
                    if close >= ma_val:
                        pos["crossed_ma"] = True
                    # 2) crossed=True 상태에서 MA 이탈 → 매도
                    if pos["crossed_ma"] and close < ma_val:
                        signal = True
                # 3) crossed=False 상태에서 손절선 -N% 도달 → 손절
                if not pos["crossed_ma"] and not pd.isna(close) and stop_pct is not None:
                    cur_ret = (close - pos["buy_price"]) / pos["buy_price"]
                    if cur_ret <= stop_pct:
                        signal = True

            if signal:
                pos["sell_signal"] = True

    # 백테스트 종료 시 미청산 강제 청산 (마지막 종가)
    if dates:
        last_d = dates[-1]
        for ticker, pos in positions.items():
            df = ticker_idx[ticker]
            if last_d not in df.index:
                continue
            row = df.loc[last_d]
            sell_price = row["close"]
            if pd.isna(sell_price) or sell_price <= 0:
                continue
            gross = (sell_price - pos["buy_price"]) / pos["buy_price"]
            net = apply_fees(gross)
            trades.append({
                "ticker": ticker,
                "buy_date": pos["buy_date"],
                "buy_price": pos["buy_price"],
                "sell_date": last_d,
                "sell_price": sell_price,
                "hold_days": pos["hold_days"],
                "gross_ret": gross,
                "net_ret": net,
                "rule": rule,
                "forced_close": True,
            })

    return trades


# ---------------------------------------------------------------------------
# 통계
# ---------------------------------------------------------------------------

def compute_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"total": 0}
    df = pd.DataFrame(trades)
    rets = df["net_ret"] * 100  # %
    wins = (rets > 0).sum()
    losses = (rets <= 0).sum()
    total = len(df)
    win_rate = wins / total * 100 if total else 0
    avg_ret = rets.mean()
    med_ret = rets.median()
    avg_win = rets[rets > 0].mean() if wins > 0 else 0
    avg_loss = rets[rets <= 0].mean() if losses > 0 else 0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    avg_hold = df["hold_days"].mean()

    # 누적수익률 (3종목 균등 분배 가정 → trade별 1/3 자본)
    # 단순 trade 곱 누적: assume 매 거래마다 1/3 자본 회전
    # 정확한 누적은 시간순 정렬 후 자본 분배 시뮬레이션 필요
    # 여기서는 단순 기하 누적 (상한 제거)
    df_sorted = df.sort_values("buy_date")
    cum_simple = (1 + df_sorted["net_ret"] / 3).prod()  # 3종목 동시 보유 가정 → 1/3씩

    return {
        "total": total, "wins": int(wins), "losses": int(losses),
        "win_rate": float(win_rate),
        "avg_ret": float(avg_ret), "med_ret": float(med_ret),
        "avg_win": float(avg_win), "avg_loss": float(avg_loss),
        "pf": float(pf), "avg_hold": float(avg_hold),
        "cum_return_x": float(cum_simple),
    }


def equity_curve_simulation(trades: list[dict], dates: list[pd.Timestamp], slots: int = 3) -> dict:
    """K슬롯 균등 자본 시뮬레이션. MDD/CAGR 계산."""
    if not trades:
        return {}
    df = pd.DataFrame(trades).sort_values("buy_date").reset_index(drop=True)

    capital = 1.0
    equity = [capital]
    eq_dates = [dates[0]]

    for _, t in df.iterrows():
        capital_used = capital / slots
        trade_pnl = capital_used * t["net_ret"]
        capital += trade_pnl
        equity.append(capital)
        eq_dates.append(t["sell_date"])

    eq_series = pd.Series(equity, index=pd.to_datetime(eq_dates))
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    mdd = dd.min() * 100  # %

    # CAGR
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
    }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run_backtest(daily_data, panel, top3, rule: str,
                 hold_n=None, ma_period=None, stop_pct=None,
                 slots: int = 3, max_concurrent: Optional[int] = None) -> tuple[list[dict], dict]:
    dates = sorted(panel.keys())
    dates = [d for d in dates if d >= pd.Timestamp(START_DATE)]
    trades = simulate_strategy(daily_data, top3, dates, rule,
                               hold_n=hold_n, ma_period=ma_period, stop_pct=stop_pct,
                               max_concurrent=max_concurrent)
    stats = compute_stats(trades)
    eq = equity_curve_simulation(trades, dates, slots=slots)
    stats.update(eq)
    return trades, stats


def yearly_stats(trades: list[dict]) -> dict[int, dict]:
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["buy_date"]).dt.year
    out = {}
    for y, g in df.groupby("year"):
        out[int(y)] = compute_stats(g.to_dict("records"))
    return out


def format_main_table(results: list[dict]) -> list[str]:
    lines = ["\n## 18 조합 비교 (전체 기간 2023~현재)\n"]
    lines.append("| 지표 | 매도규칙 | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 손익비 | 평균보유일 | CAGR(%) | MDD(%) | 누적자본 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["stats"]
        if not s or s.get("total", 0) == 0:
            continue
        lines.append(
            f"| {r['indicator']} | {r['rule_label']} | {s['total']:,} | "
            f"{s['win_rate']:.1f} | {s['avg_ret']:+.2f} | {s['med_ret']:+.2f} | "
            f"{s['pf']:.2f} | {s['avg_hold']:.1f} | "
            f"{s.get('cagr', 0):+.2f} | {s.get('mdd', 0):+.2f} | "
            f"{s.get('final_equity', 1):.2f} |"
        )
    return lines


def format_yearly_table(results: list[dict]) -> list[str]:
    lines = []
    all_years = set()
    for r in results:
        all_years.update(r.get("yearly", {}).keys())
    all_years = sorted(all_years)

    for y in all_years:
        lines.append(f"\n## {y}년 비교\n")
        lines.append("| 지표 | 매도규칙 | 거래수 | 승률(%) | 평균(%) | 손익비 | 누적자본 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in results:
            s = r.get("yearly", {}).get(y, {})
            if not s or s.get("total", 0) == 0:
                lines.append(f"| {r['indicator']} | {r['rule_label']} | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {r['indicator']} | {r['rule_label']} | {s['total']:,} | "
                f"{s['win_rate']:.1f} | {s['avg_ret']:+.2f} | "
                f"{s['pf']:.2f} | {s['cum_return_x']:.2f} |"
            )
    return lines


def main():
    print("="*60)
    print("Top 3 지표 비교 백테스트 - KOSPI200")
    print("="*60)

    print("[1] KOSPI200 ticker 로드...")
    k200 = load_kospi200_tickers()
    print(f"    {len(k200)}종목")

    print("[2] 시가총액 snapshot 로드...")
    snapshot = load_market_cap_snapshot()
    print(f"    {len(snapshot)}종목 shares 정보")

    print("[3] 가격/거래대금 데이터 로드...")
    t0 = time.time()
    price_df = load_price_data(k200["ticker"].tolist())
    print(f"    {len(price_df):,}행 ({time.time()-t0:.1f}초)")

    print("[4] daily_data 빌드...")
    daily_data = build_daily_data(price_df, snapshot)
    print(f"    {len(daily_data)}종목 시계열")

    results = []
    for indicator in INDICATORS:
        print(f"\n[5] panel/top3 빌드: {indicator}")
        panel = build_daily_indicator_panel(daily_data, indicator)
        top3 = compute_top3_per_day(panel, indicator, n=3)
        print(f"    {len(panel)}일치 panel")

        for rule in RULES:
            t0 = time.time()
            if rule.startswith("HOLD_"):
                hold_n = int(rule.split("_")[1])
                trades, stats = run_backtest(daily_data, panel, top3, "HOLD_N", hold_n=hold_n)
                rule_label = rule
            else:
                trades, stats = run_backtest(daily_data, panel, top3, rule)
                rule_label = rule
            elapsed = time.time() - t0
            yr_stats = yearly_stats(trades)
            results.append({
                "indicator": indicator,
                "rule": rule,
                "rule_label": rule_label,
                "stats": stats,
                "yearly": yr_stats,
                "trades": trades,
            })
            print(f"  {rule_label:10s}: {stats.get('total',0):>5}건, "
                  f"승률 {stats.get('win_rate',0):>5.1f}%, "
                  f"CAGR {stats.get('cagr',0):>+6.2f}%, "
                  f"MDD {stats.get('mdd',0):>+7.2f}%, "
                  f"자본 {stats.get('final_equity',1):>5.2f}x | {elapsed:.1f}s")

    # 리포트
    print("\n[6] 리포트 생성...")
    lines = ["# Top 3 지표 비교 백테스트 (KOSPI200)\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **대상**: KOSPI200 199종목 (kospi200_members 2026-04-16 기준)")
    lines.append("- **기간**: 2023-01-01 ~ 현재")
    lines.append("- **매수**: 매일 마감 후 지표 Top3 → 다음날 시가 매수 (보유 중 제외)")
    lines.append("- **매도 규칙**:")
    lines.append("  - LIST_EXIT: Top3 이탈 → 다음날 시가 매도")
    lines.append("  - MA5/MA20: 종가 < MA → 다음날 시가 매도")
    lines.append("  - HOLD_5/10/20: N영업일 보유 후 시가 매도")
    lines.append("- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%")
    lines.append("- **자본 분배**: 3슬롯 균등 (각 매수 자본의 1/3)")
    lines.append("- **시가총액**: ka10001 mac/flo_stk로 상장주식수 추정 → close[d] × 상장주식수\n")
    lines.append("---")
    lines.append("\n## 지표 정의\n")
    lines.append("- **amount**: 거래대금 (원)")
    lines.append("- **mcap**: 시가총액 = close × 상장주식수 (원)")
    lines.append("- **turnover**: 거래회전율 = amount / mcap")

    lines += format_main_table(results)
    lines += format_yearly_table(results)

    # 최우수 조합 상세
    valid = [r for r in results if r["stats"].get("total", 0) > 0]
    if valid:
        best_cagr = max(valid, key=lambda r: r["stats"].get("cagr", -999))
        best_pf = max(valid, key=lambda r: r["stats"].get("pf", 0) if r["stats"].get("pf", 0) != float("inf") else 0)
        best_mdd = max(valid, key=lambda r: r["stats"].get("mdd", -999))
        lines.append("\n## 최우수 조합 요약\n")
        lines.append(f"- **CAGR 최고**: {best_cagr['indicator']} + {best_cagr['rule_label']} → CAGR {best_cagr['stats']['cagr']:+.2f}%, MDD {best_cagr['stats']['mdd']:+.2f}%, 자본 {best_cagr['stats']['final_equity']:.2f}x")
        lines.append(f"- **손익비 최고**: {best_pf['indicator']} + {best_pf['rule_label']} → PF {best_pf['stats']['pf']:.2f}, 승률 {best_pf['stats']['win_rate']:.1f}%")
        lines.append(f"- **MDD 최저**: {best_mdd['indicator']} + {best_mdd['rule_label']} → MDD {best_mdd['stats']['mdd']:+.2f}%, CAGR {best_mdd['stats']['cagr']:+.2f}%")

        # CAGR 최고 상세 거래
        lines.append(f"\n---\n\n## 최우수(CAGR) 상세: {best_cagr['indicator']} + {best_cagr['rule_label']}\n")
        s = best_cagr["stats"]
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
        lines.append(f"| 최종 자본(x) | {s['final_equity']:.2f} |")

        tdf = pd.DataFrame(best_cagr["trades"])
        if not tdf.empty:
            tdf["return_pct"] = tdf["net_ret"] * 100
            # 종목명 매핑
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
