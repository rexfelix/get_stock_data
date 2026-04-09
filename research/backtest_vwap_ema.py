"""
VWAP-EMA 매매 백테스트 - 파라미터 최적화
- 조건 세팅: EMA가 VWAP 상향 돌파 (EMA > VWAP 상태 진입)
- 매수: EMA > VWAP 상태에서, 종가 > EMA (저가 터치 조건 제거)
- 매도: 캔들 몸통이 EMA 완전 이탈 (시가 < EMA AND 종가 < EMA)
- 파라미터:
    EMA 기간: 5, 9, 12, 20, 26
    VWAP 롤링 기간: 10, 20, 30, 60
- 기간: 2023 ~ 현재
"""

import os
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

ENGINE = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

FEE_BUY = 0.00015
FEE_SELL = 0.00015
TAX_SELL = 0.0018

EMA_PERIODS = [5, 9, 12, 20, 26]
VWAP_PERIODS = [10, 20, 30, 60]


def load_kospi200_tickers() -> set[str]:
    with ENGINE.connect() as conn:
        df = pd.read_sql(text("SELECT ticker FROM kospi200_members"), conn)
    return set(df["ticker"])


def load_data(tickers: set[str] | None = None) -> dict[str, pd.DataFrame]:
    query = text("""
        SELECT date, open, high, low, close, volume, ticker, name
        FROM stocks
        WHERE date >= '2022-01-01'
        ORDER BY ticker, date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = {}
    for t, g in df.groupby("ticker"):
        if tickers and t not in tickers:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) >= 70:
            out[t] = g
    return out


def calc_ema(close: np.ndarray, period: int) -> np.ndarray:
    """지수이동평균 계산"""
    ema = np.full(len(close), np.nan)
    if len(close) < period:
        return ema
    # 첫 EMA = SMA
    ema[period - 1] = np.mean(close[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(close)):
        ema[i] = close[i] * k + ema[i - 1] * (1 - k)
    return ema


def calc_rolling_vwap(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """롤링 VWAP = sum(close*volume, period) / sum(volume, period)"""
    n = len(close)
    vwap = np.full(n, np.nan)
    pv = close * volume
    cum_pv = pd.Series(pv).rolling(period).sum().values
    cum_vol = pd.Series(volume).rolling(period).sum().values
    mask = cum_vol > 0
    vwap[mask] = cum_pv[mask] / cum_vol[mask]
    return vwap


def simulate(df: pd.DataFrame, ticker: str, ema_period: int, vwap_period: int) -> list[dict]:
    close = df["close"].values.astype(float)
    open_ = df["open"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)

    ema = calc_ema(close, ema_period)
    vwap = calc_rolling_vwap(close, volume, vwap_period)

    n = len(df)
    trades = []
    in_position = False
    entry_price = 0.0
    buy_date = None
    entry_name = ""

    start = max(ema_period, vwap_period)

    for i in range(start, n):
        if np.isnan(ema[i]) or np.isnan(vwap[i]) or np.isnan(ema[i - 1]) or np.isnan(vwap[i - 1]):
            continue

        # EMA가 VWAP 상향 돌파 감지
        prev_above = ema[i - 1] > vwap[i - 1]
        curr_above = ema[i] > vwap[i]
        crossover = not prev_above and curr_above

        if not in_position:
            # 매수 조건: EMA가 VWAP를 상향 돌파하는 시점
            if crossover:
                entry_price = close[i]
                buy_date = df.iloc[i]["date"]
                entry_name = df.iloc[i]["name"]
                in_position = True
        else:
            # 매도 조건: 캔들 몸통이 EMA 완전 이탈 (시가 < EMA AND 종가 < EMA)
            if open_[i] < ema[i] and close[i] < ema[i]:
                sell_price = close[i]
                row = df.iloc[i]
                hold = (row["date"] - buy_date).days
                gross_ret = (sell_price - entry_price) / entry_price
                net_ret = gross_ret - FEE_BUY - FEE_SELL - TAX_SELL
                trades.append({
                    "ticker": ticker, "name": entry_name,
                    "buy_date": buy_date, "buy_price": entry_price,
                    "sell_date": row["date"], "sell_price": sell_price,
                    "return_pct": net_ret * 100, "hold_days": hold,
                })
                in_position = False

    # 미청산
    if in_position:
        last = df.iloc[-1]
        hold = (last["date"] - buy_date).days
        gross_ret = (last["close"] - entry_price) / entry_price
        net_ret = gross_ret - FEE_BUY - FEE_SELL - TAX_SELL
        trades.append({
            "ticker": ticker, "name": entry_name,
            "buy_date": buy_date, "buy_price": entry_price,
            "sell_date": last["date"], "sell_price": last["close"],
            "return_pct": net_ret * 100, "hold_days": hold,
            "reason": "미청산",
        })

    return trades


def stats(sub: pd.DataFrame) -> dict:
    total = len(sub)
    if total == 0:
        return {}
    wins = (sub["return_pct"] > 0).sum()
    losses = total - wins
    win_rate = wins / total * 100
    avg_ret = sub["return_pct"].mean()
    med_ret = sub["return_pct"].median()
    avg_win = sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
    avg_loss = sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if losses > 0 else 0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    avg_hold = sub["hold_days"].mean()
    total_return = (1 + sub["return_pct"] / 100).prod()
    return {
        "total": total, "wins": wins, "losses": losses, "win_rate": win_rate,
        "avg_ret": avg_ret, "med_ret": med_ret, "avg_win": avg_win,
        "avg_loss": avg_loss, "pf": pf, "avg_hold": avg_hold,
        "total_return": total_return,
    }


def main():
    print("KOSPI200 종목 로딩...")
    k200 = load_kospi200_tickers()
    print(f"  KOSPI200: {len(k200)}종목")
    stocks = load_data(tickers=k200)
    print(f"  데이터 로드: {len(stocks):,}종목\n")

    results = []
    best_dfs = {}
    total_combos = len(EMA_PERIODS) * len(VWAP_PERIODS)
    combo_idx = 0

    for ep in EMA_PERIODS:
        for vp in VWAP_PERIODS:
            combo_idx += 1
            t0 = time.time()
            all_trades = []
            for ticker, df in stocks.items():
                all_trades.extend(simulate(df, ticker, ep, vp))

            if not all_trades:
                continue

            tdf = pd.DataFrame(all_trades)
            tdf["buy_date"] = pd.to_datetime(tdf["buy_date"])
            tdf["sell_date"] = pd.to_datetime(tdf["sell_date"])
            tdf["buy_year"] = tdf["buy_date"].dt.year
            tdf = tdf[(tdf["buy_year"] >= 2023) & np.isfinite(tdf["return_pct"])]

            # 전체
            s = stats(tdf)
            elapsed = time.time() - t0
            if s:
                key = (ep, vp)
                results.append({"ema": ep, "vwap": vp, "filter": "전체", **s})
                best_dfs[key] = tdf

                # 단기(5~20일) 필터
                short = tdf[(tdf["hold_days"] >= 5) & (tdf["hold_days"] <= 20)]
                s_short = stats(short)
                if s_short:
                    results.append({"ema": ep, "vwap": vp, "filter": "단기(5~20일)", **s_short})

            print(f"  [{combo_idx:>2}/{total_combos}] EMA={ep:>2}, VWAP={vp:>2}: "
                  f"{s.get('total', 0):>6}건, "
                  f"승률 {s.get('win_rate', 0):>5.1f}%, "
                  f"평균 {s.get('avg_ret', 0):>+7.2f}%, "
                  f"중간 {s.get('med_ret', 0):>+7.2f}%, "
                  f"PF {s.get('pf', 0):>5.2f}, "
                  f"보유 {s.get('avg_hold', 0):>5.1f}일 | {elapsed:.1f}초")

    if not results:
        print("결과 없음!")
        return

    rdf = pd.DataFrame(results)

    # ── 리포트 ──
    lines = ["# VWAP-EMA 매매 백테스트 - KOSPI200 파라미터 최적화\n"]
    lines.append("## 매매 규칙\n")
    lines.append("1. **조건 세팅**: EMA가 롤링 VWAP를 상향 돌파 (EMA > VWAP 상태)")
    lines.append("2. **매수**: EMA가 VWAP를 상향 돌파하는 시점 (당일 종가 매수)")
    lines.append("3. **매도**: 캔들 몸통이 EMA 완전 이탈 (시가 < EMA AND 종가 < EMA)")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% + 매도 {FEE_SELL*100:.3f}% + 세금 {TAX_SELL*100:.2f}%")
    lines.append("- **기간**: 2023 ~ 현재")
    lines.append(f"- **EMA 기간**: {EMA_PERIODS}")
    lines.append(f"- **VWAP 롤링 기간**: {VWAP_PERIODS}\n")
    lines.append("---\n")

    # 전체 거래
    lines.append("## 1. 전체 거래 파라미터 비교 (중간값 수익률 순)\n")
    full = rdf[rdf["filter"] == "전체"].sort_values("med_ret", ascending=False)
    lines.append("| EMA | VWAP | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 이익(%) | 손실(%) | PF | 보유일 | 누적(x) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in full.iterrows():
        cum = r['total_return']
        cum_str = f"{cum:.2f}" if cum < 1e6 else f"{cum:.2e}"
        lines.append(
            f"| {r['ema']:.0f} | {r['vwap']:.0f} | {r['total']:,.0f} | {r['win_rate']:.1f} "
            f"| {r['avg_ret']:+.2f} | {r['med_ret']:+.2f} "
            f"| {r['avg_win']:+.2f} | {r['avg_loss']:+.2f} "
            f"| {r['pf']:.2f} | {r['avg_hold']:.1f} "
            f"| {cum_str} |"
        )

    # 단기 거래
    lines.append("\n## 2. 단기(5~20일) 거래 파라미터 비교\n")
    short = rdf[rdf["filter"] == "단기(5~20일)"].sort_values("med_ret", ascending=False)
    if not short.empty:
        lines.append("| EMA | VWAP | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 이익(%) | 손실(%) | PF | 보유일 | 누적(x) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in short.iterrows():
            cum = r['total_return']
            cum_str = f"{cum:.2f}" if cum < 1e6 else f"{cum:.2e}"
            lines.append(
                f"| {r['ema']:.0f} | {r['vwap']:.0f} | {r['total']:,.0f} | {r['win_rate']:.1f} "
                f"| {r['avg_ret']:+.2f} | {r['med_ret']:+.2f} "
                f"| {r['avg_win']:+.2f} | {r['avg_loss']:+.2f} "
                f"| {r['pf']:.2f} | {r['avg_hold']:.1f} "
                f"| {cum_str} |"
            )

    # 최적 요약
    lines.append("\n## 3. 최적 파라미터 요약\n")

    lines.append("### 전체 거래")
    if not full.empty:
        bm = full.iloc[0]
        ba = full.sort_values("avg_ret", ascending=False).iloc[0]
        bw = full.sort_values("win_rate", ascending=False).iloc[0]
        lines.append(f"- **중간값 최고**: EMA={bm['ema']:.0f}, VWAP={bm['vwap']:.0f} → 중간값 {bm['med_ret']:+.2f}%, 승률 {bm['win_rate']:.1f}%")
        lines.append(f"- **평균 최고**: EMA={ba['ema']:.0f}, VWAP={ba['vwap']:.0f} → 평균 {ba['avg_ret']:+.2f}%")
        lines.append(f"- **승률 최고**: EMA={bw['ema']:.0f}, VWAP={bw['vwap']:.0f} → 승률 {bw['win_rate']:.1f}%")

    lines.append("\n### 단기(5~20일) 거래")
    if not short.empty:
        bm = short.iloc[0]
        ba = short.sort_values("avg_ret", ascending=False).iloc[0]
        bw = short.sort_values("win_rate", ascending=False).iloc[0]
        lines.append(f"- **중간값 최고**: EMA={bm['ema']:.0f}, VWAP={bm['vwap']:.0f} → 중간값 {bm['med_ret']:+.2f}%, 승률 {bm['win_rate']:.1f}%")
        lines.append(f"- **평균 최고**: EMA={ba['ema']:.0f}, VWAP={ba['vwap']:.0f} → 평균 {ba['avg_ret']:+.2f}%")
        lines.append(f"- **승률 최고**: EMA={bw['ema']:.0f}, VWAP={bw['vwap']:.0f} → 승률 {bw['win_rate']:.1f}%")

    # 최적 상세 (전체 기준 중간값 best)
    full_best = rdf[rdf["filter"] == "전체"].sort_values("med_ret", ascending=False).iloc[0]
    bkey = (int(full_best["ema"]), int(full_best["vwap"]))
    if bkey in best_dfs:
        bdf = best_dfs[bkey]
        s = stats(bdf)

        lines.append(f"\n---\n\n# 최적 파라미터 상세: EMA={bkey[0]}, VWAP={bkey[1]}\n")
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
        cum = s['total_return']
        cum_str = f"{cum:.2f}x" if cum < 1e6 else f"{cum:.2e}x"
        lines.append(f"| 누적수익률 | {cum_str} |")

        # 연도별
        lines.append(f"\n## 연도별 성과\n")
        lines.append("| 연도 | 거래수 | 승률(%) | 평균(%) | 중간값(%) | PF | 보유일 | 누적(x) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for y in sorted(bdf["buy_year"].unique()):
            ys = stats(bdf[bdf["buy_year"] == y])
            if ys:
                yc = ys['total_return']
                yc_str = f"{yc:.2f}" if yc < 1e6 else f"{yc:.2e}"
                lines.append(
                    f"| {y} | {ys['total']:,} | {ys['win_rate']:.1f} "
                    f"| {ys['avg_ret']:+.2f} | {ys['med_ret']:+.2f} "
                    f"| {ys['pf']:.2f} | {ys['avg_hold']:.1f} | {yc_str} |"
                )

        # Top/Bottom 20
        lines.append("\n## 수익률 상위 거래 (Top 20)\n")
        lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
        lines.append("|---|---|---|---:|---:|")
        for _, r in bdf.nlargest(20, "return_pct").iterrows():
            lines.append(
                f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
                f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
                f"| {r['return_pct']:+.1f} |"
            )

        lines.append("\n## 수익률 하위 거래 (Bottom 20)\n")
        lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
        lines.append("|---|---|---|---:|---:|")
        for _, r in bdf.nsmallest(20, "return_pct").iterrows():
            lines.append(
                f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
                f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
                f"| {r['return_pct']:+.1f} |"
            )

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_vwap_ema_k200_v3.md", "w") as f:
        f.write(report)
        f.write("\n")
    print("\n결과 저장: results/backtest_vwap_ema_k200_v3.md")


if __name__ == "__main__":
    main()
