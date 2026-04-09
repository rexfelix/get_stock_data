"""
DSB (Disparity Squeeze Breakout) 매매 백테스트 - 파라미터 최적화
- 이격도 밴드폭 = StdDev(MA5,MA10,MA20,MA60) / Avg * 100
- 스무딩: 밴드폭 5일 SMA
- 백분위순위: 60일 롤링 퍼센타일
- 스퀴즈: 백분위 < squeezePct
- 매수: 스퀴즈 → 확장 전환 + 종가 > MA20
- 매도: 트레일링 스톱 / 밴드폭 재수렴 / 최대보유일 초과 손실
- 파라미터 스윕:
    squeezePct: 15, 20, 25, 30, 35
    holdDays: 10, 15, 20, 30
    trailPct: 3, 5, 7, 10
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

SQUEEZE_PCTS = [15, 20, 25, 30, 35]
HOLD_DAYS_LIST = [10, 15, 20, 30]
TRAIL_PCTS = [3, 5, 7, 10]


def load_data() -> dict[str, pd.DataFrame]:
    query = text("""
        SELECT date, open, close, ticker, name
        FROM stocks
        WHERE date >= '2022-01-01'
        ORDER BY ticker, date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) >= 80:  # MA60 + 여유
            out[t] = g
    return out


def precompute_ma(close: np.ndarray):
    """MA5, MA10, MA20, MA60 사전 계산"""
    s = pd.Series(close)
    return s.rolling(5).mean().values, s.rolling(10).mean().values, \
           s.rolling(20).mean().values, s.rolling(60).mean().values


def calc_bw_rank(close: np.ndarray, ma5, ma10, ma20, ma60):
    """밴드폭 → 5일 스무딩 → 60일 롤링 백분위순위"""
    n = len(close)

    # 밴드폭: 4개 이평선의 표준편차 / 평균 * 100
    bandwidth = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(ma60[i]):
            continue
        vals = np.array([ma5[i], ma10[i], ma20[i], ma60[i]])
        avg = vals.mean()
        if avg > 0:
            bandwidth[i] = (vals.std(ddof=0) / avg) * 100

    # 5일 스무딩
    bw_smooth = pd.Series(bandwidth).rolling(5).mean().values

    # 60일 롤링 백분위순위
    bw_rank = np.full(n, np.nan)
    for i in range(59, n):
        if np.isnan(bw_smooth[i]):
            continue
        window = bw_smooth[i - 59:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 10:
            continue
        below = np.sum(valid <= bw_smooth[i])
        bw_rank[i] = (below / len(valid)) * 100

    return bw_rank


def simulate(df: pd.DataFrame, ticker: str, bw_rank: np.ndarray,
             ma20: np.ndarray, squeeze_pct: float, hold_days: int,
             trail_pct: float) -> list[dict]:
    close = df["close"].values.astype(float)
    n = len(df)
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    peak_price = 0.0
    entry_name = ""
    buy_date = None

    for i in range(1, n):
        if np.isnan(bw_rank[i]) or np.isnan(bw_rank[i - 1]) or np.isnan(ma20[i]):
            continue

        if not in_position:
            # 스퀴즈 → 확장 전환: 전일 스퀴즈, 당일 확장 + 방향 상승
            was_squeeze = bw_rank[i - 1] < squeeze_pct
            now_expanding = bw_rank[i] >= squeeze_pct and bw_rank[i] > bw_rank[i - 1]
            direction_up = close[i] > ma20[i]

            if was_squeeze and now_expanding and direction_up:
                entry_price = close[i]
                entry_idx = i
                peak_price = close[i]
                entry_name = df.iloc[i]["name"]
                buy_date = df.iloc[i]["date"]
                in_position = True
        else:
            peak_price = max(peak_price, close[i])
            days_held = i - entry_idx
            drawdown = (peak_price - close[i]) / peak_price * 100

            # 밴드폭 재수렴: rank 하락 + rank < 50 + 5일 이상 보유
            bw_contracting = (bw_rank[i] < bw_rank[i - 1] and
                              bw_rank[i] < 50 and days_held > 5)

            # 청산 조건
            exit_trail = drawdown > trail_pct
            exit_bw = bw_contracting
            exit_hold = days_held > hold_days and close[i] <= entry_price

            if exit_trail or exit_bw or exit_hold:
                sell_price = close[i]
                row = df.iloc[i]
                hold = (row["date"] - buy_date).days
                gross_ret = (sell_price - entry_price) / entry_price
                net_ret = gross_ret - FEE_BUY - FEE_SELL - TAX_SELL
                reason = "trail" if exit_trail else ("bw_contract" if exit_bw else "max_hold")
                trades.append({
                    "ticker": ticker, "name": entry_name,
                    "buy_date": buy_date, "buy_price": entry_price,
                    "sell_date": row["date"], "sell_price": sell_price,
                    "return_pct": net_ret * 100, "hold_days": hold,
                    "reason": reason,
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
    print("데이터 로딩...")
    stocks = load_data()
    print(f"  {len(stocks):,}종목 로드")

    # 사전계산: MA, bw_rank (파라미터 무관)
    print("이격도 밴드폭/백분위 사전 계산...")
    t0 = time.time()
    precomp = {}
    for ticker, df in stocks.items():
        close = df["close"].values.astype(float)
        ma5, ma10, ma20, ma60 = precompute_ma(close)
        bw_rank = calc_bw_rank(close, ma5, ma10, ma20, ma60)
        precomp[ticker] = {"ma20": ma20, "bw_rank": bw_rank}
    print(f"  사전 계산 완료: {time.time() - t0:.1f}초\n")

    # 파라미터 스윕
    results = []
    best_dfs = {}

    total_combos = len(SQUEEZE_PCTS) * len(HOLD_DAYS_LIST) * len(TRAIL_PCTS)
    combo_idx = 0

    for sq in SQUEEZE_PCTS:
        for hd in HOLD_DAYS_LIST:
            for tp in TRAIL_PCTS:
                combo_idx += 1
                t0 = time.time()
                all_trades = []
                for ticker, df in stocks.items():
                    pc = precomp[ticker]
                    all_trades.extend(
                        simulate(df, ticker, pc["bw_rank"], pc["ma20"], sq, hd, tp)
                    )

                if not all_trades:
                    continue

                tdf = pd.DataFrame(all_trades)
                tdf["buy_date"] = pd.to_datetime(tdf["buy_date"])
                tdf["sell_date"] = pd.to_datetime(tdf["sell_date"])
                tdf["buy_year"] = tdf["buy_date"].dt.year
                tdf = tdf[(tdf["buy_year"] >= 2023) & np.isfinite(tdf["return_pct"])]

                s = stats(tdf)
                elapsed = time.time() - t0
                if s:
                    key = (sq, hd, tp)
                    results.append({"sq": sq, "hd": hd, "tp": tp, **s})
                    best_dfs[key] = tdf

                print(f"  [{combo_idx:>2}/{total_combos}] sq={sq:>2}%, hd={hd:>2}d, tp={tp:>2}%: "
                      f"{s.get('total', 0):>5}건, "
                      f"승률 {s.get('win_rate', 0):>5.1f}%, "
                      f"평균 {s.get('avg_ret', 0):>+7.2f}%, "
                      f"중간 {s.get('med_ret', 0):>+7.2f}%, "
                      f"PF {s.get('pf', 0):>5.2f}, "
                      f"보유 {s.get('avg_hold', 0):>5.1f}일 | {elapsed:.1f}초")

    if not results:
        print("결과 없음!")
        return

    rdf = pd.DataFrame(results)

    # ── 리포트 생성 ──
    lines = ["# DSB (Disparity Squeeze Breakout) 매매 백테스트 - 파라미터 최적화\n"]
    lines.append("## 매매 규칙\n")
    lines.append("```")
    lines.append("밴드폭 = StdDev(MA5, MA10, MA20, MA60) / Avg(MA5,MA10,MA20,MA60) × 100")
    lines.append("스무딩밴드폭 = SMA(밴드폭, 5)")
    lines.append("백분위순위 = 60일 롤링 Percentile Rank(스무딩밴드폭)")
    lines.append("스퀴즈 = 백분위순위 < squeezePct")
    lines.append("```")
    lines.append("- **매수**: 전일 스퀴즈 + 당일 확장(순위 상승 + ≥ 임계값) + 종가 > MA20")
    lines.append("- **매도 (3가지 중 하나)**:")
    lines.append("  1. 트레일링 스톱: 고점 대비 trailPct% 하락")
    lines.append("  2. 밴드폭 재수렴: 순위 하락 + 순위 < 50% + 5일 이상 보유")
    lines.append("  3. 최대보유일 초과 + 손실 상태")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% + 매도 {FEE_SELL*100:.3f}% + 세금 {TAX_SELL*100:.2f}%")
    lines.append("- **기간**: 2023 ~ 현재\n")
    lines.append(f"- **squeezePct**: {SQUEEZE_PCTS}")
    lines.append(f"- **holdDays**: {HOLD_DAYS_LIST}")
    lines.append(f"- **trailPct**: {TRAIL_PCTS}\n")
    lines.append("---\n")

    # 전체 파라미터 비교 (중간값 수익률 내림차순)
    lines.append("## 파라미터 비교 (중간값 수익률 순)\n")
    sorted_rdf = rdf.sort_values("med_ret", ascending=False)
    lines.append("| 순위 | sq% | hold | trail% | 거래수 | 승률(%) | 평균(%) | 중간값(%) | 이익(%) | 손실(%) | PF | 보유일 | 누적(x) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rank, (_, r) in enumerate(sorted_rdf.iterrows(), 1):
        lines.append(
            f"| {rank} | {r['sq']:.0f} | {r['hd']:.0f} | {r['tp']:.0f} "
            f"| {r['total']:,.0f} | {r['win_rate']:.1f} "
            f"| {r['avg_ret']:+.2f} | {r['med_ret']:+.2f} "
            f"| {r['avg_win']:+.2f} | {r['avg_loss']:+.2f} "
            f"| {r['pf']:.2f} | {r['avg_hold']:.1f} "
            f"| {r['total_return']:.4f} |"
        )

    # 최적 파라미터 요약
    best_med = sorted_rdf.iloc[0]
    best_avg = rdf.sort_values("avg_ret", ascending=False).iloc[0]
    best_wr = rdf.sort_values("win_rate", ascending=False).iloc[0]
    finite_pf = rdf[rdf["pf"] < float("inf")]
    best_pf = finite_pf.sort_values("pf", ascending=False).iloc[0] if len(finite_pf) > 0 else None

    lines.append("\n## 최적 파라미터 요약\n")
    lines.append(f"- **중간값 수익률 최고**: sq={best_med['sq']:.0f}%, hold={best_med['hd']:.0f}, trail={best_med['tp']:.0f}% → 중간값 {best_med['med_ret']:+.2f}%, 승률 {best_med['win_rate']:.1f}%")
    lines.append(f"- **평균 수익률 최고**: sq={best_avg['sq']:.0f}%, hold={best_avg['hd']:.0f}, trail={best_avg['tp']:.0f}% → 평균 {best_avg['avg_ret']:+.2f}%")
    lines.append(f"- **승률 최고**: sq={best_wr['sq']:.0f}%, hold={best_wr['hd']:.0f}, trail={best_wr['tp']:.0f}% → 승률 {best_wr['win_rate']:.1f}%")
    if best_pf is not None:
        lines.append(f"- **손익비 최고**: sq={best_pf['sq']:.0f}%, hold={best_pf['hd']:.0f}, trail={best_pf['tp']:.0f}% → PF {best_pf['pf']:.2f}")

    # 최적 파라미터 상세 (중간값 기준)
    bkey = (int(best_med["sq"]), int(best_med["hd"]), int(best_med["tp"]))
    if bkey in best_dfs:
        bdf = best_dfs[bkey]
        s = stats(bdf)

        lines.append(f"\n---\n\n# 최적 파라미터 상세: sq={bkey[0]}%, hold={bkey[1]}, trail={bkey[2]}%\n")
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
        lines.append(f"| 누적수익률 | {s['total_return']:.4f}x |")

        # 청산 사유 분포
        if "reason" in bdf.columns:
            lines.append("\n## 청산 사유 분포\n")
            reason_counts = bdf["reason"].value_counts()
            lines.append("| 사유 | 건수 | 비율(%) |")
            lines.append("|---|---:|---:|")
            for reason, cnt in reason_counts.items():
                lines.append(f"| {reason} | {cnt:,} | {cnt/len(bdf)*100:.1f} |")

        # 연도별 통계
        lines.append(f"\n## 연도별 성과\n")
        lines.append("| 연도 | 거래수 | 승률(%) | 평균(%) | 중간값(%) | PF | 누적(x) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for y in sorted(bdf["buy_year"].unique()):
            ys = stats(bdf[bdf["buy_year"] == y])
            if ys:
                lines.append(
                    f"| {y} | {ys['total']:,} | {ys['win_rate']:.1f} "
                    f"| {ys['avg_ret']:+.2f} | {ys['med_ret']:+.2f} "
                    f"| {ys['pf']:.2f} | {ys['total_return']:.4f} |"
                )

        # 수익률 상위/하위
        lines.append("\n## 수익률 상위 거래 (Top 20)\n")
        lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 사유 | 수익률(%) |")
        lines.append("|---|---|---|---:|---|---:|")
        for _, r in bdf.nlargest(20, "return_pct").iterrows():
            lines.append(
                f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
                f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
                f"| {r.get('reason','')} | {r['return_pct']:+.1f} |"
            )

        lines.append("\n## 수익률 하위 거래 (Bottom 20)\n")
        lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 사유 | 수익률(%) |")
        lines.append("|---|---|---|---:|---|---:|")
        for _, r in bdf.nsmallest(20, "return_pct").iterrows():
            lines.append(
                f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
                f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
                f"| {r.get('reason','')} | {r['return_pct']:+.1f} |"
            )

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_dsb.md", "w") as f:
        f.write(report)
        f.write("\n")
    print("\n결과 저장: results/backtest_dsb.md")


if __name__ == "__main__":
    main()
