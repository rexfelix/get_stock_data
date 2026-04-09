"""
DSL (Disparity Support Line) 매매 백테스트 - 승수/스무딩 파라미터 최적화
- DSL = 이평평균 - 승수 * 스무딩이격 * 이평평균 / 100
- 매수: 종가가 DSL 상향 돌파 (전일 종가 < 전일 DSL, 당일 종가 >= 당일 DSL)
- 매도: 종가가 DSL 하향 돌파 (전일 종가 >= 전일 DSL, 당일 종가 < 당일 DSL)
- 단기 매매 (5~20일) 최적 파라미터 탐색
- 승수: 0.5 ~ 3.0 (0.5 단위)
- 스무딩: 3, 5, 7, 10, 14, 20
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

MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SMOOTHINGS = [3, 5, 7, 10, 14, 20]
MIN_HOLD = 5
MAX_HOLD = 20


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
        if len(g) >= 30:
            out[t] = g
    return out


def calc_dsl(close: np.ndarray, multiplier: float, smoothing: int) -> np.ndarray:
    n = len(close)
    dsl = np.full(n, np.nan)

    # MA5, MA10, MA20
    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values

    # 이평평균
    ma_avg = (ma5 + ma10 + ma20) / 3.0

    # 이평분산 → 이평이격도
    ma_var = ((ma5 - ma_avg)**2 + (ma10 - ma_avg)**2 + (ma20 - ma_avg)**2) / 3.0
    disp = np.where(ma_avg > 0, np.sqrt(ma_var) / ma_avg * 100, 0.0)

    # 스무딩이격 = 이평이격도의 smoothing일 이동평균
    smooth_disp = pd.Series(disp).rolling(smoothing).mean().values

    # DSL = 이평평균 - 승수 * 스무딩이격 * 이평평균 / 100
    dsl = ma_avg - multiplier * smooth_disp * ma_avg / 100.0

    return dsl


def simulate(df: pd.DataFrame, ticker: str, multiplier: float, smoothing: int) -> list[dict]:
    close = df["close"].values.astype(float)
    dsl = calc_dsl(close, multiplier, smoothing)

    n = len(df)
    trades = []
    in_position = False
    buy_price = 0.0
    buy_date = None
    entry_name = ""

    start_idx = 20 + smoothing  # DSL이 유효해지는 시점

    for i in range(start_idx, n):
        c_today = close[i]
        c_prev = close[i - 1]
        dsl_today = dsl[i]
        dsl_prev = dsl[i - 1]

        if np.isnan(dsl_today) or np.isnan(dsl_prev):
            continue

        if not in_position:
            # 매수: 전일 종가 < 전일 DSL, 당일 종가 >= 당일 DSL (상향 돌파)
            if c_prev < dsl_prev and c_today >= dsl_today:
                row = df.iloc[i]
                buy_price = c_today
                buy_date = row["date"]
                entry_name = row["name"]
                in_position = True
        else:
            # 매도: 전일 종가 >= 전일 DSL, 당일 종가 < 당일 DSL (하향 돌파)
            if c_prev >= dsl_prev and c_today < dsl_today:
                row = df.iloc[i]
                sell_price = c_today
                hold_days = (row["date"] - buy_date).days
                gross_ret = (sell_price - buy_price) / buy_price
                net_ret = gross_ret - FEE_BUY - FEE_SELL - TAX_SELL
                trades.append({
                    "ticker": ticker,
                    "name": entry_name,
                    "buy_date": buy_date,
                    "buy_price": buy_price,
                    "sell_date": row["date"],
                    "sell_price": sell_price,
                    "return_pct": net_ret * 100,
                    "hold_days": hold_days,
                })
                in_position = False

    # 미청산 포지션
    if in_position:
        last = df.iloc[-1]
        hold_days = (last["date"] - buy_date).days
        gross_ret = (last["close"] - buy_price) / buy_price
        net_ret = gross_ret - FEE_BUY - FEE_SELL - TAX_SELL
        trades.append({
            "ticker": ticker,
            "name": entry_name,
            "buy_date": buy_date,
            "buy_price": buy_price,
            "sell_date": last["date"],
            "sell_price": last["close"],
            "return_pct": net_ret * 100,
            "hold_days": hold_days,
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


def run_sweep(stocks: dict, multiplier: float, smoothing: int) -> pd.DataFrame:
    all_trades = []
    for ticker, df in stocks.items():
        all_trades.extend(simulate(df, ticker, multiplier, smoothing))
    if not all_trades:
        return pd.DataFrame()
    df = pd.DataFrame(all_trades)
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df["sell_date"] = pd.to_datetime(df["sell_date"])
    df["buy_year"] = df["buy_date"].dt.year
    # 2023년 이후, 유효 수익률만
    df = df[(df["buy_year"] >= 2023) & np.isfinite(df["return_pct"])]
    return df


def main():
    print("데이터 로딩...")
    stocks = load_data()
    print(f"  {len(stocks):,}종목 로드\n")

    # === 전체 거래 파라미터 스윕 ===
    all_results = []
    all_dfs = {}

    for mult in MULTIPLIERS:
        for smooth in SMOOTHINGS:
            t0 = time.time()
            df = run_sweep(stocks, mult, smooth)
            elapsed = time.time() - t0
            s = stats(df)
            if s:
                key = (mult, smooth)
                all_results.append({"mult": mult, "smooth": smooth, "filter": "전체", **s})
                all_dfs[key] = df

                # 단기(5~20일) 필터링 버전
                short = df[(df["hold_days"] >= MIN_HOLD) & (df["hold_days"] <= MAX_HOLD)]
                s_short = stats(short)
                if s_short:
                    all_results.append({"mult": mult, "smooth": smooth, "filter": "단기(5~20일)", **s_short})

            print(f"  승수={mult:.1f}, 스무딩={smooth:>2}: "
                  f"{s.get('total', 0):>5}건, "
                  f"승률 {s.get('win_rate', 0):>5.1f}%, "
                  f"평균 {s.get('avg_ret', 0):>+7.2f}%, "
                  f"PF {s.get('pf', 0):>5.2f}, "
                  f"보유 {s.get('avg_hold', 0):>5.1f}일, "
                  f"누적 {s.get('total_return', 0):>8.2f}x | {elapsed:.1f}초")

    if not all_results:
        print("결과 없음!")
        return

    results_df = pd.DataFrame(all_results)

    # ── 리포트 생성 ──
    lines = ["# DSL (Disparity Support Line) 매매 백테스트 - 파라미터 최적화\n"]
    lines.append("## 매매 규칙\n")
    lines.append("```")
    lines.append("MA5 = ma(C, 5);  MA10 = ma(C, 10);  MA20 = ma(C, 20);")
    lines.append("이평평균 = (MA5 + MA10 + MA20) / 3;")
    lines.append("이평분산 = (pow(MA5-이평평균,2) + pow(MA10-이평평균,2) + pow(MA20-이평평균,2)) / 3;")
    lines.append("이평이격도 = sqrt(이평분산) / 이평평균 * 100;")
    lines.append("스무딩이격 = MA(이평이격도, 스무딩);")
    lines.append("DSL = 이평평균 - 승수 * 스무딩이격 * 이평평균 / 100;")
    lines.append("```")
    lines.append("- **매수**: 종가가 DSL 상향 돌파 (전일 < DSL → 당일 >= DSL)")
    lines.append("- **매도**: 종가가 DSL 하향 돌파 (전일 >= DSL → 당일 < DSL)")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% + 매도 {FEE_SELL*100:.3f}% + 세금 {TAX_SELL*100:.2f}%")
    lines.append("- **기간**: 2023 ~ 현재")
    lines.append(f"- **승수 스윕**: {MULTIPLIERS}")
    lines.append(f"- **스무딩 스윕**: {SMOOTHINGS}")
    lines.append(f"- **단기 필터**: 보유일 {MIN_HOLD}~{MAX_HOLD}일\n")
    lines.append("---\n")

    # == 전체 거래 파라미터 비교 ==
    lines.append("## 1. 전체 거래 파라미터 비교\n")
    full = results_df[results_df["filter"] == "전체"].sort_values("med_ret", ascending=False)
    lines.append("| 승수 | 스무딩 | 거래수 | 승률(%) | 평균수익(%) | 중간값(%) | 평균이익(%) | 평균손실(%) | 손익비 | 평균보유일 | 누적(x) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in full.iterrows():
        lines.append(
            f"| {r['mult']:.1f} | {r['smooth']:.0f} | {r['total']:,.0f} | {r['win_rate']:.1f} "
            f"| {r['avg_ret']:+.2f} | {r['med_ret']:+.2f} "
            f"| {r['avg_win']:+.2f} | {r['avg_loss']:+.2f} "
            f"| {r['pf']:.2f} | {r['avg_hold']:.1f} "
            f"| {r['total_return']:.2f} |"
        )

    # == 단기(5~20일) 파라미터 비교 ==
    lines.append("\n## 2. 단기(5~20일) 거래만 필터링\n")
    short = results_df[results_df["filter"] == "단기(5~20일)"].sort_values("med_ret", ascending=False)
    if not short.empty:
        lines.append("| 승수 | 스무딩 | 거래수 | 승률(%) | 평균수익(%) | 중간값(%) | 평균이익(%) | 평균손실(%) | 손익비 | 평균보유일 | 누적(x) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in short.iterrows():
            lines.append(
                f"| {r['mult']:.1f} | {r['smooth']:.0f} | {r['total']:,.0f} | {r['win_rate']:.1f} "
                f"| {r['avg_ret']:+.2f} | {r['med_ret']:+.2f} "
                f"| {r['avg_win']:+.2f} | {r['avg_loss']:+.2f} "
                f"| {r['pf']:.2f} | {r['avg_hold']:.1f} "
                f"| {r['total_return']:.2f} |"
            )
    else:
        lines.append("단기 거래 없음\n")

    # == 최적 파라미터 요약 ==
    lines.append("\n## 3. 최적 파라미터 요약\n")

    # 전체 기준
    if not full.empty:
        best_full_med = full.iloc[0]
        best_full_cum = full.sort_values("total_return", ascending=False).iloc[0]
        best_full_pf = full[full["pf"] < float("inf")].sort_values("pf", ascending=False).iloc[0] if len(full[full["pf"] < float("inf")]) > 0 else None
        lines.append("### 전체 거래 기준")
        lines.append(f"- **중간값 수익률 최고**: 승수={best_full_med['mult']:.1f}, 스무딩={best_full_med['smooth']:.0f} → 중간값 {best_full_med['med_ret']:+.2f}%")
        lines.append(f"- **누적수익 최고**: 승수={best_full_cum['mult']:.1f}, 스무딩={best_full_cum['smooth']:.0f} → {best_full_cum['total_return']:.2f}x")
        if best_full_pf is not None:
            lines.append(f"- **손익비 최고**: 승수={best_full_pf['mult']:.1f}, 스무딩={best_full_pf['smooth']:.0f} → PF {best_full_pf['pf']:.2f}")

    # 단기 기준
    if not short.empty:
        best_short_med = short.iloc[0]
        best_short_cum = short.sort_values("total_return", ascending=False).iloc[0]
        best_short_pf = short[short["pf"] < float("inf")].sort_values("pf", ascending=False).iloc[0] if len(short[short["pf"] < float("inf")]) > 0 else None
        lines.append("\n### 단기(5~20일) 거래 기준")
        lines.append(f"- **중간값 수익률 최고**: 승수={best_short_med['mult']:.1f}, 스무딩={best_short_med['smooth']:.0f} → 중간값 {best_short_med['med_ret']:+.2f}%")
        lines.append(f"- **누적수익 최고**: 승수={best_short_cum['mult']:.1f}, 스무딩={best_short_cum['smooth']:.0f} → {best_short_cum['total_return']:.2f}x")
        if best_short_pf is not None:
            lines.append(f"- **손익비 최고**: 승수={best_short_pf['mult']:.1f}, 스무딩={best_short_pf['smooth']:.0f} → PF {best_short_pf['pf']:.2f}")

    # == 최적 단기 파라미터 상세 ==
    if not short.empty:
        best = short.iloc[0]
        bm, bs = best["mult"], int(best["smooth"])
        key = (bm, bs)
        if key in all_dfs:
            bdf = all_dfs[key]
            bdf_short = bdf[(bdf["hold_days"] >= MIN_HOLD) & (bdf["hold_days"] <= MAX_HOLD)]
            s = stats(bdf_short)

            lines.append(f"\n---\n\n# 최적 단기 파라미터 상세: 승수={bm:.1f}, 스무딩={bs}\n")
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
            lines.append(f"| 누적수익률(배수) | {s['total_return']:.2f}x |")

            # 연도별 통계
            lines.append(f"\n## 연도별 성과 (승수={bm:.1f}, 스무딩={bs}, 단기 5~20일)\n")
            lines.append("| 연도 | 거래수 | 승률(%) | 평균수익(%) | 중간값(%) | 손익비 | 누적(x) |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|")
            for y in sorted(bdf_short["buy_year"].unique()):
                ys = stats(bdf_short[bdf_short["buy_year"] == y])
                if ys:
                    lines.append(
                        f"| {y} | {ys['total']:,} | {ys['win_rate']:.1f} "
                        f"| {ys['avg_ret']:+.2f} | {ys['med_ret']:+.2f} "
                        f"| {ys['pf']:.2f} | {ys['total_return']:.2f} |"
                    )

            # 수익률 상위/하위
            lines.append(f"\n## 수익률 상위 거래 (Top 20)\n")
            lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
            lines.append("|---|---|---|---:|---:|")
            for _, r in bdf_short.nlargest(20, "return_pct").iterrows():
                lines.append(
                    f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
                    f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
                    f"| {r['return_pct']:+.1f} |"
                )

            lines.append(f"\n## 수익률 하위 거래 (Bottom 20)\n")
            lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
            lines.append("|---|---|---|---:|---:|")
            for _, r in bdf_short.nsmallest(20, "return_pct").iterrows():
                lines.append(
                    f"| {r['name']}({r['ticker']}) | {r['buy_date'].strftime('%Y-%m-%d')} "
                    f"| {r['sell_date'].strftime('%Y-%m-%d')} | {r['hold_days']} "
                    f"| {r['return_pct']:+.1f} |"
                )

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_dsl.md", "w") as f:
        f.write(report)
        f.write("\n")
    print("\n결과 저장: results/backtest_dsl.md")


if __name__ == "__main__":
    main()
