"""1500억 (15,3) 2019~2023 백테스트 (amount 추정).

데이터 한계:
- DB stock_all.amount 는 2023-09-25 이후만 가용
- 본 검증은 amount = close × volume 추정값 사용
- mcap = close × snapshot shares (2026-04 시점 상장주식수 고정, 시계열 한계)

목적: (15,3) Calmar 41.88 (2024~2026 베이스) 이 2024+ 반도체 상승에 의존했는지 검증.
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env")

import backtest_top3_indicators as bt  # noqa: E402

# ⭐ START_DATE monkey patch: run_backtest 가 이를 쓰므로 변경 필요
bt.START_DATE = "2019-01-01"

from backtest_5d_amount_filter import compute_5d_filter_signals  # noqa: E402
from backtest_5d_realistic_k import equity_real_k  # noqa: E402

OUTPUT_MD = "/Volumes/SSD/project/py/invest/data_center/research/results/backtest_n15_k3_2019_2023.md"

THRESHOLD_WON = 150_000_000_000  # 1500억
N = 15
K = 3
START = "2019-01-01"
END = "2023-12-31"


# ──────────────────────────────────────────────
# 헬퍼 (단위 테스트 대상)
# ──────────────────────────────────────────────
def estimate_amount_column(df: pd.DataFrame) -> pd.DataFrame:
    """df 의 close × volume 으로 amount 컬럼 채움 (기존 amount 덮어씀).

    추정값 한계: 일중 평균가 아닌 종가 기준이라 실제 amount 와 미세 차이.
    1500억 같은 큰 임계 식별엔 충분.
    """
    out = df.copy()
    out["amount"] = out["close"].astype(float) * out["volume"].astype(float)
    return out


# ──────────────────────────────────────────────
# 데이터 로드 (stocks 직접, amount 추정)
# ──────────────────────────────────────────────
def load_price_data_estimated(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    placeholders = ",".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT ticker, name, date, open, high, low, close, volume
        FROM stocks
        WHERE ticker IN ({placeholders})
          AND date >= '{start}'::date - interval '60 days'
          AND date <= '{end}'::date
        ORDER BY ticker, date
    """
    df = pd.read_sql(query, bt.ENGINE)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = estimate_amount_column(df)
    return df


# ──────────────────────────────────────────────
# 분석 헬퍼
# ──────────────────────────────────────────────
def yearly_breakdown(trades: list[dict]) -> dict[int, dict]:
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df["buy_year"] = pd.to_datetime(df["buy_date"]).dt.year
    out = {}
    for y, g in df.groupby("buy_year"):
        wins = (g["net_ret"] > 0).sum()
        out[int(y)] = {
            "n": len(g),
            "win_rate": wins / len(g) * 100,
            "avg_ret": g["net_ret"].mean() * 100,
            "max_gain": g["net_ret"].max() * 100,
            "max_loss": g["net_ret"].min() * 100,
            "total_pnl_pct": (1 + g["net_ret"]).prod() * 100 - 100 if len(g) > 0 else 0,
        }
    return out


def trade_summary(trades: list[dict]) -> dict:
    if not trades:
        return {"total": 0, "win_rate": 0, "avg_ret": 0, "max_loss": 0,
                "max_gain": 0, "avg_hold": 0, "unique": 0,
                "samsung_count": 0, "most_t": "-", "most_n": 0}
    df = pd.DataFrame(trades)
    counts = Counter(t["ticker"] for t in trades)
    most_t, most_n = max(counts.items(), key=lambda x: x[1])
    return {
        "total": len(df),
        "win_rate": (df["net_ret"] > 0).sum() / len(df) * 100,
        "avg_ret": df["net_ret"].mean() * 100,
        "max_loss": df["net_ret"].min() * 100,
        "max_gain": df["net_ret"].max() * 100,
        "avg_hold": df["hold_days"].mean(),
        "unique": len(set(t["ticker"] for t in trades)),
        "samsung_count": counts.get("005930", 0),
        "most_t": most_t, "most_n": most_n,
    }


def main():
    print("=" * 60)
    print(f"(N=15, K=3) 1500억 LIST_EXIT — {START} ~ {END} (amount 추정)")
    print("=" * 60)

    print("[1] 데이터 로드 (stocks 직접, amount 추정)...")
    k200 = bt.load_kospi200_tickers()
    snapshot = bt.load_market_cap_snapshot()
    price_df = load_price_data_estimated(k200["ticker"].tolist(), START, END)
    print(f"    {len(price_df):,}행, {price_df['ticker'].nunique()}종목")
    print(f"    기간: {price_df['date'].min()} ~ {price_df['date'].max()}")
    print(f"    amount 추정 평균: {price_df['amount'].mean()/1e9:.1f}억 / max: {price_df['amount'].max()/1e9:.0f}억")

    daily_data = bt.build_daily_data(price_df, snapshot)
    panel = bt.build_daily_indicator_panel(daily_data, "amount")
    print(f"    daily_data {len(daily_data)}종목 / panel {len(panel)}일")

    print("\n[2] 1500억 N=15 신호 생성...")
    signals = compute_5d_filter_signals(
        daily_data, threshold_won=THRESHOLD_WON, lookback=N, top_k=200,
    )
    sig_per = [len(v) for v in signals.values()]
    print(f"    {len(signals)}일치, 평균 {np.mean(sig_per) if sig_per else 0:.2f} 종목/일")

    print("\n[3] (15,3) LIST_EXIT 백테스트...")
    t0 = time.time()
    trades, _ = bt.run_backtest(
        daily_data, panel, signals,
        rule="LIST_EXIT", slots=K, max_concurrent=K,
    )
    eq = equity_real_k(trades, K=K)
    cagr = eq.get("cagr", 0)
    mdd = eq.get("mdd", 0)
    calmar = abs(cagr / mdd) if mdd != 0 else 0
    print(f"    거래수 {len(trades)}, CAGR {cagr:+.2f}%, MDD {mdd:+.2f}%, "
          f"Calmar {calmar:.2f} ({time.time()-t0:.1f}s)")

    ts = trade_summary(trades)
    print(f"    승률 {ts['win_rate']:.1f}%, 평균 수익률 {ts['avg_ret']:+.2f}%, "
          f"최대 단일 손실 {ts['max_loss']:+.2f}%, 005930 매수 {ts['samsung_count']}회")

    yearly = yearly_breakdown(trades)
    print(f"\n[4] 연도별 분해:")
    for y in sorted(yearly.keys()):
        ys = yearly[y]
        print(f"    {y}: {ys['n']}건, 승률 {ys['win_rate']:.0f}%, "
              f"평균 {ys['avg_ret']:+.1f}%, 최대손 {ys['max_loss']:+.1f}%")

    # 리포트
    print("\n[5] 리포트 생성...")
    lines = [f"# (N=15, K=3) 1500억 LIST_EXIT — {START} ~ {END} (amount 추정)\n"]

    lines.append("## 1. 검증 목적\n")
    lines.append("(15,3) Calmar 41.88 (2024~2026 베이스) 결과가 2024+ 반도체 상승 (005930·000660) 효과인지 검증.\n")

    lines.append("## 2. 데이터 한계 (중요)\n")
    lines.append("- DB `stock_all.amount` 는 2023-09-25 이후만 가용")
    lines.append("- 본 검증은 **amount = close × volume 추정값** 사용")
    lines.append("- 추정 한계: 일중 평균가 아닌 종가 기준 → 실제 amount 와 미세 차이 가능")
    lines.append("- 1500억 같은 큰 임계 식별엔 충분히 근사하나, 결과 해석 시 한계 인지 필요")
    lines.append("- mcap = close × 2026-04 snapshot shares (상장주식수 변동 미반영)\n")

    lines.append("## 3. 핵심 결과\n")
    lines.append("| 지표 | 값 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 기간 | {START} ~ {END} |")
    lines.append(f"| 신호 발생 일수 | {len(signals)}일 |")
    lines.append(f"| 평균 신호 종목/일 | {np.mean(sig_per) if sig_per else 0:.2f} |")
    lines.append(f"| 거래수 | **{len(trades)}** |")
    lines.append(f"| CAGR | **{cagr:+.2f}%** |")
    lines.append(f"| MDD | **{mdd:+.2f}%** |")
    lines.append(f"| Calmar | **{calmar:.2f}** |")
    lines.append(f"| 자본 (5년) | {eq.get('final_equity', 1):.2f}x |")
    lines.append(f"| 승률 | {ts['win_rate']:.1f}% |")
    lines.append(f"| 평균 단일 수익률 | {ts['avg_ret']:+.2f}% |")
    lines.append(f"| 최대 단일 이익 | {ts['max_gain']:+.2f}% |")
    lines.append(f"| 최대 단일 손실 | {ts['max_loss']:+.2f}% |")
    lines.append(f"| 평균 보유일 | {ts['avg_hold']:.1f}일 |")
    lines.append(f"| 고유 종목 수 | {ts['unique']} |")
    lines.append(f"| 005930 매수 횟수 | {ts['samsung_count']} |")
    lines.append(f"| 최다 거래 종목 | {ts['most_t']} ({ts['most_n']}회) |")

    lines.append("\n## 4. 2024~2026 베이스라인과 비교\n")
    lines.append("| 지표 | 2019~2023 (추정) | 2024~2026 (실 amount) |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| 거래수 | {len(trades)} | 17 |")
    lines.append(f"| CAGR(%) | {cagr:+.2f} | +105.71 |")
    lines.append(f"| MDD(%) | {mdd:+.2f} | -2.52 |")
    lines.append(f"| Calmar | **{calmar:.2f}** | **41.88** |")
    lines.append(f"| 자본(x) | {eq.get('final_equity', 1):.2f} | 6.15 |")
    lines.append(f"| 005930 매수 | {ts['samsung_count']}회 | 1회 |")
    lines.append(f"| 평균 보유일 | {ts['avg_hold']:.1f} | 98.5 |")

    lines.append("\n## 5. 연도별 분해\n")
    lines.append("| 연도 | 거래수 | 승률(%) | 평균 수익률(%) | 최대 단일 이익(%) | 최대 단일 손실(%) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for y in sorted(yearly.keys()):
        ys = yearly[y]
        lines.append(
            f"| {y} | {ys['n']} | {ys['win_rate']:.1f} | "
            f"{ys['avg_ret']:+.2f} | {ys['max_gain']:+.2f} | {ys['max_loss']:+.2f} |"
        )

    # 거래 상세
    if trades:
        lines.append("\n## 6. 거래 상세 (시간순)\n")
        lines.append("| 종목 | 매수일 | 매도일 | 보유일 | 수익률(%) |")
        lines.append("| :---: | :---: | :---: | ---: | ---: |")
        for t in sorted(trades, key=lambda x: pd.Timestamp(x["buy_date"])):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            mark = " ⭐" if t["ticker"] == "005930" else ""
            lines.append(
                f"| {t['ticker']}{mark} | {bd} | {sd} | "
                f"{t['hold_days']} | {t['net_ret']*100:+.2f} |"
            )

    # Top/Bottom 거래
    if trades:
        df_t = pd.DataFrame(trades).sort_values("net_ret", ascending=False)
        lines.append("\n## 7. Top 5 / Bottom 5 거래\n")
        lines.append("**Top 5 (이익):**\n")
        lines.append("| 종목 | 매수일 | 보유일 | 수익률(%) |")
        lines.append("| :---: | :---: | ---: | ---: |")
        for _, t in df_t.head(5).iterrows():
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            lines.append(f"| {t['ticker']} | {bd} | {t['hold_days']} | {t['net_ret']*100:+.2f} |")

        lines.append("\n**Bottom 5 (손실):**\n")
        lines.append("| 종목 | 매수일 | 보유일 | 수익률(%) |")
        lines.append("| :---: | :---: | ---: | ---: |")
        for _, t in df_t.tail(5).iterrows():
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            lines.append(f"| {t['ticker']} | {bd} | {t['hold_days']} | {t['net_ret']*100:+.2f} |")

    # 종합 결론
    lines.append("\n---\n\n## 8. 종합 결론\n")

    lines.append("### 8.1 가설 검증\n")
    if calmar >= 10:
        lines.append(f"- **H1 (시스템 robust) 지지**: 2019~2023 Calmar **{calmar:.2f}** ≥ 10 → (15,3) 시기 무관 우위")
        lines.append(f"  005930 효과는 시스템 설계상 자연스러운 결과")
    elif calmar >= 5:
        lines.append(f"- **부분 지지**: 2019~2023 Calmar **{calmar:.2f}** (5~10) → (15,3) 시스템 작동하나 2024+보다 약함")
    elif calmar >= 1:
        lines.append(f"- **H2 (시기 의존) 지지**: 2019~2023 Calmar **{calmar:.2f}** < 5 → 2024+ 반도체 상승에 의존")
    else:
        lines.append(f"- **시스템 실패**: Calmar **{calmar:.2f}** → (15,3) 2019~2023 적용 불가")

    base_calmar = 41.88
    delta_pct = (calmar - base_calmar) / base_calmar * 100 if base_calmar > 0 else 0
    lines.append(f"\n  베이스 대비: {delta_pct:+.1f}% (베이스 {base_calmar:.2f} → 추정 기간 {calmar:.2f})")

    # MDD 코로나 충격
    lines.append("\n### 8.2 베어마켓 충격 검증\n")
    if 2020 in yearly:
        y2020 = yearly[2020]
        lines.append(f"- 2020년 (코로나): {y2020['n']}건, 평균 {y2020['avg_ret']:+.1f}%, "
                     f"최대손 {y2020['max_loss']:+.1f}%")
    if 2022 in yearly:
        y2022 = yearly[2022]
        lines.append(f"- 2022년 (베어마켓): {y2022['n']}건, 평균 {y2022['avg_ret']:+.1f}%, "
                     f"최대손 {y2022['max_loss']:+.1f}%")
    lines.append(f"- 전체 MDD: {mdd:+.2f}%")

    lines.append("\n### 8.3 005930 매수 변화\n")
    lines.append(f"- 2019~2023: {ts['samsung_count']}회")
    lines.append(f"- 2024~2026: 1회 (615일 보유, +220.8%)")

    lines.append("\n### 8.4 채택 결정 영향\n")
    if calmar >= 10:
        lines.append("- (15,3) 채택 결정 **재확정** — 시기 무관 robust")
    elif calmar >= 5:
        lines.append("- (15,3) 채택 가능, 단 2024+ 의존성 일부 인정")
    else:
        lines.append("- (15,3) 채택 결정 **재검토 필요** — 2024+ 의존이 너무 큼")

    lines.append("\n### 8.5 amount 추정 한계 영향\n")
    lines.append("- 1500억 임계 식별은 큰 수치라 추정 오차 영향 작음")
    lines.append("- 결과 차이가 베이스 대비 클 경우 추정값보다 시기 효과 가능성 높음")
    lines.append("- 정확한 검증 위해서는 amount 백필 후 재검증 필요")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
        f.write("\n")
    print(f"\n저장: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
