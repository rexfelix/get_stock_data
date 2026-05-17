"""
계단뛰기 — 당일 매수 + 당일 매도 (intraday) 전략의 기대수익 분석

비용 가정 (backtest_stair_jump.py 와 동일):
  FEE_BUY  = 0.00015 (0.015%)
  FEE_SELL = 0.00015 (0.015%)
  TAX_SELL = 0.0023  (0.23%)
  → 왕복 비용 ≈ 0.26%

분석 대상:
  1) 원본 시그널 (16·17 포함, 10,324건)  — stair_jump_events.csv
  2) 변형 시그널 (16·17 제거, 12,036건) — stair_jump_events_no1617.csv

각각:
  - Gross/Net 기대수익 (per-trade)
  - 강세장/약세장
  - 만약 일별 동일가중 분산투자로 모든 시그널 매수했을 때 누적 시뮬레이션
  - Buy-and-hold KOSPI 와 비교
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_crash import ENGINE

FEE_BUY = 0.00015
FEE_SELL = 0.00015
TAX_SELL = 0.0023
ROUND_TRIP_COST = FEE_BUY + FEE_SELL + TAX_SELL  # 0.26%
COST_PCT = ROUND_TRIP_COST * 100

BASE_DIR = os.path.dirname(__file__)
RESULTS = os.path.join(BASE_DIR, "results")


def load_kospi_regime() -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT date, close FROM market_indices WHERE symbol='^KS11' ORDER BY date",
        ENGINE,
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["ma200"] = df["close"].rolling(200).mean()
    df["ma200_up"] = df["ma200"].diff(5) > 0
    df["regime"] = np.where(
        (df["close"] > df["ma200"]) & df["ma200_up"], "강세장", "약세장"
    )
    return df[["close", "ma200", "regime"]]


def label_regime(ev_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    ev_df = ev_df.copy()
    ev_df["signal_date"] = pd.to_datetime(ev_df["signal_date"])
    reg_series = regime_df["regime"]
    unique_dates = pd.DatetimeIndex(sorted(set(ev_df["signal_date"].unique())))
    full = pd.concat([reg_series, pd.Series(index=unique_dates, dtype=object)])
    full = full[~full.index.duplicated(keep="first")].sort_index().ffill()
    ev_df["regime"] = ev_df["signal_date"].map(full).fillna("약세장")
    return ev_df


def per_trade_stats(ev_df: pd.DataFrame) -> dict:
    """체결된 거래의 gross/net 통계."""
    ent = ev_df[ev_df["nd_entered"].fillna(False).astype(bool)].copy()
    if ent.empty:
        return {}
    gross = ent["nd_entry_to_close_ret"].dropna()
    net = gross - COST_PCT
    return {
        "n": len(gross),
        "gross_mean": gross.mean(),
        "gross_median": gross.median(),
        "gross_winrate": (gross > 0).mean() * 100,
        "net_mean": net.mean(),
        "net_median": net.median(),
        "net_winrate": (net > 0).mean() * 100,
        "gross_up_mean": gross[gross > 0].mean() if (gross > 0).any() else np.nan,
        "gross_dn_mean": gross[gross < 0].mean() if (gross < 0).any() else np.nan,
        "net_up_mean": net[net > 0].mean() if (net > 0).any() else np.nan,
        "net_dn_mean": net[net < 0].mean() if (net < 0).any() else np.nan,
    }


def daily_equal_weight_sim(ev_df: pd.DataFrame, label: str) -> dict:
    """일별 동일가중 분산투자 시뮬레이션.

    - 매일 그 날 매수 체결된 모든 시그널 종목을 자본 1/N 으로 분산매수
    - 같은 날 종가 매도 (intraday)
    - Net 수익률 = gross - 왕복비용
    - 일별 수익률 = 평균(개별 net 수익률)  (모든 시그널 진입했다 가정)
    """
    ent = ev_df[ev_df["nd_entered"].fillna(False).astype(bool)].copy()
    ent["trade_date"] = pd.to_datetime(ent["nd_date"])
    ent["net_ret"] = ent["nd_entry_to_close_ret"] - COST_PCT

    daily = ent.groupby("trade_date").agg(
        n_trades=("net_ret", "size"),
        gross=("nd_entry_to_close_ret", "mean"),
        net=("net_ret", "mean"),
    )

    if daily.empty:
        return {"label": label, "n_days": 0}

    # 누적 자본 (복리)
    equity = (1 + daily["net"] / 100).cumprod()
    n_days = len(daily)

    # 연환산: 거래일 평균을 일평균으로 본 뒤 250일 환산
    avg_daily_net = daily["net"].mean()
    median_daily_net = daily["net"].median()
    std_daily_net = daily["net"].std()
    sharpe_252 = (avg_daily_net / std_daily_net * np.sqrt(252)) if std_daily_net > 0 else np.nan
    total_ret = (equity.iloc[-1] - 1) * 100

    # 전체 백테스트 기간을 분모로 한 연환산 (실제 보유는 거래일만, 무거래일 0%)
    span_days = (daily.index.max() - daily.index.min()).days + 1
    years = span_days / 365.25
    cagr = ((equity.iloc[-1]) ** (1 / years) - 1) * 100 if years > 0 else np.nan

    # MDD
    peak = equity.cummax()
    dd = (equity / peak - 1) * 100
    mdd = dd.min()

    return {
        "label": label,
        "n_days": n_days,
        "n_trades": int(daily["n_trades"].sum()),
        "avg_trades_per_day": daily["n_trades"].mean(),
        "avg_daily_gross": daily["gross"].mean(),
        "avg_daily_net": avg_daily_net,
        "median_daily_net": median_daily_net,
        "std_daily_net": std_daily_net,
        "sharpe_252": sharpe_252,
        "total_ret_pct": total_ret,
        "cagr_pct": cagr,
        "mdd_pct": mdd,
        "span_days": span_days,
        "years": years,
        "final_equity": equity.iloc[-1],
        "equity_curve": equity,
    }


def fmt_pct(x, p=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{p}f}%"


def fmt_pct1(x, p=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}%"


def fmt_int(x):
    return "—" if x is None else f"{int(x):,}"


def render_block(title: str, ev_df: pd.DataFrame, regime_df: pd.DataFrame) -> list[str]:
    lines = []
    ev_df = label_regime(ev_df, regime_df)
    lines.append(f"## {title}\n")
    lines.append("### 거래당 (per-trade) 기대수익\n")
    lines.append("| 구분 | 표본 | 승률(net) | Gross 평균 | **Net 평균** | Net 중앙 | Net 상승평균 | Net 하락평균 |")
    lines.append("|------|-----:|--------:|----------:|----------:|--------:|------------:|------------:|")
    for sub_label, sub in [("전체", ev_df),
                            ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                            ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        s = per_trade_stats(sub)
        if not s:
            lines.append(f"| {sub_label} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {sub_label} | {fmt_int(s['n'])} | {fmt_pct1(s['net_winrate'])} | "
            f"{fmt_pct(s['gross_mean'])} | **{fmt_pct(s['net_mean'])}** | {fmt_pct(s['net_median'])} | "
            f"{fmt_pct(s['net_up_mean'])} | {fmt_pct(s['net_dn_mean'])} |"
        )
    lines.append("")

    lines.append("### 일별 동일가중 분산투자 시뮬레이션 (intraday, 매일 자본 100% 활용)\n")
    lines.append("| 구분 | 거래일 | 평균 시그널/일 | 일평균 Gross | **일평균 Net** | 누적 수익 | CAGR | MDD | Sharpe |")
    lines.append("|------|-------:|--------------:|------------:|-------------:|----------:|-----:|----:|------:|")
    for sub_label, sub in [("전체", ev_df),
                            ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                            ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        sim = daily_equal_weight_sim(sub, sub_label)
        if sim["n_days"] == 0:
            lines.append(f"| {sub_label} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {sub_label} | {sim['n_days']:,} | {sim['avg_trades_per_day']:.1f} | "
            f"{fmt_pct(sim['avg_daily_gross'])} | **{fmt_pct(sim['avg_daily_net'])}** | "
            f"{fmt_pct1(sim['total_ret_pct'])} | {fmt_pct1(sim['cagr_pct'])} | "
            f"{fmt_pct1(sim['mdd_pct'])} | {sim['sharpe_252']:.2f} |"
        )
    lines.append("")
    return lines


def make_report(ev_orig: pd.DataFrame, ev_no1617: pd.DataFrame,
                regime_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# 계단뛰기 — 당일 매수+당일 매도(intraday) 기대수익 분석\n")
    lines.append("## 비용 가정\n")
    lines.append(f"- 매수 수수료 {FEE_BUY*100:.3f}% + 매도 수수료 {FEE_SELL*100:.3f}% + 매도세 {TAX_SELL*100:.2f}%")
    lines.append(f"- **왕복 비용 = {COST_PCT:.3f}%** (backtest_stair_jump.py 기본값)")
    lines.append("- Slippage·hosting fee 등 추가 마찰 없음 가정 → 실전에선 더 나쁨")
    lines.append("- Intraday 가정: 시그널 다음봉(t=+1) 시가/시그널종가 트리거에 매수, 같은 봉 종가에 매도")
    lines.append("")

    lines.append("## 데이터 기간\n")
    lines.append(f"- 2019-01-02 ~ 2026-05-15 (약 7.4 년)")
    lines.append(f"- 유니버스: 시총 ≥ 3,000억, 종가 ≥ 2,000원")
    lines.append("")

    lines.extend(render_block("1. 원본 시그널 (16·17 포함)", ev_orig, regime_df))
    lines.extend(render_block("2. 변형 시그널 (16·17 제거)", ev_no1617, regime_df))

    # ── 종합 결론 ──
    lines.append("## 3. 결론\n")

    o_stat = per_trade_stats(label_regime(ev_orig, regime_df))
    n_stat = per_trade_stats(label_regime(ev_no1617, regime_df))
    o_sim = daily_equal_weight_sim(label_regime(ev_orig, regime_df), "원본 전체")
    n_sim = daily_equal_weight_sim(label_regime(ev_no1617, regime_df), "변형 전체")

    lines.append(f"### 결론 요약\n")
    lines.append("| 지표 | 원본 (16·17 포함) | 변형 (16·17 제거) |")
    lines.append("|------|-----------------:|-----------------:|")
    lines.append(f"| 거래당 Gross 평균 | {fmt_pct(o_stat['gross_mean'])} | {fmt_pct(n_stat['gross_mean'])} |")
    lines.append(f"| **거래당 Net 평균** | **{fmt_pct(o_stat['net_mean'])}** | **{fmt_pct(n_stat['net_mean'])}** |")
    lines.append(f"| 거래당 Net 승률 | {fmt_pct1(o_stat['net_winrate'])} | {fmt_pct1(n_stat['net_winrate'])} |")
    lines.append(f"| 일평균 Net (전체) | {fmt_pct(o_sim['avg_daily_net'])} | {fmt_pct(n_sim['avg_daily_net'])} |")
    lines.append(f"| **누적 수익 (7.4년)** | **{fmt_pct1(o_sim['total_ret_pct'])}** | **{fmt_pct1(n_sim['total_ret_pct'])}** |")
    lines.append(f"| CAGR | {fmt_pct1(o_sim['cagr_pct'])} | {fmt_pct1(n_sim['cagr_pct'])} |")
    lines.append(f"| MDD | {fmt_pct1(o_sim['mdd_pct'])} | {fmt_pct1(n_sim['mdd_pct'])} |")
    lines.append(f"| Sharpe (252d) | {o_sim['sharpe_252']:.2f} | {n_sim['sharpe_252']:.2f} |")
    lines.append("")

    lines.append("### 판단\n")
    judgement_o = "❌ 손실" if o_stat["net_mean"] < 0 else "✅ 수익"
    judgement_n = "❌ 손실" if n_stat["net_mean"] < 0 else "✅ 수익"
    lines.append(f"- **원본 시그널 intraday**: 거래당 Net {o_stat['net_mean']:+.3f}% → {judgement_o}")
    lines.append(f"- **변형 시그널 intraday**: 거래당 Net {n_stat['net_mean']:+.3f}% → {judgement_n}")
    lines.append("")
    lines.append("**Intraday(당일 매수+당일 매도) 전략은 비추천**:")
    lines.append("1. **Gross 단계에서 이미 음수** (-0.21% ~ -0.22%) — 시그널 자체의 'mean-reversion'")
    lines.append(f"2. 왕복 비용 {COST_PCT:.2f}% 가 추가되면 -0.47% ~ -0.48% (확정 손실 기대값)")
    lines.append("3. 강세장/약세장 차이 없음 — 시장 regime 으로 회생 불가")
    lines.append("4. 일별 동일가중 분산투자 시뮬도 7.4년간 큰 손실 (CAGR 마이너스)")
    lines.append("")
    lines.append("**대안**:")
    lines.append("- 매도 익절/이익보전/손절을 사용하는 multi-bar 보유 전략 (현재 `backtest_stair_jump.py` 기본 모드) 이 합리적")
    lines.append("- 그러나 [[project_stair_jump]] 에서 약세장(2022~2024) 19/19 그리드 손실로 채택 부적합 결론")
    lines.append("- 즉 intraday 도, multi-bar 도 단독 채택 어려움")
    lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def main():
    ev_orig = pd.read_csv(os.path.join(RESULTS, "stair_jump_events.csv"))
    ev_no1617 = pd.read_csv(os.path.join(RESULTS, "stair_jump_events_no1617.csv"))
    regime_df = load_kospi_regime()

    report = make_report(ev_orig, ev_no1617, regime_df)
    out = os.path.join(RESULTS, "stair_jump_intraday_expect.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"리포트: {out}")


if __name__ == "__main__":
    main()
