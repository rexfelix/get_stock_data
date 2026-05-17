"""
계단뛰기 — '시그널 하루 당김 + 종가매수→다음날 종가매도' 전략

조건 하루 당김 (모든 봉 인덱스 -1):
  원본 bar(2) → 새 bar(1)
  원본 bar(1) → 새 bar(0)
  원본 bar(0) 참조 조건 모두 제거

새 조건 (10 항목):
  1) open(1) < close(1)                  [원 1]
  2) open(0) > close(1)                  [원 2]
  3) low(0)  > (open(1)+close(1))/2      [원 3]
  Branch A: 5) open(0) < close(0)        [원 5]   (오늘 양봉)
  Branch B: 7) open(0) > close(0)
            9) volume(1) > volume(0)     [원 7,9] (오늘 음봉 + 거래량 감소)
  10) close(1) > ma5(1)                  [원 10]
  11) close(0) > ma5(0)                  [원 11]
  14) 시가총액 ≥ 3000억                  [universe]
  15) close(0) ≥ 2000                    [원 15]

제거된 조건: 4, 6, 8, 12, 13, 16, 17

논리식: 1 && 2 && 3 && (5 || (7 && 9)) && 10 && 11 && 14 && 15

매매:
  - 시그널 당일 종가 매수
  - 다음날 종가 매도  (1-day hold, close-to-close)
  - 비용: 왕복 0.26%
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from backtest_stair_jump import (
    calc_indicators, load_universe, load_all,
    MCAP_THRESHOLD, PRICE_FLOOR,
)
from analyze_stair_jump_stats import load_kospi_regime


FEE_BUY = 0.00015
FEE_SELL = 0.00015
TAX_SELL = 0.0023
ROUND_TRIP = FEE_BUY + FEE_SELL + TAX_SELL
COST_PCT = ROUND_TRIP * 100  # 0.26%

DATA_START = "2019-01-01"
DATA_END = "2026-05-17"


def find_signals_shifted(df: pd.DataFrame) -> list[int]:
    """하루 당긴 조건 (bar 0=today, 1=yesterday 만 사용)."""
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["volume"].values
    ma5 = df["ma5"].values

    signals = []
    for i in range(1, len(df)):
        # 결측 체크 (ma5 of bar 0 and 1)
        if np.isnan(ma5[i]) or np.isnan(ma5[i - 1]):
            continue

        # 1) open(1) < close(1)
        if not (o[i - 1] < c[i - 1]):
            continue
        # 2) open(0) > close(1)
        if not (o[i] > c[i - 1]):
            continue
        # 3) low(0) > (open(1)+close(1))/2
        mid1 = (o[i - 1] + c[i - 1]) / 2
        if not (l[i] > mid1):
            continue

        # Branch A: 5) open(0) < close(0)
        a = o[i] < c[i]
        # Branch B: 7) open(0) > close(0) AND 9) volume(1) > volume(0)
        b = (o[i] > c[i]) and (v[i - 1] > v[i])
        if not (a or b):
            continue

        # 10) close(1) > ma5(1)
        if not (c[i - 1] > ma5[i - 1]):
            continue
        # 11) close(0) > ma5(0)
        if not (c[i] > ma5[i]):
            continue
        # 15) close(0) ≥ PRICE_FLOOR
        if c[i] < PRICE_FLOOR:
            continue

        signals.append(i)
    return signals


def collect_events(ticker: str, name: str, df: pd.DataFrame) -> list[dict]:
    df = calc_indicators(df)
    signals = find_signals_shifted(df)
    if not signals:
        return []

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    dates = df.index

    out = []
    for i in signals:
        ev = {
            "ticker": ticker, "name": name,
            "signal_date": dates[i],
            "sig_open": float(o[i]),
            "sig_high": float(h[i]),
            "sig_low": float(l[i]),
            "sig_close": float(c[i]),
            "sig_bullish": bool(c[i] > o[i]),
        }
        # 다음날: 종가매도 + 양봉/등락 통계
        if i + 1 < len(df) and not np.isnan(o[i + 1]) and o[i + 1] > 0:
            nd_open = o[i + 1]
            nd_high = h[i + 1]
            nd_low = l[i + 1]
            nd_close = c[i + 1]
            ev["nd_date"] = dates[i + 1]
            ev["nd_open"] = float(nd_open)
            ev["nd_high"] = float(nd_high)
            ev["nd_low"] = float(nd_low)
            ev["nd_close"] = float(nd_close)
            ev["nd_bullish"] = bool(nd_close > nd_open)
            ev["nd_up_vs_prev"] = bool(nd_close > c[i])  # close-to-close 기준 상승
            # 매매: 시그널 종가 매수 → 다음날 종가 매도
            gross_ret = (nd_close - c[i]) / c[i] * 100
            ev["gross_ret"] = gross_ret
            ev["net_ret"] = gross_ret - COST_PCT
        else:
            ev["nd_date"] = pd.NaT
            ev["nd_open"] = ev["nd_high"] = ev["nd_low"] = ev["nd_close"] = np.nan
            ev["nd_bullish"] = None
            ev["nd_up_vs_prev"] = None
            ev["gross_ret"] = ev["net_ret"] = np.nan
        out.append(ev)
    return out


def stat_block(sub: pd.DataFrame, col: str) -> dict:
    s = sub.dropna(subset=[col])
    n = len(s)
    if n == 0:
        return {"n": 0}
    bull = s["nd_bullish"].fillna(False).astype(bool).sum() if "nd_bullish" in s else None
    up = s[s[col] > 0]
    dn = s[s[col] < 0]
    return {
        "n": n,
        "p_bullish": (bull / n * 100) if bull is not None else np.nan,
        "p_up": len(up) / n * 100,
        "mean": s[col].mean(),
        "median": s[col].median(),
        "up_n": len(up),
        "up_mean": up[col].mean() if len(up) else np.nan,
        "up_median": up[col].median() if len(up) else np.nan,
        "up_max": up[col].max() if len(up) else np.nan,
        "dn_n": len(dn),
        "dn_mean": dn[col].mean() if len(dn) else np.nan,
        "dn_median": dn[col].median() if len(dn) else np.nan,
        "dn_min": dn[col].min() if len(dn) else np.nan,
    }


def daily_equal_weight_sim(ev_df: pd.DataFrame) -> dict:
    ent = ev_df.dropna(subset=["net_ret"]).copy()
    if ent.empty:
        return {"n_days": 0}
    ent["trade_date"] = pd.to_datetime(ent["signal_date"])  # 매수일=signal_date close
    daily = ent.groupby("trade_date").agg(
        n_trades=("net_ret", "size"),
        gross=("gross_ret", "mean"),
        net=("net_ret", "mean"),
    )
    equity = (1 + daily["net"] / 100).cumprod()
    if equity.empty:
        return {"n_days": 0}
    span_days = (daily.index.max() - daily.index.min()).days + 1
    years = span_days / 365.25
    total = (equity.iloc[-1] - 1) * 100
    cagr = (equity.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else np.nan
    peak = equity.cummax()
    dd = (equity / peak - 1) * 100
    mdd = dd.min()
    std = daily["net"].std()
    sharpe = (daily["net"].mean() / std * np.sqrt(252)) if std and std > 0 else np.nan
    return {
        "n_days": len(daily),
        "n_trades": int(daily["n_trades"].sum()),
        "avg_trades_per_day": daily["n_trades"].mean(),
        "avg_daily_gross": daily["gross"].mean(),
        "avg_daily_net": daily["net"].mean(),
        "total_ret": total,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
    }


def fmt_pct(x, p=2, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{p}f}%" if sign else f"{x:.{p}f}%"


def render_stats(label: str, s: dict) -> list[str]:
    if s.get("n", 0) == 0:
        return [f"#### {label}\n\n표본 없음.\n"]
    return [
        f"#### {label}\n",
        "| 지표 | 값 |", "|------|-----|",
        f"| 표본 수 | {s['n']:,} |",
        f"| 양봉 확률 (다음날 close>open) | {fmt_pct(s['p_bullish'], 1, False)} |",
        f"| 상승 확률 (다음날 close>시그널종가) | {fmt_pct(s['p_up'], 1, False)} |",
        f"| 평균 등락 (Gross) | {fmt_pct(s['mean'])} |",
        f"| 중앙 등락 (Gross) | {fmt_pct(s['median'])} |",
        f"| 상승일 수 | {s['up_n']:,} |",
        f"| 상승일 평균 | {fmt_pct(s['up_mean'])} |",
        f"| 상승일 중앙 | {fmt_pct(s['up_median'])} |",
        f"| 상승일 최대 | {fmt_pct(s['up_max'])} |",
        f"| 하락일 수 | {s['dn_n']:,} |",
        f"| 하락일 평균 | {fmt_pct(s['dn_mean'])} |",
        f"| 하락일 중앙 | {fmt_pct(s['dn_median'])} |",
        f"| 하락일 최저 | {fmt_pct(s['dn_min'])} |",
        "",
    ]


def make_report(ev_df: pd.DataFrame, regime_df: pd.DataFrame) -> str:
    ev_df = ev_df.copy()
    ev_df["signal_date"] = pd.to_datetime(ev_df["signal_date"])
    reg_series = regime_df["regime"]
    unique_dates = pd.DatetimeIndex(sorted(set(ev_df["signal_date"].unique())))
    full = pd.concat([reg_series, pd.Series(index=unique_dates, dtype=object)])
    full = full[~full.index.duplicated(keep="first")].sort_index().ffill()
    ev_df["regime"] = ev_df["signal_date"].map(full).fillna("약세장")

    lines = []
    lines.append("# 계단뛰기 (하루 당김) — 종가 매수→다음날 종가 매도 전략\n")
    lines.append("## 변경 사항\n")
    lines.append("- **조건 하루 당김**: 모든 봉 인덱스 -1 (원본 bar 2→1, 1→0), 원본 bar(0) 참조 조건 모두 제거")
    lines.append("- 제거: 조건 4, 6, 8, 12, 13, 16, 17")
    lines.append("- 새 논리식: `1 && 2 && 3 && (5 || (7 && 9)) && 10 && 11 && 14 && 15`")
    lines.append("")
    lines.append("**새 조건**:\n")
    lines.append("- 1) open(1) < close(1) — 어제 양봉")
    lines.append("- 2) open(0) > close(1) — 오늘 시가가 어제 종가 위에서 출발")
    lines.append("- 3) low(0) > (open(1)+close(1))/2 — 오늘 저가가 어제 캔들 중앙선 위")
    lines.append("- Branch A: 5) open(0) < close(0) — 오늘 양봉")
    lines.append("- Branch B: 7) open(0) > close(0) AND 9) volume(1) > volume(0) — 오늘 음봉 + 거래량 감소")
    lines.append("- 10) close(1) > ma5(1), 11) close(0) > ma5(0)")
    lines.append("- 14) 시가총액 ≥ 3000억, 15) close(0) ≥ 2000원")
    lines.append("")
    lines.append("**매매**:\n")
    lines.append("- 시그널 당일(t=0) 종가에 매수")
    lines.append("- 다음날(t=+1) 종가에 매도 (1-day hold, close-to-close)")
    lines.append(f"- 왕복 비용 {COST_PCT:.2f}%")
    lines.append("")

    lines.append("## 데이터\n")
    lines.append(f"- 기간: {DATA_START} ~ {DATA_END}")
    lines.append(f"- 유니버스: 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억, 종가 ≥ {PRICE_FLOOR:,}원")
    lines.append(f"- 총 시그널 수: **{len(ev_df):,}건**")
    cnt = ev_df["regime"].value_counts()
    lines.append(f"  - 강세장: {cnt.get('강세장', 0):,}건  /  약세장: {cnt.get('약세장', 0):,}건")
    lines.append("")

    # ── 1. 다음날 통계 (Gross, 비용 미반영) ──
    lines.append("## 1. 다음날 (t=+1) 양봉/등락 통계 (Gross)\n")
    lines.append("> 시그널 종가→다음날 종가 수익률 (비용 미반영)\n")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        s = stat_block(sub, "gross_ret")
        lines.extend(render_stats(label, s))

    # ── 2. Net (비용 반영) ──
    lines.append("## 2. Net 수익률 (왕복 비용 0.26% 반영)\n")
    lines.append("| 구분 | 표본 | Net 평균 | Net 중앙 | Net 승률 | Net 상승평균 | Net 하락평균 | 최대 | 최저 |")
    lines.append("|------|-----:|--------:|--------:|--------:|------------:|------------:|-----:|-----:|")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        s = stat_block(sub, "net_ret")
        if s["n"] == 0:
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {s['n']:,} | **{fmt_pct(s['mean'])}** | {fmt_pct(s['median'])} | "
            f"{fmt_pct(s['p_up'], 1, False)} | {fmt_pct(s['up_mean'])} | {fmt_pct(s['dn_mean'])} | "
            f"{fmt_pct(s['up_max'])} | {fmt_pct(s['dn_min'])} |"
        )
    lines.append("")

    # ── 3. 일별 동일가중 분산투자 시뮬 ──
    lines.append("## 3. 일별 동일가중 분산투자 시뮬 (매일 자본 100%)\n")
    lines.append("| 구분 | 거래일 | 평균 시그널/일 | 일평균 Gross | 일평균 Net | 누적 수익 | CAGR | MDD | Sharpe |")
    lines.append("|------|-------:|--------------:|------------:|----------:|---------:|----:|----:|------:|")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        r = daily_equal_weight_sim(sub)
        if r["n_days"] == 0:
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {r['n_days']:,} | {r['avg_trades_per_day']:.1f} | "
            f"{fmt_pct(r['avg_daily_gross'])} | **{fmt_pct(r['avg_daily_net'])}** | "
            f"{fmt_pct(r['total_ret'], 1)} | {fmt_pct(r['cagr'], 1)} | "
            f"{fmt_pct(r['mdd'], 1, False)} | {r['sharpe']:.2f} |"
        )
    lines.append("")

    # ── 4. 연도별 ──
    lines.append("## 4. 연도별 요약\n")
    lines.append("| 연도 | 시그널 | 양봉률 | 상승률 | Gross 평균 | Net 평균 | Net 승률 |")
    lines.append("|------|------:|------:|------:|----------:|--------:|--------:|")
    ev_df["year"] = ev_df["signal_date"].dt.year
    for yr, sub in ev_df.groupby("year"):
        sub = sub.dropna(subset=["net_ret"])
        if sub.empty:
            continue
        n = len(sub)
        bull = sub["nd_bullish"].fillna(False).astype(bool).sum()
        up = (sub["gross_ret"] > 0).sum()
        gm = sub["gross_ret"].mean()
        nm = sub["net_ret"].mean()
        nwr = (sub["net_ret"] > 0).sum() / n * 100
        lines.append(
            f"| {yr} | {n:,} | {bull/n*100:.1f}% | {up/n*100:.1f}% | "
            f"{fmt_pct(gm)} | {fmt_pct(nm)} | {nwr:.1f}% |"
        )
    lines.append("")

    # ── 5. 결론 ──
    s_all = stat_block(ev_df, "net_ret")
    sim_all = daily_equal_weight_sim(ev_df)
    lines.append("## 5. 결론\n")
    judge = "✅ 양수 기댓값" if s_all["mean"] > 0 else "❌ 음수 기댓값"
    lines.append(f"### 핵심 판단: **{judge}**\n")
    lines.append(f"- 거래당 Net 평균: **{fmt_pct(s_all['mean'])}**  (Gross {fmt_pct(stat_block(ev_df, 'gross_ret')['mean'])})")
    lines.append(f"- Net 승률: {fmt_pct(s_all['p_up'], 1, False)}")
    lines.append(f"- 일별 동일가중 분산투자 시뮬: 일평균 Net **{fmt_pct(sim_all['avg_daily_net'])}**, 누적 {fmt_pct(sim_all['total_ret'], 1)}, CAGR {fmt_pct(sim_all['cagr'], 1)}, MDD {fmt_pct(sim_all['mdd'], 1, False)}, Sharpe {sim_all['sharpe']:.2f}")
    lines.append(f"- 시그널 수: {len(ev_df):,}건, 일평균 {sim_all['avg_trades_per_day']:.1f}건")
    lines.append("")

    # 시그널 분포
    lines.append("## 6. 시그널 분포 (참고)\n")
    lines.append(f"- 종목 수: **{ev_df['ticker'].nunique():,}개**")
    top = ev_df.groupby(["ticker", "name"]).size().sort_values(ascending=False).head(15)
    lines.append("- TOP 15 종목:\n")
    lines.append("| # | 종목 | 코드 | 시그널 |")
    lines.append("|---|------|------|------:|")
    for j, ((tk, nm), n) in enumerate(top.items(), 1):
        lines.append(f"| {j} | {nm} | {tk} | {n} |")
    lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("계단뛰기 (하루 당김) — 종가→다음날 종가 매매 분석")
    print("=" * 60)

    ticker_list, name_map = load_universe()
    all_data = load_all(ticker_list, DATA_START, DATA_END)

    print(f"[3/4] 시그널 수집 ({len(all_data)} 종목)...")
    events = []
    for k, ticker in enumerate(ticker_list, 1):
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        if len(df) < 130:
            continue
        nm = name_map.get(ticker, ticker)
        events.extend(collect_events(ticker, nm, df))
        if k % 500 == 0:
            print(f"   {k}/{len(ticker_list)}, 누적 {len(events):,}")

    ev_df = pd.DataFrame(events)
    print(f"        총 시그널: {len(ev_df):,}")

    print("[4/4] 리포트 생성...")
    regime_df = load_kospi_regime()
    report = make_report(ev_df, regime_df)

    out = os.path.join(os.path.dirname(__file__), "results",
                        "stair_jump_shifted.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트: {out}")

    csv_out = os.path.join(os.path.dirname(__file__), "results",
                            "stair_jump_shifted_events.csv")
    ev_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"이벤트 CSV: {csv_out}")


if __name__ == "__main__":
    main()
