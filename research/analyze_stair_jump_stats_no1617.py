"""
계단뛰기 시그널 (조건 16, 17 제거) 통계 분석

원본 논리식: 1 && 2 && 3 && 4 && ((5 && 6 && 17) || (7 && 8 && 9 && 16)) && 10..15
변형 논리식: 1 && 2 && 3 && 4 && ((5 && 6)       || (7 && 8 && 9))      && 10..15

즉, 다음 조건만 제거:
  17) close(1) < close(0)   — Branch A 끝
  16) open(1)  < close(0)   — Branch B 끝

나머지 모든 조건 유지.

이 변경으로 시그널 당일 close(0) > close(-1) 보장이 깨지므로
'당일 상승 확률' 도 더 이상 100% 가 아님 (양봉은 여전히 100%, 조건 4 유지).
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from backtest_stair_jump import (
    calc_indicators, load_universe, load_all,
    MCAP_THRESHOLD, PRICE_FLOOR,
)
from analyze_stair_jump_stats import (
    DATA_START, DATA_END, load_kospi_regime, stat_block,
    render_stats_table, fmt_pct, fmt_pct_pos, fmt_int,
)


# ──────────────────────────────────────────────
# 변형 시그널 (16, 17 제거)
# ──────────────────────────────────────────────
def find_signals_no1617(df: pd.DataFrame) -> list[int]:
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["volume"].values
    ma5 = df["ma5"].values
    ma120 = df["ma120"].values

    signals = []
    for i in range(2, len(df)):
        if (np.isnan(ma5[i]) or np.isnan(ma5[i-1]) or np.isnan(ma5[i-2])
                or np.isnan(ma120[i])):
            continue

        # 1) open(2) < close(2)
        if not (o[i-2] < c[i-2]):
            continue
        # 2) open(1) > close(2)
        if not (o[i-1] > c[i-2]):
            continue
        # 3) low(1) > (open(2)+close(2))/2
        if not (l[i-1] > (o[i-2] + c[i-2]) / 2):
            continue
        # 4) open(0) < close(0)
        if not (o[i] < c[i]):
            continue

        mid2 = (o[i-2] + c[i-2]) / 2
        mid1 = (o[i-1] + c[i-1]) / 2

        # Branch A (변형): 5 && 6    (17 제거)
        a = (o[i-1] < c[i-1]) and (l[i] > mid1)
        # Branch B (변형): 7 && 8 && 9    (16 제거)
        b = (o[i-1] > c[i-1]) and (l[i] >= mid2) and (v[i-2] > v[i-1])
        if not (a or b):
            continue

        # 10, 11, 12) close > ma5 (-2, -1, 0)
        if not (c[i-2] > ma5[i-2] and c[i-1] > ma5[i-1] and c[i] > ma5[i]):
            continue
        # 13) close(0) > ma120(0)
        if not (c[i] > ma120[i]):
            continue
        # 15) close >= PRICE_FLOOR
        if c[i] < PRICE_FLOOR:
            continue

        signals.append(i)
    return signals


def collect_events(ticker: str, name: str, df: pd.DataFrame) -> list[dict]:
    df = calc_indicators(df)
    signals = find_signals_no1617(df)
    if not signals:
        return []

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    dates = df.index

    events = []
    for i in signals:
        prev_close = c[i - 1]
        sig_close_open_ret = (c[i] - o[i]) / o[i] * 100
        sig_prev_close_ret = (c[i] - prev_close) / prev_close * 100
        sig_is_bullish = c[i] > o[i]

        ev = {
            "ticker": ticker,
            "name": name,
            "signal_date": dates[i],
            "sig_close": float(c[i]),
            "sig_intraday_ret": sig_close_open_ret,
            "sig_daily_ret": sig_prev_close_ret,
            "sig_bullish": bool(sig_is_bullish),
            "sig_up_vs_prev": bool(c[i] > prev_close),
        }

        if i + 1 < len(df) and not np.isnan(o[i + 1]) and o[i + 1] > 0:
            nd_open = o[i + 1]
            nd_high = h[i + 1]
            nd_close = c[i + 1]
            ev["nd_date"] = dates[i + 1]
            ev["nd_intraday_ret"] = (nd_close - nd_open) / nd_open * 100
            ev["nd_daily_ret"] = (nd_close - c[i]) / c[i] * 100
            ev["nd_bullish"] = bool(nd_close > nd_open)
            ev["nd_up_vs_prev"] = bool(nd_close > c[i])
            ev["nd_gap"] = (nd_open - c[i]) / c[i] * 100

            trigger = c[i]
            if nd_open >= trigger:
                ev["nd_entry"] = nd_open
                ev["nd_entered"] = True
            elif nd_high >= trigger:
                ev["nd_entry"] = trigger
                ev["nd_entered"] = True
            else:
                ev["nd_entry"] = np.nan
                ev["nd_entered"] = False

            if ev["nd_entered"]:
                ev["nd_entry_to_close_ret"] = (nd_close - ev["nd_entry"]) / ev["nd_entry"] * 100
            else:
                ev["nd_entry_to_close_ret"] = np.nan
        else:
            ev["nd_date"] = pd.NaT
            for k in ("nd_intraday_ret", "nd_daily_ret", "nd_gap",
                      "nd_entry", "nd_entry_to_close_ret"):
                ev[k] = np.nan
            ev["nd_bullish"] = None
            ev["nd_up_vs_prev"] = None
            ev["nd_entered"] = False

        events.append(ev)
    return events


# ──────────────────────────────────────────────
# 리포트
# ──────────────────────────────────────────────
def make_report(ev_df: pd.DataFrame, regime_df: pd.DataFrame, elapsed: float,
                cmp_orig: dict | None = None) -> str:
    ev_df = ev_df.copy()
    ev_df["signal_date"] = pd.to_datetime(ev_df["signal_date"])
    reg_series = regime_df["regime"]
    unique_dates = pd.DatetimeIndex(sorted(set(ev_df["signal_date"].unique())))
    full = pd.concat([reg_series, pd.Series(index=unique_dates, dtype=object)])
    full = full[~full.index.duplicated(keep="first")].sort_index().ffill()
    ev_df["regime"] = ev_df["signal_date"].map(full)
    ev_df["regime"] = ev_df["regime"].fillna("약세장")

    lines = []
    lines.append("# 계단뛰기 시그널 (조건 16·17 제거) 통계 분석\n")
    lines.append("## 변경 사항\n")
    lines.append("- **원본 논리식**: `1 && 2 && 3 && 4 && ((5 && 6 && 17) || (7 && 8 && 9 && 16)) && 10..15`")
    lines.append("- **변형 논리식**: `1 && 2 && 3 && 4 && ((5 && 6) || (7 && 8 && 9)) && 10..15`")
    lines.append("- 제거: 17) close(1) < close(0), 16) open(1) < close(0)")
    lines.append("- 결과: 시그널 당일 'close(0) > 전일종가' 보장 깨짐 (양봉은 조건 4로 100% 유지)\n")

    lines.append("## 분석 개요\n")
    lines.append(f"- **데이터 기간**: {DATA_START} ~ {DATA_END}")
    lines.append(f"- **유니버스**: 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억, 종가 ≥ {PRICE_FLOOR:,}원")
    lines.append(f"- **총 시그널 수**: {len(ev_df):,}건")
    cnt_reg = ev_df["regime"].value_counts()
    lines.append(f"  - 강세장: {cnt_reg.get('강세장', 0):,}건  /  약세장: {cnt_reg.get('약세장', 0):,}건")
    if cmp_orig:
        lines.append(f"  - 원본(16·17 포함) 대비: {cmp_orig['n']:,} → {len(ev_df):,} ({(len(ev_df)/cmp_orig['n']-1)*100:+.1f}%)")
    lines.append("")

    # ── 1. 당일 통계 ──
    lines.append("## 1. 당일 (시그널일, t=0) 통계\n")
    lines.append("> 양봉(close>open)은 조건 4로 100%. 상승(close>전일종가)은 이제 100% 아님.\n")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        s = stat_block(sub, "sig_daily_ret", "sig_bullish")
        lines.extend(render_stats_table(label, s))

    # ── 2. 다음날 통계 ──
    lines.append("## 2. 다음날 (t=+1) 통계 — 매수 체결 여부 무관\n")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        s = stat_block(sub, "nd_daily_ret", "nd_bullish")
        lines.extend(render_stats_table(label, s))

    # ── 3. 매수 체결 통계 ──
    lines.append("## 3. 다음날 매수 체결 통계\n")
    lines.append("> 시그널 종가 상향 돌파 시 매수 (시가 GAP-UP 이면 시가, 아니면 시그널종가).\n")
    lines.append("| 구분 | 시그널 | 체결 | 체결률 | 갭 평균 | 매수→당일종가 평균 | 매수→당일종가 승률 |")
    lines.append("|------|------:|-----:|------:|--------:|-------------------:|-----------------:|")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        n_total = len(sub)
        entered = sub[sub["nd_entered"].fillna(False)]
        n_ent = len(entered)
        fill_rate = n_ent / n_total * 100 if n_total else np.nan
        gap_mean = sub["nd_gap"].mean()
        ent_ret = entered["nd_entry_to_close_ret"]
        ent_mean = ent_ret.mean() if len(ent_ret) else np.nan
        ent_winrate = (ent_ret > 0).sum() / len(ent_ret) * 100 if len(ent_ret) else np.nan
        lines.append(
            f"| {label} | {n_total:,} | {n_ent:,} | "
            f"{fmt_pct_pos(fill_rate)} | {fmt_pct(gap_mean)} | "
            f"{fmt_pct(ent_mean)} | {fmt_pct_pos(ent_winrate)} |"
        )
    lines.append("")

    # ── 4. 매수 체결된 경우 분포 ──
    lines.append("## 4. 매수 체결된 경우 — 매수가→당일종가 등락률 분포\n")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        ent = sub[sub["nd_entered"].fillna(False)].copy()
        if ent.empty:
            lines.append(f"### {label}\n\n표본 없음.\n")
            continue
        ent["_bull"] = ent["nd_entry_to_close_ret"] > 0
        s = stat_block(ent, "nd_entry_to_close_ret", "_bull")
        lines.append(f"#### {label}\n")
        lines.append("| 지표 | 값 |")
        lines.append("|------|-----|")
        lines.append(f"| 체결 표본 | {fmt_int(s['n'])} |")
        lines.append(f"| 승률 (매수가 < 당일종가) | {fmt_pct_pos(s['p_up'])} |")
        lines.append(f"| 평균 수익률 | {fmt_pct(s['ret_mean'])} |")
        lines.append(f"| 중앙 수익률 | {fmt_pct(s['ret_median'])} |")
        lines.append(f"| 상승 평균 | {fmt_pct(s['up_mean'])} |")
        lines.append(f"| 상승 중앙 | {fmt_pct(s['up_median'])} |")
        lines.append(f"| 상승 최대 | {fmt_pct(s['up_max'])} |")
        lines.append(f"| 하락 평균 | {fmt_pct(s['dn_mean'])} |")
        lines.append(f"| 하락 중앙 | {fmt_pct(s['dn_median'])} |")
        lines.append(f"| 하락 최저 | {fmt_pct(s['dn_min'])} |")
        lines.append("")

    # ── 5. 연도별 ──
    lines.append("## 5. 연도별 요약 (다음날 등락률 기준)\n")
    lines.append("| 연도 | 시그널 | 양봉률 | 상승률 | 평균 등락 | 상승 평균 | 하락 평균 | 매수체결률 | 진입→종가 평균 |")
    lines.append("|------|------:|------:|------:|----------:|----------:|----------:|----------:|-------------:|")
    ev_df["year"] = ev_df["signal_date"].dt.year
    for yr, sub in ev_df.groupby("year"):
        nd_total = len(sub)
        nd_bull = sub["nd_bullish"].fillna(False).astype(bool).sum()
        nd_up = (sub["nd_daily_ret"] > 0).sum()
        nd_ret = sub["nd_daily_ret"].dropna()
        up_mean = nd_ret[nd_ret > 0].mean() if (nd_ret > 0).any() else np.nan
        dn_mean = nd_ret[nd_ret < 0].mean() if (nd_ret < 0).any() else np.nan
        ent = sub[sub["nd_entered"].fillna(False)]
        fill = len(ent) / nd_total * 100 if nd_total else np.nan
        ent_mean = ent["nd_entry_to_close_ret"].mean() if len(ent) else np.nan
        lines.append(
            f"| {yr} | {nd_total:,} | "
            f"{fmt_pct_pos(nd_bull / nd_total * 100 if nd_total else np.nan)} | "
            f"{fmt_pct_pos(nd_up / nd_total * 100 if nd_total else np.nan)} | "
            f"{fmt_pct(nd_ret.mean())} | {fmt_pct(up_mean)} | {fmt_pct(dn_mean)} | "
            f"{fmt_pct_pos(fill)} | {fmt_pct(ent_mean)} |"
        )
    lines.append("")

    # ── 6. 원본 대비 비교 ──
    if cmp_orig:
        lines.append("## 6. 원본 시그널 (16·17 포함) 대비 비교\n")
        lines.append("| 항목 | 원본 (16·17 포함) | 변형 (16·17 제거) | 변화 |")
        lines.append("|------|------------------:|------------------:|-----:|")
        rows = [
            ("총 시그널", cmp_orig["n"], len(ev_df)),
            ("당일 양봉률", cmp_orig["sig_bull"], 100.0),
            ("당일 상승률", cmp_orig["sig_up"],
             (ev_df["sig_up_vs_prev"].sum() / len(ev_df) * 100) if len(ev_df) else np.nan),
            ("다음날 양봉률", cmp_orig["nd_bull"],
             ev_df["nd_bullish"].fillna(False).astype(bool).sum() / len(ev_df.dropna(subset=["nd_daily_ret"])) * 100),
            ("다음날 상승률", cmp_orig["nd_up"],
             (ev_df["nd_daily_ret"] > 0).sum() / len(ev_df.dropna(subset=["nd_daily_ret"])) * 100),
            ("다음날 평균 등락", cmp_orig["nd_mean"], ev_df["nd_daily_ret"].mean()),
            ("매수체결률", cmp_orig["fill_rate"],
             ev_df["nd_entered"].fillna(False).astype(bool).sum() / len(ev_df) * 100),
            ("진입→종가 평균", cmp_orig["entry_ret"],
             ev_df[ev_df["nd_entered"].fillna(False)]["nd_entry_to_close_ret"].mean()),
            ("진입→종가 승률", cmp_orig["entry_win"],
             (ev_df[ev_df["nd_entered"].fillna(False)]["nd_entry_to_close_ret"] > 0).sum() /
             max(1, ev_df["nd_entered"].fillna(False).astype(bool).sum()) * 100),
        ]
        for name, a, b in rows:
            if isinstance(a, float) and ("등락" in name or "평균" in name) and "률" not in name:
                a_s = fmt_pct(a); b_s = fmt_pct(b)
                d_s = f"{b - a:+.2f}%p"
            elif name == "총 시그널":
                a_s = f"{int(a):,}"; b_s = f"{int(b):,}"
                d_s = f"{(b/a-1)*100:+.1f}%"
            else:
                a_s = fmt_pct_pos(a); b_s = fmt_pct_pos(b)
                d_s = f"{b - a:+.2f}%p"
            lines.append(f"| {name} | {a_s} | {b_s} | {d_s} |")
        lines.append("")

    # ── 7. 시그널 분포 ──
    lines.append("## 7. 시그널 분포 (참고)\n")
    lines.append(f"- 시그널 발생 종목 수: **{ev_df['ticker'].nunique():,}개**")
    top = ev_df.groupby(["ticker", "name"]).size().sort_values(ascending=False).head(15)
    lines.append("- TOP 15 종목:\n")
    lines.append("| # | 종목 | 코드 | 시그널 |")
    lines.append("|---|------|------|------:|")
    for j, ((tk, nm), n) in enumerate(top.items(), 1):
        lines.append(f"| {j} | {nm} | {tk} | {n} |")
    lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- 실행 시간: {elapsed:.2f}초")
    lines.append(f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
ORIG_CMP = {
    "n": 10324,
    "sig_bull": 100.0,
    "sig_up": 100.0,
    "nd_bull": 42.5,
    "nd_up": 46.3,
    "nd_mean": 0.25,
    "fill_rate": 90.9,
    "entry_ret": -0.21,
    "entry_win": 40.1,
}


def main():
    t0 = time.time()
    print("=" * 60)
    print("계단뛰기 시그널 (16·17 제거) 통계 분석")
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
    print(f"        총 시그널: {len(ev_df):,} (원본 10,324 대비 {(len(ev_df)/10324-1)*100:+.1f}%)")

    print("[4/4] 리포트 생성...")
    regime_df = load_kospi_regime()
    elapsed = time.time() - t0
    report = make_report(ev_df, regime_df, elapsed, cmp_orig=ORIG_CMP)

    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "stair_jump_stats_no1617.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트: {out} ({elapsed:.2f}초)")

    csv_out = os.path.join(out_dir, "stair_jump_events_no1617.csv")
    ev_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"이벤트 CSV: {csv_out}")


if __name__ == "__main__":
    main()
