"""
계단뛰기 시그널 (조건 7, 8, 9, 16, 17 제거) 통계 분석

원본 논리식: 1 && 2 && 3 && 4 && ((5 && 6 && 17) || (7 && 8 && 9 && 16)) && 10..15
변형 논리식: 1 && 2 && 3 && 4 && (5 && 6) && 10..15

→ Branch B 완전 소멸 (7, 8, 9, 16 모두 제거)
→ Branch A 의 17 추가 제거 → (5 && 6) 만 남음
→ 즉 "전날 양봉 + 오늘 양봉 + 오늘 저가 > 전날 중앙선" 패턴
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


def find_signals_no789_1617(df: pd.DataFrame) -> list[int]:
    """7, 8, 9, 16, 17 제거 → (5 && 6) 만 남음."""
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
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

        mid1 = (o[i-1] + c[i-1]) / 2

        # 5) open(1) < close(1)  AND  6) low(0) > (open(1)+close(1))/2
        # (Branch B 완전 소멸)
        if not ((o[i-1] < c[i-1]) and (l[i] > mid1)):
            continue

        # 10, 11, 12) close > ma5
        if not (c[i-2] > ma5[i-2] and c[i-1] > ma5[i-1] and c[i] > ma5[i]):
            continue
        # 13) close(0) > ma120(0)
        if not (c[i] > ma120[i]):
            continue
        # 15) close ≥ PRICE_FLOOR
        if c[i] < PRICE_FLOOR:
            continue

        signals.append(i)
    return signals


def collect_events(ticker: str, name: str, df: pd.DataFrame) -> list[dict]:
    df = calc_indicators(df)
    signals = find_signals_no789_1617(df)
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
            "sig_open": float(o[i]),
            "sig_high": float(h[i]),
            "sig_low": float(l[i]),
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


def make_report(ev_df: pd.DataFrame, regime_df: pd.DataFrame, elapsed: float,
                cmp_orig: dict, cmp_no1617: dict) -> str:
    ev_df = ev_df.copy()
    ev_df["signal_date"] = pd.to_datetime(ev_df["signal_date"])
    reg_series = regime_df["regime"]
    unique_dates = pd.DatetimeIndex(sorted(set(ev_df["signal_date"].unique())))
    full = pd.concat([reg_series, pd.Series(index=unique_dates, dtype=object)])
    full = full[~full.index.duplicated(keep="first")].sort_index().ffill()
    ev_df["regime"] = ev_df["signal_date"].map(full).fillna("약세장")

    lines = []
    lines.append("# 계단뛰기 시그널 (조건 7·8·9·16·17 제거) 통계 분석\n")
    lines.append("## 변경 사항\n")
    lines.append("- **원본 논리식**: `1 && 2 && 3 && 4 && ((5 && 6 && 17) || (7 && 8 && 9 && 16)) && 10..15`")
    lines.append("- **변형 논리식**: `1 && 2 && 3 && 4 && (5 && 6) && 10..15`")
    lines.append("- 제거: 7, 8, 9, 16, 17 → **Branch B 완전 소멸**, Branch A 의 17 추가 제거")
    lines.append("- 남은 시그널 의미: 전날 양봉(5) + 오늘 저가 > 전날 중앙선(6) + 오늘 양봉(4) + 3봉 MA5 위 + MA120 위\n")

    lines.append("## 분석 개요\n")
    lines.append(f"- **데이터 기간**: {DATA_START} ~ {DATA_END}")
    lines.append(f"- **유니버스**: 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억, 종가 ≥ {PRICE_FLOOR:,}원")
    lines.append(f"- **총 시그널 수**: {len(ev_df):,}건")
    cnt_reg = ev_df["regime"].value_counts()
    lines.append(f"  - 강세장: {cnt_reg.get('강세장', 0):,}건  /  약세장: {cnt_reg.get('약세장', 0):,}건")
    lines.append(f"  - 원본(전체 조건) 대비: {cmp_orig['n']:,} → {len(ev_df):,} ({(len(ev_df)/cmp_orig['n']-1)*100:+.1f}%)")
    lines.append(f"  - 16·17 제거 변형 대비: {cmp_no1617['n']:,} → {len(ev_df):,} ({(len(ev_df)/cmp_no1617['n']-1)*100:+.1f}%)")
    lines.append("")

    # ── 1. 당일 ──
    lines.append("## 1. 당일 (시그널일, t=0) 통계\n")
    lines.append("> 조건 4 (open<close) 로 양봉률 100% 유지. 17 제거되었으므로 '상승률(close>전일종가)' 은 100% 아님.\n")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        s = stat_block(sub, "sig_daily_ret", "sig_bullish")
        lines.extend(render_stats_table(label, s))

    # ── 2. 다음날 ──
    lines.append("## 2. 다음날 (t=+1) 통계 — 매수 체결 여부 무관\n")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        s = stat_block(sub, "nd_daily_ret", "nd_bullish")
        lines.extend(render_stats_table(label, s))

    # ── 3. 매수 체결 ──
    lines.append("## 3. 다음날 매수 체결 통계\n")
    lines.append("> 시그널 종가 상향 돌파 시 매수 (시가 GAP-UP 이면 시가, 아니면 시그널종가).\n")
    lines.append("| 구분 | 시그널 | 체결 | 체결률 | 갭 평균 | 매수→당일종가 평균 | 매수→당일종가 승률 |")
    lines.append("|------|------:|-----:|------:|--------:|-------------------:|-----------------:|")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        n_total = len(sub)
        entered = sub[sub["nd_entered"].fillna(False).astype(bool)]
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

    # ── 4. 연도별 ──
    lines.append("## 4. 연도별 요약 (다음날 등락률 기준)\n")
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
        ent = sub[sub["nd_entered"].fillna(False).astype(bool)]
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

    # ── 5. 3가지 변형 비교 ──
    lines.append("## 5. 변형 비교 (원본 vs 16·17 제거 vs 7·8·9·16·17 제거)\n")

    def safe(d, k, default=np.nan):
        return d.get(k, default)

    n_curr = len(ev_df)
    sig_up_curr = (ev_df["sig_up_vs_prev"].sum() / n_curr * 100) if n_curr else np.nan
    nd_count = len(ev_df.dropna(subset=["nd_daily_ret"]))
    nd_bull_curr = (ev_df["nd_bullish"].fillna(False).astype(bool).sum() / nd_count * 100) if nd_count else np.nan
    nd_up_curr = ((ev_df["nd_daily_ret"] > 0).sum() / nd_count * 100) if nd_count else np.nan
    nd_mean_curr = ev_df["nd_daily_ret"].mean()
    fill_curr = (ev_df["nd_entered"].fillna(False).astype(bool).sum() / n_curr * 100) if n_curr else np.nan
    ent_mask = ev_df["nd_entered"].fillna(False).astype(bool)
    n_ent_curr = ent_mask.sum()
    ent_mean_curr = ev_df[ent_mask]["nd_entry_to_close_ret"].mean()
    ent_win_curr = ((ev_df[ent_mask]["nd_entry_to_close_ret"] > 0).sum() / max(1, n_ent_curr) * 100)

    lines.append("| 항목 | 원본 | 16·17 제거 | **7·8·9·16·17 제거** |")
    lines.append("|------|----:|----------:|--------------------:|")
    rows = [
        ("총 시그널", cmp_orig["n"], cmp_no1617["n"], n_curr, "int"),
        ("당일 양봉률", 100.0, 100.0, 100.0, "pct"),
        ("당일 상승률", 100.0, cmp_no1617["sig_up"], sig_up_curr, "pct"),
        ("다음날 양봉률", cmp_orig["nd_bull"], cmp_no1617["nd_bull"], nd_bull_curr, "pct"),
        ("다음날 상승률", cmp_orig["nd_up"], cmp_no1617["nd_up"], nd_up_curr, "pct"),
        ("다음날 평균 등락", cmp_orig["nd_mean"], cmp_no1617["nd_mean"], nd_mean_curr, "ret"),
        ("매수체결률", cmp_orig["fill_rate"], cmp_no1617["fill_rate"], fill_curr, "pct"),
        ("진입→종가 평균", cmp_orig["entry_ret"], cmp_no1617["entry_ret"], ent_mean_curr, "ret"),
        ("진입→종가 승률", cmp_orig["entry_win"], cmp_no1617["entry_win"], ent_win_curr, "pct"),
    ]
    for name, a, b, c_val, kind in rows:
        if kind == "int":
            a_s, b_s, c_s = f"{int(a):,}", f"{int(b):,}", f"{int(c_val):,}"
        elif kind == "pct":
            a_s, b_s, c_s = fmt_pct_pos(a), fmt_pct_pos(b), fmt_pct_pos(c_val)
        else:  # ret
            a_s, b_s, c_s = fmt_pct(a), fmt_pct(b), fmt_pct(c_val)
        lines.append(f"| {name} | {a_s} | {b_s} | **{c_s}** |")
    lines.append("")

    # ── 6. 시그널 분포 ──
    lines.append("## 6. 시그널 분포 (참고)\n")
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


# 비교용 (이전 리포트 수치)
CMP_ORIG = {
    "n": 10324, "sig_bull": 100.0, "sig_up": 100.0,
    "nd_bull": 42.5, "nd_up": 46.3, "nd_mean": 0.25,
    "fill_rate": 90.9, "entry_ret": -0.21, "entry_win": 40.1,
}
CMP_NO1617 = {
    "n": 12036, "sig_bull": 100.0, "sig_up": 95.1,
    "nd_bull": 42.4, "nd_up": 46.1, "nd_mean": 0.23,
    "fill_rate": 90.7, "entry_ret": -0.22, "entry_win": 40.0,
}


def main():
    t0 = time.time()
    print("=" * 60)
    print("계단뛰기 시그널 (7·8·9·16·17 제거) 통계 분석")
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
    elapsed = time.time() - t0
    report = make_report(ev_df, regime_df, elapsed, CMP_ORIG, CMP_NO1617)

    out = os.path.join(os.path.dirname(__file__), "results",
                        "stair_jump_stats_no789_1617.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트: {out} ({elapsed:.2f}초)")

    csv_out = os.path.join(os.path.dirname(__file__), "results",
                            "stair_jump_events_no789_1617.csv")
    ev_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"이벤트 CSV: {csv_out}")


if __name__ == "__main__":
    main()
