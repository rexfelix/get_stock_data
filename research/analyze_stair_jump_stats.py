"""
계단뛰기 시그널 통계 분석

목적: 계단뛰기.md 종목 조건을 만족하는 종목의 당일/다음날 통계.
  - 양봉 확률 (close > open)
  - 등락률 (전일종가 대비 close) — 상승/하락 분리
  - 전체 + 강세장/약세장 구분 (KOSPI MA200 기준)

데이터: stocks 테이블 전 기간 (2019-01-02 ~ 2026-05-15)
유니버스: 시총 ≥ 3000억, 종가 ≥ 2000원 (backtest_stair_jump.py 와 동일)

출력: results/stair_jump_stats.md
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_stair_jump import (
    calc_indicators, find_signals, load_universe, load_all,
    MCAP_THRESHOLD, PRICE_FLOOR,
)
from backtest_crash import ENGINE


DATA_START = "2019-01-01"
DATA_END = "2026-05-17"


# ──────────────────────────────────────────────
# KOSPI 강/약세 분류
# ──────────────────────────────────────────────
def load_kospi_regime() -> pd.DataFrame:
    """KOSPI 일봉 → 강세/약세 라벨링.

    분류 기준 (둘 다 만족 = 강세, 그 외 = 약세):
      - close > MA200
      - MA200 5일 상승
    """
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


# ──────────────────────────────────────────────
# 시그널 → 통계 이벤트
# ──────────────────────────────────────────────
def collect_events(ticker: str, name: str, df: pd.DataFrame) -> list[dict]:
    """find_signals 가 찾은 시그널 인덱스마다 통계 이벤트 생성."""
    df = calc_indicators(df)
    signals = find_signals(df)
    if not signals:
        return []

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    dates = df.index

    events = []
    for i in signals:
        # PRICE_FLOOR 는 find_signals 내부에서 이미 체크됨 (mcap 은 유니버스로 필터)

        prev_close = c[i - 1]
        # 당일 (signal day)
        sig_close_open_ret = (c[i] - o[i]) / o[i] * 100        # 시가→종가 (조건4로 양수)
        sig_prev_close_ret = (c[i] - prev_close) / prev_close * 100  # 전일종가→종가
        sig_is_bullish = c[i] > o[i]  # 조건4로 True

        ev = {
            "ticker": ticker,
            "name": name,
            "signal_date": dates[i],
            "sig_close": float(c[i]),
            "sig_intraday_ret": sig_close_open_ret,
            "sig_daily_ret": sig_prev_close_ret,
            "sig_bullish": bool(sig_is_bullish),
            "sig_up_vs_prev": c[i] > prev_close,
        }

        # 다음날 (buy day per 매매규칙 1-2: 시그널 종가 상향돌파시 진입)
        if i + 1 < len(df) and not np.isnan(o[i + 1]) and o[i + 1] > 0:
            nd_open = o[i + 1]
            nd_high = h[i + 1]
            nd_low = l[i + 1]
            nd_close = c[i + 1]
            ev["nd_date"] = dates[i + 1]
            ev["nd_intraday_ret"] = (nd_close - nd_open) / nd_open * 100
            ev["nd_daily_ret"] = (nd_close - c[i]) / c[i] * 100  # 전일종가=sig_close
            ev["nd_bullish"] = bool(nd_close > nd_open)
            ev["nd_up_vs_prev"] = bool(nd_close > c[i])
            ev["nd_gap"] = (nd_open - c[i]) / c[i] * 100  # 갭

            # 매매규칙 1-2: 시그널 종가 상향 돌파 시 매수 (당일 진입체결 여부)
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
# 통계 계산
# ──────────────────────────────────────────────
def stat_block(df: pd.DataFrame, ret_col: str, bull_col: str) -> dict:
    """주어진 등락률 컬럼/양봉 컬럼에 대한 통계."""
    d = df.dropna(subset=[ret_col])
    n = len(d)
    if n == 0:
        return {"n": 0, "p_bullish": np.nan, "p_up": np.nan,
                "ret_mean": np.nan, "ret_median": np.nan,
                "up_n": 0, "up_mean": np.nan, "up_median": np.nan,
                "dn_n": 0, "dn_mean": np.nan, "dn_median": np.nan,
                "flat_n": 0,
                "up_max": np.nan, "dn_min": np.nan}
    bullish = d[bull_col].fillna(False).astype(bool).sum() if bull_col in d.columns else np.nan
    up = d[d[ret_col] > 0]
    dn = d[d[ret_col] < 0]
    flat = d[d[ret_col] == 0]
    return {
        "n": n,
        "p_bullish": (bullish / n * 100) if not np.isnan(bullish) else np.nan,
        "p_up": len(up) / n * 100,
        "ret_mean": d[ret_col].mean(),
        "ret_median": d[ret_col].median(),
        "up_n": len(up),
        "up_mean": up[ret_col].mean() if len(up) else np.nan,
        "up_median": up[ret_col].median() if len(up) else np.nan,
        "up_max": up[ret_col].max() if len(up) else np.nan,
        "dn_n": len(dn),
        "dn_mean": dn[ret_col].mean() if len(dn) else np.nan,
        "dn_median": dn[ret_col].median() if len(dn) else np.nan,
        "dn_min": dn[ret_col].min() if len(dn) else np.nan,
        "flat_n": len(flat),
    }


def fmt_pct(x, p=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{p}f}%"


def fmt_pct_pos(x, p=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}%"


def fmt_int(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{int(x):,}"


def render_stats_table(label: str, stats: dict) -> list[str]:
    lines = []
    lines.append(f"#### {label}\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 표본 수 | {fmt_int(stats['n'])} |")
    lines.append(f"| 양봉 확률 (close > open) | {fmt_pct_pos(stats['p_bullish'])} |")
    lines.append(f"| 상승 확률 (close > 전일종가) | {fmt_pct_pos(stats['p_up'])} |")
    lines.append(f"| 평균 등락률 | {fmt_pct(stats['ret_mean'])} |")
    lines.append(f"| 중앙 등락률 | {fmt_pct(stats['ret_median'])} |")
    lines.append(f"| 상승일 수 | {fmt_int(stats['up_n'])} |")
    lines.append(f"| 상승일 평균 | {fmt_pct(stats['up_mean'])} |")
    lines.append(f"| 상승일 중앙 | {fmt_pct(stats['up_median'])} |")
    lines.append(f"| 상승일 최대 | {fmt_pct(stats['up_max'])} |")
    lines.append(f"| 하락일 수 | {fmt_int(stats['dn_n'])} |")
    lines.append(f"| 하락일 평균 | {fmt_pct(stats['dn_mean'])} |")
    lines.append(f"| 하락일 중앙 | {fmt_pct(stats['dn_median'])} |")
    lines.append(f"| 하락일 최저 | {fmt_pct(stats['dn_min'])} |")
    lines.append(f"| 보합일 수 | {fmt_int(stats['flat_n'])} |")
    lines.append("")
    return lines


# ──────────────────────────────────────────────
# 리포트
# ──────────────────────────────────────────────
def make_report(ev_df: pd.DataFrame, regime_df: pd.DataFrame, elapsed: float) -> str:
    # signal_date 기준 regime 매핑 (KOSPI 휴장일은 forward-fill)
    ev_df = ev_df.copy()
    ev_df["signal_date"] = pd.to_datetime(ev_df["signal_date"])
    reg_series = regime_df["regime"]
    unique_dates = pd.DatetimeIndex(sorted(set(ev_df["signal_date"].unique())))
    full = pd.concat([reg_series, pd.Series(index=unique_dates, dtype=object)])
    full = full[~full.index.duplicated(keep="first")].sort_index().ffill()
    ev_df["regime"] = ev_df["signal_date"].map(full)
    ev_df["regime"] = ev_df["regime"].fillna("약세장")  # MA200 결측 초기 구간

    lines = []
    lines.append("# 계단뛰기 시그널 통계 분석\n")
    lines.append("## 분석 개요\n")
    lines.append(f"- **데이터 기간**: {DATA_START} ~ {DATA_END}")
    lines.append(f"- **유니버스**: 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억, 종가 ≥ {PRICE_FLOOR:,}원")
    lines.append("- **시그널 조건**: 계단뛰기.md 종목 조건 1~15 (Branch A 또는 B)")
    lines.append("- **강세장/약세장**: KOSPI close > MA200 AND MA200 5일 상승 = 강세장, 그 외 = 약세장")
    lines.append(f"- **총 시그널 수**: {len(ev_df):,}건")
    cnt_reg = ev_df["regime"].value_counts()
    lines.append(f"  - 강세장: {cnt_reg.get('강세장', 0):,}건  /  약세장: {cnt_reg.get('약세장', 0):,}건")
    lines.append("")

    lines.append("## 정의\n")
    lines.append("- **당일 (signal day, t=0)**: 시그널 발생한 봉. 조건 4로 항상 양봉(close>open).")
    lines.append("- **다음날 (t=+1)**: 매매규칙상 매수 후보일. 시그널 종가 상향돌파 시 매수.")
    lines.append("- **양봉 확률**: P(close > open) — 봉의 형태")
    lines.append("- **상승 확률**: P(close > 전일종가) — 일간 등락 방향")
    lines.append("- **등락률**: (close − 전일종가) / 전일종가 × 100  (당일=시그널전일대비, 다음날=시그널종가대비)")
    lines.append("- **갭**: 다음날 시가가 시그널 종가 대비 몇 % 위/아래")
    lines.append("- **매수체결률**: 다음날 시가≥시그널종가 OR 다음날 고가≥시그널종가 비율")
    lines.append("")

    # ── 1. 당일 (시그널일) 통계 ──
    lines.append("## 1. 당일 (시그널일, t=0) 통계\n")
    lines.append("> 조건 4 (open<close) 로 양봉 확률은 100%. 일간 등락률(전일종가 대비)은 변동 가능.\n")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        s = stat_block(sub, "sig_daily_ret", "sig_bullish")
        lines.extend(render_stats_table(label, s))

    # ── 2. 다음날 (매수일) 통계 — 모든 시그널 ──
    lines.append("## 2. 다음날 (t=+1) 통계 — 매수 체결 여부 무관\n")
    lines.append("> 시그널 발생 다음날 봉의 통계. 양봉/상승 모두 의미 있음.\n")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        s = stat_block(sub, "nd_daily_ret", "nd_bullish")
        lines.extend(render_stats_table(label, s))

    # ── 3. 다음날 매수체결 통계 ──
    lines.append("## 3. 다음날 매수 체결 통계\n")
    lines.append("> 매매규칙: 시그널 종가 상향 돌파 시 매수 (시가 GAP-UP 이면 시가, 아니면 시그널종가).\n")
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

    # ── 4. 매수 체결된 경우 다음날 통계 (실전 분포) ──
    lines.append("## 4. 매수 체결된 경우 — 매수가→당일종가 등락률 분포\n")
    lines.append("> 매매규칙 1-2 따라 시그널 종가에 진입했을 때, 그 날 종가까지의 수익률 분포.\n")
    for label, sub in [("전체", ev_df),
                       ("강세장", ev_df[ev_df["regime"] == "강세장"]),
                       ("약세장", ev_df[ev_df["regime"] == "약세장"])]:
        ent = sub[sub["nd_entered"].fillna(False)].copy()
        if ent.empty:
            lines.append(f"### {label}\n\n표본 없음.\n")
            continue
        # bullish 정의: 종가 > 진입가
        ent["_bull"] = ent["nd_entry_to_close_ret"] > 0
        s = stat_block(ent, "nd_entry_to_close_ret", "_bull")
        # 양봉 확률은 사용하지 않음 (의미 동일) — 별도 라벨링
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

    # ── 6. 시그널 분포 ──
    lines.append("## 6. 시그널 분포 (참고)\n")
    lines.append(f"- 시그널 발생 종목 수: **{ev_df['ticker'].nunique():,}개**")
    top = ev_df.groupby(["ticker", "name"]).size().sort_values(ascending=False).head(15)
    lines.append("- TOP 15 종목 (시그널 횟수):\n")
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
def main():
    t0 = time.time()
    print("=" * 60)
    print("계단뛰기 시그널 통계 분석")
    print("=" * 60)

    ticker_list, name_map = load_universe()
    all_data = load_all(ticker_list, DATA_START, DATA_END)

    print(f"[3/4] 시그널 수집 & 이벤트 계산 ({len(all_data)} 종목)...")
    events = []
    for k, ticker in enumerate(ticker_list, 1):
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        if len(df) < 130:
            continue
        nm = name_map.get(ticker, ticker)
        evs = collect_events(ticker, nm, df)
        events.extend(evs)
        if k % 500 == 0:
            print(f"   {k}/{len(ticker_list)} 처리, 누적 이벤트 {len(events):,}")

    ev_df = pd.DataFrame(events)
    print(f"        총 시그널 이벤트: {len(ev_df):,}")

    print("[4/4] KOSPI 강/약세 라벨링 & 리포트 생성...")
    regime_df = load_kospi_regime()
    elapsed = time.time() - t0
    report = make_report(ev_df, regime_df, elapsed)

    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "stair_jump_stats.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트: {out} ({elapsed:.2f}초)")

    # CSV 도 같이 저장 (디버깅용)
    csv_out = os.path.join(out_dir, "stair_jump_events.csv")
    ev_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"이벤트 CSV: {csv_out}")


if __name__ == "__main__":
    main()
