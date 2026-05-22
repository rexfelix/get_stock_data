"""
계단뛰기 매매 백테스트 (주봉)

일봉 데이터로 주봉(W-FRI)을 만들어 동일한 계단뛰기 시그널 + 매도 규칙을 적용.
  - OHLC: first / max / min / last,  Volume: sum
  - MA5  = 5 주, MA120 = 120 주
  - 매수: 시그널 다음 주봉의 주봉 종가 상향 돌파 (만료 1봉 = 1주)
  - 매도: tpsl / ma5 / sameday (일봉 버전과 동일 의미, 단위만 주봉)

테스트 기간 (디폴트): 2025-01-01 ~ today
주봉 warmup: 1,000일 (~143주, MA120 안정화 위해 충분)
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from backtest_crash import (
    INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    load_all_data,
)
from backtest_stair_jump import (
    find_signals, run_backtest, _metrics,
    load_universe,
    PRICE_FLOOR, MCAP_THRESHOLD,
)


START_DATE = "2025-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

WARMUP_DAYS = 1000

DEFAULT_TGT = 7.0
DEFAULT_MIN = 3.0
DEFAULT_SL = 3.0


# ──────────────────────────────────────────────
# 주봉 변환 + 지표
# ──────────────────────────────────────────────
def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """W-FRI 주봉 변환. 휴장 등으로 거래일 1~4일만 있는 주도 그대로 한 봉."""
    if daily.empty:
        return daily
    w = daily.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return w.dropna(subset=["open", "high", "low", "close"])


def calc_indicators_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma120"] = df["close"].rolling(window=120).mean()
    return df


# ──────────────────────────────────────────────
# 데이터 로딩 (warmup 포함)
# ──────────────────────────────────────────────
def load_all_with_warmup(ticker_list, start, end):
    warm = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    print(f"[2/3] 데이터 로딩 (warmup {warm} ~ {end})...")
    all_data: dict[str, pd.DataFrame] = {}
    batch = 500
    for i in range(0, len(ticker_list), batch):
        chunk = ticker_list[i:i + batch]
        d = load_all_data(chunk, warm, end)
        all_data.update(d)
    print(f"        {len(all_data)}종목 로딩 완료")
    return all_data


# ──────────────────────────────────────────────
# 풀-유니버스 실행
# ──────────────────────────────────────────────
def run_universe_weekly(tgt_pct, min_pct, sl_pct,
                        ticker_list, name_map, all_data,
                        start_date, end_date,
                        exit_mode="tpsl"):
    summaries = []
    all_trades = []
    n_signals_total = 0

    for ticker in ticker_list:
        if ticker not in all_data:
            continue
        df_daily = all_data[ticker]
        if len(df_daily) < 30:
            continue
        df_w = calc_indicators_weekly(resample_weekly(df_daily))
        df_test = df_w.loc[start_date:end_date]
        if len(df_test) < 5:
            continue
        nm = name_map.get(ticker, ticker)
        cap, trades, equity, n_sig = run_backtest(
            df_test, ticker, nm, tgt_pct, min_pct, sl_pct,
            exit_mode=exit_mode,
        )
        n_signals_total += n_sig
        if trades:
            t_df = pd.DataFrame(trades)
            wins = (t_df["pnl"] > 0).sum()
            summaries.append({
                "ticker": ticker, "name": nm,
                "n_signals": n_sig,
                "n_trades": len(trades),
                "final_equity": equity[-1]["equity"] if equity else cap,
                "win_rate": wins / len(trades) * 100,
                "total_pnl": t_df["pnl"].sum(),
                "avg_return": t_df["return_pct"].mean(),
            })
            all_trades.append(t_df)
        else:
            summaries.append({
                "ticker": ticker, "name": nm,
                "n_signals": n_sig,
                "n_trades": 0,
                "final_equity": INITIAL_CAPITAL,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_return": 0,
            })

    sum_df = pd.DataFrame(summaries)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return sum_df, trades_df, n_signals_total


# ──────────────────────────────────────────────
# 리포트
# ──────────────────────────────────────────────
def _exit_desc(exit_mode, tgt, min_p, sl):
    return {
        "tpsl": f"익절 {tgt}% / 손절 {sl}% / 이익보전 {min_p}%",
        "ma5": f"익절 {tgt}% / MA5(주) 종가이탈 손절 / 이익보전 {min_p}%",
        "sameday": "매수 주봉 종가 무조건 매도 (당일=당주)",
    }[exit_mode]


def _title_tag(exit_mode):
    return {"tpsl": "Tgt/Min/SL", "ma5": "Tgt/Min/MA5", "sameday": "당주종가"}[exit_mode]


def _suffix(exit_mode):
    return {"tpsl": "", "ma5": "_ma5", "sameday": "_sameday"}[exit_mode]


def make_single_report(sum_df, trades_df, n_signals, tgt, min_p, sl,
                       elapsed, exit_mode, start_date, end_date) -> str:
    m = _metrics(trades_df, sum_df)
    lines = []
    lines.append(f"# 계단뛰기 매매 백테스트 (주봉, {_title_tag(exit_mode)})\n")
    lines.append("## 전략 개요\n")
    lines.append("- **봉**: 일봉 → 주봉(W-FRI) OHLC=first/max/min/last, Vol=sum")
    lines.append("- **MA**: MA5 = 5주, MA120 = 120주")
    lines.append(f"- **기간**: {start_date} ~ {end_date}")
    lines.append(f"- **유니버스**: 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억, 종가 ≥ {PRICE_FLOOR:,}원")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append("- **매수**: 시그널 다음 주봉의 시그널 주봉 종가 상향 돌파 (만료 1봉=1주)")
    lines.append(f"- **매도**: {_exit_desc(exit_mode, tgt, min_p, sl)}\n")

    lines.append("## 성과 요약\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 유니버스 종목 | {m['n_stocks']:,} |")
    lines.append(f"| 거래 발생 종목 | {m['n_traded']:,} |")
    lines.append(f"| 총 시그널 | {n_signals:,} |")
    lines.append(f"| 총 거래 | {m['n_trades']:,} |")
    if m["n_trades"]:
        pf_s = f"{m['pf']:.2f}" if m['pf'] != float('inf') else "∞"
        lines.append(f"| 종목 평균 수익률 | {m['stock_avg_ret']:.2f}% |")
        lines.append(f"| 종목 중위 수익률 | {m['stock_med_ret']:.2f}% |")
        lines.append(f"| 전체 승률 | {m['win_rate']:.1f}% |")
        lines.append(f"| 손익비 (PF) | {pf_s} |")
        lines.append(f"| 총 손익 | {m['total_pnl']:,.0f}원 |")
        lines.append(f"| 거래당 평균 수익률 | {m['avg_ret']:.2f}% |")
        lines.append(f"| 평균 보유 (캘린더일) | {m['hold_days']:.1f}일 |")
    lines.append("")

    if not trades_df.empty:
        lines.append("## 매도 사유 분포\n")
        lines.append("| 사유 | 건수 | 평균 수익률 | 승률 | 총 손익 |")
        lines.append("|------|------|------------|------|---------|")
        for reason, grp in trades_df.groupby("reason"):
            n = len(grp)
            wr = (grp["pnl"] > 0).sum() / n * 100
            lines.append(f"| {reason} | {n:,} | {grp['return_pct'].mean():.2f}% | {wr:.1f}% | {grp['pnl'].sum():,.0f} |")
        lines.append("")

        traded = sum_df[sum_df["n_trades"] > 0].copy()
        if not traded.empty:
            traded["stock_ret"] = (traded["final_equity"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            lines.append("## TOP 10 종목\n")
            lines.append("| # | 종목 | 코드 | 수익률 | 거래 | 승률 |")
            lines.append("|---|------|------|--------|------|------|")
            for j, (_, r) in enumerate(traded.nlargest(10, "stock_ret").iterrows()):
                lines.append(f"| {j+1} | {r['name']} | {r['ticker']} | {r['stock_ret']:.2f}% | {r['n_trades']} | {r['win_rate']:.0f}% |")
            lines.append("")
            lines.append("## BOTTOM 10 종목\n")
            lines.append("| # | 종목 | 코드 | 수익률 | 거래 | 승률 |")
            lines.append("|---|------|------|--------|------|------|")
            for j, (_, r) in enumerate(traded.nsmallest(10, "stock_ret").iterrows()):
                lines.append(f"| {j+1} | {r['name']} | {r['ticker']} | {r['stock_ret']:.2f}% | {r['n_trades']} | {r['win_rate']:.0f}% |")
            lines.append("")

        lines.append("## 전체 거래 기록 (최근 200건)\n")
        lines.append("| # | 종목 | 매수일 | 매수가 | 매도일 | 매도가 | 수익률 | 사유 |")
        lines.append("|---|------|--------|--------|--------|--------|--------|------|")
        td = trades_df.sort_values("sell_date").tail(200)
        for j, (_, t) in enumerate(td.iterrows()):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            lines.append(f"| {j+1} | {t['name']} | {bd} | {t['buy_price']:,.0f} | {sd} | {t['sell_price']:,.0f} | {t['return_pct']:.2f}% | {t['reason']} |")
        lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- 실행 시간: {elapsed:.2f}초")
    lines.append(f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def make_opt_report(rows, elapsed, exit_mode, start_date, end_date) -> str:
    df = pd.DataFrame(rows)
    lines = []
    title_tag = _title_tag(exit_mode)
    stop_label = "SL" if exit_mode == "tpsl" else "MA5"
    stop_desc = "고정 손절 %" if exit_mode == "tpsl" else "MA5(주) 종가이탈 손절"
    lines.append(f"# 계단뛰기 매매 백테스트 (주봉) — 매도 최적화 ({title_tag})\n")
    lines.append("## 전략 개요\n")
    lines.append("- **봉**: 일봉 → 주봉(W-FRI)")
    lines.append(f"- **기간**: {start_date} ~ {end_date}")
    lines.append(f"- **유니버스**: 시총 ≥ {MCAP_THRESHOLD/1e8:,.0f}억, 종가 ≥ {PRICE_FLOOR:,}원")
    lines.append("- **매수**: 시그널 다음 주봉의 주봉 종가 상향 돌파")
    lines.append(f"- **매도 후보**: 익절(Tgt) × 이익보전(Min) × {stop_desc}\n")

    def stop_cell(sl_val):
        return f"{sl_val:.1f}%" if exit_mode == "tpsl" else "MA5"

    lines.append("## 전체 그리드 결과\n")
    lines.append(f"| # | Tgt | Min | {stop_label} | 거래 | 승률 | PF | 총손익 | 거래당% | 종목평균% | 보유일 |")
    lines.append("|---|----:|----:|---:|-----:|-----:|----:|--------:|--------:|----------:|------:|")
    df_sorted = df.sort_values("total_pnl", ascending=False).reset_index(drop=True)
    for j, r in df_sorted.iterrows():
        pf_s = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "∞"
        lines.append(
            f"| {j+1} | {r['tgt']:.1f}% | {r['min']:.1f}% | {stop_cell(r['sl'])} "
            f"| {int(r['n_trades']):,} | {r['win_rate']:.1f}% | {pf_s} "
            f"| {r['total_pnl']:,.0f} | {r['avg_ret']:.2f}% | {r['stock_avg_ret']:.2f}% | {r['hold_days']:.1f} |"
        )
    lines.append("")

    for metric, asc, title in [
        ("total_pnl", False, "총손익 TOP 10"),
        ("pf", False, "손익비 TOP 10"),
        ("stock_avg_ret", False, "종목 평균 수익률 TOP 10"),
    ]:
        d = df.copy()
        if metric == "pf":
            d = d[d["pf"] != float("inf")]
        d = d.sort_values(metric, ascending=asc).head(10).reset_index(drop=True)
        lines.append(f"## {title}\n")
        lines.append(f"| # | Tgt | Min | {stop_label} | 거래 | 승률 | PF | 총손익 | 거래당% | 종목평균% |")
        lines.append("|---|----:|----:|---:|-----:|-----:|----:|--------:|--------:|----------:|")
        for j, r in d.iterrows():
            pf_s = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "∞"
            lines.append(
                f"| {j+1} | {r['tgt']:.1f}% | {r['min']:.1f}% | {stop_cell(r['sl'])} "
                f"| {int(r['n_trades']):,} | {r['win_rate']:.1f}% | {pf_s} "
                f"| {r['total_pnl']:,.0f} | {r['avg_ret']:.2f}% | {r['stock_avg_ret']:.2f}% |"
            )
        lines.append("")

    best_pnl = df.sort_values("total_pnl", ascending=False).iloc[0]
    df_pf = df[df["pf"] != float("inf")]
    best_pf = df_pf.sort_values("pf", ascending=False).iloc[0] if not df_pf.empty else best_pnl
    lines.append("## 최적 파라미터 추천\n")
    lines.append(f"- **총손익 최대**: Tgt={best_pnl['tgt']:.1f}%, Min={best_pnl['min']:.1f}%, {stop_label}={stop_cell(best_pnl['sl'])} → 총손익 {best_pnl['total_pnl']:,.0f}원, PF {best_pnl['pf']:.2f}, 승률 {best_pnl['win_rate']:.1f}%")
    lines.append(f"- **손익비 최대**: Tgt={best_pf['tgt']:.1f}%, Min={best_pf['min']:.1f}%, {stop_label}={stop_cell(best_pf['sl'])} → PF {best_pf['pf']:.2f}, 총손익 {best_pf['total_pnl']:,.0f}원, 거래 {int(best_pf['n_trades'])}건")
    lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- 실행 시간: {elapsed:.2f}초")
    lines.append(f"- 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def run_single(tgt, min_p, sl, exit_mode="tpsl"):
    t0 = time.time()
    base_dir = os.path.dirname(__file__)
    ticker_list, name_map = load_universe()
    all_data = load_all_with_warmup(ticker_list, START_DATE, END_DATE)

    label = _title_tag(exit_mode)
    if exit_mode == "sameday":
        print(f"[3/3] 백테스트 (주봉, {label})...")
    elif exit_mode == "ma5":
        print(f"[3/3] 백테스트 (주봉, {label}: Tgt={tgt}% / Min={min_p}% / MA5손절)...")
    else:
        print(f"[3/3] 백테스트 (주봉, {label}: Tgt={tgt}% / Min={min_p}% / SL={sl}%)...")

    sum_df, trades_df, n_sig = run_universe_weekly(
        tgt, min_p, sl, ticker_list, name_map, all_data,
        START_DATE, END_DATE, exit_mode=exit_mode,
    )
    elapsed = time.time() - t0
    n_traded = int((sum_df["n_trades"] > 0).sum()) if not sum_df.empty else 0
    print(f"        시그널 {n_sig:,}건, 거래종목 {n_traded}, 총거래 {len(trades_df):,}건")

    report = make_single_report(sum_df, trades_df, n_sig, tgt, min_p, sl,
                                elapsed, exit_mode, START_DATE, END_DATE)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"backtest_stair_jump_weekly{_suffix(exit_mode)}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트: {out} ({elapsed:.2f}초)")


def run_optimize(exit_mode="tpsl"):
    t0 = time.time()
    base_dir = os.path.dirname(__file__)
    ticker_list, name_map = load_universe()
    all_data = load_all_with_warmup(ticker_list, START_DATE, END_DATE)

    tgt_grid = [5.0, 7.0, 10.0, 15.0, 20.0]
    min_grid = [1.0, 2.0, 3.0, 5.0]
    if exit_mode == "tpsl":
        sl_grid = [2.0, 3.0, 4.0, 5.0]
        combos = [(t, mn, s) for t in tgt_grid for mn in min_grid for s in sl_grid if mn < t]
    else:  # ma5
        combos = [(t, mn, 0.0) for t in tgt_grid for mn in min_grid if mn < t]
    print(f"[3/3] 그리드 최적화 (주봉, {_title_tag(exit_mode)}): {len(combos)} 조합")

    rows = []
    for k, (t, mn, s) in enumerate(combos, 1):
        sub_t = time.time()
        sum_df, trades_df, n_sig = run_universe_weekly(
            t, mn, s, ticker_list, name_map, all_data,
            START_DATE, END_DATE, exit_mode=exit_mode,
        )
        m = _metrics(trades_df, sum_df)
        row = {
            "tgt": t, "min": mn, "sl": s,
            "n_signals": n_sig, "n_trades": m["n_trades"],
            "win_rate": m["win_rate"], "pf": m["pf"],
            "total_pnl": m["total_pnl"], "avg_ret": m["avg_ret"],
            "stock_avg_ret": m["stock_avg_ret"], "stock_med_ret": m["stock_med_ret"],
            "hold_days": m["hold_days"],
        }
        rows.append(row)
        sub_e = time.time() - sub_t
        pf_s = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "∞"
        s_tag = f"S={s:.1f}" if exit_mode == "tpsl" else "MA5"
        print(f"  [{k:>3}/{len(combos)}] T={t:.1f} M={mn:.1f} {s_tag} | "
              f"거래 {m['n_trades']:,} 승률 {m['win_rate']:.1f}% PF {pf_s} "
              f"PNL {m['total_pnl']:>14,.0f} ({sub_e:.1f}s)")

    elapsed = time.time() - t0
    report = make_opt_report(rows, elapsed, exit_mode, START_DATE, END_DATE)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"backtest_stair_jump_weekly_optimize{_suffix(exit_mode)}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트: {out} ({elapsed:.2f}초)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="계단뛰기 매매 백테스트 (주봉)")
    parser.add_argument("--optimize", action="store_true",
                        help="매도 파라미터 그리드 최적화")
    parser.add_argument("--exit", dest="exit_mode", choices=["tpsl", "ma5", "sameday"],
                        default="tpsl", help="매도 방식 (default tpsl)")
    parser.add_argument("--tgt", type=float, default=DEFAULT_TGT, help="익절 %% (default 7)")
    parser.add_argument("--min", dest="min_p", type=float, default=DEFAULT_MIN, help="이익보전 %% (default 3)")
    parser.add_argument("--sl", type=float, default=DEFAULT_SL, help="손절 %% (default 3)")
    parser.add_argument("--start", default=START_DATE, help=f"시작일 (default {START_DATE})")
    parser.add_argument("--end", default=END_DATE, help=f"종료일 (default {END_DATE})")
    args = parser.parse_args()

    START_DATE = args.start
    END_DATE = args.end

    if args.optimize:
        if args.exit_mode == "sameday":
            print("sameday 모드는 최적화 파라미터 없음 → 단일 실행")
            run_single(args.tgt, args.min_p, args.sl, exit_mode="sameday")
        else:
            run_optimize(exit_mode=args.exit_mode)
    else:
        run_single(args.tgt, args.min_p, args.sl, exit_mode=args.exit_mode)
