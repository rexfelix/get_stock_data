"""
MTL/MTH 돌파 매매법 포트폴리오 백테스트

매매 규칙:
- 매수: 당일 종가가 최신 확정 MTH 고가 상향 돌파 (전일종가≤MTH고가 < 당일종가)
- 매도: 당일 종가가 최신 확정 MTL 저가 하향 돌파 (당일종가 < 최신 MTL 저가)
- 대상: KOSPI200 ∪ 시총≥3조 (349종목)
- 슬롯: 10종목, 거래대금 상위 매수, 슬롯당 1천만원

시나리오 비교:
  S10       : 지수필터 없음, 10슬롯 고정
  S10+F1    : KOSPI>MA60 일 때만 매수
  H10/5     : 상승장 10슬롯 / 하락장 5슬롯 (F1 기반)
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_3bar import get_mcap_tickers, load_amount_data, ENGINE
from backtest_3bar_portfolio_idx import load_kospi_filters, build_universe
from backtest_crash import load_all_data, FEE_BUY, FEE_SELL, TAX_SELL
from backtest_larry_mtl import find_swing_points, find_medium_points

START_DATE = "2023-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
SLOT_CAPITAL = 10_000_000


# ──────────────────────────────────────────────
# 종목별 MTL/MTH 사전 계산
# ──────────────────────────────────────────────
def precompute_mtl_mth(df: pd.DataFrame) -> dict:
    low = df["low"].values
    high = df["high"].values
    stls, sths = find_swing_points(df)
    mtls = find_medium_points(stls, low, is_low=True)
    mths = find_medium_points(sths, high, is_low=False)
    return {"mtls": mtls, "mths": mths}


def latest_confirmed(points: list, current_idx: int):
    """현 인덱스까지 확정된 최신 (pivot_idx, conf_idx) 반환."""
    latest = None
    for piv, conf in points:
        if conf <= current_idx:
            latest = (piv, conf)
        else:
            break
    return latest


# ──────────────────────────────────────────────
# 포트폴리오 엔진
# ──────────────────────────────────────────────
def run_portfolio(data, amount_data, ticker_list, name_map,
                   cap_up: int, cap_down: int, index_filter: pd.Series = None):
    initial_capital = cap_up * SLOT_CAPITAL

    # 데이터 준비 (DataFrame + MTL/MTH precompute)
    stock_info = {}
    for t in ticker_list:
        if t not in data:
            continue
        df = data[t].reset_index()
        df = df.rename(columns={"index": "date", "date": "date"})
        # data[t]는 index가 date. reset_index 후 'date' column 있음
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        df = df[~df["date"].duplicated(keep="last")].reset_index(drop=True)
        if len(df) < 10:
            continue
        pc = precompute_mtl_mth(df)
        date_to_idx = {d: i for i, d in enumerate(df["date"].values)}
        stock_info[t] = {"df": df, "pc": pc, "date_to_idx": date_to_idx}

    all_dates = set()
    for info in stock_info.values():
        all_dates.update(info["df"]["date"].values)
    all_dates = sorted(d for d in all_dates
                       if pd.Timestamp(START_DATE) <= pd.Timestamp(d) <= pd.Timestamp(END_DATE))

    positions = {}
    cash = initial_capital
    trades = []
    equity_curve = []

    for d in all_dates:
        d_ts = pd.Timestamp(d)

        # (A) 매도: 당일 종가가 최신 MTL 저가 하향 돌파
        to_close = []
        for t, pos in positions.items():
            info = stock_info.get(t)
            if info is None:
                continue
            idx = info["date_to_idx"].get(d)
            if idx is None or idx <= pos["entry_idx"]:
                continue
            row = info["df"].iloc[idx]
            close_now = row["close"]
            mtl_latest = latest_confirmed(info["pc"]["mtls"], idx)
            if mtl_latest is None:
                continue
            mtl_low = info["df"].iloc[mtl_latest[0]]["low"]
            if close_now < mtl_low:
                # 매도
                qty = pos["quantity"]
                revenue = qty * close_now
                net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
                buy_cost = pos["entry_price"] * qty
                buy_fee = buy_cost * FEE_BUY
                pnl = net - buy_cost - buy_fee
                ret_pct = pnl / (buy_cost + buy_fee) * 100
                trades.append({
                    "ticker": t, "name": name_map.get(t, t),
                    "buy_date": pos["entry_date"], "buy_price": pos["entry_price"],
                    "sell_date": d_ts, "sell_price": close_now, "quantity": qty,
                    "pnl": pnl, "return_pct": ret_pct, "reason": "MTL이탈",
                })
                cash += net
                to_close.append(t)
        for t in to_close:
            del positions[t]

        # (B) 지수필터로 cap 결정
        if index_filter is not None and d_ts in index_filter.index:
            cap_today = cap_up if bool(index_filter.loc[d_ts]) else cap_down
        else:
            cap_today = cap_up

        # (C) 매수 시그널: MTH 상향 돌파
        if len(positions) < cap_today:
            slots_free = cap_today - len(positions)
            candidates = []
            for t, info in stock_info.items():
                if t in positions:
                    continue
                idx = info["date_to_idx"].get(d)
                if idx is None or idx < 1:
                    continue
                row = info["df"].iloc[idx]
                close_now = row["close"]
                prev_close = info["df"].iloc[idx - 1]["close"]

                mtl_latest = latest_confirmed(info["pc"]["mtls"], idx)
                if mtl_latest is None:
                    continue
                # MTL 이후 확정된 MTH
                mth_latest = None
                for piv, conf in info["pc"]["mths"]:
                    if conf <= idx and piv > mtl_latest[0]:
                        mth_latest = (piv, conf)
                if mth_latest is None:
                    continue
                mth_high = info["df"].iloc[mth_latest[0]]["high"]

                if prev_close <= mth_high and close_now > mth_high:
                    # 당일 거래대금
                    amt = 0
                    if t in amount_data and d_ts in amount_data[t].index:
                        v = amount_data[t].loc[d_ts, "amount"]
                        amt = float(v) if pd.notna(v) else 0
                    candidates.append((t, amt, idx, close_now))

            candidates.sort(key=lambda x: -x[1])  # 거래대금 내림차순
            for t, amt, idx, close_p in candidates:
                if slots_free == 0 or cash < SLOT_CAPITAL * 0.9:
                    break
                qty = int(SLOT_CAPITAL / (close_p * (1 + FEE_BUY)))
                if qty <= 0:
                    continue
                cost = qty * close_p
                fee = cost * FEE_BUY
                if cash < cost + fee:
                    continue
                cash -= (cost + fee)
                positions[t] = {
                    "entry_price": close_p, "quantity": qty, "entry_date": d_ts,
                    "entry_idx": idx,
                }
                slots_free -= 1

        # (D) 자산 평가
        pos_val = 0
        for t, pos in positions.items():
            info = stock_info.get(t)
            if info is None:
                continue
            idx = info["date_to_idx"].get(d)
            if idx is None:
                continue
            pos_val += pos["quantity"] * info["df"].iloc[idx]["close"]
        equity_curve.append({"date": d_ts, "equity": cash + pos_val,
                              "n_positions": len(positions),
                              "cap_today": cap_today})

    # 미청산 종가 청산
    last_d = all_dates[-1] if all_dates else pd.Timestamp(END_DATE)
    for t, pos in list(positions.items()):
        info = stock_info.get(t)
        if info is None:
            continue
        idx = info["date_to_idx"].get(last_d)
        if idx is None:
            continue
        last_close = info["df"].iloc[idx]["close"]
        qty = pos["quantity"]
        revenue = qty * last_close
        net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
        buy_cost = pos["entry_price"] * qty
        buy_fee = buy_cost * FEE_BUY
        pnl = net - buy_cost - buy_fee
        ret_pct = pnl / (buy_cost + buy_fee) * 100
        trades.append({
            "ticker": t, "name": name_map.get(t, t),
            "buy_date": pos["entry_date"], "buy_price": pos["entry_price"],
            "sell_date": pd.Timestamp(last_d), "sell_price": last_close,
            "quantity": qty, "pnl": pnl, "return_pct": ret_pct,
            "reason": "미청산",
        })
        cash += net

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("date") if equity_curve else pd.DataFrame()
    return trades_df, equity_df, initial_capital


def summarize(trades_df, equity_df, initial_capital, label=""):
    final_eq = equity_df["equity"].iloc[-1] if not equity_df.empty else initial_capital
    total_ret = (final_eq - initial_capital) / initial_capital * 100
    if not equity_df.empty:
        dur = (equity_df.index[-1] - equity_df.index[0]).days
        years = dur / 365.25
        cagr = ((final_eq / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        roll_max = equity_df["equity"].cummax()
        dd = equity_df["equity"] / roll_max - 1
        mdd = dd.min() * 100
        avg_pos = equity_df["n_positions"].mean()
    else:
        cagr = mdd = avg_pos = 0
    m = {"label": label, "initial_capital": initial_capital,
         "total_return": total_ret, "cagr": cagr, "final_equity": final_eq,
         "mdd": mdd, "n_trades": len(trades_df), "avg_positions": avg_pos}
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        m["win_rate"] = len(wins) / len(trades_df) * 100
        gp = wins["pnl"].sum() if len(wins) > 0 else 0
        gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        m["pf"] = gp / gl if gl > 0 else float("inf")
        tc = trades_df.copy()
        tc["hd"] = (pd.to_datetime(tc["sell_date"]) - pd.to_datetime(tc["buy_date"])).dt.days
        m["hold_days"] = tc["hd"].mean()
    else:
        m["win_rate"] = m["pf"] = m["hold_days"] = 0
    m["calmar"] = m["cagr"] / abs(m["mdd"]) if m["mdd"] != 0 else 0
    return m


def year_metrics(equity_df, trades_df, year):
    eq = equity_df[equity_df.index.year == year]
    if eq.empty:
        return None
    yr_ret = (eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1) * 100
    roll_max = eq["equity"].cummax()
    dd = eq["equity"] / roll_max - 1
    mdd = dd.min() * 100
    t = trades_df[(pd.to_datetime(trades_df["buy_date"]).dt.year == year)]
    n_tr = len(t)
    wr = (t["pnl"] > 0).mean() * 100 if n_tr > 0 else 0
    return {"year": year, "return": yr_ret, "mdd": mdd,
            "n_trades": n_tr, "win_rate": wr}


def main():
    t0 = time.time()
    print("[1/4] 유니버스 구성...")
    tickers, name_map = build_universe()
    print(f"     {len(tickers)}종목")

    print("[2/4] KOSPI 필터...")
    _, filters = load_kospi_filters()
    f1 = filters["F1_MA60위"]

    print("[3/4] 데이터 로딩...")
    all_data, amt_data = {}, {}
    for i in range(0, len(tickers), 500):
        batch = tickers[i:i+500]
        all_data.update(load_all_data(batch, START_DATE, END_DATE))
        amt_data.update(load_amount_data(batch, START_DATE, END_DATE))

    scenarios = [
        {"name": "S10", "cap_up": 10, "cap_down": 10, "filter": None},
        {"name": "S10+F1", "cap_up": 10, "cap_down": 0, "filter": f1},
        {"name": "H10/5", "cap_up": 10, "cap_down": 5, "filter": f1},
    ]

    print(f"[4/4] {len(scenarios)}개 시나리오 실행...")
    results = {}
    for sc in scenarios:
        print(f"  ▸ {sc['name']:8s}...", end=" ", flush=True)
        ts = time.time()
        tdf, edf, init_cap = run_portfolio(
            all_data, amt_data, tickers, name_map,
            cap_up=sc["cap_up"], cap_down=sc["cap_down"],
            index_filter=sc["filter"],
        )
        m = summarize(tdf, edf, init_cap, sc["name"])
        results[sc["name"]] = {"trades": tdf, "equity": edf, "summary": m}
        pf = "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}"
        print(f"초기 {init_cap/1e8:.1f}억 → {m['final_equity']/1e8:.2f}억 "
              f"| CAGR {m['cagr']:.2f}% | MDD {m['mdd']:.2f}% | Calmar {m['calmar']:.2f} "
              f"| 거래 {m['n_trades']}건 PF {pf} | 승률 {m['win_rate']:.1f}% ({time.time()-ts:.1f}s)")

    elapsed = time.time() - t0

    # 리포트
    lines = []
    lines.append("# MTL/MTH 돌파 매매법 포트폴리오 백테스트\n")
    lines.append("## 매매 규칙\n")
    lines.append("| 항목 | 내용 |")
    lines.append("|---|---|")
    lines.append("| 매수 | 당일 종가가 최신 확정 MTH 고가 상향 돌파 (전일 종가 ≤ MTH고가 < 당일 종가) |")
    lines.append("| 매도 | 당일 종가가 최신 확정 MTL 저가 하향 돌파 |")
    lines.append("| MTL/MTH | STL/STH 기반 중기 스윙 포인트 (연속 3 STL/STH 중 가운데 극값) |")
    lines.append("")

    lines.append("## 개요\n")
    lines.append(f"- **유니버스**: KOSPI200 ∪ 시총≥3조 ({len(tickers)}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **슬롯당 자본**: {SLOT_CAPITAL:,}원 (초기자본 = 최대슬롯×1천만)")
    lines.append(f"- **매수 우선순위**: 당일 거래대금 순")
    lines.append(f"- **F1 지수필터**: KOSPI > KOSPI MA60")
    lines.append("")

    # 종합 비교
    lines.append("## 시나리오 종합 비교\n")
    lines.append("| 시나리오 | 초기자본 | 최종자본 | 총수익률 | CAGR | MDD | Calmar | 거래수 | 승률 | PF | 평균보유 | 평균포지션 |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for sc in scenarios:
        name = sc["name"]
        m = results[name]["summary"]
        pf = "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}"
        lines.append(
            f"| {name} | {m['initial_capital']/1e8:.1f}억 "
            f"| {m['final_equity']/1e8:.2f}억 | {m['total_return']:.2f}% "
            f"| {m['cagr']:.2f}% | {m['mdd']:.2f}% | {m['calmar']:.2f} "
            f"| {m['n_trades']}건 | {m['win_rate']:.1f}% | {pf} "
            f"| {m['hold_days']:.1f}일 | {m['avg_positions']:.1f}/{sc['cap_up']} |"
        )
    lines.append("")

    # 연도별
    for sc in scenarios:
        name = sc["name"]
        lines.append(f"## 시나리오 · {name} — 연도별 성과\n")
        lines.append("| 연도 | 수익률 | MDD | 거래수 | 승률 |")
        lines.append("|---|:---:|:---:|:---:|:---:|")
        eq_df = results[name]["equity"]
        tr_df = results[name]["trades"]
        for yr in [2023, 2024, 2025, 2026]:
            ym = year_metrics(eq_df, tr_df, yr)
            if ym is None:
                continue
            lines.append(f"| {yr} | {ym['return']:.2f}% | {ym['mdd']:.2f}% "
                         f"| {ym['n_trades']}건 | {ym['win_rate']:.1f}% |")
        lines.append("")

    # TOP 거래 (최고 CAGR 전략)
    best = max(results, key=lambda k: results[k]["summary"]["cagr"])
    lines.append(f"## 최고 CAGR · {best} — TOP 10 수익 거래\n")
    bt = results[best]["trades"]
    if not bt.empty:
        top = bt.nlargest(10, "return_pct")
        lines.append("| # | 종목 | 매수일 | 매도일 | 매수가 | 매도가 | 수익률 | 보유일 | 사유 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for j, (_, r) in enumerate(top.iterrows()):
            hd = (pd.Timestamp(r["sell_date"]) - pd.Timestamp(r["buy_date"])).days
            lines.append(
                f"| {j+1} | {r['name']} "
                f"| {pd.Timestamp(r['buy_date']).strftime('%Y-%m-%d')} "
                f"| {pd.Timestamp(r['sell_date']).strftime('%Y-%m-%d')} "
                f"| {r['buy_price']:,.0f} | {r['sell_price']:,.0f} "
                f"| {r['return_pct']:.2f}% | {hd}일 | {r['reason']} |"
            )
    lines.append("")

    # 3-bar와 비교 (선행 실험 있음을 가정, 수치 하드코딩)
    lines.append("## 3-bar 매매규칙 (H10/5) 대비 비교\n")
    lines.append("| 전략 | CAGR | MDD | Calmar | 거래수 | 평균보유 |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    m_h = results.get("H10/5", {}).get("summary", {})
    if m_h:
        lines.append(f"| MTL/MTH 돌파 · H10/5 | {m_h['cagr']:.2f}% | {m_h['mdd']:.2f}% "
                     f"| {m_h['calmar']:.2f} | {m_h['n_trades']}건 | {m_h['hold_days']:.1f}일 |")
    lines.append("| 3-bar · H10/5 (기준선) | 25.34% | -55.63% | 0.46 | 936건 | 11.3일 |")
    lines.append("")

    lines.append(f"## 실행 정보\n- 실행시간: {elapsed:.1f}초")
    lines.append(f"- 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    out = os.path.join(os.path.dirname(__file__), "results",
                       "backtest_mtl_mth_portfolio.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n리포트: {out}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
