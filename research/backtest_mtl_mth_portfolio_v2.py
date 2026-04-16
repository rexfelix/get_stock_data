"""
MTL/MTH 돌파 매매법 V2 — 진입 MTH 캔들 저가 손절 추가

매매 규칙 (v1과 차이: 손절선 추가):
- 매수: 당일 종가가 최신 확정 MTH 고가 상향 돌파
- **손절: 당일 종가 < 진입 시 MTH 캔들(pivot)의 저가** (NEW)
- 매도: 당일 종가 < 최신 확정 MTL 저가 (기존 MTL 트레일링)
- 손절 조건이 먼저 체크됨

시나리오: S10, S10+F1, H10/5
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


def precompute_mtl_mth(df: pd.DataFrame) -> dict:
    low = df["low"].values
    high = df["high"].values
    stls, sths = find_swing_points(df)
    mtls = find_medium_points(stls, low, is_low=True)
    mths = find_medium_points(sths, high, is_low=False)
    return {"mtls": mtls, "mths": mths}


def latest_confirmed(points: list, current_idx: int):
    latest = None
    for piv, conf in points:
        if conf <= current_idx:
            latest = (piv, conf)
        else:
            break
    return latest


def run_portfolio(data, amount_data, ticker_list, name_map,
                   cap_up: int, cap_down: int, index_filter: pd.Series = None):
    initial_capital = cap_up * SLOT_CAPITAL

    stock_info = {}
    for t in ticker_list:
        if t not in data:
            continue
        df = data[t].reset_index()
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

        # (A) 매도: 손절(MTH 저가) 우선 → MTL 이탈
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

            sell_price = None
            reason = None

            # 1) 손절: 진입 MTH 캔들 저가 이탈
            if close_now < pos["mth_low"]:
                sell_price = close_now
                reason = "MTH저가손절"
            else:
                # 2) 매도: 최신 MTL 저가 이탈 (트레일링)
                mtl_latest = latest_confirmed(info["pc"]["mtls"], idx)
                if mtl_latest is not None:
                    mtl_low = info["df"].iloc[mtl_latest[0]]["low"]
                    if close_now < mtl_low:
                        sell_price = close_now
                        reason = "MTL이탈"

            if sell_price is not None:
                qty = pos["quantity"]
                revenue = qty * sell_price
                net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
                buy_cost = pos["entry_price"] * qty
                buy_fee = buy_cost * FEE_BUY
                pnl = net - buy_cost - buy_fee
                ret_pct = pnl / (buy_cost + buy_fee) * 100
                trades.append({
                    "ticker": t, "name": name_map.get(t, t),
                    "buy_date": pos["entry_date"], "buy_price": pos["entry_price"],
                    "sell_date": d_ts, "sell_price": sell_price, "quantity": qty,
                    "pnl": pnl, "return_pct": ret_pct, "reason": reason,
                })
                cash += net
                to_close.append(t)
        for t in to_close:
            del positions[t]

        # (B) cap 결정
        if index_filter is not None and d_ts in index_filter.index:
            cap_today = cap_up if bool(index_filter.loc[d_ts]) else cap_down
        else:
            cap_today = cap_up

        # (C) 매수: MTH 상향 돌파
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
                mth_latest = None
                for piv, conf in info["pc"]["mths"]:
                    if conf <= idx and piv > mtl_latest[0]:
                        mth_latest = (piv, conf)
                if mth_latest is None:
                    continue
                mth_high = info["df"].iloc[mth_latest[0]]["high"]
                mth_low = info["df"].iloc[mth_latest[0]]["low"]

                if prev_close <= mth_high and close_now > mth_high:
                    amt = 0
                    if t in amount_data and d_ts in amount_data[t].index:
                        v = amount_data[t].loc[d_ts, "amount"]
                        amt = float(v) if pd.notna(v) else 0
                    candidates.append((t, amt, idx, close_now, mth_low))

            candidates.sort(key=lambda x: -x[1])
            for t, amt, idx, close_p, mth_low in candidates:
                if slots_free == 0 or cash < SLOT_CAPITAL * 0.9:
                    break
                if close_p <= mth_low:
                    continue  # 이론적 방어
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
                    "entry_idx": idx, "mth_low": mth_low,
                }
                slots_free -= 1

        # (D) 자산
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
                              "n_positions": len(positions)})

    # 미청산 청산
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
        # 사유별 분포
        m["reasons"] = trades_df["reason"].value_counts().to_dict()
    else:
        m["win_rate"] = m["pf"] = m["hold_days"] = 0
        m["reasons"] = {}
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

    print(f"[4/4] {len(scenarios)}개 시나리오 실행 (MTH 저가 손절 추가)...")
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
    lines.append("# MTL/MTH 돌파 매매법 V2 — MTH 저가 손절 추가\n")
    lines.append("## 매매 규칙\n")
    lines.append("| 항목 | 내용 |")
    lines.append("|---|---|")
    lines.append("| 매수 | 당일 종가가 최신 확정 MTH 고가 상향 돌파 |")
    lines.append("| **손절** | **당일 종가 < 진입 시 MTH 캔들(pivot)의 저가** (NEW) |")
    lines.append("| 매도 | 당일 종가 < 최신 확정 MTL 저가 (트레일링) |")
    lines.append("| 우선순위 | 손절 먼저 체크 → 미발동 시 MTL 이탈 체크 |")
    lines.append("")

    lines.append("## 개요\n")
    lines.append(f"- **유니버스**: KOSPI200 ∪ 시총≥3조 ({len(tickers)}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **슬롯당 자본**: {SLOT_CAPITAL:,}원 (초기자본 = 최대슬롯×1천만)")
    lines.append(f"- **매수 우선순위**: 당일 거래대금 순")
    lines.append("")

    # 종합 비교
    lines.append("## 시나리오 종합 비교\n")
    lines.append("| 시나리오 | 초기자본 | 최종자본 | 총수익률 | CAGR | MDD | Calmar | 거래수 | 승률 | PF | 평균보유 |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for sc in scenarios:
        name = sc["name"]
        m = results[name]["summary"]
        pf = "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}"
        lines.append(
            f"| {name} | {m['initial_capital']/1e8:.1f}억 "
            f"| {m['final_equity']/1e8:.2f}억 | {m['total_return']:.2f}% "
            f"| {m['cagr']:.2f}% | {m['mdd']:.2f}% | {m['calmar']:.2f} "
            f"| {m['n_trades']}건 | {m['win_rate']:.1f}% | {pf} "
            f"| {m['hold_days']:.1f}일 |"
        )
    lines.append("")

    # 매도 사유 분포
    lines.append("## 매도 사유 분포\n")
    lines.append("| 시나리오 | MTH저가손절 | MTL이탈 | 미청산 | 총거래 |")
    lines.append("|---|:---:|:---:|:---:|:---:|")
    for sc in scenarios:
        name = sc["name"]
        rr = results[name]["summary"]["reasons"]
        stop = rr.get("MTH저가손절", 0)
        mtl = rr.get("MTL이탈", 0)
        open_ = rr.get("미청산", 0)
        tot = stop + mtl + open_
        lines.append(f"| {name} | {stop}건 ({stop/tot*100:.1f}%) "
                     f"| {mtl}건 ({mtl/tot*100:.1f}%) "
                     f"| {open_}건 | {tot}건 |")
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

    # 최고 CAGR TOP 거래
    best = max(results, key=lambda k: results[k]["summary"]["cagr"])
    lines.append(f"## 최고 CAGR · {best} — TOP 10 수익 거래\n")
    bt = results[best]["trades"]
    if not bt.empty:
        top = bt.nlargest(10, "return_pct")
        lines.append("| # | 종목 | 매수일 | 매도일 | 수익률 | 보유일 | 사유 |")
        lines.append("|---|---|---|---|:---:|:---:|---|")
        for j, (_, r) in enumerate(top.iterrows()):
            hd = (pd.Timestamp(r["sell_date"]) - pd.Timestamp(r["buy_date"])).days
            lines.append(
                f"| {j+1} | {r['name']} "
                f"| {pd.Timestamp(r['buy_date']).strftime('%Y-%m-%d')} "
                f"| {pd.Timestamp(r['sell_date']).strftime('%Y-%m-%d')} "
                f"| {r['return_pct']:.2f}% | {hd}일 | {r['reason']} |"
            )
    lines.append("")

    # V1 대비 비교 (V1 수치 하드코딩)
    lines.append("## V1(손절 無) vs V2(MTH 저가 손절) 비교\n")
    lines.append("| 시나리오 | V1 CAGR | V2 CAGR | V1 MDD | V2 MDD | V1 PF | V2 PF |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    v1_map = {
        "S10": {"cagr": 31.51, "mdd": -57.51, "pf": 2.26},
        "S10+F1": {"cagr": 20.62, "mdd": -67.66, "pf": 1.74},
        "H10/5": {"cagr": 8.80, "mdd": -74.28, "pf": 1.26},
    }
    for sc in scenarios:
        name = sc["name"]
        v2 = results[name]["summary"]
        v1 = v1_map[name]
        pf_v2 = "∞" if np.isinf(v2["pf"]) else f"{v2['pf']:.2f}"
        lines.append(
            f"| {name} | {v1['cagr']:.2f}% | {v2['cagr']:.2f}% "
            f"| {v1['mdd']:.2f}% | {v2['mdd']:.2f}% "
            f"| {v1['pf']:.2f} | {pf_v2} |"
        )
    lines.append("")

    lines.append(f"## 실행 정보\n- 실행시간: {elapsed:.1f}초")
    lines.append(f"- 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    out = os.path.join(os.path.dirname(__file__), "results",
                       "backtest_mtl_mth_portfolio_v2.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n리포트: {out}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
