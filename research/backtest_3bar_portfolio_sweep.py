"""
3-bar 포트폴리오 백테스트 — 슬롯 수 민감도 + 하이브리드 슬롯 조절

비교 시나리오:
  [A] 슬롯 스윕 (지수필터 없음): S5, S10, S15, S20
  [B] 슬롯 스윕 (F1 KOSPI>MA60 필터): S5+F1, S10+F1, S15+F1, S20+F1
  [C] 하이브리드 (상승장/하락장 슬롯 조절):
      H10/3 = 상승장 10슬롯, 하락장 3슬롯
      H10/5, H10/7
      H15/5, H20/10

매도 전략: MA5이탈 (원규칙)
각 시나리오 초기자본 = 최대 슬롯 수 × 1천만원
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_3bar import calc_indicators, find_signals, get_mcap_tickers, load_amount_data, ENGINE
from backtest_3bar_portfolio_idx import load_kospi_filters, build_universe
from backtest_crash import load_all_data, FEE_BUY, FEE_SELL, TAX_SELL

START_DATE = "2023-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
SLOT_CAPITAL = 10_000_000


def run_portfolio(ind_data, amount_data, ticker_list, name_map,
                   cap_up: int, cap_down: int, index_filter: pd.Series = None):
    """cap_up/cap_down: 지수필터 통과 시/미통과 시 최대 포지션 수.
    index_filter=None이면 항상 cap_up 사용.
    """
    initial_capital = cap_up * SLOT_CAPITAL
    all_dates = set()
    for df in ind_data.values():
        all_dates.update(df.index)
    all_dates = sorted(d for d in all_dates
                       if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE))

    signals_by_date = {}
    for t in ticker_list:
        if t not in ind_data:
            continue
        df = ind_data[t]
        sigs = find_signals(df)
        for i in sigs:
            d = df.index[i]
            if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE):
                signals_by_date.setdefault(d, []).append(t)

    positions = {}
    cash = initial_capital
    trades = []
    equity_curve = []

    for d in all_dates:
        # (A) 매도
        to_close = []
        for t, pos in positions.items():
            if t not in ind_data or d not in ind_data[t].index:
                continue
            df_t = ind_data[t]
            try:
                i = df_t.index.get_loc(d)
            except KeyError:
                continue
            if isinstance(i, slice):
                i = i.start
            if i <= pos["entry_idx"]:
                continue
            c = df_t["close"].iloc[i]
            ma5 = df_t["sma5"].iloc[i]
            if not np.isnan(ma5):
                if c > ma5:
                    pos["ma5_crossed"] = True
                elif pos["ma5_crossed"] and c < ma5:
                    qty = pos["quantity"]
                    revenue = qty * c
                    net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
                    buy_cost = pos["entry_price"] * qty
                    buy_fee = buy_cost * FEE_BUY
                    pnl = net - buy_cost - buy_fee
                    ret_pct = pnl / (buy_cost + buy_fee) * 100
                    trades.append({
                        "ticker": t, "name": name_map.get(t, t),
                        "buy_date": pos["entry_date"], "buy_price": pos["entry_price"],
                        "sell_date": d, "sell_price": c, "quantity": qty,
                        "pnl": pnl, "return_pct": ret_pct, "reason": "MA5이탈",
                    })
                    cash += net
                    to_close.append(t)
        for t in to_close:
            del positions[t]

        # (B) 당일 cap 결정
        if index_filter is not None and d in index_filter.index:
            cap_today = cap_up if bool(index_filter.loc[d]) else cap_down
        else:
            cap_today = cap_up

        # (C) 매수
        if d in signals_by_date and len(positions) < cap_today:
            slots_free = cap_today - len(positions)
            candidates = [c for c in signals_by_date[d] if c not in positions]
            cand_amt = []
            for c in candidates:
                amt = 0
                if c in amount_data and d in amount_data[c].index:
                    v = amount_data[c].loc[d, "amount"]
                    amt = float(v) if pd.notna(v) else 0
                cand_amt.append((c, amt))
            cand_amt.sort(key=lambda x: -x[1])
            for t, _ in cand_amt:
                if slots_free == 0 or cash < SLOT_CAPITAL * 0.9:
                    break
                df_t = ind_data[t]
                try:
                    i = df_t.index.get_loc(d)
                except KeyError:
                    continue
                if isinstance(i, slice):
                    i = i.start
                close_p = df_t["close"].iloc[i]
                qty = int(SLOT_CAPITAL / (close_p * (1 + FEE_BUY)))
                if qty <= 0:
                    continue
                cost = qty * close_p
                fee = cost * FEE_BUY
                if cash < cost + fee:
                    continue
                cash -= (cost + fee)
                positions[t] = {
                    "entry_price": close_p, "quantity": qty, "entry_date": d,
                    "entry_idx": i, "ma5_crossed": False,
                }
                slots_free -= 1

        # (D) 자산 평가
        pos_val = 0
        for t, pos in positions.items():
            if t in ind_data and d in ind_data[t].index:
                c_now = ind_data[t]["close"].loc[d]
                if isinstance(c_now, pd.Series):
                    c_now = c_now.iloc[-1]
                pos_val += pos["quantity"] * c_now
        equity_curve.append({"date": d, "equity": cash + pos_val,
                              "n_positions": len(positions),
                              "cap_today": cap_today})

    # 미청산 청산
    last_d = all_dates[-1] if all_dates else pd.Timestamp(END_DATE)
    for t, pos in list(positions.items()):
        if t in ind_data and last_d in ind_data[t].index:
            last_close = ind_data[t]["close"].loc[last_d]
            if isinstance(last_close, pd.Series):
                last_close = last_close.iloc[-1]
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
                "sell_date": last_d, "sell_price": last_close, "quantity": qty,
                "pnl": pnl, "return_pct": ret_pct, "reason": "미청산",
            })
            cash += net

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("date") if equity_curve else pd.DataFrame()
    return trades_df, equity_df, initial_capital


def summarize(trades_df, equity_df, initial_capital, label=""):
    final_eq = equity_df["equity"].iloc[-1] if not equity_df.empty else initial_capital
    total_ret = (final_eq - initial_capital) / initial_capital * 100
    if not equity_df.empty:
        dur_days = (equity_df.index[-1] - equity_df.index[0]).days
        years = dur_days / 365.25
        cagr = ((final_eq / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        roll_max = equity_df["equity"].cummax()
        dd = equity_df["equity"] / roll_max - 1
        mdd = dd.min() * 100
        avg_pos = equity_df["n_positions"].mean()
        max_pos = equity_df["n_positions"].max()
    else:
        cagr = mdd = avg_pos = max_pos = 0
    m = {"label": label, "initial_capital": initial_capital,
         "total_return": total_ret, "cagr": cagr, "final_equity": final_eq,
         "mdd": mdd, "n_trades": len(trades_df),
         "avg_positions": avg_pos, "max_positions": int(max_pos)}
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        m["win_rate"] = len(wins) / len(trades_df) * 100
        gp = wins["pnl"].sum() if len(wins) > 0 else 0
        gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        m["pf"] = gp / gl if gl > 0 else float("inf")
    else:
        m["win_rate"] = m["pf"] = 0

    # MDD-adjusted return (Calmar-like)
    m["calmar"] = m["cagr"] / abs(m["mdd"]) if m["mdd"] != 0 else 0
    return m


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
    ind_data = {}
    for t, df in all_data.items():
        df = df[~df.index.duplicated(keep="last")]
        if t in amt_data:
            df = df.join(amt_data[t], how="left")
        ind_data[t] = calc_indicators(df, apply_amount_filter=False)

    # 시나리오 정의
    scenarios = []
    # A: 슬롯 스윕 (필터 없음)
    for s in [5, 10, 15, 20]:
        scenarios.append({"name": f"S{s}", "cap_up": s, "cap_down": s,
                          "filter": None, "group": "A_슬롯스윕"})
    # B: 슬롯 스윕 + F1 (미통과 시 신규 매수 전면 차단)
    for s in [5, 10, 15, 20]:
        scenarios.append({"name": f"S{s}+F1", "cap_up": s, "cap_down": 0,
                          "filter": f1, "group": "B_F1슬롯스윕"})
    # C: 하이브리드
    for up, dn in [(10, 3), (10, 5), (10, 7), (15, 5), (20, 10)]:
        scenarios.append({"name": f"H{up}/{dn}", "cap_up": up, "cap_down": dn,
                          "filter": f1, "group": "C_하이브리드"})

    print(f"[4/4] {len(scenarios)}개 시나리오 실행...")
    results = {}
    for sc in scenarios:
        print(f"  ▸ {sc['name']:10s}...", end=" ", flush=True)
        ts = time.time()
        tdf, edf, init_cap = run_portfolio(
            ind_data, amt_data, tickers, name_map,
            cap_up=sc["cap_up"], cap_down=sc["cap_down"],
            index_filter=sc["filter"],
        )
        m = summarize(tdf, edf, init_cap, sc["name"])
        m["group"] = sc["group"]
        m["cap_up"] = sc["cap_up"]
        m["cap_down"] = sc["cap_down"]
        results[sc["name"]] = {"trades": tdf, "equity": edf, "summary": m}
        pf = "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}"
        print(f"초기 {init_cap/1e8:.1f}억 → {m['final_equity']/1e8:.2f}억 "
              f"| CAGR {m['cagr']:.1f}% | MDD {m['mdd']:.1f}% | Calmar {m['calmar']:.2f} "
              f"| 거래 {m['n_trades']}건 PF {pf} ({time.time()-ts:.1f}s)")

    elapsed = time.time() - t0

    # 리포트
    lines = []
    lines.append("# 3-bar 포트폴리오 — 슬롯 민감도 + 하이브리드 비교\n")
    lines.append("## 개요\n")
    lines.append(f"- **유니버스**: KOSPI200 ∪ 시총≥3조 ({len(tickers)}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **슬롯당 자본**: {SLOT_CAPITAL:,}원 고정 (초기자본 = 최대슬롯×{SLOT_CAPITAL:,})")
    lines.append(f"- **매도**: MA5이탈 · **매수**: 거래대금 순 상위")
    lines.append(f"- **F1 필터**: KOSPI > KOSPI MA60 (통과율 70.6%)")
    lines.append("")

    for group in ["A_슬롯스윕", "B_F1슬롯스윕", "C_하이브리드"]:
        lines.append(f"## {group}\n")
        lines.append("| 시나리오 | 초기자본 | 최종자본 | 총수익률 | CAGR | MDD | Calmar | 거래수 | 승률 | PF | 평균포지션 |")
        lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for sc in scenarios:
            if sc["group"] != group:
                continue
            m = results[sc["name"]]["summary"]
            pf = "∞" if np.isinf(m["pf"]) else f"{m['pf']:.2f}"
            lines.append(
                f"| {sc['name']} | {m['initial_capital']/1e8:.1f}억 "
                f"| {m['final_equity']/1e8:.2f}억 | {m['total_return']:.2f}% "
                f"| {m['cagr']:.2f}% | {m['mdd']:.2f}% | {m['calmar']:.2f} "
                f"| {m['n_trades']}건 | {m['win_rate']:.1f}% | {pf} "
                f"| {m['avg_positions']:.1f}/{sc['cap_up']} |"
            )
        lines.append("")

    # 전체 랭킹 (Calmar 기준)
    lines.append("## 전체 Calmar 비율 랭킹 (CAGR ÷ |MDD|)\n")
    lines.append("| 순위 | 시나리오 | CAGR | MDD | Calmar | 그룹 |")
    lines.append("|---|---|:---:|:---:|:---:|---|")
    ranked = sorted(results.items(), key=lambda x: -x[1]["summary"]["calmar"])
    for rank, (name, r) in enumerate(ranked, 1):
        m = r["summary"]
        lines.append(
            f"| {rank} | {name} | {m['cagr']:.2f}% | {m['mdd']:.2f}% "
            f"| {m['calmar']:.2f} | {m['group']} |"
        )
    lines.append("")

    # 2024년 방어 성능 랭킹
    lines.append("## 2024년(하락장) 수익률 랭킹\n")
    lines.append("| 순위 | 시나리오 | 2024 수익률 | 전체 CAGR | 그룹 |")
    lines.append("|---|---|:---:|:---:|---|")
    yr2024 = []
    for name, r in results.items():
        eq = r["equity"][r["equity"].index.year == 2024]
        if eq.empty:
            continue
        yr_ret = (eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1) * 100
        yr2024.append((name, yr_ret, r["summary"]["cagr"], r["summary"]["group"]))
    yr2024.sort(key=lambda x: -x[1])
    for rank, (name, yr_ret, cagr, grp) in enumerate(yr2024, 1):
        lines.append(f"| {rank} | {name} | {yr_ret:.2f}% | {cagr:.2f}% | {grp} |")
    lines.append("")

    # 최고 CAGR
    best_cagr = max(results.items(), key=lambda x: x[1]["summary"]["cagr"])
    best_calmar = ranked[0]
    lines.append("## 결론\n")
    m1 = best_cagr[1]["summary"]
    m2 = best_calmar[1]["summary"]
    lines.append(f"- **최고 CAGR**: {best_cagr[0]} — CAGR {m1['cagr']:.2f}%, MDD {m1['mdd']:.2f}%")
    lines.append(f"- **최고 Calmar**: {best_calmar[0]} — CAGR {m2['cagr']:.2f}%, MDD {m2['mdd']:.2f}%, Calmar {m2['calmar']:.2f}")

    lines.append("")
    lines.append(f"## 실행 정보\n- 실행시간: {elapsed:.1f}초")
    lines.append(f"- 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    out = os.path.join(os.path.dirname(__file__), "results",
                       "backtest_3bar_portfolio_sweep.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n리포트: {out}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
