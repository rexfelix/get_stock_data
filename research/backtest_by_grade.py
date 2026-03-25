"""
재무 등급(rs_ratings.grade)별 매매 규칙 백테스트
- 전체 종목 대상, grade(A/B/F/N)별 그룹 분석
- 전략1: V6 급락주 매매 (이격도15일 92/125, 기준봉+양봉+갭하락)
- 전략2: 2봉 연속 급락 V1 (2봉 연속 -6%)
- 전략3: 2봉 연속 급락 V2 (V1 OR 총낙폭 -12%)
"""

import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text as sql_text

from backtest_crash import (
    ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    START_DATE, END_DATE, load_all_data,
)
from backtest_simple_buy import calc_indicators_15

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# V6 파라미터
MA_PERIOD = 15
CRASH_TH = 92
SURGE_TH = 125
GAP_DOWN_PCT = 0.03

# 2봉 연속 급락 파라미터
CRASH_PCT = -6.0
CONSEC_DAYS = 2
TOTAL_DROP_PCT = -12.0
STOP_LOSS_PCT = -2.0


# ──────────────────────────────────────────────
# 종목-grade 매핑 조회
# ──────────────────────────────────────────────
def get_ticker_grades() -> dict[str, dict]:
    """rs_ratings에서 최신 grade 매핑 (ticker → {name, grade})"""
    query = sql_text("""
        SELECT DISTINCT ON (ticker) ticker, name, grade
        FROM rs_ratings
        ORDER BY ticker, calculated_at DESC
    """)
    with ENGINE.connect() as conn:
        rows = conn.execute(query).fetchall()
    return {r[0]: {"name": r[1], "grade": r[2]} for r in rows}


# ──────────────────────────────────────────────
# 전략1: V6 급락주 매매
# ──────────────────────────────────────────────
def run_v6(df: pd.DataFrame, ticker: str, name: str):
    """V6: 기준봉돌파 + 양봉즉시 + 갭하락즉시, 매도: 매수봉저가/급등/15MA이탈"""
    trades = []
    dates = df.index.tolist()
    gijoongbong = None
    position = None
    above_ma = False

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        close = row["close"]
        sma15 = row["sma15"]
        disp = row["disparity15"]
        date = dates[i]

        if pd.isna(sma15):
            continue

        # 매도 체크
        if position is not None:
            sell_reason = None
            if not above_ma and close > sma15:
                above_ma = True
            if close < position["buy_candle_low"]:
                sell_reason = "손절(매수봉저가)"
            elif disp >= SURGE_TH:
                sell_reason = "급등신호"
            elif above_ma and close < sma15:
                sell_reason = "15MA이탈"

            if sell_reason:
                qty = position["quantity"]
                buy_cost = position["entry_price"] * qty
                buy_fee = buy_cost * FEE_BUY
                revenue = qty * close
                net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
                pnl = net - buy_cost - buy_fee
                ret = pnl / (buy_cost + buy_fee) * 100
                hold_days = (pd.Timestamp(date) - pd.Timestamp(position["entry_date"])).days
                trades.append({
                    "ticker": ticker, "name": name,
                    "buy_date": position["entry_date"], "buy_price": position["entry_price"],
                    "sell_date": date, "sell_price": close,
                    "quantity": qty, "pnl": pnl, "return_pct": ret,
                    "reason": sell_reason, "buy_type": position["buy_type"],
                    "hold_days": hold_days,
                })
                position = None
                above_ma = False
        else:
            bought = False
            prev_close = prev_row["close"]
            gap_pct = (row["open"] - prev_close) / prev_close if prev_close > 0 else 0
            is_bearish = close < row["open"]
            is_bullish = close >= row["open"]
            vol_up = row["volume"] > prev_row["volume"]

            # 경로A: 기준봉 돌파
            if gijoongbong is not None:
                if close > gijoongbong["open_price"]:
                    max_qty = int(INITIAL_CAPITAL / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                            "buy_type": "기준봉돌파",
                        }
                        above_ma = False
                        bought = True
                    gijoongbong = None
            if not bought and gijoongbong is not None:
                if close > sma15:
                    gijoongbong = None

            if not bought and disp <= CRASH_TH:
                # 경로C: 갭하락≥3% 즉시매수
                if gap_pct <= -GAP_DOWN_PCT:
                    max_qty = int(INITIAL_CAPITAL / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                            "buy_type": "갭하락즉시",
                        }
                        above_ma = False
                        bought = True
                    gijoongbong = None
                elif is_bearish and vol_up:
                    gijoongbong = {"date": date, "open_price": row["open"]}
                elif is_bullish:
                    # 경로B: 양봉 즉시매수
                    max_qty = int(INITIAL_CAPITAL / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                            "buy_type": "양봉즉시",
                        }
                        above_ma = False
                        bought = True

    # 미청산
    if position is not None:
        last = df.iloc[-1]
        qty = position["quantity"]
        buy_cost = position["entry_price"] * qty
        buy_fee = buy_cost * FEE_BUY
        revenue = qty * last["close"]
        net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
        pnl = net - buy_cost - buy_fee
        ret = pnl / (buy_cost + buy_fee) * 100
        hold_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(position["entry_date"])).days
        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": position["entry_date"], "buy_price": position["entry_price"],
            "sell_date": dates[-1], "sell_price": last["close"],
            "quantity": qty, "pnl": pnl, "return_pct": ret,
            "reason": "미청산", "buy_type": position["buy_type"],
            "hold_days": hold_days,
        })

    return pd.DataFrame(trades)


# ──────────────────────────────────────────────
# 전략2/3: 2봉 연속 급락
# ──────────────────────────────────────────────
def run_consec(df: pd.DataFrame, ticker: str, name: str, use_total_drop: bool = False):
    """2봉 연속 급락 매수 + 15MA이탈 매도 + 손절-2%"""
    df = df.copy()
    df["sma"] = df["close"].rolling(window=MA_PERIOD).mean()

    intraday_ret = (df["close"] - df["open"]) / df["open"] * 100
    is_crash = intraday_ret <= CRASH_PCT

    trades = []
    dates = df.index.tolist()
    occupied_until = -1

    for i in range(CONSEC_DAYS - 1, len(df)):
        if i + 1 >= len(df):
            continue
        if i <= occupied_until:
            continue

        triggered = False
        reason = ""

        # 조건A: 2봉 연속 -6%
        if all(is_crash.iloc[i - j] for j in range(CONSEC_DAYS)):
            triggered = True
            reason = "A"

        # 조건B: 총낙폭 -12% (V2만)
        if use_total_drop and i >= 2:
            open_2d = df.iloc[i - 2]["open"]
            close_today = df.iloc[i]["close"]
            if open_2d > 0:
                total_drop = (close_today - open_2d) / open_2d * 100
                if total_drop <= TOTAL_DROP_PCT:
                    if triggered:
                        reason = "A+B"
                    else:
                        triggered = True
                        reason = "B"

        if not triggered:
            continue

        buy_price = df.iloc[i]["close"]
        buy_date = dates[i]
        max_qty = int(INITIAL_CAPITAL / (buy_price * (1 + FEE_BUY)))
        if max_qty <= 0:
            continue

        # 매도 탐색
        sell_idx = None
        sell_reason = None
        above_ma = False

        for j in range(i + 1, len(df)):
            close = df.iloc[j]["close"]
            sma = df.iloc[j]["sma"]

            loss_pct = (close - buy_price) / buy_price * 100
            if loss_pct <= STOP_LOSS_PCT:
                sell_idx = j
                sell_reason = f"손절({STOP_LOSS_PCT}%)"
                break

            if not pd.isna(sma):
                if not above_ma and close > sma:
                    above_ma = True
                if above_ma and close < sma:
                    sell_idx = j
                    sell_reason = f"{MA_PERIOD}MA이탈"
                    break

        if sell_idx is None:
            sell_idx = len(df) - 1
            sell_reason = "미청산"

        occupied_until = sell_idx
        sell_price = df.iloc[sell_idx]["close"]
        sell_date = dates[sell_idx]

        buy_cost = max_qty * buy_price
        buy_fee = buy_cost * FEE_BUY
        sell_revenue = max_qty * sell_price
        net = sell_revenue - sell_revenue * FEE_SELL - sell_revenue * TAX_SELL
        pnl = net - buy_cost - buy_fee
        ret = pnl / (buy_cost + buy_fee) * 100
        hold_days = (pd.Timestamp(sell_date) - pd.Timestamp(buy_date)).days

        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": buy_date, "buy_price": buy_price,
            "sell_date": sell_date, "sell_price": sell_price,
            "quantity": max_qty, "pnl": pnl, "return_pct": ret,
            "reason": sell_reason, "signal_type": reason,
            "hold_days": hold_days,
        })

    return pd.DataFrame(trades)


# ──────────────────────────────────────────────
# 그룹 요약 함수
# ──────────────────────────────────────────────
def grade_summary(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"n": 0, "tickers": 0, "win_rate": 0, "pf": 0,
                "avg_ret": 0, "med_ret": 0, "total_pnl": 0, "expect": 0, "hold": 0}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else float("inf")
    return {
        "n": len(trades_df),
        "tickers": trades_df["ticker"].nunique(),
        "win_rate": len(wins) / len(trades_df) * 100,
        "pf": pf,
        "avg_ret": trades_df["return_pct"].mean(),
        "med_ret": trades_df["return_pct"].median(),
        "total_pnl": trades_df["pnl"].sum(),
        "expect": trades_df["pnl"].mean(),
        "hold": trades_df["hold_days"].mean(),
    }


def format_pf(pf):
    return f"{pf:.2f}" if pf != float("inf") else "∞"


# ──────────────────────────────────────────────
# 리포트 생성
# ──────────────────────────────────────────────
def generate_report(strategy_results: dict, grade_counts: dict, elapsed: float) -> str:
    lines = []
    lines.append("# 재무 등급(grade)별 매매 규칙 백테스트 리포트\n")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **대상**: rs_ratings 전체 종목 (grade별 그룹)")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append("")

    lines.append("## Grade 분포\n")
    lines.append("| Grade | 종목수 | 설명 |")
    lines.append("|-------|--------|------|")
    desc = {"A": "우수", "B": "보통", "F": "부진", "N": "데이터 부족"}
    for g in ["A", "B", "F", "N"]:
        lines.append(f"| {g} | {grade_counts.get(g, 0)} | {desc.get(g, '')} |")
    lines.append("")

    # 전략별 grade 비교
    for strat_name, grade_trades in strategy_results.items():
        lines.append(f"## {strat_name}\n")

        # 요약 테이블
        lines.append("### Grade별 성과 비교\n")
        lines.append("| Grade | 종목수 | 거래수 | 승률 | 손익비 | 평균수익률 | 중위수익률 | 총손익 | 거래당기대수익 | 평균보유일 |")
        lines.append("|-------|--------|--------|------|--------|----------|----------|--------|------------|----------|")

        for g in ["A", "B", "F", "N", "전체"]:
            if g not in grade_trades:
                continue
            s = grade_summary(grade_trades[g])
            if s["n"] == 0:
                lines.append(f"| **{g}** | {grade_counts.get(g, '-')} | 0 | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| **{g}** | {s['tickers']} | {s['n']} | {s['win_rate']:.1f}% | "
                f"{format_pf(s['pf'])} | {s['avg_ret']:.2f}% | {s['med_ret']:.2f}% | "
                f"{s['total_pnl']:,.0f}원 | {s['expect']:,.0f}원 | {s['hold']:.1f}일 |"
            )
        lines.append("")

        # 매도 사유별 분석 (grade별)
        lines.append("### Grade별 매도 사유\n")
        for g in ["A", "B", "F", "N"]:
            if g not in grade_trades or grade_trades[g].empty:
                continue
            tdf = grade_trades[g]
            lines.append(f"#### Grade {g}\n")
            lines.append("| 사유 | 건수 | 승률 | 평균수익률 | 총손익 |")
            lines.append("|------|------|------|----------|--------|")
            for reason, grp in tdf.groupby("reason"):
                n = len(grp)
                wr = (grp["pnl"] > 0).sum() / n * 100
                ar = grp["return_pct"].mean()
                tp = grp["pnl"].sum()
                lines.append(f"| {reason} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 |")
            lines.append("")

        # V6 매수 유형별 (grade별)
        if "buy_type" in grade_trades.get("전체", pd.DataFrame()).columns:
            lines.append("### Grade별 매수 유형\n")
            for g in ["A", "B", "F", "N"]:
                if g not in grade_trades or grade_trades[g].empty:
                    continue
                tdf = grade_trades[g]
                lines.append(f"#### Grade {g}\n")
                lines.append("| 유형 | 건수 | 승률 | 평균수익률 | 총손익 |")
                lines.append("|------|------|------|----------|--------|")
                for bt, grp in tdf.groupby("buy_type"):
                    n = len(grp)
                    wr = (grp["pnl"] > 0).sum() / n * 100
                    ar = grp["return_pct"].mean()
                    tp = grp["pnl"].sum()
                    lines.append(f"| {bt} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 |")
                lines.append("")

    lines.append(f"---\n실행 시간: {elapsed:.2f}초")
    lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def generate_charts(strategy_results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    grades = ["A", "B", "F", "N"]
    colors = {"A": "#4CAF50", "B": "#2196F3", "F": "#f44336", "N": "#9E9E9E"}

    for strat_name, grade_trades in strategy_results.items():
        safe_name = strat_name.replace(" ", "_").replace("/", "_")

        # 차트1: Grade별 총손익 비교
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 총손익
        pnls = [grade_trades[g]["pnl"].sum() if g in grade_trades and not grade_trades[g].empty else 0 for g in grades]
        bar_colors = [colors[g] for g in grades]
        axes[0].bar(grades, pnls, color=bar_colors, alpha=0.7, edgecolor="black")
        axes[0].set_title(f"{strat_name}\n총손익", fontsize=12)
        axes[0].set_ylabel("원")
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].grid(True, alpha=0.3, axis="y")
        for i, (g, p) in enumerate(zip(grades, pnls)):
            n = len(grade_trades[g]) if g in grade_trades and not grade_trades[g].empty else 0
            axes[0].text(i, p, f"{p/10000:,.0f}만\n({n}건)", ha="center",
                        va="bottom" if p >= 0 else "top", fontsize=9)

        # 승률
        wrs = []
        for g in grades:
            if g in grade_trades and not grade_trades[g].empty:
                t = grade_trades[g]
                wrs.append((t["pnl"] > 0).sum() / len(t) * 100)
            else:
                wrs.append(0)
        axes[1].bar(grades, wrs, color=bar_colors, alpha=0.7, edgecolor="black")
        axes[1].set_title(f"{strat_name}\n승률", fontsize=12)
        axes[1].set_ylabel("%")
        axes[1].set_ylim(0, 100)
        axes[1].axhline(50, color="gray", linewidth=0.8, linestyle="--")
        axes[1].grid(True, alpha=0.3, axis="y")
        for i, (g, w) in enumerate(zip(grades, wrs)):
            axes[1].text(i, w + 1, f"{w:.1f}%", ha="center", fontsize=10)

        # 거래당 기대수익
        expects = []
        for g in grades:
            if g in grade_trades and not grade_trades[g].empty:
                expects.append(grade_trades[g]["pnl"].mean())
            else:
                expects.append(0)
        exp_colors = ["green" if e >= 0 else "red" for e in expects]
        axes[2].bar(grades, expects, color=exp_colors, alpha=0.7, edgecolor="black")
        axes[2].set_title(f"{strat_name}\n거래당 기대수익", fontsize=12)
        axes[2].set_ylabel("원")
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].grid(True, alpha=0.3, axis="y")
        for i, (g, e) in enumerate(zip(grades, expects)):
            axes[2].text(i, e, f"{e:,.0f}", ha="center",
                        va="bottom" if e >= 0 else "top", fontsize=9)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 전략 통합 비교 차트
    fig, ax = plt.subplots(figsize=(14, 6))
    strat_names = list(strategy_results.keys())
    x = np.arange(len(grades))
    width = 0.25

    for si, strat_name in enumerate(strat_names):
        grade_trades = strategy_results[strat_name]
        avg_rets = []
        for g in grades:
            if g in grade_trades and not grade_trades[g].empty:
                avg_rets.append(grade_trades[g]["return_pct"].mean())
            else:
                avg_rets.append(0)
        bars = ax.bar(x + si * width, avg_rets, width, label=strat_name, alpha=0.7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(grades, fontsize=12)
    ax.set_title("전략별 × Grade별 평균 수익률 비교", fontsize=14)
    ax.set_ylabel("평균 수익률 (%)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "all_strategies_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 1) 종목-grade 매핑
    print("[1/4] rs_ratings grade 매핑 조회...")
    ticker_info = get_ticker_grades()
    grade_counts = {}
    for info in ticker_info.values():
        g = info["grade"]
        grade_counts[g] = grade_counts.get(g, 0) + 1
    print(f"      전체: {len(ticker_info)}종목")
    for g in ["A", "B", "F", "N"]:
        print(f"      Grade {g}: {grade_counts.get(g, 0)}종목")

    # 2) stocks 테이블에 있는 종목만 필터
    print("\n[2/4] 전체 데이터 로딩...")
    all_tickers = list(ticker_info.keys())

    # 배치 로딩 (전체 한번에)
    all_data = load_all_data(all_tickers, START_DATE, END_DATE)
    available = set(all_data.keys())
    print(f"      {len(available)}종목 데이터 로딩 완료")

    # grade별 ticker 분류
    grade_tickers = {"A": [], "B": [], "F": [], "N": []}
    for ticker, info in ticker_info.items():
        if ticker in available:
            grade_tickers[info["grade"]].append(ticker)
    for g in ["A", "B", "F", "N"]:
        print(f"      Grade {g}: {len(grade_tickers[g])}종목 (데이터 있음)")

    # 3) 백테스트 실행
    print("\n[3/4] 백테스트 실행...")
    strategy_results = {}

    for strat_idx, (strat_name, strat_func) in enumerate([
        ("V6 급락주 매매", "v6"),
        ("2봉연속급락 V1", "v1"),
        ("2봉연속급락 V2", "v2"),
    ]):
        print(f"\n  === {strat_name} ===")
        grade_trades = {}
        all_strat_trades = []

        total = sum(len(v) for v in grade_tickers.values())
        done = 0

        for grade in ["A", "B", "F", "N"]:
            g_trades = []
            for ticker in grade_tickers[grade]:
                df = all_data[ticker]
                df_test = df.loc[START_DATE:]
                if len(df_test) < 20:
                    done += 1
                    continue

                name = ticker_info[ticker]["name"]

                if strat_func == "v6":
                    df_ind = calc_indicators_15(df_test)
                    trades = run_v6(df_ind, ticker, name)
                elif strat_func == "v1":
                    trades = run_consec(df_test, ticker, name, use_total_drop=False)
                else:  # v2
                    trades = run_consec(df_test, ticker, name, use_total_drop=True)

                if not trades.empty:
                    g_trades.append(trades)

                done += 1
                if done % 500 == 0:
                    print(f"      {done}/{total}...")

            grade_trades[grade] = pd.concat(g_trades, ignore_index=True) if g_trades else pd.DataFrame()
            n = len(grade_trades[grade])
            if n > 0:
                wr = (grade_trades[grade]["pnl"] > 0).sum() / n * 100
                pnl = grade_trades[grade]["pnl"].sum()
                print(f"      Grade {grade}: {n}건, 승률 {wr:.1f}%, 총손익 {pnl:,.0f}원")

        # 전체 합산
        all_dfs = [grade_trades[g] for g in ["A", "B", "F", "N"] if not grade_trades[g].empty]
        grade_trades["전체"] = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        strategy_results[strat_name] = grade_trades

    # 4) 리포트 & 차트
    print("\n[4/4] 리포트 & 차트 생성...")
    chart_dir = os.path.join(base_dir, "charts_by_grade")
    generate_charts(strategy_results, chart_dir)

    elapsed = time.time() - start_time
    report = generate_report(strategy_results, grade_counts, elapsed)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_by_grade.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 콘솔 요약
    print(f"\n{'='*70}")
    print("재무 등급별 백테스트 완료!")
    print(f"{'='*70}")
    for strat_name, grade_trades in strategy_results.items():
        print(f"\n  [{strat_name}]")
        for g in ["A", "B", "F", "N"]:
            t = grade_trades.get(g, pd.DataFrame())
            if t.empty:
                print(f"    Grade {g}: 거래 없음")
                continue
            w = (t["pnl"] > 0).sum()
            print(f"    Grade {g}: {len(t)}건, 승률 {w/len(t)*100:.1f}%, "
                  f"평균수익률 {t['return_pct'].mean():.2f}%, "
                  f"총손익 {t['pnl'].sum():,.0f}원")

    print(f"\n리포트: {report_path}")
    print(f"실행 시간: {elapsed:.2f}초")


if __name__ == "__main__":
    run_main()
