"""
V6 급락주 매매 경로별(A/B/C) × 연도별(2023/2024/2025) 백테스트
- Leaders≥70 종목 대상
- 경로A: 기준봉돌파, 경로B: 양봉즉시, 경로C: 갭하락즉시
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

from backtest_crash import ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL, load_all_data
from backtest_simple_buy import calc_indicators_15, get_leaders_tickers

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# V6 파라미터
MA_PERIOD = 15
CRASH_TH = 92
SURGE_TH = 125
GAP_DOWN_PCT = 0.03

PERIODS = [
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", datetime.today().strftime("%Y-%m-%d")),
]


def run_v6(df: pd.DataFrame, ticker: str, name: str):
    """V6 전략 실행 - 경로A/B/C 태깅"""
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


def calc_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "tickers": 0, "win_rate": 0, "pf": 0,
                "avg_ret": 0, "med_ret": 0, "total_pnl": 0, "expect": 0, "hold": 0}
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else float("inf")
    return {
        "n": len(df),
        "tickers": df["ticker"].nunique(),
        "win_rate": len(wins) / len(df) * 100,
        "pf": pf,
        "avg_ret": df["return_pct"].mean(),
        "med_ret": df["return_pct"].median(),
        "total_pnl": df["pnl"].sum(),
        "expect": df["pnl"].mean(),
        "hold": df["hold_days"].mean(),
    }


def fmt_pf(pf):
    return f"{pf:.2f}" if pf != float("inf") else "∞"


def fmt_pnl(v):
    """원 단위를 만원/억원으로 포매팅"""
    if abs(v) >= 1e8:
        return f"{v/1e8:.2f}억"
    return f"{v/1e4:,.0f}만"


def generate_report(yearly_trades: dict, leaders_count: int, elapsed: float) -> str:
    lines = []
    lines.append("# V6 급락주 매매 - 경로별 × 연도별 백테스트 리포트\n")
    lines.append(f"- **대상**: Leaders ≥ 70 ({leaders_count}종목)")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append(f"- **V6 파라미터**: MA{MA_PERIOD}, 급락≤{CRASH_TH}%, 급등≥{SURGE_TH}%, 갭하락≥{GAP_DOWN_PCT*100:.0f}%")
    lines.append("")

    buy_types = ["기준봉돌파", "양봉즉시", "갭하락즉시"]
    period_names = [p[0] for p in PERIODS]

    # ──────────────────────────────────────────
    # 1. 연도별 통합 성과
    # ──────────────────────────────────────────
    lines.append("## 1. 연도별 통합 성과\n")
    lines.append("| 연도 | 거래수 | 종목수 | 승률 | 손익비 | 평균수익률 | 중위수익률 | 총손익 | 거래당기대수익 | 평균보유일 |")
    lines.append("|------|--------|--------|------|--------|----------|----------|--------|------------|----------|")

    all_trades_list = []
    for period_name in period_names:
        tdf = yearly_trades[period_name]
        s = calc_summary(tdf)
        all_trades_list.append(tdf)
        if s["n"] == 0:
            lines.append(f"| **{period_name}** | 0 | - | - | - | - | - | - | - | - |")
        else:
            lines.append(
                f"| **{period_name}** | {s['n']} | {s['tickers']} | {s['win_rate']:.1f}% | "
                f"{fmt_pf(s['pf'])} | {s['avg_ret']:.2f}% | {s['med_ret']:.2f}% | "
                f"{fmt_pnl(s['total_pnl'])} | {s['expect']:,.0f}원 | {s['hold']:.1f}일 |"
            )

    all_df = pd.concat(all_trades_list, ignore_index=True) if all_trades_list else pd.DataFrame()
    s = calc_summary(all_df)
    if s["n"] > 0:
        lines.append(
            f"| **전체** | {s['n']} | {s['tickers']} | {s['win_rate']:.1f}% | "
            f"{fmt_pf(s['pf'])} | {s['avg_ret']:.2f}% | {s['med_ret']:.2f}% | "
            f"{fmt_pnl(s['total_pnl'])} | {s['expect']:,.0f}원 | {s['hold']:.1f}일 |"
        )
    lines.append("")

    # ──────────────────────────────────────────
    # 2. 경로별 × 연도별 성과
    # ──────────────────────────────────────────
    lines.append("## 2. 경로별 × 연도별 성과\n")

    for bt in buy_types:
        label = {"기준봉돌파": "A. 기준봉돌파", "양봉즉시": "B. 양봉즉시", "갭하락즉시": "C. 갭하락즉시"}[bt]
        lines.append(f"### 경로{label}\n")
        lines.append("| 연도 | 거래수 | 종목수 | 승률 | 손익비 | 평균수익률 | 중위수익률 | 총손익 | 거래당기대수익 | 평균보유일 |")
        lines.append("|------|--------|--------|------|--------|----------|----------|--------|------------|----------|")

        bt_all = []
        for period_name in period_names:
            tdf = yearly_trades[period_name]
            bt_df = tdf[tdf["buy_type"] == bt] if not tdf.empty else pd.DataFrame()
            bt_all.append(bt_df)
            s = calc_summary(bt_df)
            if s["n"] == 0:
                lines.append(f"| {period_name} | 0 | - | - | - | - | - | - | - | - |")
            else:
                lines.append(
                    f"| {period_name} | {s['n']} | {s['tickers']} | {s['win_rate']:.1f}% | "
                    f"{fmt_pf(s['pf'])} | {s['avg_ret']:.2f}% | {s['med_ret']:.2f}% | "
                    f"{fmt_pnl(s['total_pnl'])} | {s['expect']:,.0f}원 | {s['hold']:.1f}일 |"
                )

        bt_total = pd.concat(bt_all, ignore_index=True) if bt_all else pd.DataFrame()
        s = calc_summary(bt_total)
        if s["n"] > 0:
            lines.append(
                f"| **전체** | {s['n']} | {s['tickers']} | {s['win_rate']:.1f}% | "
                f"{fmt_pf(s['pf'])} | {s['avg_ret']:.2f}% | {s['med_ret']:.2f}% | "
                f"{fmt_pnl(s['total_pnl'])} | {s['expect']:,.0f}원 | {s['hold']:.1f}일 |"
            )
        lines.append("")

    # ──────────────────────────────────────────
    # 3. 연도별 경로 비교 (가로 비교)
    # ──────────────────────────────────────────
    lines.append("## 3. 연도별 경로 비교 (핵심 지표)\n")
    lines.append("| 연도 | 경로 | 거래수 | 승률 | 손익비 | 평균수익률 | 총손익 | 거래당기대수익 |")
    lines.append("|------|------|--------|------|--------|----------|--------|------------|")

    for period_name in period_names:
        tdf = yearly_trades[period_name]
        for bt in buy_types:
            bt_df = tdf[tdf["buy_type"] == bt] if not tdf.empty else pd.DataFrame()
            s = calc_summary(bt_df)
            label = {"기준봉돌파": "A.기준봉돌파", "양봉즉시": "B.양봉즉시", "갭하락즉시": "C.갭하락즉시"}[bt]
            if s["n"] == 0:
                lines.append(f"| {period_name} | {label} | 0 | - | - | - | - | - |")
            else:
                lines.append(
                    f"| {period_name} | {label} | {s['n']} | {s['win_rate']:.1f}% | "
                    f"{fmt_pf(s['pf'])} | {s['avg_ret']:.2f}% | {fmt_pnl(s['total_pnl'])} | {s['expect']:,.0f}원 |"
                )
    lines.append("")

    # ──────────────────────────────────────────
    # 4. 매도 사유별 분석
    # ──────────────────────────────────────────
    lines.append("## 4. 연도별 × 경로별 매도 사유 분석\n")

    for period_name in period_names:
        tdf = yearly_trades[period_name]
        if tdf.empty:
            continue
        lines.append(f"### {period_name}년\n")
        lines.append("| 경로 | 매도사유 | 건수 | 승률 | 평균수익률 | 총손익 |")
        lines.append("|------|---------|------|------|----------|--------|")
        for bt in buy_types:
            bt_df = tdf[tdf["buy_type"] == bt]
            if bt_df.empty:
                continue
            label = {"기준봉돌파": "A", "양봉즉시": "B", "갭하락즉시": "C"}[bt]
            for reason, grp in bt_df.groupby("reason"):
                n = len(grp)
                wr = (grp["pnl"] > 0).sum() / n * 100
                ar = grp["return_pct"].mean()
                tp = grp["pnl"].sum()
                lines.append(f"| {label} | {reason} | {n} | {wr:.1f}% | {ar:.2f}% | {fmt_pnl(tp)} |")
        lines.append("")

    # ──────────────────────────────────────────
    # 5. 종합 고찰
    # ──────────────────────────────────────────
    lines.append("## 5. 종합 고찰\n")

    # 경로별 전체 성과 정리
    path_stats = {}
    for bt in buy_types:
        bt_df = all_df[all_df["buy_type"] == bt] if not all_df.empty else pd.DataFrame()
        path_stats[bt] = calc_summary(bt_df)

    # 최고 경로 판별
    best_expect = max(path_stats.items(), key=lambda x: x[1]["expect"] if x[1]["n"] > 0 else -999)
    best_pf = max(path_stats.items(), key=lambda x: x[1]["pf"] if x[1]["n"] > 0 else -999)
    best_wr = max(path_stats.items(), key=lambda x: x[1]["win_rate"] if x[1]["n"] > 0 else -999)
    best_total = max(path_stats.items(), key=lambda x: x[1]["total_pnl"] if x[1]["n"] > 0 else -999)

    lines.append(f"- **거래당 기대수익 최고**: {best_expect[0]} ({best_expect[1]['expect']:,.0f}원)")
    lines.append(f"- **손익비 최고**: {best_pf[0]} ({fmt_pf(best_pf[1]['pf'])})")
    lines.append(f"- **승률 최고**: {best_wr[0]} ({best_wr[1]['win_rate']:.1f}%)")
    lines.append(f"- **총손익 최고**: {best_total[0]} ({fmt_pnl(best_total[1]['total_pnl'])})")
    lines.append("")

    # 연도별 안정성 분석
    lines.append("### 연도별 안정성\n")
    for bt in buy_types:
        label = {"기준봉돌파": "A", "양봉즉시": "B", "갭하락즉시": "C"}[bt]
        rets = []
        for period_name in period_names:
            tdf = yearly_trades[period_name]
            bt_df = tdf[tdf["buy_type"] == bt] if not tdf.empty else pd.DataFrame()
            s = calc_summary(bt_df)
            rets.append(s["avg_ret"])
        avg = np.mean(rets)
        std = np.std(rets) if len(rets) > 1 else 0
        lines.append(f"- **경로{label}**: 연도별 평균수익률 = {', '.join(f'{r:.2f}%' for r in rets)} → 평균 {avg:.2f}%, 표준편차 {std:.2f}%p")
    lines.append("")

    lines.append(f"---\n실행 시간: {elapsed:.2f}초")
    lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def generate_charts(yearly_trades: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    buy_types = ["기준봉돌파", "양봉즉시", "갭하락즉시"]
    bt_labels = {"기준봉돌파": "A.기준봉돌파", "양봉즉시": "B.양봉즉시", "갭하락즉시": "C.갭하락즉시"}
    period_names = [p[0] for p in PERIODS]
    colors = {"기준봉돌파": "#2196F3", "양봉즉시": "#4CAF50", "갭하락즉시": "#FF9800"}

    # 차트1: 경로별 × 연도별 총손익
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [
        ("총손익", lambda s: s["total_pnl"] / 10000, "만원"),
        ("승률", lambda s: s["win_rate"], "%"),
        ("거래당 기대수익", lambda s: s["expect"], "원"),
    ]

    for ax, (title, fn, unit) in zip(axes, metrics):
        x = np.arange(len(period_names))
        width = 0.25
        for bi, bt in enumerate(buy_types):
            vals = []
            for pn in period_names:
                tdf = yearly_trades[pn]
                bt_df = tdf[tdf["buy_type"] == bt] if not tdf.empty else pd.DataFrame()
                s = calc_summary(bt_df)
                vals.append(fn(s))
            bars = ax.bar(x + bi * width, vals, width, label=bt_labels[bt],
                         color=colors[bt], alpha=0.8, edgecolor="black", linewidth=0.5)
            for xi, v in zip(x + bi * width, vals):
                va = "bottom" if v >= 0 else "top"
                if unit == "만원":
                    txt = f"{v:,.0f}"
                elif unit == "%":
                    txt = f"{v:.1f}"
                else:
                    txt = f"{v:,.0f}"
                ax.text(xi, v, txt, ha="center", va=va, fontsize=7)

        ax.set_xticks(x + width)
        ax.set_xticklabels(period_names)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(unit)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("V6 급락주 매매 - 경로별 × 연도별 성과", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "v6_yearly_path_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 차트2: 경로별 손익비 + 평균수익률 비교
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 손익비
    x = np.arange(len(period_names))
    width = 0.25
    for bi, bt in enumerate(buy_types):
        vals = []
        for pn in period_names:
            tdf = yearly_trades[pn]
            bt_df = tdf[tdf["buy_type"] == bt] if not tdf.empty else pd.DataFrame()
            s = calc_summary(bt_df)
            pf = s["pf"] if s["pf"] != float("inf") else 0
            vals.append(min(pf, 10))  # cap at 10 for display
        axes[0].bar(x + bi * width, vals, width, label=bt_labels[bt],
                   color=colors[bt], alpha=0.8, edgecolor="black", linewidth=0.5)
        for xi, v in zip(x + bi * width, vals):
            axes[0].text(xi, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(period_names)
    axes[0].set_title("손익비", fontsize=13)
    axes[0].axhline(1, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    # 평균수익률
    for bi, bt in enumerate(buy_types):
        vals = []
        for pn in period_names:
            tdf = yearly_trades[pn]
            bt_df = tdf[tdf["buy_type"] == bt] if not tdf.empty else pd.DataFrame()
            s = calc_summary(bt_df)
            vals.append(s["avg_ret"])
        axes[1].bar(x + bi * width, vals, width, label=bt_labels[bt],
                   color=colors[bt], alpha=0.8, edgecolor="black", linewidth=0.5)
        for xi, v in zip(x + bi * width, vals):
            va = "bottom" if v >= 0 else "top"
            axes[1].text(xi, v, f"{v:.1f}%", ha="center", va=va, fontsize=8)

    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(period_names)
    axes[1].set_title("평균 수익률", fontsize=13)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("V6 경로별 손익비 & 평균수익률 연도 추이", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "v6_yearly_pf_return.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 차트3: 매도사유별 비중 (연도별 스택바)
    fig, axes = plt.subplots(1, len(period_names), figsize=(6 * len(period_names), 6))
    if len(period_names) == 1:
        axes = [axes]
    reasons_order = ["15MA이탈", "급등신호", "손절(매수봉저가)", "미청산"]
    reason_colors = {"15MA이탈": "#4CAF50", "급등신호": "#FF9800",
                     "손절(매수봉저가)": "#f44336", "미청산": "#9E9E9E"}

    for ax, pn in zip(axes, period_names):
        tdf = yearly_trades[pn]
        if tdf.empty:
            ax.set_title(f"{pn} - 데이터 없음")
            continue

        bt_reasons = {}
        for bt in buy_types:
            bt_df = tdf[tdf["buy_type"] == bt]
            counts = bt_df.groupby("reason").size()
            bt_reasons[bt] = counts

        bt_idx = np.arange(len(buy_types))
        bottom = np.zeros(len(buy_types))
        for reason in reasons_order:
            vals = [bt_reasons.get(bt, pd.Series()).get(reason, 0) for bt in buy_types]
            color = reason_colors.get(reason, "#666666")
            ax.bar(bt_idx, vals, bottom=bottom, label=reason, color=color, alpha=0.8, edgecolor="white")
            for xi, v, b in zip(bt_idx, vals, bottom):
                if v > 0:
                    ax.text(xi, b + v / 2, str(int(v)), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            bottom += vals

        ax.set_xticks(bt_idx)
        ax.set_xticklabels(["A.기준봉돌파", "B.양봉즉시", "C.갭하락즉시"], fontsize=9)
        ax.set_title(f"{pn}년 매도사유 분포", fontsize=12)
        ax.set_ylabel("건수")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "v6_yearly_sell_reasons.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 1) Leaders ≥ 70 종목
    print("[1/3] Leaders ≥ 70 종목 조회...")
    leaders = get_leaders_tickers(70)
    tickers = [l["ticker"] for l in leaders]
    print(f"      {len(leaders)}종목")

    # 2) 전체 기간 데이터 로딩 (2023~현재)
    print("\n[2/3] 전체 데이터 로딩 (2023-01-01 ~ 현재)...")
    all_data = load_all_data(tickers, "2023-01-01", datetime.today().strftime("%Y-%m-%d"))
    print(f"      {len(all_data)}종목 데이터 로딩 완료")

    # 3) 연도별 백테스트
    print("\n[3/3] 연도별 백테스트 실행...")
    yearly_trades = {}

    for period_name, start_date, end_date in PERIODS:
        print(f"\n  === {period_name} ({start_date} ~ {end_date}) ===")
        all_period_trades = []

        for ticker, df_raw in all_data.items():
            name_val = df_raw["name"].iloc[0] if "name" in df_raw.columns else ticker
            df_test = df_raw.loc[start_date:end_date]
            if len(df_test) < 20:
                continue

            # 지표는 전체 데이터로 계산 후 기간 슬라이싱
            df_ind = calc_indicators_15(df_raw)
            df_period = df_ind.loc[start_date:end_date]
            if len(df_period) < 20:
                continue

            trades = run_v6(df_period, ticker, name_val)
            if not trades.empty:
                all_period_trades.append(trades)

        period_df = pd.concat(all_period_trades, ignore_index=True) if all_period_trades else pd.DataFrame()
        yearly_trades[period_name] = period_df

        if not period_df.empty:
            n = len(period_df)
            wr = (period_df["pnl"] > 0).sum() / n * 100
            pnl = period_df["pnl"].sum()
            for bt in ["기준봉돌파", "양봉즉시", "갭하락즉시"]:
                bt_df = period_df[period_df["buy_type"] == bt]
                bn = len(bt_df)
                if bn > 0:
                    bwr = (bt_df["pnl"] > 0).sum() / bn * 100
                    bpnl = bt_df["pnl"].sum()
                    print(f"      {bt}: {bn}건, 승률 {bwr:.1f}%, 총손익 {bpnl:,.0f}원")
            print(f"      → 합계: {n}건, 승률 {wr:.1f}%, 총손익 {pnl:,.0f}원")

    # 리포트 & 차트
    print("\n리포트 & 차트 생성...")
    chart_dir = os.path.join(base_dir, "charts_v6_yearly")
    generate_charts(yearly_trades, chart_dir)

    elapsed = time.time() - start_time
    report = generate_report(yearly_trades, len(leaders), elapsed)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_v6_yearly.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*70}")
    print("V6 연도별 백테스트 완료!")
    print(f"리포트: {report_path}")
    print(f"차트: {chart_dir}/")
    print(f"실행 시간: {elapsed:.2f}초")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_main()
