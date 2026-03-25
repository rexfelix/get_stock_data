"""
2봉 연속 급락 매수 V2 - 매수 조건 확장
- 매수 A: 시가대비 종가 -6% 이하가 2봉 연속
- 매수 B: 2일전 시가 → 오늘 종가 등락률 -12% 이하
- 매수: A 또는 B 충족 시 종가 매수
- 매도: 15일선 종가 이탈 (위로 올라간 후)
- 손절: 매입단가 -2%
- 대상: KOSPI200
"""

import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest_crash import (
    ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    START_DATE, END_DATE,
    get_kospi200_tickers, load_all_data,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 매수 파라미터
CRASH_PCT = -6.0       # 시가대비 종가 하락률 (%)
CONSEC_DAYS = 2        # 연속 급락일 수
TOTAL_DROP_PCT = -12.0 # 2일전 시가 → 오늘 종가 등락률 기준 (%)

# 매도 파라미터
MA_PERIOD = 15         # 이평선
STOP_LOSS_PCT = -2.0   # 손절 (%)


def find_signals(df: pd.DataFrame) -> list[dict]:
    """매수 시그널 탐색 - A 또는 B 조건
    Returns: list of {index, reason}
    """
    intraday_ret = (df["close"] - df["open"]) / df["open"] * 100
    is_crash = intraday_ret <= CRASH_PCT

    signals = []
    seen = set()

    for i in range(CONSEC_DAYS - 1, len(df)):
        if i + 1 >= len(df):
            continue

        triggered = False
        reason = ""

        # 조건 A: 2봉 연속 시가대비 종가 -6% 이하
        if all(is_crash.iloc[i - j] for j in range(CONSEC_DAYS)):
            triggered = True
            d1 = intraday_ret.iloc[i - 1]
            d2 = intraday_ret.iloc[i]
            reason = f"A(연속급락 {d1:.1f}%/{d2:.1f}%)"

        # 조건 B: 2일전 시가 → 오늘 종가 등락률 -12% 이하
        if i >= 2:
            open_2d_ago = df.iloc[i - 2]["open"]
            close_today = df.iloc[i]["close"]
            if open_2d_ago > 0:
                total_drop = (close_today - open_2d_ago) / open_2d_ago * 100
                if total_drop <= TOTAL_DROP_PCT:
                    if triggered:
                        reason = f"A+B(연속급락+총낙폭{total_drop:.1f}%)"
                    else:
                        triggered = True
                        reason = f"B(총낙폭 {total_drop:.1f}%)"

        if triggered and i not in seen:
            seen.add(i)
            signals.append({"index": i, "reason": reason})

    return signals


def run_backtest(df: pd.DataFrame, ticker: str, name: str):
    """15MA 이탈 매도 + 손절 백테스트"""
    df = df.copy()
    df["sma"] = df["close"].rolling(window=MA_PERIOD).mean()

    signals = find_signals(df)
    trades = []
    dates = df.index.tolist()

    occupied_until = -1

    for sig in signals:
        sig_idx = sig["index"]
        if sig_idx <= occupied_until:
            continue

        buy_price = df.iloc[sig_idx]["close"]
        buy_date = dates[sig_idx]
        max_qty = int(INITIAL_CAPITAL / (buy_price * (1 + FEE_BUY)))
        if max_qty <= 0:
            continue

        # 매도 탐색
        sell_idx = None
        sell_reason = None
        above_ma = False

        for j in range(sig_idx + 1, len(df)):
            close = df.iloc[j]["close"]
            sma = df.iloc[j]["sma"]

            # 손절 체크
            loss_pct = (close - buy_price) / buy_price * 100
            if loss_pct <= STOP_LOSS_PCT:
                sell_idx = j
                sell_reason = f"손절({STOP_LOSS_PCT}%)"
                break

            # 15MA 이탈 체크
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
        sell_fee = sell_revenue * FEE_SELL
        sell_tax = sell_revenue * TAX_SELL
        net_sell = sell_revenue - sell_fee - sell_tax
        pnl = net_sell - buy_cost - buy_fee
        ret_pct = pnl / (buy_cost + buy_fee) * 100

        hold_days = (pd.Timestamp(sell_date) - pd.Timestamp(buy_date)).days

        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": buy_date, "buy_price": buy_price,
            "sell_date": sell_date, "sell_price": sell_price,
            "quantity": max_qty, "pnl": pnl,
            "return_pct": ret_pct, "reason": sell_reason,
            "hold_days": hold_days,
            "signal_type": sig["reason"],
        })

    return pd.DataFrame(trades)


def generate_report(all_trades_df: pd.DataFrame, n_tickers: int, elapsed: float) -> str:
    lines = []
    lines.append("# 2봉 연속 급락 매수 V2 백테스트 리포트 (KOSPI200)\n")

    lines.append("## 전략 개요\n")
    lines.append(f"- **대상**: KOSPI200 ({n_tickers}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append("")

    lines.append("### 매매 규칙\n")
    lines.append("| 구분 | 조건 |")
    lines.append("|------|------|")
    lines.append(f"| 매수 A | 시가대비 종가 {CRASH_PCT}%이하 {CONSEC_DAYS}봉 연속 → 종가 매수 |")
    lines.append(f"| 매수 B | 2일전 시가 → 오늘 종가 등락률 {TOTAL_DROP_PCT}% 이하 → 종가 매수 |")
    lines.append("| **매수** | **A 또는 B 충족 시 매수** |")
    lines.append(f"| **매도** | **{MA_PERIOD}일선 위로 올라간 후 종가 이탈 시 매도** |")
    lines.append(f"| **손절** | **매입단가 대비 {STOP_LOSS_PCT}% 하락 시 손절** |")
    lines.append("")

    if all_trades_df.empty:
        lines.append("거래 없음\n")
        return "\n".join(lines)

    wins = all_trades_df[all_trades_df["pnl"] > 0]
    losses = all_trades_df[all_trades_df["pnl"] <= 0]
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else float("inf")
    traded_tickers = all_trades_df["ticker"].nunique()

    lines.append("## 통합 성과 요약\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 분석 종목 수 | {n_tickers} |")
    lines.append(f"| 거래 발생 종목 | {traded_tickers} |")
    lines.append(f"| 총 거래 건수 | {len(all_trades_df)} |")
    lines.append(f"| **전체 승률** | **{len(wins)/len(all_trades_df)*100:.1f}% ({len(wins)}승 / {len(losses)}패)** |")
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
    lines.append(f"| **전체 손익비** | **{pf_str}** |")
    lines.append(f"| **전체 총 손익** | **{all_trades_df['pnl'].sum():,.0f}원** |")
    lines.append(f"| 거래당 평균 수익률 | {all_trades_df['return_pct'].mean():.2f}% |")
    lines.append(f"| 거래당 중위 수익률 | {all_trades_df['return_pct'].median():.2f}% |")
    lines.append(f"| 거래당 기대수익 | {all_trades_df['pnl'].mean():,.0f}원 |")
    lines.append(f"| 평균 보유일 | {all_trades_df['hold_days'].mean():.1f}일 |")
    if len(wins) > 0:
        lines.append(f"| 평균 수익 (승) | {wins['pnl'].mean():,.0f}원 ({wins['return_pct'].mean():.2f}%) |")
    if len(losses) > 0:
        lines.append(f"| 평균 손실 (패) | {losses['pnl'].mean():,.0f}원 ({losses['return_pct'].mean():.2f}%) |")
    lines.append("")

    # 시그널 유형별 분석
    lines.append("## 시그널 유형별 분석\n")
    lines.append("| 유형 | 건수 | 승률 | 평균수익률 | 총손익 | 평균보유일 |")
    lines.append("|------|------|------|----------|--------|----------|")

    # A만, B만, A+B로 분류
    def classify(sig_type):
        if sig_type.startswith("A+B"):
            return "A+B (양쪽 충족)"
        elif sig_type.startswith("A"):
            return "A (연속급락)"
        else:
            return "B (총낙폭)"

    tc = all_trades_df.copy()
    tc["sig_class"] = tc["signal_type"].apply(classify)
    for cls, grp in tc.groupby("sig_class"):
        n = len(grp)
        wr = (grp["pnl"] > 0).sum() / n * 100
        ar = grp["return_pct"].mean()
        tp = grp["pnl"].sum()
        hd = grp["hold_days"].mean()
        lines.append(f"| {cls} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 | {hd:.1f}일 |")
    lines.append("")

    # 매도 사유별 분석
    lines.append("## 매도 사유별 분석\n")
    lines.append("| 사유 | 건수 | 승률 | 평균수익률 | 총손익 | 평균보유일 |")
    lines.append("|------|------|------|----------|--------|----------|")
    for reason, grp in all_trades_df.groupby("reason"):
        n = len(grp)
        wr = (grp["pnl"] > 0).sum() / n * 100
        ar = grp["return_pct"].mean()
        tp = grp["pnl"].sum()
        hd = grp["hold_days"].mean()
        lines.append(f"| {reason} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 | {hd:.1f}일 |")
    lines.append("")

    # 월별 성과
    lines.append("## 월별 성과\n")
    lines.append("| 월 | 거래수 | 승률 | 평균수익률 | 총손익 |")
    lines.append("|----|--------|------|----------|--------|")
    tc["month"] = pd.to_datetime(tc["buy_date"]).dt.to_period("M")
    for month, grp in tc.groupby("month"):
        n = len(grp)
        wr = (grp["pnl"] > 0).sum() / n * 100
        ar = grp["return_pct"].mean()
        tp = grp["pnl"].sum()
        lines.append(f"| {month} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 |")
    lines.append("")

    # 차트
    lines.append("## 차트\n")
    lines.append("### 거래별 수익률 분포")
    lines.append("![수익률분포](charts_consec_crash_v2/return_distribution.png)\n")
    lines.append("### 시그널 유형별 수익률 비교")
    lines.append("![시그널유형](charts_consec_crash_v2/signal_type_comparison.png)\n")
    lines.append("### 월별 손익")
    lines.append("![월별손익](charts_consec_crash_v2/monthly_pnl.png)\n")

    # 전체 매매 기록
    lines.append("## 전체 매매 기록\n")
    lines.append("| # | 종목 | 시그널 | 매수일 | 매수가 | 매도일 | 매도가 | 수익률 | 손익 | 보유일 | 사유 |")
    lines.append("|---|------|--------|--------|--------|--------|--------|--------|------|--------|------|")
    sorted_trades = all_trades_df.sort_values("buy_date")
    for i, (_, t) in enumerate(sorted_trades.iterrows()):
        bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
        sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
        sig_short = classify(t["signal_type"]).split(" ")[0]
        lines.append(
            f"| {i+1} | {t['name']} | {sig_short} | {bd} | {t['buy_price']:,.0f} "
            f"| {sd} | {t['sell_price']:,.0f} | {t['return_pct']:.2f}% | {t['pnl']:,.0f} "
            f"| {t['hold_days']}일 | {t['reason']} |"
        )
    lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- **실행 시간**: {elapsed:.2f}초")
    lines.append(f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    return "\n".join(lines)


def generate_charts(all_trades_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if all_trades_df.empty:
        return

    # 차트1: 거래별 수익률 분포
    fig, ax = plt.subplots(figsize=(14, 6))
    bins = np.arange(
        max(all_trades_df["return_pct"].min() - 1, -50),
        min(all_trades_df["return_pct"].max() + 1, 50), 1
    )
    n, bins_out, patches = ax.hist(all_trades_df["return_pct"], bins=bins, edgecolor="black", alpha=0.7)
    for patch, b in zip(patches, bins_out):
        if b + (bins_out[1] - bins_out[0]) / 2 >= 0:
            patch.set_facecolor("green")
        else:
            patch.set_facecolor("red")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    avg_ret = all_trades_df["return_pct"].mean()
    ax.axvline(avg_ret, color="blue", linewidth=1.5, label=f"평균: {avg_ret:.2f}%")
    ax.set_title(f"거래별 수익률 분포 (총 {len(all_trades_df)}건)", fontsize=14)
    ax.set_xlabel("수익률 (%)")
    ax.set_ylabel("거래 건수")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "return_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 차트2: 시그널 유형별 수익률 비교 (박스플롯)
    def classify(sig_type):
        if sig_type.startswith("A+B"):
            return "A+B"
        elif sig_type.startswith("A"):
            return "A(연속급락)"
        else:
            return "B(총낙폭)"

    tc = all_trades_df.copy()
    tc["sig_class"] = tc["signal_type"].apply(classify)
    classes = sorted(tc["sig_class"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 박스플롯
    data_by_class = [tc[tc["sig_class"] == c]["return_pct"].values for c in classes]
    bp = axes[0].boxplot(data_by_class, labels=classes, patch_artist=True)
    colors_box = ["#4CAF50", "#2196F3", "#FF9800"]
    for patch, color in zip(bp["boxes"], colors_box[:len(classes)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("시그널 유형별 수익률 분포", fontsize=13)
    axes[0].set_ylabel("수익률 (%)")
    axes[0].grid(True, alpha=0.3, axis="y")

    # 바차트 (건수, 승률, 총손익)
    summary_data = []
    for c in classes:
        grp = tc[tc["sig_class"] == c]
        summary_data.append({
            "class": c,
            "n": len(grp),
            "win_rate": (grp["pnl"] > 0).sum() / len(grp) * 100 if len(grp) > 0 else 0,
            "total_pnl": grp["pnl"].sum(),
        })
    x = range(len(classes))
    bars = axes[1].bar(x, [s["total_pnl"] for s in summary_data],
                       color=colors_box[:len(classes)], alpha=0.7, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes)
    axes[1].set_title("시그널 유형별 총손익", fontsize=13)
    axes[1].set_ylabel("총손익 (원)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, s in enumerate(summary_data):
        axes[1].text(i, s["total_pnl"],
                     f"{s['n']}건\n승률{s['win_rate']:.0f}%",
                     ha="center", va="bottom" if s["total_pnl"] >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "signal_type_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 차트3: 월별 손익
    fig, ax = plt.subplots(figsize=(14, 6))
    tc["month"] = pd.to_datetime(tc["buy_date"]).dt.to_period("M")
    monthly = tc.groupby("month").agg(
        total_pnl=("pnl", "sum"),
        n_trades=("pnl", "count"),
    )
    colors_bar = ["green" if p >= 0 else "red" for p in monthly["total_pnl"]]
    x = range(len(monthly))
    ax.bar(x, monthly["total_pnl"], color=colors_bar, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha="right")
    ax.set_title("월별 총 손익", fontsize=14)
    ax.set_ylabel("손익 (원)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (_, row) in enumerate(monthly.iterrows()):
        ax.text(i, row["total_pnl"], f"{row['n_trades']}건", ha="center",
                va="bottom" if row["total_pnl"] >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "monthly_pnl.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 1) KOSPI200 종목 조회
    print("[1/4] KOSPI200 종목 조회...")
    kospi200 = get_kospi200_tickers()
    with ENGINE.connect() as conn:
        db_rows = conn.execute(
            text("SELECT DISTINCT ticker FROM stocks")
        ).fetchall()
    db_tickers = set(r[0] for r in db_rows)
    ticker_list = [t["ticker"] for t in kospi200 if t["ticker"] in db_tickers]
    name_map = {t["ticker"]: t["name"] for t in kospi200 if t["ticker"] in db_tickers}
    print(f"      KOSPI200 대상: {len(ticker_list)}종목")

    # 2) 데이터 로딩
    print("[2/4] 데이터 로딩...")
    all_data = load_all_data(ticker_list, START_DATE, END_DATE)
    print(f"      {len(all_data)}종목 로딩 완료")

    # 3) 백테스트 실행
    print("[3/4] 종목별 백테스트 실행...")
    all_trades = []

    for idx, ticker in enumerate(ticker_list):
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        df_test = df.loc[START_DATE:]
        if len(df_test) < 5:
            continue

        name = name_map.get(ticker, ticker)
        trades_df = run_backtest(df_test, ticker, name)
        if not trades_df.empty:
            all_trades.append(trades_df)

        if (idx + 1) % 50 == 0:
            print(f"      {idx+1}/{len(ticker_list)} 완료...")

    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    traded_count = all_trades_df["ticker"].nunique() if not all_trades_df.empty else 0
    print(f"      완료! 거래 발생: {traded_count}종목, 총 {len(all_trades_df)}건")

    # 4) 차트 & 리포트
    chart_dir = os.path.join(base_dir, "charts_consec_crash_v2")
    print("[4/4] 차트 & 리포트 생성...")
    generate_charts(all_trades_df, chart_dir)

    elapsed = time.time() - start_time
    report = generate_report(all_trades_df, len(ticker_list), elapsed)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_consec_crash_v2.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*50}")
    print(f"2봉 연속 급락 V2 백테스트 완료! ({elapsed:.2f}초)")
    print(f"리포트: {report_path}")
    if not all_trades_df.empty:
        w = (all_trades_df["pnl"] > 0).sum()
        avg = all_trades_df["return_pct"].mean()
        total_pnl = all_trades_df["pnl"].sum()
        print(f"  거래: {len(all_trades_df)}건 ({traded_count}종목)")
        print(f"  승률: {w}/{len(all_trades_df)} = {w/len(all_trades_df)*100:.1f}%")
        print(f"  거래당 평균 수익률: {avg:.2f}%")
        print(f"  총 손익: {total_pnl:,.0f}원")
        print(f"  평균 보유일: {all_trades_df['hold_days'].mean():.1f}일")

        # 시그널 유형별 요약
        def classify(sig_type):
            if sig_type.startswith("A+B"):
                return "A+B"
            elif sig_type.startswith("A"):
                return "A"
            return "B"
        tc = all_trades_df.copy()
        tc["sig_class"] = tc["signal_type"].apply(classify)
        print("\n  [시그널 유형별]")
        for cls, grp in tc.groupby("sig_class"):
            n = len(grp)
            wr = (grp["pnl"] > 0).sum() / n * 100
            tp = grp["pnl"].sum()
            print(f"    {cls}: {n}건, 승률 {wr:.1f}%, 총손익 {tp:,.0f}원")


if __name__ == "__main__":
    run_main()
