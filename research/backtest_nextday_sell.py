"""
매수 다음날 종가 매도 백테스트
- 기존 매수 규칙 3가지 전략을 그대로 사용하되, 매도만 "다음날 종가"로 변경
- 손절 규칙은 그대로 유지
- 전략1: V6 급락주 매매 (Leaders≥70) - 손절: 매수봉 저가 이탈
- 전략2: 2봉 연속 급락 V1 (KOSPI200) - 손절: -2%
- 전략3: 2봉 연속 급락 V2 (KOSPI200) - 손절: -2%
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
    START_DATE, END_DATE,
    get_kospi200_tickers, load_all_data,
)
from backtest_simple_buy import get_leaders_tickers, calc_indicators_15

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
# 전략1: V6 급락주 매매 - 다음날 종가 매도
# ──────────────────────────────────────────────
def run_v6_nextday(df: pd.DataFrame, ticker: str, name: str):
    """V6 매수 규칙 + 다음날 종가 매도 (손절: 매수봉 저가 이탈)"""
    trades = []
    dates = df.index.tolist()
    gijoongbong = None
    occupied_until = -1

    for i in range(1, len(df)):
        if i <= occupied_until:
            continue

        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        close = row["close"]
        sma15 = row["sma15"]
        disp = row["disparity15"]
        date = dates[i]

        if pd.isna(sma15):
            continue

        bought = False
        buy_price = None
        buy_type = None
        buy_candle_low = None

        # 경로A: 기존 기준봉 돌파 매수
        if gijoongbong is not None:
            if close > gijoongbong["open_price"]:
                buy_price = close
                buy_type = "기준봉돌파"
                buy_candle_low = row["low"]
                bought = True
                gijoongbong = None
        if not bought and gijoongbong is not None:
            if close > sma15:
                gijoongbong = None

        if not bought and disp <= CRASH_TH:
            prev_close = prev_row["close"]
            gap_pct = (row["open"] - prev_close) / prev_close if prev_close > 0 else 0
            is_bearish = close < row["open"]
            is_bullish = close >= row["open"]
            vol_up = row["volume"] > prev_row["volume"]

            # 경로C: 갭하락≥3% → 즉시매수
            if gap_pct <= -GAP_DOWN_PCT:
                buy_price = close
                buy_type = "갭하락즉시"
                buy_candle_low = row["low"]
                bought = True
                gijoongbong = None
            elif is_bearish and vol_up:
                # 기준봉 설정
                gijoongbong = {"date": date, "open_price": row["open"]}
            elif is_bullish:
                # 경로B: 양봉 즉시 매수
                buy_price = close
                buy_type = "양봉즉시"
                buy_candle_low = row["low"]
                bought = True

        if not bought:
            continue

        # 매수 실행
        max_qty = int(INITIAL_CAPITAL / (buy_price * (1 + FEE_BUY)))
        if max_qty <= 0:
            continue

        # 다음날 존재 확인
        if i + 1 >= len(df):
            continue

        next_row = df.iloc[i + 1]
        next_close = next_row["close"]
        next_date = dates[i + 1]

        # 손절 체크: 다음날 종가가 매수봉 저가 아래이면 손절
        if next_close < buy_candle_low:
            sell_price = next_close
            sell_reason = "손절(매수봉저가)"
        else:
            sell_price = next_close
            sell_reason = "다음날매도"

        occupied_until = i + 1

        buy_cost = max_qty * buy_price
        buy_fee = buy_cost * FEE_BUY
        sell_revenue = max_qty * sell_price
        sell_fee = sell_revenue * FEE_SELL
        sell_tax = sell_revenue * TAX_SELL
        net_sell = sell_revenue - sell_fee - sell_tax
        pnl = net_sell - buy_cost - buy_fee
        ret_pct = pnl / (buy_cost + buy_fee) * 100
        hold_days = (pd.Timestamp(next_date) - pd.Timestamp(date)).days

        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": date, "buy_price": buy_price,
            "sell_date": next_date, "sell_price": sell_price,
            "quantity": max_qty, "pnl": pnl,
            "return_pct": ret_pct, "reason": sell_reason,
            "hold_days": hold_days, "buy_type": buy_type,
        })

    return pd.DataFrame(trades)


# ──────────────────────────────────────────────
# 전략2/3: 2봉 연속 급락 - 다음날 종가 매도
# ──────────────────────────────────────────────
def find_consec_signals(df: pd.DataFrame, use_total_drop: bool = False) -> list[dict]:
    """매수 시그널 탐색
    use_total_drop=False: V1 (2봉 연속 -6%만)
    use_total_drop=True: V2 (A or B)
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

        # 조건 A: 2봉 연속 -6%
        if all(is_crash.iloc[i - j] for j in range(CONSEC_DAYS)):
            triggered = True
            d1 = intraday_ret.iloc[i - 1]
            d2 = intraday_ret.iloc[i]
            reason = f"A(연속급락 {d1:.1f}%/{d2:.1f}%)"

        # 조건 B: 2일전 시가 → 오늘 종가 -12% (V2만)
        if use_total_drop and i >= 2:
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


def run_consec_nextday(df: pd.DataFrame, ticker: str, name: str, use_total_drop: bool = False):
    """2봉 연속 급락 매수 + 다음날 종가 매도 (손절: -2%)"""
    df = df.copy()
    df["sma"] = df["close"].rolling(window=MA_PERIOD).mean()

    signals = find_consec_signals(df, use_total_drop)
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

        # 다음날 존재 확인
        if sig_idx + 1 >= len(df):
            continue

        next_row = df.iloc[sig_idx + 1]
        next_close = next_row["close"]
        next_date = dates[sig_idx + 1]

        # 손절 체크: 다음날 종가 기준 -2% 이하이면 손절
        loss_pct = (next_close - buy_price) / buy_price * 100
        if loss_pct <= STOP_LOSS_PCT:
            sell_reason = f"손절({STOP_LOSS_PCT}%)"
        else:
            sell_reason = "다음날매도"

        occupied_until = sig_idx + 1

        buy_cost = max_qty * buy_price
        buy_fee = buy_cost * FEE_BUY
        sell_revenue = max_qty * next_close
        sell_fee = sell_revenue * FEE_SELL
        sell_tax = sell_revenue * TAX_SELL
        net_sell = sell_revenue - sell_fee - sell_tax
        pnl = net_sell - buy_cost - buy_fee
        ret_pct = pnl / (buy_cost + buy_fee) * 100
        hold_days = (pd.Timestamp(next_date) - pd.Timestamp(buy_date)).days

        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": buy_date, "buy_price": buy_price,
            "sell_date": next_date, "sell_price": next_close,
            "quantity": max_qty, "pnl": pnl,
            "return_pct": ret_pct, "reason": sell_reason,
            "hold_days": hold_days, "signal_type": sig["reason"],
        })

    return pd.DataFrame(trades)


# ──────────────────────────────────────────────
# 리포트 생성
# ──────────────────────────────────────────────
def summarize(trades_df: pd.DataFrame, label: str) -> str:
    if trades_df.empty:
        return f"### {label}\n거래 없음\n"

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else float("inf")
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
    traded = trades_df["ticker"].nunique()

    lines = [f"### {label}\n"]
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 거래 발생 종목 | {traded} |")
    lines.append(f"| 총 거래 건수 | {len(trades_df)} |")
    lines.append(f"| 승률 | {len(wins)/len(trades_df)*100:.1f}% ({len(wins)}승/{len(losses)}패) |")
    lines.append(f"| 손익비 | {pf_str} |")
    lines.append(f"| 평균 수익률 | {trades_df['return_pct'].mean():.2f}% |")
    lines.append(f"| 중위 수익률 | {trades_df['return_pct'].median():.2f}% |")
    lines.append(f"| 총 손익 | {trades_df['pnl'].sum():,.0f}원 |")
    lines.append(f"| 거래당 기대수익 | {trades_df['pnl'].mean():,.0f}원 |")
    lines.append(f"| 평균 보유일 | {trades_df['hold_days'].mean():.1f}일 |")
    lines.append("")

    # 매도 사유별
    lines.append("#### 매도 사유별\n")
    lines.append("| 사유 | 건수 | 승률 | 평균수익률 | 총손익 |")
    lines.append("|------|------|------|----------|--------|")
    for reason, grp in trades_df.groupby("reason"):
        n = len(grp)
        wr = (grp["pnl"] > 0).sum() / n * 100
        ar = grp["return_pct"].mean()
        tp = grp["pnl"].sum()
        lines.append(f"| {reason} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 |")
    lines.append("")

    # 매수 유형별 (V6용)
    if "buy_type" in trades_df.columns:
        lines.append("#### 매수 유형별\n")
        lines.append("| 유형 | 건수 | 승률 | 평균수익률 | 총손익 |")
        lines.append("|------|------|------|----------|--------|")
        for bt, grp in trades_df.groupby("buy_type"):
            n = len(grp)
            wr = (grp["pnl"] > 0).sum() / n * 100
            ar = grp["return_pct"].mean()
            tp = grp["pnl"].sum()
            lines.append(f"| {bt} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 |")
        lines.append("")

    # 시그널 유형별 (V1/V2용)
    if "signal_type" in trades_df.columns:
        def classify(sig_type):
            if sig_type.startswith("A+B"):
                return "A+B"
            elif sig_type.startswith("A"):
                return "A(연속급락)"
            return "B(총낙폭)"

        lines.append("#### 시그널 유형별\n")
        lines.append("| 유형 | 건수 | 승률 | 평균수익률 | 총손익 |")
        lines.append("|------|------|------|----------|--------|")
        tc = trades_df.copy()
        tc["sig_class"] = tc["signal_type"].apply(classify)
        for cls, grp in tc.groupby("sig_class"):
            n = len(grp)
            wr = (grp["pnl"] > 0).sum() / n * 100
            ar = grp["return_pct"].mean()
            tp = grp["pnl"].sum()
            lines.append(f"| {cls} | {n} | {wr:.1f}% | {ar:.2f}% | {tp:,.0f}원 |")
        lines.append("")

    return "\n".join(lines)


def generate_report(results: dict, elapsed: float) -> str:
    lines = []
    lines.append("# 매수 다음날 종가 매도 백테스트 리포트\n")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append(f"- **매도 규칙**: 매수 다음날 종가에 매도 (손절 조건 해당 시 손절 처리)")
    lines.append("")

    # 비교 요약 테이블
    lines.append("## 전략별 비교 요약\n")
    lines.append("| 전략 | 대상 | 거래수 | 승률 | 손익비 | 평균수익률 | 총손익 | 거래당기대수익 |")
    lines.append("|------|------|--------|------|--------|----------|--------|------------|")
    for key, trades_df in results.items():
        if trades_df.empty:
            lines.append(f"| {key} | - | 0 | - | - | - | - | - |")
            continue
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        gp = wins["pnl"].sum() if len(wins) > 0 else 0
        gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        pf = gp / gl if gl > 0 else float("inf")
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        target = "Leaders≥70" if "V6" in key else "KOSPI200"
        lines.append(
            f"| {key} | {target} | {len(trades_df)} | "
            f"{len(wins)/len(trades_df)*100:.1f}% | {pf_str} | "
            f"{trades_df['return_pct'].mean():.2f}% | "
            f"{trades_df['pnl'].sum():,.0f}원 | "
            f"{trades_df['pnl'].mean():,.0f}원 |"
        )
    lines.append("")

    # 각 전략 상세
    for key, trades_df in results.items():
        lines.append(summarize(trades_df, key))

    lines.append(f"---\n실행 시간: {elapsed:.2f}초")
    lines.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def generate_charts(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 차트1: 전략별 수익률 분포 비교
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (key, trades_df) in enumerate(results.items()):
        ax = axes[idx]
        if trades_df.empty:
            ax.set_title(f"{key}\n(거래 없음)")
            continue
        rets = trades_df["return_pct"].clip(-30, 30)
        bins = np.arange(rets.min() - 1, rets.max() + 1, 1)
        n, bins_out, patches = ax.hist(rets, bins=bins, edgecolor="black", alpha=0.7)
        for patch, b in zip(patches, bins_out):
            if b + (bins_out[1] - bins_out[0]) / 2 >= 0:
                patch.set_facecolor("green")
            else:
                patch.set_facecolor("red")
        avg = trades_df["return_pct"].mean()
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.axvline(avg, color="blue", linewidth=1.5, label=f"평균: {avg:.2f}%")
        wins = (trades_df["pnl"] > 0).sum()
        wr = wins / len(trades_df) * 100
        ax.set_title(f"{key}\n({len(trades_df)}건, 승률 {wr:.1f}%)", fontsize=11)
        ax.set_xlabel("수익률 (%)")
        ax.set_ylabel("거래 건수")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "return_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 차트2: 전략별 총손익 비교 바차트
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(results.keys())
    pnls = [df["pnl"].sum() if not df.empty else 0 for df in results.values()]
    colors = ["green" if p >= 0 else "red" for p in pnls]
    bars = ax.bar(range(len(labels)), pnls, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("전략별 총손익 비교 (다음날 종가 매도)", fontsize=14)
    ax.set_ylabel("총손익 (원)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (p, label) in enumerate(zip(pnls, labels)):
        n = len(results[label]) if not results[label].empty else 0
        ax.text(i, p, f"{p:,.0f}원\n({n}건)", ha="center",
                va="bottom" if p >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "strategy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_main():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)
    results = {}

    # ── 전략1: V6 급락주 매매 (Leaders≥70) ──
    print("=" * 50)
    print("[전략1] V6 급락주 매매 - 다음날 종가 매도 (Leaders≥70)")
    print("=" * 50)
    print("[1/3] Leaders≥70 종목 조회...")
    leaders = get_leaders_tickers(70)
    with ENGINE.connect() as conn:
        db_rows = conn.execute(sql_text("SELECT DISTINCT ticker FROM stocks")).fetchall()
    db_tickers = set(r[0] for r in db_rows)
    leader_tickers = [t["ticker"] for t in leaders if t["ticker"] in db_tickers]
    leader_names = {t["ticker"]: t["name"] for t in leaders if t["ticker"] in db_tickers}
    print(f"      대상: {len(leader_tickers)}종목")

    print("[2/3] 데이터 로딩...")
    leader_data = load_all_data(leader_tickers, START_DATE, END_DATE)
    print(f"      {len(leader_data)}종목 로딩 완료")

    print("[3/3] V6 백테스트 실행...")
    v6_trades = []
    for idx, ticker in enumerate(leader_tickers):
        if ticker not in leader_data:
            continue
        df = leader_data[ticker]
        df_test = calc_indicators_15(df.loc[START_DATE:])
        if len(df_test) < 20:
            continue
        name = leader_names.get(ticker, ticker)
        trades = run_v6_nextday(df_test, ticker, name)
        if not trades.empty:
            v6_trades.append(trades)
        if (idx + 1) % 20 == 0:
            print(f"      {idx+1}/{len(leader_tickers)}...")

    v6_df = pd.concat(v6_trades, ignore_index=True) if v6_trades else pd.DataFrame()
    results["V6 급락주(Leaders)"] = v6_df
    print(f"      V6: {len(v6_df)}건, {v6_df['ticker'].nunique() if not v6_df.empty else 0}종목")

    # ── 전략2/3: 2봉 연속 급락 (KOSPI200) ──
    print("\n" + "=" * 50)
    print("[전략2/3] 2봉 연속 급락 V1/V2 - 다음날 종가 매도 (KOSPI200)")
    print("=" * 50)
    print("[1/3] KOSPI200 종목 조회...")
    kospi200 = get_kospi200_tickers()
    kospi_tickers = [t["ticker"] for t in kospi200 if t["ticker"] in db_tickers]
    kospi_names = {t["ticker"]: t["name"] for t in kospi200 if t["ticker"] in db_tickers}
    print(f"      대상: {len(kospi_tickers)}종목")

    print("[2/3] 데이터 로딩...")
    kospi_data = load_all_data(kospi_tickers, START_DATE, END_DATE)
    print(f"      {len(kospi_data)}종목 로딩 완료")

    print("[3/3] V1/V2 백테스트 실행...")
    v1_trades = []
    v2_trades = []
    for idx, ticker in enumerate(kospi_tickers):
        if ticker not in kospi_data:
            continue
        df = kospi_data[ticker]
        df_test = df.loc[START_DATE:]
        if len(df_test) < 5:
            continue
        name = kospi_names.get(ticker, ticker)

        # V1
        t1 = run_consec_nextday(df_test, ticker, name, use_total_drop=False)
        if not t1.empty:
            v1_trades.append(t1)

        # V2
        t2 = run_consec_nextday(df_test, ticker, name, use_total_drop=True)
        if not t2.empty:
            v2_trades.append(t2)

        if (idx + 1) % 50 == 0:
            print(f"      {idx+1}/{len(kospi_tickers)}...")

    v1_df = pd.concat(v1_trades, ignore_index=True) if v1_trades else pd.DataFrame()
    v2_df = pd.concat(v2_trades, ignore_index=True) if v2_trades else pd.DataFrame()
    results["2봉연속급락 V1(KOSPI200)"] = v1_df
    results["2봉연속급락 V2(KOSPI200)"] = v2_df
    print(f"      V1: {len(v1_df)}건, V2: {len(v2_df)}건")

    # ── 리포트 & 차트 ──
    chart_dir = os.path.join(base_dir, "charts_nextday_sell")
    print("\n차트 생성...")
    generate_charts(results, chart_dir)

    elapsed = time.time() - start_time
    report = generate_report(results, elapsed)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "backtest_nextday_sell.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 콘솔 요약
    print(f"\n{'='*60}")
    print("매수 다음날 종가 매도 백테스트 완료!")
    print(f"{'='*60}")
    for key, trades_df in results.items():
        if trades_df.empty:
            print(f"  {key}: 거래 없음")
            continue
        w = (trades_df["pnl"] > 0).sum()
        n = len(trades_df)
        print(f"  {key}: {n}건, 승률 {w/n*100:.1f}%, "
              f"평균수익률 {trades_df['return_pct'].mean():.2f}%, "
              f"총손익 {trades_df['pnl'].sum():,.0f}원")
    print(f"\n리포트: {report_path}")
    print(f"실행 시간: {elapsed:.2f}초")


if __name__ == "__main__":
    run_main()
