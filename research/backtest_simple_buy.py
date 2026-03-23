"""
단순 급락 매수 백테스트
- 이격도(15일, 92%) 이하 발생 → 조건 없이 종가에 매수
- 매도: 급등신호(이격도>=125%) 또는 15MA 위 진입 후 15MA 종가 이탈
- 손절: 매수봉 저가를 종가로 이탈
- Leaders≥70 종목 대상
"""

import argparse
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from backtest_crash import (
    DB_URL, ENGINE, INITIAL_CAPITAL, FEE_BUY, FEE_SELL, TAX_SELL,
    START_DATE, END_DATE,
    get_kospi200_tickers, load_all_data,
    generate_summary_charts,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 파라미터
MA_PERIOD = 15
CRASH_TH = 92
SURGE_TH = 125


def calc_indicators_15(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma15"] = df["close"].rolling(window=MA_PERIOD).mean()
    df["disparity15"] = df["close"] / df["sma15"] * 100
    return df


def run_backtest_simple(df: pd.DataFrame, ticker: str, name: str):
    capital = INITIAL_CAPITAL
    position = None
    above_ma = False

    trades = []
    equity_curve = []
    dates = df.index.tolist()

    for i in range(1, len(df)):
        row = df.iloc[i]
        close = row["close"]
        sma15 = row["sma15"]
        disp = row["disparity15"]
        date = dates[i]

        if pd.isna(sma15):
            eq = capital + (position["quantity"] * close if position else 0)
            equity_curve.append({"date": date, "equity": eq})
            continue

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
                revenue = qty * close
                fee = revenue * FEE_SELL
                tax = revenue * TAX_SELL
                net = revenue - fee - tax
                capital += net
                buy_cost = position["entry_price"] * qty
                buy_fee = buy_cost * FEE_BUY
                pnl = net - buy_cost - buy_fee
                ret = pnl / (buy_cost + buy_fee) * 100
                trades.append({
                    "ticker": ticker, "name": name,
                    "buy_date": position["entry_date"],
                    "buy_price": position["entry_price"],
                    "sell_date": date, "sell_price": close,
                    "quantity": qty, "pnl": pnl,
                    "return_pct": ret, "reason": sell_reason,
                })
                position = None
                above_ma = False
        else:
            # 이격도 92% 이하면 조건 없이 종가 매수
            if disp <= CRASH_TH:
                max_qty = int(capital / (close * (1 + FEE_BUY)))
                if max_qty > 0:
                    cost = max_qty * close
                    fee = cost * FEE_BUY
                    capital -= cost + fee
                    position = {
                        "entry_price": close, "quantity": max_qty,
                        "entry_date": date, "buy_candle_low": row["low"],
                    }
                    above_ma = False

        eq = capital + (position["quantity"] * close if position else 0)
        equity_curve.append({"date": date, "equity": eq})

    # 미청산 포지션
    if position is not None:
        last = df.iloc[-1]
        qty = position["quantity"]
        revenue = qty * last["close"]
        net = revenue - revenue * FEE_SELL - revenue * TAX_SELL
        capital += net
        buy_cost = position["entry_price"] * qty
        buy_fee = buy_cost * FEE_BUY
        pnl = net - buy_cost - buy_fee
        ret = pnl / (buy_cost + buy_fee) * 100
        trades.append({
            "ticker": ticker, "name": name,
            "buy_date": position["entry_date"],
            "buy_price": position["entry_price"],
            "sell_date": dates[-1], "sell_price": last["close"],
            "quantity": qty, "pnl": pnl,
            "return_pct": ret, "reason": "미청산",
        })

    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df = equity_df.set_index("date")
    trades_df = pd.DataFrame(trades)

    end_eq = equity_df["equity"].iloc[-1] if not equity_df.empty else INITIAL_CAPITAL
    total_ret = (end_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    bnh_ret = (df["close"].iloc[-1] / df.loc[START_DATE:]["close"].iloc[0] - 1) * 100 if len(df.loc[START_DATE:]) > 0 else 0

    summary = {
        "ticker": ticker, "name": name,
        "total_return": total_ret,
        "bnh_return": bnh_ret,
        "n_trades": len(trades_df),
        "final_equity": end_eq,
    }

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        summary["win_rate"] = len(wins) / len(trades_df) * 100
        summary["total_pnl"] = trades_df["pnl"].sum()
        summary["avg_return"] = trades_df["return_pct"].mean()
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        summary["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        summary["win_rate"] = 0
        summary["total_pnl"] = 0
        summary["avg_return"] = 0
        summary["profit_factor"] = 0

    if not equity_df.empty:
        roll_max = equity_df["equity"].cummax()
        dd = equity_df["equity"] / roll_max - 1
        summary["mdd"] = dd.min() * 100
    else:
        summary["mdd"] = 0

    return summary, trades_df, equity_df


def get_leaders_tickers(min_score: int = 70) -> list[dict]:
    query = text("""
        SELECT ticker, name, leader_score
        FROM leaders
        WHERE leader_score >= :min_score
        ORDER BY leader_score DESC
    """)
    with ENGINE.connect() as conn:
        rows = conn.execute(query, {"min_score": min_score}).fetchall()
    return [{"ticker": r[0], "name": r[1], "score": r[2]} for r in rows]


def generate_report(summaries_df, all_trades_df, elapsed, label):
    traded = summaries_df[summaries_df["n_trades"] > 0]
    no_trade = summaries_df[summaries_df["n_trades"] == 0]

    lines = []
    lines.append(f"# 단순 급락 매수 백테스트 리포트 ({label})\n")

    lines.append("## 전략 개요\n")
    lines.append(f"- **대상**: {label} ({len(summaries_df)}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append("")

    lines.append("### 매매 규칙\n")
    lines.append("| 구분 | 조건 |")
    lines.append("|------|------|")
    lines.append(f"| 급락 신호 | {MA_PERIOD}일 이동평균 이격도 ≤ {CRASH_TH}% |")
    lines.append(f"| **매수** | **급락 신호 발생 시 조건 없이 종가 매수** |")
    lines.append(f"| 매도 | 급등신호(이격도≥{SURGE_TH}%) 또는 {MA_PERIOD}MA 위 진입 후 {MA_PERIOD}MA 종가 이탈 |")
    lines.append("| 손절 | 매수봉 저가를 종가로 이탈 |")
    lines.append("")

    lines.append("## 통합 성과 요약\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 분석 종목 수 | {len(summaries_df)} |")
    lines.append(f"| 거래 발생 종목 | {len(traded)} ({len(traded)/len(summaries_df)*100:.1f}%) |")
    lines.append(f"| 미거래 종목 | {len(no_trade)} |")
    lines.append(f"| 총 거래 건수 | {len(all_trades_df) if not all_trades_df.empty else 0} |")

    if not traded.empty:
        avg_ret = traded["total_return"].mean()
        med_ret = traded["total_return"].median()
        pos_count = (traded["total_return"] > 0).sum()
        neg_count = (traded["total_return"] <= 0).sum()
        lines.append(f"| 종목 평균 수익률 | {avg_ret:.2f}% |")
        lines.append(f"| 종목 중위 수익률 | {med_ret:.2f}% |")
        lines.append(f"| 수익 종목 / 손실 종목 | {pos_count} / {neg_count} |")
        lines.append(f"| 종목 평균 MDD | {traded['mdd'].mean():.2f}% |")

    if not all_trades_df.empty:
        all_wins = all_trades_df[all_trades_df["pnl"] > 0]
        all_losses = all_trades_df[all_trades_df["pnl"] <= 0]
        total_win_rate = len(all_wins) / len(all_trades_df) * 100
        gp = all_wins["pnl"].sum() if len(all_wins) > 0 else 0
        gl = abs(all_losses["pnl"].sum()) if len(all_losses) > 0 else 0
        pf = gp / gl if gl > 0 else float("inf")
        total_pnl = all_trades_df["pnl"].sum()
        avg_trade_ret = all_trades_df["return_pct"].mean()

        lines.append(f"| 전체 승률 | {total_win_rate:.1f}% ({len(all_wins)}승 / {len(all_losses)}패) |")
        lines.append(f"| 전체 손익비 | {pf:.2f} |")
        lines.append(f"| 전체 총 손익 | {total_pnl:,.0f}원 |")
        lines.append(f"| 거래당 평균 수익률 | {avg_trade_ret:.2f}% |")
        lines.append(f"| 평균 수익 (승) | {all_wins['pnl'].mean():,.0f}원 |" if len(all_wins) > 0 else "| 평균 수익 (승) | N/A |")
        lines.append(f"| 평균 손실 (패) | {all_losses['pnl'].mean():,.0f}원 |" if len(all_losses) > 0 else "| 평균 손실 (패) | N/A |")

        tc = all_trades_df.copy()
        tc["hold_days"] = (pd.to_datetime(tc["sell_date"]) - pd.to_datetime(tc["buy_date"])).dt.days
        lines.append(f"| 평균 보유일 | {tc['hold_days'].mean():.1f}일 |")
    lines.append("")

    # 바이앤홀드 비교
    if not traded.empty:
        lines.append("## 전략 vs 바이앤홀드 비교\n")
        lines.append("| 지표 | 전략 (거래종목 평균) | 바이앤홀드 (거래종목 평균) |")
        lines.append("|------|-----|-----|")
        lines.append(f"| 평균 수익률 | {traded['total_return'].mean():.2f}% | {traded['bnh_return'].mean():.2f}% |")
        lines.append(f"| 중위 수익률 | {traded['total_return'].median():.2f}% | {traded['bnh_return'].median():.2f}% |")
        outperform = (traded["total_return"] > traded["bnh_return"]).sum()
        lines.append(f"| 전략 우위 종목 수 | {outperform} / {len(traded)} ({outperform/len(traded)*100:.1f}%) | |")
        lines.append("")

    # 매도 사유별 분석
    if not all_trades_df.empty:
        lines.append("## 매도 사유별 분석\n")
        lines.append("| 사유 | 건수 | 평균 수익률 | 승률 | 총 손익 |")
        lines.append("|------|------|------------|------|---------|")
        for reason, grp in all_trades_df.groupby("reason"):
            n = len(grp)
            avg_r = grp["return_pct"].mean()
            wr = (grp["pnl"] > 0).sum() / n * 100
            tp = grp["pnl"].sum()
            lines.append(f"| {reason} | {n} | {avg_r:.2f}% | {wr:.1f}% | {tp:,.0f}원 |")
        lines.append("")

    # 차트
    lines.append("## 차트\n")
    lines.append("### 종목별 수익률 분포")
    lines.append("![수익률분포](charts_simple_buy/return_distribution.png)\n")
    lines.append("### 전략 vs 바이앤홀드")
    lines.append("![전략vs바이앤홀드](charts_simple_buy/strategy_vs_bnh.png)\n")
    lines.append("### 매도 사유별 수익률")
    lines.append("![매도사유](charts_simple_buy/reason_boxplot.png)\n")
    lines.append("### TOP / BOTTOM 10 종목")
    lines.append("![TOP_BOTTOM](charts_simple_buy/top_bottom.png)\n")
    lines.append("### 월별 손익")
    lines.append("![월별손익](charts_simple_buy/monthly_pnl.png)\n")

    # TOP/BOTTOM 종목 상세
    if not traded.empty:
        lines.append("## TOP 10 수익 종목\n")
        lines.append("| # | 종목 | 코드 | 수익률 | 거래수 | 승률 | MDD | B&H |")
        lines.append("|---|------|------|--------|--------|------|-----|-----|")
        for i, (_, r) in enumerate(traded.nlargest(10, "total_return").iterrows()):
            lines.append(
                f"| {i+1} | {r['name']} | {r['ticker']} | {r['total_return']:.2f}% "
                f"| {r['n_trades']} | {r['win_rate']:.0f}% | {r['mdd']:.2f}% | {r['bnh_return']:.2f}% |"
            )
        lines.append("")

        lines.append("## BOTTOM 10 손실 종목\n")
        lines.append("| # | 종목 | 코드 | 수익률 | 거래수 | 승률 | MDD | B&H |")
        lines.append("|---|------|------|--------|--------|------|-----|-----|")
        for i, (_, r) in enumerate(traded.nsmallest(10, "total_return").iterrows()):
            lines.append(
                f"| {i+1} | {r['name']} | {r['ticker']} | {r['total_return']:.2f}% "
                f"| {r['n_trades']} | {r['win_rate']:.0f}% | {r['mdd']:.2f}% | {r['bnh_return']:.2f}% |"
            )
        lines.append("")

    # 전체 매매 기록
    if not all_trades_df.empty:
        lines.append("## 전체 매매 기록\n")
        lines.append("| # | 종목 | 매수일 | 매수가 | 매도일 | 매도가 | 수익률 | 손익 | 사유 |")
        lines.append("|---|------|--------|--------|--------|--------|--------|------|------|")
        sorted_trades = all_trades_df.sort_values("sell_date")
        for i, (_, t) in enumerate(sorted_trades.iterrows()):
            bd = pd.Timestamp(t["buy_date"]).strftime("%Y-%m-%d")
            sd = pd.Timestamp(t["sell_date"]).strftime("%Y-%m-%d")
            lines.append(
                f"| {i+1} | {t['name']} | {bd} | {t['buy_price']:,.0f} "
                f"| {sd} | {t['sell_price']:,.0f} | {t['return_pct']:.2f}% | {t['pnl']:,.0f} | {t['reason']} |"
            )
        lines.append("")

    lines.append("## 실행 정보\n")
    lines.append(f"- **실행 시간**: {elapsed:.2f}초")
    lines.append(f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **분석 종목 수**: {len(summaries_df)}")
    lines.append("")

    return "\n".join(lines)


def run_main(mode: str):
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 1) 종목 선택
    if mode == "leaders":
        print("[1/5] Leaders 종목 조회...")
        leaders = get_leaders_tickers(70)
        ticker_list = [t["ticker"] for t in leaders]
        name_map = {t["ticker"]: t["name"] for t in leaders}
        label = "Leaders≥70"
    elif mode == "kospi200":
        print("[1/5] KOSPI200 종목 조회...")
        kospi200 = get_kospi200_tickers()
        with ENGINE.connect() as conn:
            db_rows = conn.execute(
                text("SELECT DISTINCT ticker FROM stocks")
            ).fetchall()
        db_tickers = set(r[0] for r in db_rows)
        ticker_list = [t["ticker"] for t in kospi200 if t["ticker"] in db_tickers]
        name_map = {t["ticker"]: t["name"] for t in kospi200 if t["ticker"] in db_tickers}
        label = "KOSPI200"
    else:
        print("[1/5] 전종목 조회...")
        with ENGINE.connect() as conn:
            db_rows = conn.execute(
                text("SELECT DISTINCT ticker, name FROM stocks ORDER BY ticker")
            ).fetchall()
        ticker_list = [r[0] for r in db_rows]
        name_map = {r[0]: r[1] for r in db_rows}
        label = "전종목"

    print(f"      [{label}] 대상: {len(ticker_list)}종목")

    # 2) 데이터 로딩
    print("[2/5] 데이터 로딩...")
    all_data = {}
    batch_size = 500
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i:i + batch_size]
        batch_data = load_all_data(batch, START_DATE, END_DATE)
        all_data.update(batch_data)
        if len(ticker_list) > batch_size:
            print(f"      {min(i + batch_size, len(ticker_list))}/{len(ticker_list)} 로딩...")
    print(f"      {len(all_data)}종목 로딩 완료")

    # 3) 백테스트 실행
    print("[3/5] 종목별 백테스트 실행...")
    summaries = []
    all_trades = []
    for idx, ticker in enumerate(ticker_list):
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        df = calc_indicators_15(df)
        df_test = df.loc[START_DATE:]
        if len(df_test) < 30:
            continue

        name = name_map.get(ticker, ticker)
        summary, trades_df, equity_df = run_backtest_simple(df_test, ticker, name)
        summaries.append(summary)
        if not trades_df.empty:
            all_trades.append(trades_df)

        if (idx + 1) % 500 == 0:
            print(f"      {idx+1}/{len(ticker_list)} 완료...")

    summaries_df = pd.DataFrame(summaries)
    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    traded_count = (summaries_df["n_trades"] > 0).sum()
    print(f"      완료! 거래 발생: {traded_count}종목, 총 {len(all_trades_df)}건")

    # 4) 차트 & 리포트
    chart_dir = os.path.join(base_dir, "charts_simple_buy")
    print("[4/5] 차트 생성...")
    generate_summary_charts(summaries_df, all_trades_df, chart_dir)

    elapsed = time.time() - start_time
    print("[5/5] 리포트 생성...")
    report = generate_report(summaries_df, all_trades_df, elapsed, label)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, f"backtest_simple_buy_{mode}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*50}")
    print(f"[{label}] 단순 급락 매수 백테스트 완료! ({elapsed:.2f}초)")
    print(f"리포트: {report_path}")
    print(f"분석 종목: {len(summaries_df)}, 거래 발생: {traded_count}, 총 거래: {len(all_trades_df)}")
    if not summaries_df.empty and traded_count > 0:
        traded = summaries_df[summaries_df["n_trades"] > 0]
        print(f"  종목 평균 수익률: {traded['total_return'].mean():.2f}%")
        if not all_trades_df.empty:
            w = (all_trades_df["pnl"] > 0).sum()
            print(f"  전체 승률: {w}/{len(all_trades_df)} = {w/len(all_trades_df)*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단순 급락 매수 백테스트")
    parser.add_argument(
        "mode", nargs="?", default="leaders",
        choices=["leaders", "kospi200", "all"],
        help="leaders: Leaders≥70 / kospi200: KOSPI200 / all: 전종목",
    )
    args = parser.parse_args()
    run_main(args.mode)
