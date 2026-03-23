"""
급락주 매매 백테스트
- 매매규칙.md 기반 전략
- 모드: kospi200 / ex_kospi200 (KOSPI200 제외 전종목)
- 종목별 독립 백테스트 후 통합 리포트 생성
"""

import argparse
import os
import re
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
load_dotenv()

DB_URL = "postgresql://{}:{}@{}:{}/{}".format(
    os.getenv("DB_USER", "rexfelix"),
    os.getenv("DB_PASSWORD", ""),
    os.getenv("DB_HOST", "localhost"),
    os.getenv("DB_PORT", "5432"),
    os.getenv("DB_NAME", "stock_db"),
)

INITIAL_CAPITAL = 10_000_000
FEE_BUY = 0.00015
FEE_SELL = 0.00015
TAX_SELL = 0.0023
STOP_LOSS_PCT = None  # None이면 기준봉 저가만, 숫자면 추가 %손절

START_DATE = "2024-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ENGINE = create_engine(DB_URL)


# ──────────────────────────────────────────────
# KOSPI200 종목 조회
# ──────────────────────────────────────────────
def get_kospi200_tickers() -> list[dict]:
    tickers = []
    for page in range(1, 22):
        url = f"https://finance.naver.com/sise/entryJongmok.naver?&page={page}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.select('td a[href*="main.naver?code="]')
        for link in links:
            code = re.search(r"code=(\d+)", link["href"])
            if code:
                tickers.append({"ticker": code.group(1), "name": link.text.strip()})
    return tickers


# ──────────────────────────────────────────────
# 데이터 로딩 (전종목 일괄)
# ──────────────────────────────────────────────
def load_all_data(tickers: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    placeholders = ",".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT date, open, high, low, close, volume, ticker, name
        FROM stocks
        WHERE ticker IN ({placeholders})
          AND date >= '{start_date}'::date - interval '90 days'
          AND date <= '{end_date}'
        ORDER BY ticker, date ASC
    """
    df_all = pd.read_sql(query, ENGINE)
    df_all["date"] = pd.to_datetime(df_all["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    result = {}
    for ticker, group in df_all.groupby("ticker"):
        g = group.set_index("date").sort_index()
        result[ticker] = g
    return result


# ──────────────────────────────────────────────
# 지표 계산
# ──────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma20"] = df["close"].rolling(window=20).mean()
    df["sma5"] = df["close"].rolling(window=5).mean()
    df["disparity"] = df["close"] / df["sma20"] * 100
    df["is_bearish"] = df["close"] < df["open"]
    df["vol_up"] = df["volume"] > df["volume"].shift(1)
    return df


# ──────────────────────────────────────────────
# 백테스트 엔진
# ──────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, ticker: str, name: str):
    capital = INITIAL_CAPITAL
    position = None
    gijoongbong = None
    above_20ma = False

    trades = []
    equity_curve = []
    dates = df.index.tolist()

    for i in range(1, len(df)):
        row = df.iloc[i]
        close = row["close"]
        sma20 = row["sma20"]
        sma5 = row["sma5"]
        disp = row["disparity"]
        date = dates[i]

        if pd.isna(sma20) or pd.isna(sma5):
            eq = capital + (position["quantity"] * close if position else 0)
            equity_curve.append({"date": date, "equity": eq})
            continue

        if position is not None:
            sell_reason = None
            if not above_20ma and close > sma20:
                above_20ma = True
            if close < position["buy_candle_low"]:
                sell_reason = "손절(기준봉저가)"
            elif STOP_LOSS_PCT is not None and close <= position["entry_price"] * (1 - STOP_LOSS_PCT):
                sell_reason = f"손절({STOP_LOSS_PCT*100:.0f}%)"
            elif disp >= 120:
                sell_reason = "급등신호"
            elif above_20ma and close < sma5:
                sell_reason = "5MA이탈"

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
                above_20ma = False
        else:
            bought = False
            if gijoongbong is not None:
                if close > gijoongbong["open_price"]:
                    max_qty = int(capital / (close * (1 + FEE_BUY)))
                    if max_qty > 0:
                        cost = max_qty * close
                        fee = cost * FEE_BUY
                        capital -= cost + fee
                        position = {
                            "entry_price": close, "quantity": max_qty,
                            "entry_date": date, "buy_candle_low": row["low"],
                        }
                        above_20ma = False
                        bought = True
                    gijoongbong = None
            if not bought and gijoongbong is not None:
                if close > sma20:
                    gijoongbong = None
            if not bought:
                if disp <= 90 and row["is_bearish"] and row["vol_up"]:
                    gijoongbong = {"date": date, "open_price": row["open"]}

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

    # 종목별 요약
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

    # MDD
    if not equity_df.empty:
        roll_max = equity_df["equity"].cummax()
        dd = equity_df["equity"] / roll_max - 1
        summary["mdd"] = dd.min() * 100
    else:
        summary["mdd"] = 0

    return summary, trades_df, equity_df


# ──────────────────────────────────────────────
# 차트 생성 (통합)
# ──────────────────────────────────────────────
def generate_summary_charts(summaries_df, all_trades_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    traded = summaries_df[summaries_df["n_trades"] > 0].copy()

    # ── 차트1: 종목별 수익률 분포 히스토그램 ──
    fig, ax = plt.subplots(figsize=(14, 6))
    if not traded.empty:
        bins = np.arange(
            traded["total_return"].min() - 2,
            traded["total_return"].max() + 2, 2
        )
        colors_hist = ["green" if x >= 0 else "red" for x in np.histogram(traded["total_return"], bins=bins)[0]]
        n, bins_out, patches = ax.hist(traded["total_return"], bins=bins, edgecolor="black", alpha=0.7)
        for patch, b in zip(patches, bins_out):
            if b + (bins_out[1] - bins_out[0]) / 2 >= 0:
                patch.set_facecolor("green")
            else:
                patch.set_facecolor("red")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"종목별 전략 수익률 분포 (거래 발생 {len(traded)}종목)", fontsize=14)
    ax.set_xlabel("수익률 (%)")
    ax.set_ylabel("종목 수")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "return_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 차트2: 전략 vs 바이앤홀드 산점도 ──
    fig, ax = plt.subplots(figsize=(10, 10))
    if not traded.empty:
        ax.scatter(traded["bnh_return"], traded["total_return"], alpha=0.6, s=30, color="steelblue")
        # 대각선 (전략=바이앤홀드)
        lim_min = min(traded["bnh_return"].min(), traded["total_return"].min()) - 5
        lim_max = max(traded["bnh_return"].max(), traded["total_return"].max()) + 5
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, label="전략=B&H")
        # 상위/하위 종목 라벨
        for _, row in traded.nlargest(5, "total_return").iterrows():
            ax.annotate(row["name"], (row["bnh_return"], row["total_return"]), fontsize=7, alpha=0.8)
        for _, row in traded.nsmallest(3, "total_return").iterrows():
            ax.annotate(row["name"], (row["bnh_return"], row["total_return"]), fontsize=7, alpha=0.8, color="red")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("바이앤홀드 수익률 (%)")
    ax.set_ylabel("전략 수익률 (%)")
    ax.set_title("전략 수익률 vs 바이앤홀드 수익률", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "strategy_vs_bnh.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 차트3: 매도 사유별 수익률 박스플롯 ──
    if not all_trades_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        reasons = all_trades_df["reason"].unique()
        data_by_reason = [all_trades_df[all_trades_df["reason"] == r]["return_pct"] for r in reasons]
        bp = ax.boxplot(data_by_reason, labels=reasons, patch_artist=True)
        palette = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#9b59b6"]
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("매도 사유별 수익률 분포", fontsize=14)
        ax.set_ylabel("수익률 (%)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(os.path.join(output_dir, "reason_boxplot.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 차트4: TOP/BOTTOM 종목 바 차트 ──
    if not traded.empty and len(traded) >= 10:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        top10 = traded.nlargest(10, "total_return")
        ax1.barh(range(len(top10)), top10["total_return"], color="green", alpha=0.7)
        ax1.set_yticks(range(len(top10)))
        ax1.set_yticklabels([f"{r['name']}({r['ticker']})" for _, r in top10.iterrows()], fontsize=9)
        ax1.set_xlabel("수익률 (%)")
        ax1.set_title("TOP 10 수익 종목", fontsize=13)
        ax1.grid(True, alpha=0.3, axis="x")
        ax1.invert_yaxis()

        bot10 = traded.nsmallest(10, "total_return")
        ax2.barh(range(len(bot10)), bot10["total_return"], color="red", alpha=0.7)
        ax2.set_yticks(range(len(bot10)))
        ax2.set_yticklabels([f"{r['name']}({r['ticker']})" for _, r in bot10.iterrows()], fontsize=9)
        ax2.set_xlabel("수익률 (%)")
        ax2.set_title("BOTTOM 10 손실 종목", fontsize=13)
        ax2.grid(True, alpha=0.3, axis="x")
        ax2.invert_yaxis()

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "top_bottom.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 차트5: 월별 거래 수익률 히트맵 ──
    if not all_trades_df.empty:
        trades_copy = all_trades_df.copy()
        trades_copy["sell_month"] = pd.to_datetime(trades_copy["sell_date"]).dt.to_period("M")
        monthly = trades_copy.groupby("sell_month").agg(
            n_trades=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_return=("return_pct", "mean"),
            win_rate=("pnl", lambda x: (x > 0).sum() / len(x) * 100),
        )
        if len(monthly) > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
            months_str = [str(m) for m in monthly.index]
            colors_bar = ["green" if v >= 0 else "red" for v in monthly["total_pnl"]]
            ax1.bar(months_str, monthly["total_pnl"], color=colors_bar, alpha=0.7)
            ax1.set_title("월별 총 손익", fontsize=13)
            ax1.set_ylabel("손익 (원)")
            ax1.grid(True, alpha=0.3, axis="y")
            ax1.axhline(0, color="black", linewidth=0.5)

            ax2.bar(months_str, monthly["n_trades"], color="steelblue", alpha=0.7)
            ax2.set_title("월별 거래 건수", fontsize=13)
            ax2.set_ylabel("거래 수")
            ax2.grid(True, alpha=0.3, axis="y")
            plt.xticks(rotation=45)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "monthly_pnl.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)


# ──────────────────────────────────────────────
# 리포트 생성
# ──────────────────────────────────────────────
def generate_report(summaries_df, all_trades_df, elapsed, label="KOSPI200", chart_dir="charts"):
    traded = summaries_df[summaries_df["n_trades"] > 0]
    no_trade = summaries_df[summaries_df["n_trades"] == 0]

    lines = []
    lines.append(f"# 급락주 매매 백테스트 리포트 ({label})\n")

    lines.append("## 전략 개요\n")
    lines.append(f"- **대상**: {label} ({len(summaries_df)}종목)")
    lines.append(f"- **기간**: {START_DATE} ~ {END_DATE}")
    lines.append(f"- **초기자본**: 종목당 {INITIAL_CAPITAL:,}원")
    lines.append(f"- **수수료**: 매수 {FEE_BUY*100:.3f}% / 매도 {FEE_SELL*100:.3f}% / 세금 {TAX_SELL*100:.2f}%")
    lines.append("")

    lines.append("### 매매 규칙\n")
    lines.append("| 구분 | 조건 |")
    lines.append("|------|------|")
    lines.append("| 급락 신호 | 20일 이동평균 이격도 ≤ 90% |")
    lines.append("| 기준봉 | 급락신호 + 음봉 + 전일대비 거래량 증가 |")
    lines.append("| 매수 | 기준봉의 시가를 종가로 돌파 |")
    lines.append("| 매도 | 급등신호(이격도≥120%) 또는 20MA 위 진입 후 5MA 종가 이탈 |")
    lines.append("| 손절 | 매수봉 저가를 종가로 이탈 |")
    lines.append("")

    # ── 통합 성과 ──
    lines.append("## 통합 성과 요약\n")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 분석 종목 수 | {len(summaries_df)} |")
    lines.append(f"| 거래 발생 종목 | {len(traded)} ({len(traded)/len(summaries_df)*100:.1f}%) |")
    lines.append(f"| 미거래 종목 | {len(no_trade)} |")
    lines.append(f"| 총 거래 건수 | {all_trades_df.shape[0] if not all_trades_df.empty else 0} |")

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

        # 보유일
        tc = all_trades_df.copy()
        tc["hold_days"] = (pd.to_datetime(tc["sell_date"]) - pd.to_datetime(tc["buy_date"])).dt.days
        lines.append(f"| 평균 보유일 | {tc['hold_days'].mean():.1f}일 |")
    lines.append("")

    # ── 바이앤홀드 비교 ──
    if not traded.empty:
        lines.append("## 전략 vs 바이앤홀드 비교\n")
        lines.append("| 지표 | 전략 (거래종목 평균) | 바이앤홀드 (거래종목 평균) |")
        lines.append("|------|-----|-----|")
        lines.append(f"| 평균 수익률 | {traded['total_return'].mean():.2f}% | {traded['bnh_return'].mean():.2f}% |")
        lines.append(f"| 중위 수익률 | {traded['total_return'].median():.2f}% | {traded['bnh_return'].median():.2f}% |")
        outperform = (traded["total_return"] > traded["bnh_return"]).sum()
        lines.append(f"| 전략 우위 종목 수 | {outperform} / {len(traded)} ({outperform/len(traded)*100:.1f}%) | |")
        lines.append("")

    # ── 매도 사유별 분석 ──
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

    # ── 차트 ──
    chart_rel = chart_dir
    lines.append("## 차트\n")
    lines.append("### 종목별 수익률 분포")
    lines.append(f"![수익률분포]({chart_rel}/return_distribution.png)\n")
    lines.append("### 전략 vs 바이앤홀드")
    lines.append(f"![전략vs바이앤홀드]({chart_rel}/strategy_vs_bnh.png)\n")
    lines.append("### 매도 사유별 수익률")
    lines.append(f"![매도사유]({chart_rel}/reason_boxplot.png)\n")
    lines.append("### TOP / BOTTOM 10 종목")
    lines.append(f"![TOP_BOTTOM]({chart_rel}/top_bottom.png)\n")
    lines.append("### 월별 손익")
    lines.append(f"![월별손익]({chart_rel}/monthly_pnl.png)\n")

    # ── TOP/BOTTOM 종목 상세 ──
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

    # ── 전체 매매 기록 (요약) ──
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

    # ── 실행 정보 ──
    lines.append("## 실행 정보\n")
    lines.append(f"- **실행 시간**: {elapsed:.2f}초")
    lines.append(f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **분석 종목 수**: {len(summaries_df)}")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def run_main(mode: str):
    start_time = time.time()
    base_dir = os.path.dirname(__file__)

    # 1) KOSPI200 목록 조회
    print("[1/5] KOSPI200 종목 목록 조회...")
    kospi200 = get_kospi200_tickers()
    kospi200_tickers = set(t["ticker"] for t in kospi200)
    print(f"      KOSPI200: {len(kospi200)}종목")

    # 2) DB 전종목 조회
    with ENGINE.connect() as conn:
        db_rows = conn.execute(
            text("SELECT DISTINCT ticker, name FROM stocks ORDER BY ticker")
        ).fetchall()
    db_all = {r[0]: r[1] for r in db_rows}
    print(f"      DB 전체: {len(db_all)}종목")

    # 3) 모드별 종목 선택
    if mode == "kospi200":
        label = "KOSPI200"
        ticker_list = [t["ticker"] for t in kospi200 if t["ticker"] in db_all]
        name_map = {t["ticker"]: t["name"] for t in kospi200 if t["ticker"] in db_all}
    elif mode == "ex_kospi200":
        label = "KOSPI200 제외"
        ticker_list = [t for t in db_all if t not in kospi200_tickers]
        name_map = {t: db_all[t] for t in ticker_list}
    else:
        label = "전종목"
        ticker_list = list(db_all.keys())
        name_map = db_all

    print(f"      [{label}] 대상: {len(ticker_list)}종목")

    # 4) 데이터 로딩 (대량일 때 배치로)
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

    # 5) 백테스트 실행
    print("[3/5] 종목별 백테스트 실행...")
    summaries = []
    all_trades = []
    for idx, ticker in enumerate(ticker_list):
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        df = calc_indicators(df)
        df_test = df.loc[START_DATE:]
        if len(df_test) < 30:
            continue

        name = name_map.get(ticker, ticker)
        summary, trades_df, equity_df = run_backtest(df_test, ticker, name)
        summaries.append(summary)
        if not trades_df.empty:
            all_trades.append(trades_df)

        if (idx + 1) % 500 == 0:
            print(f"      {idx+1}/{len(ticker_list)} 완료...")

    summaries_df = pd.DataFrame(summaries)
    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    traded_count = (summaries_df["n_trades"] > 0).sum()
    print(f"      완료! 거래 발생: {traded_count}종목, 총 {len(all_trades_df)}건")

    # 6) 차트 & 리포트 (손절 옵션에 따라 파일명 구분)
    sl_tag = f"_sl{int(STOP_LOSS_PCT*100)}" if STOP_LOSS_PCT else ""
    chart_suffix = {"kospi200": "", "ex_kospi200": "_ex_kospi200", "all": "_all"}
    chart_dir_name = f"charts{chart_suffix.get(mode, '')}{sl_tag}"
    chart_dir = os.path.join(base_dir, chart_dir_name)

    # label에 손절 조건 표시
    if STOP_LOSS_PCT:
        label += f" (손절 {STOP_LOSS_PCT*100:.0f}% 추가)"

    print("[4/5] 차트 생성...")
    generate_summary_charts(summaries_df, all_trades_df, chart_dir)

    elapsed = time.time() - start_time
    print(f"[5/5] 리포트 생성...")
    report = generate_report(summaries_df, all_trades_df, elapsed, label=label, chart_dir=chart_dir_name)

    report_suffix = {"kospi200": "kospi200", "ex_kospi200": "ex_kospi200", "all": "all"}
    report_name = f"backtest_{report_suffix.get(mode, mode)}{sl_tag}.md"
    report_path = os.path.join(base_dir, "results", report_name)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*50}")
    print(f"[{label}] 백테스트 완료! ({elapsed:.2f}초)")
    print(f"리포트: {report_path}")
    print(f"분석 종목: {len(summaries_df)}, 거래 발생: {traded_count}, 총 거래: {len(all_trades_df)}")
    if not summaries_df.empty and traded_count > 0:
        traded = summaries_df[summaries_df["n_trades"] > 0]
        print(f"  종목 평균 수익률: {traded['total_return'].mean():.2f}%")
        if not all_trades_df.empty:
            print(f"  전체 승률: {(all_trades_df['pnl']>0).sum()}/{len(all_trades_df)} = {(all_trades_df['pnl']>0).sum()/len(all_trades_df)*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="급락주 매매 백테스트")
    parser.add_argument(
        "mode", nargs="?", default="kospi200",
        choices=["kospi200", "ex_kospi200", "all"],
        help="kospi200: KOSPI200 종목 / ex_kospi200: KOSPI200 제외 / all: 전종목",
    )
    parser.add_argument(
        "--stoploss", type=float, default=None,
        help="추가 손절 비율 (예: 0.03 = -3%%)",
    )
    args = parser.parse_args()
    STOP_LOSS_PCT = args.stoploss
    run_main(args.mode)
