"""
스윙 돌파 매매 백테스트 - 120MA 필터 비교
- 기본 규칙: backtest_swing_breakout.py 와 동일
- 추가 비교: 매수 시점에 종가(또는 매수가)가 120일 이동평균선 이상인 경우만 매수
"""

import os
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

ENGINE = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)


def load_stocks() -> dict[str, pd.DataFrame]:
    # MA120 계산을 위해 충분한 과거 데이터 필요 → 2022-01부터 로드
    query = text("""
        SELECT date, open, high, low, close, ticker
        FROM stocks WHERE date >= '2022-01-01'
        ORDER BY ticker, date
    """)
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for t, g in df.groupby("ticker"):
        g = g.reset_index(drop=True)
        g["ma120"] = g["close"].rolling(120, min_periods=120).mean()
        out[t] = g
    return out


def simulate_trades(df: pd.DataFrame, ticker: str = "", use_ma120: bool = False) -> list[dict]:
    n = len(df)
    if n < 5:
        return []

    low = df["low"]
    high = df["high"]
    is_swing_low = (low < low.shift(1)) & (low < low.shift(-1))
    is_swing_high = (high > high.shift(1)) & (high > high.shift(-1))

    trades = []
    in_position = False
    buy_price = 0.0
    buy_date = None
    stop_loss = 0.0

    last_swing_low_range = 0.0
    last_swing_high_low = 0.0
    has_buy_signal = False
    has_sell_signal = False
    buy_signal_day = -1

    i = 2
    while i < n:
        row = df.iloc[i]

        if is_swing_low.iloc[i - 1]:
            swing_candle = df.iloc[i - 1]
            last_swing_low_range = (swing_candle["high"] - swing_candle["low"]) / 2
            has_buy_signal = True
            buy_signal_day = i

        if is_swing_high.iloc[i - 1]:
            last_swing_high_low = df.iloc[i - 1]["low"]
            if in_position:
                has_sell_signal = True

        if in_position:
            if row["low"] <= stop_loss:
                ret = (stop_loss - buy_price) / buy_price * 100
                trades.append({
                    "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
                    "sell_date": row["date"], "sell_price": stop_loss,
                    "return_pct": ret, "hold_days": (row["date"] - buy_date).days,
                    "reason": "손절",
                })
                in_position = False
                has_sell_signal = False
                i += 1
                continue

            if has_sell_signal and last_swing_high_low > 0 and row["low"] <= last_swing_high_low:
                sell_price = last_swing_high_low
                ret = (sell_price - buy_price) / buy_price * 100
                trades.append({
                    "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
                    "sell_date": row["date"], "sell_price": sell_price,
                    "return_pct": ret, "hold_days": (row["date"] - buy_date).days,
                    "reason": "고점저가이탈",
                })
                in_position = False
                has_sell_signal = False
                i += 1
                continue

            i += 1

        else:
            if has_buy_signal and buy_signal_day == i and last_swing_low_range > 0:
                trigger = row["open"] + last_swing_low_range
                if row["high"] >= trigger and trigger > 0:
                    # 120MA 필터: 매수가(trigger)가 120일선 이상이어야 함
                    if use_ma120:
                        ma = row["ma120"]
                        if pd.isna(ma) or trigger < ma:
                            i += 1
                            continue

                    buy_price = trigger
                    buy_date = row["date"]
                    stop_loss = row["low"]
                    has_buy_signal = False
                    has_sell_signal = False
                    last_swing_high_low = 0.0
                    in_position = True
                    i += 1
                    continue

            i += 1

    if in_position:
        last = df.iloc[-1]
        ret = (last["close"] - buy_price) / buy_price * 100
        trades.append({
            "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
            "sell_date": last["date"], "sell_price": last["close"],
            "return_pct": ret, "hold_days": (last["date"] - buy_date).days,
            "reason": "미청산",
        })

    return trades


def stats(sub: pd.DataFrame) -> dict:
    total = len(sub)
    if total == 0:
        return {}
    wins = (sub["return_pct"] > 0).sum()
    losses = total - wins
    win_rate = wins / total * 100
    avg_ret = sub["return_pct"].mean()
    med_ret = sub["return_pct"].median()
    avg_win = sub.loc[sub["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
    avg_loss = sub.loc[sub["return_pct"] <= 0, "return_pct"].mean() if losses > 0 else 0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    avg_hold = sub["hold_days"].mean()
    return {
        "total": total, "wins": wins, "losses": losses, "win_rate": win_rate,
        "avg_ret": avg_ret, "med_ret": med_ret, "avg_win": avg_win,
        "avg_loss": avg_loss, "pf": pf, "avg_hold": avg_hold,
    }


def section_table(label: str, s_base: dict, s_ma: dict) -> list[str]:
    if not s_base and not s_ma:
        return []
    lines = [f"\n## {label}\n"]
    lines.append("| 지표 | 기본 | +120MA필터 | 차이 |")
    lines.append("|---|---:|---:|---:|")

    def fmt(d, k, f):
        return f.format(d[k]) if d else "-"

    def diff(k, f):
        if not (s_base and s_ma):
            return "-"
        v = s_ma[k] - s_base[k]
        return f.format(v)

    rows = [
        ("총 거래수", "total", "{:,}"),
        ("승리", "wins", "{:,}"),
        ("패배", "losses", "{:,}"),
        ("승률(%)", "win_rate", "{:.1f}"),
        ("평균 수익률(%)", "avg_ret", "{:+.2f}"),
        ("중간값 수익률(%)", "med_ret", "{:+.2f}"),
        ("평균이익(%)", "avg_win", "{:+.2f}"),
        ("평균손실(%)", "avg_loss", "{:+.2f}"),
        ("손익비", "pf", "{:.2f}"),
        ("평균 보유일", "avg_hold", "{:.1f}"),
    ]
    for name, k, f in rows:
        lines.append(f"| {name} | {fmt(s_base, k, f)} | {fmt(s_ma, k, f)} | {diff(k, f)} |")
    return lines


def analyze_compare(base_df: pd.DataFrame, ma_df: pd.DataFrame) -> str:
    lines = ["# 스윙 돌파 매매 - 120MA 필터 비교\n"]
    lines.append("## 매매 규칙\n")
    lines.append("- **매수**: 단기저점 확정 후, 다음봉 시가 + (고가-저가)/2 돌파")
    lines.append("- **매도**: 단기고점 캔들 저가 하향 돌파")
    lines.append("- **손절**: 매수일 저가 이탈")
    lines.append("- **120MA 필터**: 매수가(trigger)가 120일 이동평균선 이상일 때만 매수\n")
    lines.append("---")

    base_df = base_df.copy()
    ma_df = ma_df.copy()
    base_df["buy_year"] = pd.to_datetime(base_df["buy_date"]).dt.year
    ma_df["buy_year"] = pd.to_datetime(ma_df["buy_date"]).dt.year

    lines += section_table("전체 기간 종합", stats(base_df), stats(ma_df))

    years = sorted(set(base_df["buy_year"].unique()) | set(ma_df["buy_year"].unique()))
    for y in years:
        sb = base_df[base_df["buy_year"] == y]
        sm = ma_df[ma_df["buy_year"] == y]
        lines += section_table(f"{y}년", stats(sb), stats(sm))

    return "\n".join(lines)


def run(stocks: dict, use_ma120: bool) -> pd.DataFrame:
    label = "120MA필터" if use_ma120 else "기본"
    print(f"\n[{label}] 시뮬레이션 시작...")
    all_trades = []
    start = time.time()
    for i, (ticker, df) in enumerate(stocks.items()):
        all_trades.extend(simulate_trades(df, ticker, use_ma120=use_ma120))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(stocks)}] 거래: {len(all_trades):,}건 | {(i+1)/elapsed:.0f}종목/초")
    print(f"  완료: {len(all_trades):,}건, {time.time()-start:.1f}초")

    df = pd.DataFrame(all_trades)
    if df.empty:
        return df
    df["buy_year"] = pd.to_datetime(df["buy_date"]).dt.year
    df = df[
        (df["buy_year"] >= 2023)
        & (df["buy_price"] > 0)
        & np.isfinite(df["return_pct"])
    ]
    print(f"  2023~ 유효 거래: {len(df):,}건")
    return df


def main():
    print("데이터 로딩...")
    stocks = load_stocks()
    print(f"  {len(stocks):,}종목 로드")

    base_df = run(stocks, use_ma120=False)
    ma_df = run(stocks, use_ma120=True)

    print("\n비교 분석...")
    report = analyze_compare(base_df, ma_df)
    print(report)

    os.makedirs("results", exist_ok=True)
    with open("results/backtest_swing_breakout_ma120.md", "w") as f:
        f.write(report)
        f.write("\n")
    print("\n결과 저장: results/backtest_swing_breakout_ma120.md")


if __name__ == "__main__":
    main()
