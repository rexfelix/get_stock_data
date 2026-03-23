"""기준봉 신호 발생 시 갭하락(전일종가→당일시가) 분포 확인"""

import numpy as np
import pandas as pd
from backtest_crash import ENGINE, START_DATE, END_DATE, load_all_data
from backtest_simple_buy import get_leaders_tickers, calc_indicators_15

MA_PERIOD = 15
CRASH_TH = 92

leaders = get_leaders_tickers(70)
ticker_list = [t["ticker"] for t in leaders]
name_map = {t["ticker"]: t["name"] for t in leaders}

all_data = load_all_data(ticker_list, START_DATE, END_DATE)
print(f"{len(all_data)}종목 로딩\n")

# 기준봉 조건: 이격도<=92 + 음봉 + 거래량 증가
gijoong_gaps = []
# 이격도<=92 전체 (기준봉 조건 불문)
all_crash_gaps = []

for ticker in ticker_list:
    if ticker not in all_data:
        continue
    df = calc_indicators_15(all_data[ticker])
    df_test = df.loc[START_DATE:]
    if len(df_test) < 30:
        continue

    name = name_map.get(ticker, ticker)
    dates = df_test.index.tolist()

    for i in range(1, len(df_test)):
        row = df_test.iloc[i]
        prev = df_test.iloc[i - 1]
        disp = row["disparity15"]

        if pd.isna(disp):
            continue

        prev_close = prev["close"]
        if prev_close <= 0:
            continue
        gap_pct = (row["open"] - prev_close) / prev_close * 100

        if disp <= CRASH_TH:
            all_crash_gaps.append({
                "ticker": ticker, "name": name, "date": dates[i],
                "gap_pct": gap_pct, "disp": disp,
                "is_bearish": row["close"] < row["open"],
                "vol_up": row["volume"] > prev["volume"],
            })

            # 기준봉 조건
            if row["close"] < row["open"] and row["volume"] > prev["volume"]:
                gijoong_gaps.append({
                    "ticker": ticker, "name": name, "date": dates[i],
                    "gap_pct": gap_pct, "disp": disp,
                })

crash_df = pd.DataFrame(all_crash_gaps)
gij_df = pd.DataFrame(gijoong_gaps)

print("=" * 60)
print(f"[이격도 ≤ {CRASH_TH}% 전체 신호]  {len(crash_df)}건")
print(f"  갭하락 분포: 평균 {crash_df['gap_pct'].mean():.2f}%, 중위 {crash_df['gap_pct'].median():.2f}%")
print(f"  min {crash_df['gap_pct'].min():.2f}% / max {crash_df['gap_pct'].max():.2f}%")
print()

print(f"[기준봉 신호 (음봉+거래량증가)]  {len(gij_df)}건")
print(f"  갭하락 분포: 평균 {gij_df['gap_pct'].mean():.2f}%, 중위 {gij_df['gap_pct'].median():.2f}%")
print(f"  min {gij_df['gap_pct'].min():.2f}% / max {gij_df['gap_pct'].max():.2f}%")
print()

# 갭 구간별 분포
bins = [float("-inf"), -10, -7, -5, -3, -1, 0, 1, 3, float("inf")]
labels = ["<-10%", "-10~-7%", "-7~-5%", "-5~-3%", "-3~-1%", "-1~0%", "0~1%", "1~3%", ">3%"]

print("=" * 60)
print("[기준봉 신호 - 갭 구간별 분포]")
print(f"{'구간':<12} {'건수':>6} {'비율':>8} {'누적':>8}")
print("-" * 40)
gij_df["gap_bin"] = pd.cut(gij_df["gap_pct"], bins=bins, labels=labels)
counts = gij_df["gap_bin"].value_counts().reindex(labels, fill_value=0)
total = len(gij_df)
cumul = 0
for label, cnt in counts.items():
    cumul += cnt
    print(f"  {label:<12} {cnt:>4}   {cnt/total*100:>6.1f}%   {cumul/total*100:>6.1f}%")

print()
print("=" * 60)
print("[이격도 ≤ 92% 전체 - 갭 구간별 분포]")
print(f"{'구간':<12} {'건수':>6} {'비율':>8} {'누적':>8}")
print("-" * 40)
crash_df["gap_bin"] = pd.cut(crash_df["gap_pct"], bins=bins, labels=labels)
counts2 = crash_df["gap_bin"].value_counts().reindex(labels, fill_value=0)
total2 = len(crash_df)
cumul2 = 0
for label, cnt in counts2.items():
    cumul2 += cnt
    print(f"  {label:<12} {cnt:>4}   {cnt/total2*100:>6.1f}%   {cumul2/total2*100:>6.1f}%")

# 기준봉 중 갭하락 3% 이상 비율
g3 = (gij_df["gap_pct"] <= -3).sum()
print(f"\n기준봉 신호 중 갭하락 ≥3%: {g3}/{len(gij_df)} ({g3/len(gij_df)*100:.1f}%)")
g5 = (gij_df["gap_pct"] <= -5).sum()
print(f"기준봉 신호 중 갭하락 ≥5%: {g5}/{len(gij_df)} ({g5/len(gij_df)*100:.1f}%)")
