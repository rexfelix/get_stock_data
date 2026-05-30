"""
한의 법칙 생애주기 — '현재 2단계 확산기' 종목 스크리너 (KOSPI200)
=================================================================
verify_hans_rule.py 에서 검증된 정의를 그대로 사용해, 현재 정배열이
'살아있는' 주도주를 4단계 생애주기로 분류하고 2단계(확산기)를 추출한다.

생애주기 (results/Hans_rule.md):
  1단계 형성기 : 정배열(MA4>13>26>52) 막 완성 (온셋 직후)
  2단계 확산기 : 강력한 추세 상승 = 공세 황금기 (신고가 갱신 지속, 둔화신호 無)
  3단계 둔화기 : 정배열은 유지되나 고점 후 정체 / 거래량 데드크로스 (기술적 균열)
  4단계 종결기 : 시간의 중력(2년=104주) 근접

조작적 정의 (최신 주봉 기준, MA4>13>26>52 정배열 '현재 유지'가 전제):
  elapsed        = 최근 온셋(정배열 거짓→참)부터 현재까지 경과 주수
  ret_onset      = 온셋 종가 대비 현재 종가 수익률
  wks_since_peak = 온셋 이후 최고 종가(공세 현재 정점)로부터 경과 주수
  dd_from_peak   = 현재 종가 / 온셋이후 최고 종가 - 1
  vol_dead       = 거래량 5주MA < 13주MA (데드크로스 = 둔화 신호)

  · 4단계 종결기 : elapsed >= 96주 (2년 임박)
  · 3단계 둔화기 : (wks_since_peak >= 8) 또는 (dd_from_peak <= -12%) 또는 vol_dead
  · 1단계 형성기 : elapsed <= 4주
  · 2단계 확산기 : 그 외 (신고가 최근 8주내 + 고점 -12% 이내 + 거래량 살아있음
                   + 5주 < elapsed < 96주) AND 온셋후 수익률 >= 15%

실적(보조): financial_annual.operating_income 으로 최근 연간 영업이익 성장률 표시.
            온셋연도=FY0, FY1=온셋+1 (가격이 실적 ~1년 선행 — 검증됨)

실행: python screen_diffusion_stage.py
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import ENGINE
from verify_hans_rule import (
    MA_PERIODS, RESAMPLE_RULE, MIN_WEEKLY_BARS, MIN_GAP_BETWEEN_ONSETS,
    get_universe, load_financials, _growth,
)

ELAPSED_END = 96       # 4단계 종결기 임계(주)
ELAPSED_FORM = 4       # 1단계 형성기 상한(주)
PEAK_STALE = 8         # 신고가 미갱신 허용(주) — 초과 시 둔화기
DD_TOL = -0.12         # 고점 대비 허용 낙폭 — 초과 하락 시 둔화기
MIN_RET = 15.0         # 확산기 최소 온셋후 수익률(%)


def load_daily_vol(tickers):
    """거래량 포함 일봉 로딩."""
    data = {}
    batch = 300
    with ENGINE.connect() as conn:
        for i in range(0, len(tickers), batch):
            chunk = tickers[i:i + batch]
            ph = ",".join([f":t{j}" for j in range(len(chunk))])
            params = {f"t{j}": t for j, t in enumerate(chunk)}
            q = (f"SELECT ticker, date, high, low, close, volume "
                 f"FROM stocks WHERE ticker IN ({ph}) ORDER BY ticker, date")
            df = pd.read_sql(text(q), conn, params=params, parse_dates=["date"])
            for col in ("high", "low", "close", "volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            for tk, g in df.groupby("ticker"):
                data[tk] = g.set_index("date").sort_index()
    return data


def to_weekly_vol(df):
    return pd.DataFrame({
        "high": df["high"].resample(RESAMPLE_RULE).max(),
        "low": df["low"].resample(RESAMPLE_RULE).min(),
        "close": df["close"].resample(RESAMPLE_RULE).last(),
        "volume": df["volume"].resample(RESAMPLE_RULE).sum(),
    }).dropna(subset=["close"])


def classify(w):
    """최신 주봉 기준 생애주기 단계를 판정. 정배열 미유지면 None."""
    c = w["close"]
    m4 = c.rolling(4).mean()
    m13 = c.rolling(13).mean()
    m26 = c.rolling(26).mean()
    m52 = c.rolling(52).mean()
    aligned = ((m4 > m13) & (m13 > m26) & (m26 > m52)).fillna(False).values
    close = c.values
    n = len(close)
    last = n - 1
    if not aligned[last]:
        return None  # 현재 정배열 아님 → 공세 진행 종목 아님

    # 최근 온셋: 현재까지 이어진 정배열 구간의 시작(거짓→참 전환주)
    onset = last
    while onset - 1 >= 0 and aligned[onset - 1]:
        onset -= 1
    # onset 은 현재 정배열 구간의 첫 주. 그 시작이 실제 전환인지 확인용으로 충분.

    elapsed = last - onset
    onset_price = close[onset]
    cur = close[last]
    ret_onset = (cur / onset_price - 1) * 100

    seg = close[onset:last + 1]
    peak_off = int(np.argmax(seg))
    peak_idx = onset + peak_off
    peak_price = close[peak_idx]
    wks_since_peak = last - peak_idx
    dd_from_peak = (cur / peak_price - 1)

    vol = w["volume"].values
    vma_s = pd.Series(vol).rolling(5).mean().values
    vma_l = pd.Series(vol).rolling(13).mean().values
    vol_dead = (not np.isnan(vma_s[last]) and not np.isnan(vma_l[last])
                and vma_s[last] < vma_l[last])

    # 단계 판정
    if elapsed >= ELAPSED_END:
        stage = "4_종결기"
    elif (wks_since_peak >= PEAK_STALE) or (dd_from_peak <= DD_TOL) or vol_dead:
        stage = "3_둔화기"
    elif elapsed <= ELAPSED_FORM:
        stage = "1_형성기"
    else:
        stage = "2_확산기" if ret_onset >= MIN_RET else "1_형성기"

    return {
        "stage": stage,
        "onset_date": c.index[onset],
        "peak_date": c.index[peak_idx],
        "elapsed_wk": elapsed,
        "ret_onset": ret_onset,
        "wks_since_peak": wks_since_peak,
        "dd_from_peak": dd_from_peak * 100,
        "vol_dead": vol_dead,
        "cur_date": c.index[last],
    }


def recent_oi_growth(fin, ticker, onset_year):
    """온셋연도=FY0 기준 FY0→FY1 영업이익 성장률(가능하면), 없으면 최근 가용 성장."""
    d = fin.get(ticker)
    if not d:
        return None
    g = _growth(d.get(onset_year), d.get(onset_year + 1))
    if g is not None:
        return g
    # fallback: 가장 최근 두 해
    yrs = sorted(y for y in d if d[y] is not None)
    if len(yrs) >= 2:
        return _growth(d[yrs[-2]], d[yrs[-1]])
    return None


def main():
    t0 = time.time()
    print("[1/4] universe")
    tickers, name_map = get_universe()
    print(f"      {len(tickers)} tickers")
    print("[2/4] daily(+vol) -> weekly")
    daily = load_daily_vol(tickers)
    weekly = {tk: to_weekly_vol(daily[tk]) for tk in tickers if tk in daily}
    weekly = {tk: w for tk, w in weekly.items() if len(w) >= MIN_WEEKLY_BARS}
    print(f"      {len(weekly)} tickers")
    print("[3/4] financials")
    fin = load_financials()
    print("[4/4] classify")

    rows = []
    for tk, w in weekly.items():
        info = classify(w)
        if info is None:
            continue
        oy = pd.Timestamp(info["onset_date"]).year
        info["ticker"] = tk
        info["name"] = name_map.get(tk, tk)
        info["oi_growth"] = recent_oi_growth(fin, tk, oy)
        rows.append(info)
    df = pd.DataFrame(rows)

    rep = build_report(df, time.time() - t0)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "Hans_rule_diffusion_stage.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(rep)
    n_diff = (df["stage"] == "2_확산기").sum() if not df.empty else 0
    print(f"report -> {out} ({time.time()-t0:.1f}s, 확산기 {n_diff}종목)")


def build_report(df, elapsed):
    L = []
    L.append("# 한의 법칙 생애주기 — 현재 '2단계 확산기' 주도주 (KOSPI200)\n")
    L.append(f"- 작성: {datetime.now():%Y-%m-%d %H:%M:%S} / 유니버스 kospi200_members 200종목")
    L.append("- 기준: 최신 주봉(W-FRI) MA4>13>26>52 **정배열 현재 유지** 종목을 4단계로 분류")
    L.append(f"- 확산기 조건: 온셋 {ELAPSED_FORM}~{ELAPSED_END}주 + 온셋후 수익 ≥ {MIN_RET:.0f}% "
             f"+ 신고가 {PEAK_STALE}주내 갱신 + 고점대비 낙폭 ≤ {abs(DD_TOL)*100:.0f}% + 거래량 살아있음")
    L.append("- 영업이익 성장: 온셋연도=FY0 → FY1 성장률(가격이 실적 ~1년 선행, verify §2 검증)\n")

    if df.empty:
        return "\n".join(L + ["> 정배열 유지 종목 없음"])

    order = {"1_형성기": 1, "2_확산기": 2, "3_둔화기": 3, "4_종결기": 4}
    df = df.copy()
    df["ord"] = df["stage"].map(order)

    # 단계 분포
    L.append("## 단계 분포 (정배열 유지 종목)\n")
    L.append("| 단계 | 종목수 |")
    L.append("|---|---|")
    for st in ["1_형성기", "2_확산기", "3_둔화기", "4_종결기"]:
        L.append(f"| {st.replace('_',' ')} | {(df['stage']==st).sum()} |")
    L.append(f"| **정배열 유지 합계** | **{len(df)}** |")
    L.append("")

    # 2단계 확산기 메인 리스트
    diff = df[df["stage"] == "2_확산기"].sort_values("ret_onset", ascending=False)
    L.append(f"## ★ 2단계 확산기 — {len(diff)}종목 (온셋후 수익률 내림차순)\n")
    L.append("| # | 종목 | 코드 | 온셋 | 경과(주) | 온셋후수익 | 고점대비 | 신고가경과(주) | FY0→1 OI성장 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(diff.iterrows(), 1):
        oi = f"{r['oi_growth']:+.0f}%" if pd.notna(r['oi_growth']) else "-"
        L.append(f"| {i} | {r['name']} | {r['ticker']} | {pd.Timestamp(r['onset_date']):%Y-%m} "
                 f"| {r['elapsed_wk']} | {r['ret_onset']:+.0f}% | {r['dd_from_peak']:+.1f}% "
                 f"| {r['wks_since_peak']} | {oi} |")
    L.append("")

    # 참고: 3단계 둔화기 (공세 후반 — 비중축소 후보)
    slow = df[df["stage"] == "3_둔화기"].sort_values("ret_onset", ascending=False)
    L.append(f"## (참고) 3단계 둔화기 — {len(slow)}종목 (기술적 균열, 비중축소 검토)\n")
    L.append("| 종목 | 코드 | 온셋 | 경과(주) | 온셋후수익 | 고점대비 | 신고가경과 | 거래량데드 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in slow.head(25).iterrows():
        L.append(f"| {r['name']} | {r['ticker']} | {pd.Timestamp(r['onset_date']):%Y-%m} "
                 f"| {r['elapsed_wk']} | {r['ret_onset']:+.0f}% | {r['dd_from_peak']:+.1f}% "
                 f"| {r['wks_since_peak']}주 | {'예' if r['vol_dead'] else '-'} |")
    L.append("")

    # 참고: 4단계 종결기 (2년 임박)
    endd = df[df["stage"] == "4_종결기"].sort_values("elapsed_wk", ascending=False)
    if len(endd):
        L.append(f"## (참고) 4단계 종결기 — {len(endd)}종목 (2년 임박, 공세종말점 경계)\n")
        L.append("| 종목 | 코드 | 온셋 | 경과(주) | 온셋후수익 | 고점대비 |")
        L.append("|---|---|---|---|---|---|")
        for _, r in endd.iterrows():
            L.append(f"| {r['name']} | {r['ticker']} | {pd.Timestamp(r['onset_date']):%Y-%m} "
                     f"| {r['elapsed_wk']} | {r['ret_onset']:+.0f}% | {r['dd_from_peak']:+.1f}% |")
        L.append("")

    L.append("## 해석\n")
    L.append("- **2단계 확산기**가 한의 법칙상 '공세의 황금기' — 정배열 유지 + 신고가 지속 + "
             "거래량 동반 구간. 검증 결과 주도주 공세는 중앙 30주·104주내 종료되므로, "
             "경과 주수가 짧을수록 잔여 공세기간이 길다.")
    L.append("- **3단계 둔화기**는 고점 정체·거래량 데드크로스 등 균열 발생 — 신규진입 부적합, "
             "보유분 비중축소 검토.")
    L.append("- 영업이익(FY0→FY1) 성장이 큰 종목일수록 확산기 펀더멘털 뒷받침이 강하다.\n")
    L.append("## 한계\n")
    L.append("- 단계 임계(경과 96주·신고가 8주·낙폭 12%·거래량 5/13주)는 휴리스틱이며 민감도 미검정.")
    L.append("- 데이터 2019~, 현재구성 200종목 생존편향. 영업이익은 연 1회 갱신·추정치 혼재.")
    L.append("- 실시간 가격이 아닌 직전 주봉 종가 기준. 진입 전 개별 차트·수급 확인 필요.")
    L.append("")
    L.append(f"- 소요 {elapsed:.1f}초 / 재현 `python research/screen_diffusion_stage.py`")
    return "\n".join(L)


if __name__ == "__main__":
    main()
