"""
섹터 ETF 로테이션 백테스트 (단일 보유, 거래대금 1위)
================================================================
매매 대상: etf_sector.csv 의 섹터 ETF 24종목 (SK텔레콤 017670 = 주식이라 제외)

매매규칙:
  - 매수: 최근 3거래일 연속 종가가 20일 이평(MA20) 위에 있는 종목들 중
          '거래대금 1위' 종목을 다음날 시가에 매수 (동시 1종목만 보유)
  - 매도: 보유 종목이 3거래일 연속 종가가 MA20 아래로 내려가면 당일 종가 매도
  - 재진입: 매도한 날 종가 기준으로 재평가 → 매수 후보가 있으면 다음날 시가 매수

데이터: 키움 ka10081 (일봉 OHLC + 거래대금 trde_prica, 백만원 단위), base_dt 기준 과거 600일
비용: ETF 는 증권거래세(0.23%) 면제 → 매수/매도 수수료만 (편도 0.015%)
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))  # 키움 키는 상위 .env

DOMAIN = "https://api.kiwoom.com"
APPKEY = os.getenv("KIWOOM_APPKEY")
SECRET = os.getenv("KIWOOM_SECRETKEY")

# etf_sector.csv 의 ETF 목록 (SK텔레콤 017670 주식 제외)
ETFS = {
    "091160": "KODEX 반도체",
    "471990": "KODEX AI반도체핵심장비",
    "487240": "KODEX AI전력핵심설비",
    "091180": "KODEX 자동차",
    "305720": "KODEX 2차전지산업",
    "0115D0": "KODEX 조선TOP10",
    "0080G0": "KODEX 방산TOP10",
    "117680": "KODEX 철강",
    "102960": "KODEX 기계장비",
    "117700": "KODEX 건설",
    "117460": "KODEX 에너지화학",
    "0098F0": "KODEX 원자력SMR",
    "385510": "KODEX 신재생에너지액티브",
    "445290": "KODEX 로봇액티브",
    "091170": "KODEX 은행",
    "140700": "KODEX 보험",
    "102970": "KODEX 증권",
    "228790": "TIGER 화장품",
    "365000": "TIGER 인터넷TOP10",
    "300950": "KODEX 게임산업",
    "395290": "HANARO Fn K-POP&미디어",
    "244580": "KODEX 바이오",
    "140710": "KODEX 운송",
    "438900": "HANARO Fn K-푸드",
}

BASE_DT = "20260724"
MA = 20
CONSEC = 3               # 3거래일 연속
INIT_CAPITAL = 10_000_000
BUY_FEE = 0.00015        # 편도 수수료 0.015%
SELL_FEE = 0.00015       # ETF: 증권거래세 면제
CACHE = "results/etf_sector_data.csv"


def clean_num(v):
    if v is None or v == "":
        return np.nan
    try:
        return float(str(v).replace("+", "").replace(",", ""))
    except (ValueError, TypeError):
        return np.nan


def get_token():
    r = requests.post(
        f"{DOMAIN}/oauth2/token",
        json={"grant_type": "client_credentials", "appkey": APPKEY, "secretkey": SECRET},
        headers={"Content-Type": "application/json;charset=UTF-8", "api-id": "au10001"},
    ).json()
    if r.get("return_code") != 0:
        raise RuntimeError(f"토큰 발급 실패: {r.get('return_msg')}")
    return r["token"]


def fetch_one(token, ticker):
    r = requests.post(
        f"{DOMAIN}/api/dostk/chart",
        json={"stk_cd": ticker, "base_dt": BASE_DT, "upd_stkpc_tp": "1"},
        headers={"Content-Type": "application/json;charset=UTF-8",
                 "api-id": "ka10081", "authorization": f"Bearer {token}"},
        timeout=20,
    ).json()
    if r.get("return_code") != 0:
        return pd.DataFrame()
    rows = r.get("stk_dt_pole_chart_qry", [])
    recs = []
    for row in rows:
        dt = row.get("dt", "")
        if len(dt) != 8:
            continue
        recs.append({
            "ticker": ticker,
            "date": pd.Timestamp(f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}"),
            "open": clean_num(row.get("open_pric")),
            "high": clean_num(row.get("high_pric")),
            "low": clean_num(row.get("low_pric")),
            "close": clean_num(row.get("cur_prc")),
            "amount": clean_num(row.get("trde_prica")),  # 백만원
        })
    return pd.DataFrame(recs)


def load_data(force=False):
    if os.path.exists(CACHE) and not force:
        df = pd.read_csv(CACHE, dtype={"ticker": str})
        df["date"] = pd.to_datetime(df["date"])
        return df
    token = get_token()
    frames = []
    for i, t in enumerate(ETFS, 1):
        d = fetch_one(token, t)
        frames.append(d)
        print(f"  [{i:2}/{len(ETFS)}] {t} {ETFS[t]:22} rows={len(d)}")
        time.sleep(0.35)
    df = pd.concat(frames, ignore_index=True)
    os.makedirs("results", exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def add_signals(g):
    g = g.sort_values("date").reset_index(drop=True)
    g["ma20"] = g["close"].rolling(MA).mean()
    above = g["close"] > g["ma20"]
    below = (g["close"] < g["ma20"]) & g["ma20"].notna()
    # 연속 카운트
    a_streak, b_streak = [], []
    ca = cb = 0
    for i in range(len(g)):
        if pd.isna(g["ma20"].iloc[i]):
            ca = cb = 0
            a_streak.append(0); b_streak.append(0); continue
        ca = ca + 1 if above.iloc[i] else 0
        cb = cb + 1 if below.iloc[i] else 0
        a_streak.append(ca); b_streak.append(cb)
    g["above_streak"] = a_streak
    g["below_streak"] = b_streak
    return g


def simulate(data, buy_consec=CONSEC, sell_consec=CONSEC):
    # 종목별 인덱스
    sd = {}
    all_dates = set()
    for t, g in data.items():
        g = add_signals(g)
        sd[t] = {"df": g, "idx": {d: i for i, d in enumerate(g["date"].values)}}
        all_dates.update(g["date"].values)
    all_dates = sorted(all_dates)

    cash = float(INIT_CAPITAL)
    holding = None           # {ticker, qty, entry_price, entry_date, name}
    pending = None           # 다음날 시가 매수 예정 ticker
    trades = []
    equity = []

    def row_at(t, d):
        i = sd[t]["idx"].get(d)
        return None if i is None else sd[t]["df"].iloc[i]

    for d in all_dates:
        # 1) 전일 결정된 매수를 오늘 시가에 체결
        if holding is None and pending is not None:
            r = row_at(pending, d)
            if r is not None and not pd.isna(r["open"]) and r["open"] > 0:
                px = float(r["open"])
                qty = int(cash // (px * (1 + BUY_FEE)))
                if qty > 0:
                    cost = px * qty
                    cash -= cost + cost * BUY_FEE
                    holding = {"ticker": pending, "qty": qty, "entry_price": px,
                               "entry_date": d, "name": ETFS[pending]}
            pending = None

        # 2) 보유 종목 매도 판정 (3일 연속 MA20 아래 → 종가 매도)
        if holding is not None:
            r = row_at(holding["ticker"], d)
            if r is not None and r["below_streak"] >= sell_consec:
                px = float(r["close"])
                qty = holding["qty"]
                revenue = px * qty
                cash += revenue - revenue * SELL_FEE
                cost = holding["entry_price"] * qty
                pnl = revenue - revenue * SELL_FEE - cost - cost * BUY_FEE
                trades.append({
                    "ticker": holding["ticker"], "name": holding["name"],
                    "buy_date": pd.Timestamp(holding["entry_date"]),
                    "sell_date": pd.Timestamp(d),
                    "buy_price": holding["entry_price"], "sell_price": px,
                    "qty": qty, "pnl": pnl,
                    "return_pct": (px * (1 - SELL_FEE)) / (holding["entry_price"] * (1 + BUY_FEE)) * 100 - 100,
                    "hold_days": (pd.Timestamp(d) - pd.Timestamp(holding["entry_date"])).days,
                })
                holding = None

        # 3) 재평가: flat 이면 오늘 종가 기준 매수후보(3일연속 MA20 위) 중 거래대금 1위 → 내일 매수
        if holding is None:
            best_t, best_amt = None, -1
            for t in ETFS:
                r = row_at(t, d)
                if r is None:
                    continue
                if r["above_streak"] >= buy_consec and not pd.isna(r["amount"]):
                    if r["amount"] > best_amt:
                        best_amt, best_t = r["amount"], t
            pending = best_t
        else:
            pending = None

        # 4) 일별 평가금액
        if holding is not None:
            r = row_at(holding["ticker"], d)
            px = float(r["close"]) if (r is not None and not pd.isna(r["close"])) else holding["entry_price"]
            equity.append((pd.Timestamp(d), cash + px * holding["qty"]))
        else:
            equity.append((pd.Timestamp(d), cash))

    # 마지막날 강제 청산
    if holding is not None:
        d = all_dates[-1]
        r = row_at(holding["ticker"], d)
        px = float(r["close"]) if r is not None else holding["entry_price"]
        qty = holding["qty"]
        revenue = px * qty
        cash += revenue - revenue * SELL_FEE
        cost = holding["entry_price"] * qty
        pnl = revenue - revenue * SELL_FEE - cost - cost * BUY_FEE
        trades.append({
            "ticker": holding["ticker"], "name": holding["name"],
            "buy_date": pd.Timestamp(holding["entry_date"]), "sell_date": pd.Timestamp(d),
            "buy_price": holding["entry_price"], "sell_price": px, "qty": qty, "pnl": pnl,
            "return_pct": (px * (1 - SELL_FEE)) / (holding["entry_price"] * (1 + BUY_FEE)) * 100 - 100,
            "hold_days": (pd.Timestamp(d) - pd.Timestamp(holding["entry_date"])).days,
        })

    eq = pd.DataFrame(equity, columns=["date", "equity"]).set_index("date")
    return pd.DataFrame(trades), eq


def perf(eq):
    if eq.empty:
        return {}
    init, final = eq["equity"].iloc[0], eq["equity"].iloc[-1]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (final / init) ** (1 / years) - 1 if years > 0 and init > 0 else 0
    roll_max = eq["equity"].cummax()
    mdd = ((eq["equity"] - roll_max) / roll_max).min()
    calmar = cagr / abs(mdd) if mdd < 0 else float("inf")
    return {"init": init, "final": final, "total_ret": final / init - 1,
            "cagr": cagr, "mdd": mdd, "calmar": calmar, "years": years}


def main():
    print("데이터 로딩 (키움 ka10081)...")
    df = load_data(force=os.environ.get("FORCE") == "1")
    cover = df.groupby("ticker")["date"].agg(["min", "max", "count"])
    print(f"  종목 {df['ticker'].nunique()}개, 전체 {df['date'].min().date()} ~ {df['date'].max().date()}")

    data = {t: g.reset_index(drop=True) for t, g in df.groupby("ticker")}
    tdf, eq = simulate(data)
    pm = perf(eq)

    # 벤치마크 계산
    bstart, bend = eq.index[0], eq.index[-1]
    def bh(t):
        g = df[df.ticker == t].sort_values("date")
        gg = g[(g.date >= bstart) & (g.date <= bend)]
        if len(gg) < 2:
            return None
        return gg["close"].iloc[-1] / gg["close"].iloc[0] - 1
    ew = [bh(t) for t in ETFS if bh(t) is not None
          and df[df.ticker == t]["date"].min() <= bstart]
    bench = {
        "반도체_bh": bh("091160"),
        "2차전지_bh": bh("305720"),
        "동일비중_bh": np.mean(ew) if ew else None,
        "n_ew": len(ew),
    }

    lines = []
    lines.append("# 섹터 ETF 로테이션 백테스트 (단일 보유·거래대금 1위)\n")
    lines.append("## 매매규칙\n")
    lines.append(f"- 대상: etf_sector.csv 의 섹터 ETF **{len(ETFS)}종목** (SK텔레콤=주식이라 제외)")
    lines.append(f"- 매수: 최근 **{CONSEC}거래일 연속** 종가가 **MA{MA} 위** → 그 중 **거래대금 1위**를 다음날 **시가** 매수 (동시 1종목)")
    lines.append(f"- 매도: 보유종목이 **{CONSEC}거래일 연속** 종가가 **MA{MA} 아래** → 당일 **종가** 매도")
    lines.append("- 재진입: 매도한 날 종가 기준 재평가 → 후보 있으면 다음날 시가 매수")
    lines.append(f"- 비용: ETF 증권거래세 면제, 편도 수수료 {BUY_FEE*100:.3f}%")
    lines.append(f"- 초기자본 {INIT_CAPITAL:,}원 (매수 시 전액 투입)")
    lines.append(f"- 기간: {eq.index[0].date()} ~ {eq.index[-1].date()} ({pm['years']:.2f}년, MA{MA} 워밍업 포함)\n")
    lines.append("---\n## 성과\n")
    lines.append("| 지표 | 값 |\n|---|---:|")
    lines.append(f"| 초기자본 | {pm['init']:,.0f}원 |")
    lines.append(f"| 최종자본 | {pm['final']:,.0f}원 |")
    lines.append(f"| 총수익률 | {pm['total_ret']*100:+.2f}% |")
    lines.append(f"| CAGR | {pm['cagr']*100:+.2f}% |")
    lines.append(f"| MDD | {pm['mdd']*100:.2f}% |")
    lines.append(f"| Calmar | {pm['calmar']:.2f} |")
    lines.append(f"| 운용기간 | {pm['years']:.2f}년 |\n")

    lines.append("## 벤치마크 비교 (동일 기간 매수후보유)\n")
    lines.append("| 벤치마크 | 총수익률 |\n|---|---:|")
    lines.append(f"| **본 전략 (로테이션)** | **{pm['total_ret']*100:+.2f}%** |")
    if bench["동일비중_bh"] is not None:
        lines.append(f"| 동일비중 {bench['n_ew']}종목 B&H | {bench['동일비중_bh']*100:+.2f}% |")
    if bench["반도체_bh"] is not None:
        lines.append(f"| KODEX 반도체 단독 B&H (사후 최고) | {bench['반도체_bh']*100:+.2f}% |")
    if bench["2차전지_bh"] is not None:
        lines.append(f"| KODEX 2차전지 단독 B&H | {bench['2차전지_bh']*100:+.2f}% |")
    lines.append("")

    if not tdf.empty:
        win = (tdf["return_pct"] > 0).mean() * 100
        avg = tdf["return_pct"].mean()
        med = tdf["return_pct"].median()
        wins = tdf[tdf["return_pct"] > 0]["return_pct"]
        losses = tdf[tdf["return_pct"] <= 0]["return_pct"]
        pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
        lines.append("## 거래 통계\n")
        lines.append("| 지표 | 값 |\n|---|---:|")
        lines.append(f"| 총 거래수 | {len(tdf)} |")
        lines.append(f"| 승률 | {win:.1f}% |")
        lines.append(f"| 평균 수익률 | {avg:+.2f}% |")
        lines.append(f"| 중간값 수익률 | {med:+.2f}% |")
        lines.append(f"| 평균 이익 | {wins.mean() if len(wins) else 0:+.2f}% |")
        lines.append(f"| 평균 손실 | {losses.mean() if len(losses) else 0:+.2f}% |")
        lines.append(f"| 손익비(PF) | {pf:.2f} |")
        lines.append(f"| 평균 보유일 | {tdf['hold_days'].mean():.1f}일 |\n")
        lines.append("## 전체 거래 내역\n")
        lines.append("| # | 종목 | 매수일 | 매도일 | 매수가 | 매도가 | 수익률 | 보유일 |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|")
        for i, r in enumerate(tdf.itertuples(), 1):
            lines.append(f"| {i} | {r.name} | {r.buy_date.date()} | {r.sell_date.date()} | "
                         f"{r.buy_price:,.0f} | {r.sell_price:,.0f} | {r.return_pct:+.2f}% | {r.hold_days} |")

    lines.append("\n---\n## 종목별 데이터 커버리지\n")
    lines.append("| 종목 | 시작 | 종료 | 봉수 |\n|---|---|---|---:|")
    for t in ETFS:
        if t in cover.index:
            c = cover.loc[t]
            lines.append(f"| {ETFS[t]} | {pd.Timestamp(c['min']).date()} | {pd.Timestamp(c['max']).date()} | {int(c['count'])} |")
        else:
            lines.append(f"| {ETFS[t]} | - | - | 0 |")

    lines.append("\n---\n## 평가 및 한계\n")
    lines.append("- **손익구조**: 승률은 낮으나(40%) 평균이익 ≫ 평균손실(PF 2.09)인 전형적 추세추종형. "
                 "큰 추세를 잡고(화장품·반도체·로봇·2차전지) MA20 3일 이탈로 손실을 빨리 끊는 구조.")
    lines.append("- **집중 리스크**: 전액 단일 ETF 올인이라 MDD −40%로 깊음. Calmar 0.76 은 위험조정 성과가 낮은 편.")
    lines.append("- **로테이션의 부가가치**: 동일비중 B&H(+72.6%) 대비 소폭 우위(+93.2%)이나, "
                 "사후 최고 종목(반도체 +254.8%)에는 크게 열위 — 단, 종목 사전선택은 불가능.")
    lines.append("- **⚠️ 시기 의존성**: 검증구간(2024~2026)이 반도체·2차전지·조선·방산 대세상승장에 편중. "
                 "약세장(2022 등) 미포함 → 강세장 한정 성과일 가능성 높음. ETF 상장 이력 한계로 확장 불가.")
    lines.append("- **해석 규칙**: '3일 이상'=3거래일 연속 종가 기준, '거래대금 1위'=평가일 당일 거래대금 기준.")

    report = "\n".join(lines) + "\n"
    os.makedirs("results", exist_ok=True)
    with open("results/backtest_etf_sector.md", "w") as f:
        f.write(report)
    print("\n" + report)
    print("결과 저장: results/backtest_etf_sector.md")


if __name__ == "__main__":
    main()
