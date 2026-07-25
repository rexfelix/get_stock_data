"""섹터 ETF 로테이션: 매도(이탈) 연속일 2일 vs 3일 비교 (매수는 3일 고정)."""
import pandas as pd
from backtest_etf_sector import load_data, simulate, perf, ETFS


def trade_stats(tdf):
    if tdf.empty:
        return dict(n=0, win=0, avg=0, med=0, pf=0, hold=0)
    win = (tdf["return_pct"] > 0).mean() * 100
    w = tdf[tdf["return_pct"] > 0]["return_pct"]
    l = tdf[tdf["return_pct"] <= 0]["return_pct"]
    pf = (w.sum() / abs(l.sum())) if l.sum() != 0 else float("inf")
    return dict(n=len(tdf), win=win, avg=tdf["return_pct"].mean(),
                med=tdf["return_pct"].median(), pf=pf, hold=tdf["hold_days"].mean())


def main():
    df = load_data()
    data = {t: g.reset_index(drop=True) for t, g in df.groupby("ticker")}

    rows = []
    for sell_c in (2, 3):
        tdf, eq = simulate(data, buy_consec=3, sell_consec=sell_c)
        pm = perf(eq)
        ts = trade_stats(tdf)
        rows.append((sell_c, pm, ts, tdf))

    L = []
    L.append("# 섹터 ETF 로테이션: 매도(20일선 이탈) 연속일 2일 vs 3일 비교\n")
    L.append("매수 조건(3거래일 연속 MA20 위 + 거래대금 1위)은 동일, 매도 이탈 연속일만 변경.\n")
    L.append(f"기간: {rows[0][1]['years']:.2f}년 (2024-02-02~2026-07-24), 대상 ETF {len(ETFS)}종목\n")
    L.append("| 이탈 조건 | 총수익률 | CAGR | MDD | Calmar | 거래수 | 승률 | PF | 평균수익 | 평균보유 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sell_c, pm, ts, _ in rows:
        tag = f"**{sell_c}일 연속**" + (" (기존)" if sell_c == 3 else " (변경)")
        L.append(f"| {tag} | {pm['total_ret']*100:+.2f}% | {pm['cagr']*100:+.2f}% | "
                 f"{pm['mdd']*100:.2f}% | {pm['calmar']:.2f} | {ts['n']} | {ts['win']:.1f}% | "
                 f"{ts['pf']:.2f} | {ts['avg']:+.2f}% | {ts['hold']:.1f}일 |")
    L.append("")

    report = "\n".join(L) + "\n"
    with open("results/compare_etf_sell_consec.md", "w") as f:
        f.write(report)
    print(report)
    print("저장: results/compare_etf_sell_consec.md")


if __name__ == "__main__":
    main()
