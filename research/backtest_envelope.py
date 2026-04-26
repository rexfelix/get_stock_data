"""Envelope 매매 전략 백테스트.

매수: close[d] < MA[d] × (1 - pct) → 익일 시가 매수
매도: 보유 중 close[d] < MA[d] → 익일 시가 매도

PRD: report/envelope/PRD.md
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# 신호 함수
# ---------------------------------------------------------------------------

def compute_ma(prices: pd.Series, n: int) -> pd.Series:
    """단순이동평균 (MA-N). 윈도우 미달 구간은 NaN."""
    return prices.rolling(window=n, min_periods=n).mean()


def make_buy_signal(close: pd.Series, ma: pd.Series, pct: float) -> pd.Series:
    """매수 신호: close < ma * (1 - pct). ma가 NaN인 구간은 False."""
    threshold = ma * (1.0 - pct)
    sig = close < threshold
    sig = sig & ma.notna()
    return sig


def make_sell_signal(close: pd.Series, ma: pd.Series) -> pd.Series:
    """매도 신호: close < ma. ma가 NaN인 구간은 False."""
    sig = close < ma
    sig = sig & ma.notna()
    return sig


# ---------------------------------------------------------------------------
# 메트릭
# ---------------------------------------------------------------------------

def compute_cagr(initial: float, final: float, years: float) -> float:
    """CAGR = (final/initial) ** (1/years) - 1."""
    if initial <= 0 or years <= 0:
        return 0.0
    return (final / initial) ** (1.0 / years) - 1.0


# ---------------------------------------------------------------------------
# 시뮬레이션 엔진
# ---------------------------------------------------------------------------

def _precompute_per_ticker(df: pd.DataFrame, ma_n: int, pct: float) -> dict[str, dict[str, Any]]:
    """ticker별로 MA, 매수신호, 매도신호, 날짜→인덱스 맵을 사전 계산."""
    by_ticker: dict[str, dict[str, Any]] = {}
    for ticker, gdf in df.groupby("ticker", sort=False):
        gdf = gdf.sort_values("date").reset_index(drop=True)
        ma = compute_ma(gdf["close"], ma_n)
        buy = make_buy_signal(gdf["close"], ma, pct)
        sell = make_sell_signal(gdf["close"], ma)
        date_to_idx = {d: i for i, d in enumerate(gdf["date"])}
        by_ticker[ticker] = {
            "df": gdf,
            "ma": ma.to_numpy(),
            "buy": buy.to_numpy(),
            "sell": sell.to_numpy(),
            "open": gdf["open"].to_numpy(dtype=float),
            "close": gdf["close"].to_numpy(dtype=float),
            "date_to_idx": date_to_idx,
            "dates": gdf["date"].to_numpy(),
        }
    return by_ticker


def simulate(
    df: pd.DataFrame,
    ma_n: int,
    pct: float,
    slot_capital: int,
    max_positions: int,
    commission: float,
    sell_commission: float,
    tax: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    """포트폴리오 백테스트 실행.

    매도 우선, 매수 후순. 같은 날 매수 신호가 빈 슬롯보다 많으면
    이격률 (MA-close)/MA 내림차순으로 선별, 동률 시 ticker 오름차순.

    종료 시 보유 종목은 마지막 종가로 강제 청산.
    """
    by_ticker = _precompute_per_ticker(df, ma_n, pct)

    all_dates = pd.DatetimeIndex(np.sort(df["date"].unique()))
    all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]

    positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    initial_capital = max_positions * slot_capital
    realized_pnl = 0.0

    for d_idx in range(len(all_dates) - 1):
        d = all_dates[d_idx]
        next_d = all_dates[d_idx + 1]

        # 1) 매도: 보유 종목에 d에서 매도 신호 → next_d 시가 매도
        for tk in list(positions.keys()):
            data = by_ticker.get(tk)
            if data is None or d not in data["date_to_idx"]:
                continue
            i = data["date_to_idx"][d]
            if not data["sell"][i]:
                continue
            ni = data["date_to_idx"].get(next_d)
            if ni is None:
                continue
            sell_price = data["open"][ni]
            if not np.isfinite(sell_price) or sell_price <= 0:
                continue

            pos = positions.pop(tk)
            shares = pos["shares"]
            sell_value = shares * sell_price
            sell_net = sell_value * (1.0 - sell_commission - tax)
            pnl = sell_net - pos["buy_net"]
            realized_pnl += pnl
            trades.append({
                "ticker": tk,
                "buy_date": pos["buy_date"],
                "buy_price": pos["buy_price"],
                "sell_date": next_d,
                "sell_price": sell_price,
                "shares": shares,
                "pnl": pnl,
                "exit_reason": "sell_signal",
            })

        # 2) 매수: d에서 매수 신호 발생 → next_d 시가 매수, 슬롯 제한
        candidates: list[tuple[float, str, float]] = []
        for tk, data in by_ticker.items():
            if tk in positions:
                continue
            i = data["date_to_idx"].get(d)
            if i is None or not data["buy"][i]:
                continue
            ni = data["date_to_idx"].get(next_d)
            if ni is None:
                continue
            close_v = data["close"][i]
            ma_v = data["ma"][i]
            if not np.isfinite(ma_v) or ma_v <= 0:
                continue
            buy_price = data["open"][ni]
            if not np.isfinite(buy_price) or buy_price <= 0:
                continue
            disparity = (ma_v - close_v) / ma_v
            candidates.append((disparity, tk, buy_price))

        # 정렬: 이격률 내림차순, 동률 시 ticker 오름차순
        candidates.sort(key=lambda x: (-x[0], x[1]))

        free_slots = max_positions - len(positions)
        for _disp, tk, buy_price in candidates[:free_slots]:
            shares = int(slot_capital / (buy_price * (1.0 + commission)))
            if shares <= 0:
                continue
            buy_value = shares * buy_price
            buy_net = buy_value * (1.0 + commission)
            positions[tk] = {
                "buy_date": next_d,
                "buy_price": buy_price,
                "shares": shares,
                "buy_net": buy_net,
            }

    # 시뮬 종료 시 강제 청산 (마지막 종가)
    for tk, pos in list(positions.items()):
        data = by_ticker[tk]
        last_price = data["close"][-1]
        last_date = data["dates"][-1]
        shares = pos["shares"]
        sell_value = shares * last_price
        sell_net = sell_value * (1.0 - sell_commission - tax)
        pnl = sell_net - pos["buy_net"]
        realized_pnl += pnl
        trades.append({
            "ticker": tk,
            "buy_date": pos["buy_date"],
            "buy_price": pos["buy_price"],
            "sell_date": last_date,
            "sell_price": last_price,
            "shares": shares,
            "pnl": pnl,
            "exit_reason": "force_close",
        })
        del positions[tk]

    final_equity = initial_capital + realized_pnl
    return {
        "trades": trades,
        "final_equity": final_equity,
        "initial_capital": initial_capital,
        "realized_pnl": realized_pnl,
    }


# ---------------------------------------------------------------------------
# 메트릭 종합
# ---------------------------------------------------------------------------

def compute_metrics(
    trades: list[dict[str, Any]],
    initial_capital: float,
    final_equity: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, float]:
    """백테스트 결과 메트릭 (CAGR, MDD, PF, 승률 등)."""
    if len(trades) == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "n_wins": 0,
            "n_losses": 0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
        }

    years = max((end_date - start_date).days / 365.25, 1e-9)
    total_return = (final_equity / initial_capital) - 1.0
    cagr = compute_cagr(initial_capital, final_equity, years)

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    win_rate = len(wins) / len(trades)
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    payoff_ratio = (avg_win / abs(avg_loss)) if losses else float("inf")

    # MDD: 거래 청산 시점 기준 누적 자본 곡선
    sorted_trades = sorted(trades, key=lambda t: t["sell_date"])
    equity = initial_capital
    peak = initial_capital
    mdd = 0.0
    for t in sorted_trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        mdd = min(mdd, dd)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "mdd": mdd,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "n_trades": len(trades),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "avg_pnl": sum(pnls) / len(pnls),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
    }


# ---------------------------------------------------------------------------
# DB 로드
# ---------------------------------------------------------------------------

def _get_engine():
    from dotenv import load_dotenv
    load_dotenv()
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


def load_data(load_start: str = "2019-07-01") -> pd.DataFrame:
    """stocks 테이블에서 OHLCV 로드 (워밍업 포함)."""
    engine = _get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT date, open, high, low, close, volume, ticker, name
                FROM stocks
                WHERE date >= :start
                ORDER BY ticker, date
            """),
            conn,
            params={"start": load_start},
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def _fdr_reader(ticker: str, start: str | None = None, end: str | None = None):
    """FinanceDataReader.DataReader 래퍼 (테스트 monkeypatch용)."""
    import FinanceDataReader as fdr
    return fdr.DataReader(ticker, start, end)


def load_kodex_lev(load_start: str = "2019-07-01") -> pd.DataFrame:
    """KODEX 레버리지(122630) 일봉을 fdr로 로드 → 표준 OHLCV 형식."""
    raw = _fdr_reader("122630", load_start)
    df = pd.DataFrame({
        "date": pd.to_datetime(raw.index),
        "open": raw["Open"].astype(float).to_numpy(),
        "high": raw["High"].astype(float).to_numpy(),
        "low": raw["Low"].astype(float).to_numpy(),
        "close": raw["Close"].astype(float).to_numpy(),
        "volume": raw["Volume"].astype(float).to_numpy(),
    })
    df["ticker"] = "122630"
    df["name"] = "KODEX 레버리지"
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 그리드 러너
# ---------------------------------------------------------------------------

MA_GRID = [10, 20, 30, 40, 60, 80, 100, 120]
PCT_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SLOT_CAPITAL = 10_000_000
MAX_POSITIONS = 10
COMMISSION = 0.00015
SELL_COMMISSION = 0.00015
TAX = 0.0018


def _precompute_ma_per_ticker(df: pd.DataFrame, ma_grid: list[int]) -> dict[str, dict[str, Any]]:
    """ticker별로 모든 ma_n에 대한 MA를 사전 계산. 신호는 매 조합마다 산출."""
    ticker_data: dict[str, dict[str, Any]] = {}
    for ticker, gdf in df.groupby("ticker", sort=False):
        gdf = gdf.sort_values("date").reset_index(drop=True)
        close = gdf["close"]
        mas = {n: compute_ma(close, n).to_numpy() for n in ma_grid}
        ticker_data[ticker] = {
            "df": gdf,
            "mas": mas,
            "open": gdf["open"].to_numpy(dtype=float),
            "close": close.to_numpy(dtype=float),
            "date_to_idx": {d: i for i, d in enumerate(gdf["date"])},
            "dates": gdf["date"].to_numpy(),
        }
    return ticker_data


def _simulate_with_precomputed(
    ticker_data: dict[str, dict[str, Any]],
    all_dates: pd.DatetimeIndex,
    ma_n: int,
    pct: float,
    slot_capital: int = SLOT_CAPITAL,
    max_positions: int = MAX_POSITIONS,
    commission: float = COMMISSION,
    sell_commission: float = SELL_COMMISSION,
    tax: float = TAX,
) -> dict[str, Any]:
    """사전계산된 MA를 사용하여 (ma_n, pct) 조합 백테스트.

    거래비용/슬롯은 파라미터로 주입 (호환 유지를 위해 모듈 상수가 default).
    """
    positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    initial_capital = max_positions * slot_capital
    realized_pnl = 0.0

    # 각 종목별 매수/매도 신호 boolean 배열 사전 산출
    sigs: dict[str, dict[str, np.ndarray]] = {}
    for tk, data in ticker_data.items():
        ma = data["mas"][ma_n]
        close = data["close"]
        threshold = ma * (1.0 - pct)
        with np.errstate(invalid="ignore"):
            buy = (close < threshold) & ~np.isnan(ma)
            sell = (close < ma) & ~np.isnan(ma)
        sigs[tk] = {"buy": buy, "sell": sell, "ma": ma}

    for d_idx in range(len(all_dates) - 1):
        d = all_dates[d_idx]
        next_d = all_dates[d_idx + 1]

        # 매도
        for tk in list(positions.keys()):
            data = ticker_data.get(tk)
            if data is None or d not in data["date_to_idx"]:
                continue
            i = data["date_to_idx"][d]
            if not sigs[tk]["sell"][i]:
                continue
            ni = data["date_to_idx"].get(next_d)
            if ni is None:
                continue
            sell_price = data["open"][ni]
            if not np.isfinite(sell_price) or sell_price <= 0:
                continue
            pos = positions.pop(tk)
            shares = pos["shares"]
            sell_value = shares * sell_price
            sell_net = sell_value * (1.0 - sell_commission - tax)
            pnl = sell_net - pos["buy_net"]
            realized_pnl += pnl
            trades.append({
                "ticker": tk,
                "buy_date": pos["buy_date"],
                "buy_price": pos["buy_price"],
                "sell_date": next_d,
                "sell_price": sell_price,
                "shares": shares,
                "pnl": pnl,
                "exit_reason": "sell_signal",
            })

        # 매수
        candidates: list[tuple[float, str, float]] = []
        for tk, data in ticker_data.items():
            if tk in positions:
                continue
            i = data["date_to_idx"].get(d)
            if i is None or not sigs[tk]["buy"][i]:
                continue
            ni = data["date_to_idx"].get(next_d)
            if ni is None:
                continue
            ma_v = sigs[tk]["ma"][i]
            close_v = data["close"][i]
            buy_price = data["open"][ni]
            if not np.isfinite(buy_price) or buy_price <= 0 or ma_v <= 0:
                continue
            disparity = (ma_v - close_v) / ma_v
            candidates.append((disparity, tk, buy_price))

        candidates.sort(key=lambda x: (-x[0], x[1]))
        free_slots = max_positions - len(positions)
        for _disp, tk, buy_price in candidates[:free_slots]:
            shares = int(slot_capital / (buy_price * (1.0 + commission)))
            if shares <= 0:
                continue
            buy_value = shares * buy_price
            buy_net = buy_value * (1.0 + commission)
            positions[tk] = {
                "buy_date": next_d,
                "buy_price": buy_price,
                "shares": shares,
                "buy_net": buy_net,
            }

    # 강제 청산
    for tk, pos in list(positions.items()):
        data = ticker_data[tk]
        last_price = data["close"][-1]
        last_date = data["dates"][-1]
        shares = pos["shares"]
        sell_value = shares * last_price
        sell_net = sell_value * (1.0 - SELL_COMMISSION - TAX)
        pnl = sell_net - pos["buy_net"]
        realized_pnl += pnl
        trades.append({
            "ticker": tk,
            "buy_date": pos["buy_date"],
            "buy_price": pos["buy_price"],
            "sell_date": last_date,
            "sell_price": last_price,
            "shares": shares,
            "pnl": pnl,
            "exit_reason": "force_close",
        })
        del positions[tk]

    final_equity = initial_capital + realized_pnl
    return {
        "trades": trades,
        "final_equity": final_equity,
        "initial_capital": initial_capital,
        "realized_pnl": realized_pnl,
    }


def run_grid(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ma_grid: list[int] = MA_GRID,
    pct_grid: list[float] = PCT_GRID,
    slot_capital: int = SLOT_CAPITAL,
    max_positions: int = MAX_POSITIONS,
    commission: float = COMMISSION,
    sell_commission: float = SELL_COMMISSION,
    tax: float = TAX,
) -> pd.DataFrame:
    """그리드 백테스트 실행. (MA × X%) 조합별 메트릭 DataFrame 반환."""
    import time

    print(f"[1/3] ticker별 MA 사전계산 시작 (종목 {df['ticker'].nunique():,}, MA {len(ma_grid)}개)")
    t0 = time.time()
    ticker_data = _precompute_ma_per_ticker(df, ma_grid)
    print(f"      완료: {time.time() - t0:.1f}초")

    all_dates_full = pd.DatetimeIndex(np.sort(df["date"].unique()))
    all_dates = all_dates_full[(all_dates_full >= start_date) & (all_dates_full <= end_date)]
    years = max((end_date - start_date).days / 365.25, 1e-9)

    print(f"[2/3] 그리드 백테스트 ({len(ma_grid) * len(pct_grid)}조합) 시작")
    rows = []
    total = len(ma_grid) * len(pct_grid)
    cnt = 0
    t_grid = time.time()
    for ma_n in ma_grid:
        for pct in pct_grid:
            cnt += 1
            t1 = time.time()
            result = _simulate_with_precomputed(
                ticker_data, all_dates, ma_n, pct,
                slot_capital=slot_capital,
                max_positions=max_positions,
                commission=commission,
                sell_commission=sell_commission,
                tax=tax,
            )
            metrics = compute_metrics(
                result["trades"],
                result["initial_capital"],
                result["final_equity"],
                start_date,
                end_date,
            )
            rows.append({
                "ma_n": ma_n,
                "pct": pct,
                "final_equity": result["final_equity"],
                **metrics,
            })
            elapsed = time.time() - t1
            print(
                f"      [{cnt:2d}/{total}] MA{ma_n:>3d} ±{int(pct * 100):>2d}% "
                f"CAGR={metrics['cagr']:.4f} MDD={metrics['mdd']:.4f} "
                f"PF={metrics['profit_factor']:.2f} 거래={metrics['n_trades']:,} "
                f"승률={metrics['win_rate']:.3f} ({elapsed:.1f}s)"
            )
    print(f"      그리드 총 시간: {time.time() - t_grid:.1f}초")

    print("[3/3] 결과 정렬")
    out = pd.DataFrame(rows).sort_values("cagr", ascending=False).reset_index(drop=True)
    return out


def write_report(grid_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp,
                 path: str) -> None:
    """그리드 결과를 마크다운 리포트로 저장."""
    years = (end_date - start_date).days / 365.25
    df = grid_df.copy()
    df["calmar"] = df.apply(
        lambda r: (r["cagr"] / abs(r["mdd"])) if r["mdd"] < 0 else float("nan"), axis=1
    )

    lines: list[str] = []
    lines.append("# Envelope 매매 전략 백테스트 결과\n")
    lines.append(f"- **기간**: {start_date.date()} ~ {end_date.date()} ({years:.2f}년)")
    lines.append("- **유니버스**: KOSPI/KOSDAQ 전체")
    lines.append(
        f"- **포트폴리오**: 슬롯 {MAX_POSITIONS}개 × 슬롯당 {SLOT_CAPITAL:,}원, "
        f"초기자본 {MAX_POSITIONS * SLOT_CAPITAL:,}원"
    )
    lines.append(
        f"- **거래비용**: 매수 수수료 {COMMISSION:.4%}, 매도 수수료 {SELL_COMMISSION:.4%}, "
        f"매도세 {TAX:.4%}"
    )
    lines.append("- **매수**: close < MA × (1 − X%) → 익일 시가 매수")
    lines.append("- **매도**: 보유 중 close < MA → 익일 시가 매도 (MA 상향 제약 없음)")
    lines.append(f"- **그리드**: MA ∈ {MA_GRID}, X% ∈ {[int(p * 100) for p in PCT_GRID]}")
    lines.append("- **목적함수**: CAGR 최대화\n")

    lines.append("## 조합별 성과 (CAGR 내림차순)\n")
    lines.append("| 순위 | MA | X% | CAGR | MDD | Calmar | PF | 승률 | 거래수 | 최종자산 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i, r in df.iterrows():
        calmar_s = f"{r['calmar']:.3f}" if pd.notna(r["calmar"]) else "—"
        lines.append(
            f"| {i + 1} | {int(r['ma_n'])} | {int(r['pct'] * 100)} | "
            f"{r['cagr']:.4f} | {r['mdd']:.4f} | {calmar_s} | "
            f"{r['profit_factor']:.2f} | {r['win_rate']:.3f} | "
            f"{int(r['n_trades']):,} | {r['final_equity']:,.0f} |"
        )

    # 결과 동일 그룹 식별: cagr/mdd/거래수가 사실상 같으면 같은 그룹으로 본다
    df["_grp"] = (
        df["cagr"].round(6).astype(str) + "_" +
        df["mdd"].round(6).astype(str) + "_" +
        df["n_trades"].astype(str)
    )
    unique_groups = df.drop_duplicates(subset=["_grp"]).reset_index(drop=True)

    lines.append("\n## 유효 결과 상위 5 (중복 그룹 1건씩)\n")
    lines.append("같은 결과를 내는 X% 그룹은 1행으로 축약. 매일 매수 후보가 슬롯 한도(10개)보다")
    lines.append("훨씬 많아 이격률 정렬 후 동일 종목들이 진입했기 때문이며, 같은 MA에서")
    lines.append("X%만 다르면 결과가 같은 경우가 다수 관찰된다.\n")
    for i in range(min(5, len(unique_groups))):
        r = unique_groups.iloc[i]
        x_in_group = sorted(
            df.loc[df["_grp"] == r["_grp"], "pct"].apply(lambda p: int(p * 100)).tolist()
        )
        x_str = (
            f"X% ∈ {{{', '.join(map(str, x_in_group))}}}"
            if len(x_in_group) > 1
            else f"X% = {x_in_group[0]}"
        )
        calmar_s = f"{r['calmar']:.3f}" if pd.notna(r["calmar"]) else "—"
        lines.append(f"### {i + 1}. MA{int(r['ma_n'])} · {x_str}\n")
        lines.append(f"- CAGR: **{r['cagr']:.4%}**")
        lines.append(f"- 총수익률: {r['total_return']:.4%}")
        lines.append(f"- MDD: {r['mdd']:.4%}")
        lines.append(f"- Calmar (CAGR/|MDD|): {calmar_s}")
        lines.append(f"- Profit Factor: {r['profit_factor']:.3f}")
        lines.append(
            f"- 승률: {r['win_rate']:.3%} ({int(r['n_wins'])}/{int(r['n_trades'])})"
        )
        lines.append(f"- 거래당 평균 손익: {r['avg_pnl']:,.0f}원")
        lines.append(f"- 최종자산: {r['final_equity']:,.0f}원")
        lines.append("")

    # Calmar 상위 5
    df_calmar = df.sort_values("calmar", ascending=False).reset_index(drop=True)
    lines.append("## Calmar (CAGR/|MDD|) 상위 5\n")
    lines.append("| 순위 | MA | X% | CAGR | MDD | Calmar | 거래수 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i in range(min(5, len(df_calmar))):
        r = df_calmar.iloc[i]
        calmar_s = f"{r['calmar']:.3f}" if pd.notna(r["calmar"]) else "—"
        lines.append(
            f"| {i + 1} | {int(r['ma_n'])} | {int(r['pct'] * 100)} | "
            f"{r['cagr']:.4f} | {r['mdd']:.4f} | {calmar_s} | {int(r['n_trades']):,} |"
        )
    lines.append("")

    # 결론
    best_cagr = df.iloc[0]
    best_calmar = df_calmar.iloc[0]
    lines.append("## 결론\n")
    lines.append(
        f"- **CAGR 기준 최적**: MA{int(best_cagr['ma_n'])} ± {int(best_cagr['pct'] * 100)}% "
        f"(동일 CAGR 그룹 다수), CAGR {best_cagr['cagr']:.4%}, MDD {best_cagr['mdd']:.4%}, "
        f"Calmar {best_cagr['calmar']:.3f}."
    )
    lines.append(
        f"- **Calmar(위험조정) 기준 최적**: MA{int(best_calmar['ma_n'])} "
        f"± {int(best_calmar['pct'] * 100)}%, CAGR {best_calmar['cagr']:.4%}, "
        f"MDD {best_calmar['mdd']:.4%}, Calmar {best_calmar['calmar']:.3f}."
    )
    n_neg = int((df["cagr"] < 0).sum())
    lines.append(f"- 48 조합 중 CAGR 음수 조합 {n_neg}개 — 본 매매법은 파라미터 선택에 매우 민감.")
    lines.append(
        "- **X% 임계값 무효성**: 매일 매수 후보가 슬롯(10) 대비 훨씬 많아 "
        "이격률 정렬 후 상위 종목만 진입 → X% 변화에도 동일 종목 매수, 결과 동일. "
        "차별화하려면 슬롯 수 확대, 종목 필터(거래대금 등), 또는 X%를 진입 가격대 "
        "정렬 키로 활용하는 변형이 필요."
    )
    lines.append(
        f"- **타 전략 대비**: 본 백테스트 최고 CAGR {best_cagr['cagr']:.2%}는 "
        "MTL/MTH(26.31%), 3-bar(25.34%) 대비 크게 열위. MDD도 -47% ~ -82%로 광범위. "
        "Envelope 단독은 평균회귀 추세에서만 동작하며 한국 시장 6년 구간에선 매력 낮음."
    )
    lines.append(
        "- **개선 후보**: ① MA 상향 이력 후 매도(원래 PRD에서 검토했던 매도 룰), "
        "② 손절선 추가, ③ 거래대금/시총 상위 필터, "
        "④ 매수 신호 발생 후 N일 이내 미진입 시 폐기."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------

def main():
    import time
    start_date = pd.Timestamp("2020-01-01")
    print(f"DB 로드 시작 (워밍업 2019-07-01부터)...")
    t0 = time.time()
    df = load_data("2019-07-01")
    print(f"  로드 완료: {len(df):,}행, {df['ticker'].nunique():,}종목, {time.time() - t0:.1f}초")

    end_date = df["date"].max()
    print(f"  기간: {df['date'].min().date()} ~ {end_date.date()}")
    print(f"  백테스트 구간: {start_date.date()} ~ {end_date.date()}")

    grid_df = run_grid(df, start_date, end_date)

    # 결과 저장
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "backtest_envelope.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_report(grid_df, start_date, end_date, out_path)
    print(f"\n리포트 저장: {out_path}")

    # CSV도 함께
    csv_path = out_path.replace(".md", ".csv")
    grid_df.to_csv(csv_path, index=False)
    print(f"CSV 저장: {csv_path}")


def write_report_kodex_lev(
    grid_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    path: str,
    slot_capital: int,
    commission: float,
    sell_commission: float,
    tax: float,
) -> None:
    """KODEX 레버리지 단일 종목 백테스트 리포트."""
    years = (end_date - start_date).days / 365.25
    df = grid_df.copy()
    df["calmar"] = df.apply(
        lambda r: (r["cagr"] / abs(r["mdd"])) if r["mdd"] < 0 else float("nan"), axis=1
    )

    lines: list[str] = []
    lines.append("# Envelope 매매 KODEX 레버리지(122630) 단일종목 백테스트 결과\n")
    lines.append(f"- **기간**: {start_date.date()} ~ {end_date.date()} ({years:.2f}년)")
    lines.append("- **종목**: KODEX 레버리지 (티커 122630, KOSPI200 일일 2배 추종 ETF)")
    lines.append("- **데이터 소스**: FinanceDataReader (조정주가 기준)")
    lines.append(
        f"- **자금 운용**: 매 매수 시 고정 {slot_capital:,}원 풀투입, 수익은 별도 현금 적립"
    )
    lines.append(
        f"- **거래비용**: 매수 수수료 {commission:.4%}, 매도 수수료 {sell_commission:.4%}, "
        f"매도세 {tax:.4%} (ETF 면제)"
    )
    lines.append("- **매수**: close < MA × (1 − X%) → 익일 시가 매수")
    lines.append("- **매도**: 보유 중 close < MA → 익일 시가 매도 (MA 상향 제약 없음)")
    lines.append(f"- **그리드**: MA ∈ {MA_GRID}, X% ∈ {[int(p * 100) for p in PCT_GRID]}")
    lines.append("- **목적함수**: CAGR 최대화\n")

    lines.append("## 조합별 성과 (CAGR 내림차순)\n")
    lines.append("| 순위 | MA | X% | CAGR | MDD | Calmar | PF | 승률 | 거래수 | 최종자본 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i, r in df.iterrows():
        calmar_s = f"{r['calmar']:.3f}" if pd.notna(r["calmar"]) else "—"
        lines.append(
            f"| {i + 1} | {int(r['ma_n'])} | {int(r['pct'] * 100)} | "
            f"{r['cagr']:.4f} | {r['mdd']:.4f} | {calmar_s} | "
            f"{r['profit_factor']:.2f} | {r['win_rate']:.3f} | "
            f"{int(r['n_trades']):,} | {r['final_equity']:,.0f} |"
        )

    # 유효 결과 그룹
    df["_grp"] = (
        df["cagr"].round(6).astype(str) + "_" +
        df["mdd"].round(6).astype(str) + "_" +
        df["n_trades"].astype(str)
    )
    unique_groups = df.drop_duplicates(subset=["_grp"]).reset_index(drop=True)

    lines.append("\n## 유효 결과 상위 5 (중복 그룹 1건씩)\n")
    lines.append(
        "단일 종목이라 슬롯 한도 효과는 없으나, X%만 다른 조합이 동일 결과를 내는 경우"
    )
    lines.append("(같은 종목의 매수 후보 발생 시점이 동일할 때) 1행으로 축약.\n")
    for i in range(min(5, len(unique_groups))):
        r = unique_groups.iloc[i]
        x_in_group = sorted(
            df.loc[df["_grp"] == r["_grp"], "pct"].apply(lambda p: int(p * 100)).tolist()
        )
        x_str = (
            f"X% ∈ {{{', '.join(map(str, x_in_group))}}}"
            if len(x_in_group) > 1
            else f"X% = {x_in_group[0]}"
        )
        calmar_s = f"{r['calmar']:.3f}" if pd.notna(r["calmar"]) else "—"
        lines.append(f"### {i + 1}. MA{int(r['ma_n'])} · {x_str}\n")
        lines.append(f"- CAGR: **{r['cagr']:.4%}**")
        lines.append(f"- 총수익률: {r['total_return']:.4%}")
        lines.append(f"- MDD: {r['mdd']:.4%}")
        lines.append(f"- Calmar (CAGR/|MDD|): {calmar_s}")
        lines.append(f"- Profit Factor: {r['profit_factor']:.3f}")
        lines.append(
            f"- 승률: {r['win_rate']:.3%} ({int(r['n_wins'])}/{int(r['n_trades'])})"
        )
        lines.append(f"- 거래당 평균 손익: {r['avg_pnl']:,.0f}원")
        lines.append(f"- 최종자본: {r['final_equity']:,.0f}원")
        lines.append("")

    df_calmar = df.sort_values("calmar", ascending=False).reset_index(drop=True)
    lines.append("## Calmar (CAGR/|MDD|) 상위 5\n")
    lines.append("| 순위 | MA | X% | CAGR | MDD | Calmar | 거래수 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i in range(min(5, len(df_calmar))):
        r = df_calmar.iloc[i]
        calmar_s = f"{r['calmar']:.3f}" if pd.notna(r["calmar"]) else "—"
        lines.append(
            f"| {i + 1} | {int(r['ma_n'])} | {int(r['pct'] * 100)} | "
            f"{r['cagr']:.4f} | {r['mdd']:.4f} | {calmar_s} | {int(r['n_trades']):,} |"
        )
    lines.append("")

    # 거래 통계 표 (CAGR 내림차순) — 사용자 요청
    lines.append("## 거래 통계 표 (CAGR 내림차순)\n")
    lines.append(
        "각 조합의 매매 횟수·승/패 분해·승률·평균이익/평균손실·**손익비(평균이익÷|평균손실|)**·"
        "Profit Factor(총이익÷총손실)·CAGR을 정리. 거래수가 0건인 조합은 제외."
    )
    lines.append("")
    lines.append(
        "| 순위 | MA | X% | 거래수 | 승 | 패 | 승률 | 평균이익(원) | 평균손실(원) | 손익비 | PF | CAGR |"
    )
    lines.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for i, r in df.iterrows():
        if r["n_trades"] == 0:
            continue
        if pd.notna(r["payoff_ratio"]) and r["payoff_ratio"] != float("inf"):
            payoff_s = f"{r['payoff_ratio']:.3f}"
        elif r["payoff_ratio"] == float("inf"):
            payoff_s = "∞"
        else:
            payoff_s = "—"
        lines.append(
            f"| {i + 1} | {int(r['ma_n'])} | {int(r['pct'] * 100)} | "
            f"{int(r['n_trades']):,} | {int(r['n_wins'])} | {int(r['n_losses'])} | "
            f"{r['win_rate']:.3f} | {r['avg_win']:,.0f} | {r['avg_loss']:,.0f} | "
            f"{payoff_s} | {r['profit_factor']:.2f} | {r['cagr']:.4f} |"
        )
    lines.append("")

    best_cagr = df.iloc[0]
    best_calmar = df_calmar.iloc[0]
    n_neg = int((df["cagr"] < 0).sum())
    lines.append("## 결론\n")
    lines.append(
        f"- **CAGR 기준 최적**: MA{int(best_cagr['ma_n'])} ± {int(best_cagr['pct'] * 100)}%, "
        f"CAGR {best_cagr['cagr']:.4%}, MDD {best_cagr['mdd']:.4%}, "
        f"Calmar {best_cagr['calmar']:.3f}, 거래수 {int(best_cagr['n_trades']):,}, "
        f"최종자본 {best_cagr['final_equity']:,.0f}원."
    )
    lines.append(
        f"- **Calmar 기준 최적**: MA{int(best_calmar['ma_n'])} "
        f"± {int(best_calmar['pct'] * 100)}%, CAGR {best_calmar['cagr']:.4%}, "
        f"MDD {best_calmar['mdd']:.4%}, Calmar {best_calmar['calmar']:.3f}."
    )
    lines.append(f"- 48 조합 중 CAGR 음수 조합 {n_neg}개.")
    lines.append(
        "- **이전 KOSPI/KOSDAQ 사이클과의 비교**: 슬롯 한도가 1로 제거된 단일 종목"
        " 환경에서 X% 임계값이 실제로 차별화되는지 본 사이클의 핵심 검증 포인트."
    )
    lines.append(
        "- **레버리지 ETF 특성 고려**: 변동성 누적/감쇠(volatility decay)로 인해"
        " 평균회귀 매매가 단순 buy & hold 대비 우위/열위인지 결과 표에서 함께 판단."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main_kodex_lev():
    """KODEX 레버리지 단일종목 envelope 그리드 백테스트 진입점."""
    import time
    print("[KODEX 레버리지 122630] fdr 로드 시작...")
    t0 = time.time()
    df = load_kodex_lev("2019-07-01")
    print(f"  로드 완료: {len(df):,}행, {time.time() - t0:.1f}초")
    print(f"  기간: {df['date'].min().date()} ~ {df['date'].max().date()}")

    start_date = pd.Timestamp("2020-01-01")
    end_date = df["date"].max()
    print(f"  백테스트 구간: {start_date.date()} ~ {end_date.date()}")

    SLOT = 10_000_000
    grid_df = run_grid(
        df, start_date, end_date,
        slot_capital=SLOT,
        max_positions=1,
        commission=0.00015,
        sell_commission=0.00015,
        tax=0.0,  # ETF 매도세 면제
    )
    grid_df = grid_df.sort_values("cagr", ascending=False).reset_index(drop=True)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "backtest_envelope_kodex_lev.md")
    csv_path = os.path.join(out_dir, "backtest_envelope_kodex_lev.csv")

    write_report_kodex_lev(
        grid_df, start_date, end_date, md_path,
        slot_capital=SLOT,
        commission=0.00015,
        sell_commission=0.00015,
        tax=0.0,
    )
    grid_df.to_csv(csv_path, index=False)
    print(f"\n리포트 저장: {md_path}")
    print(f"CSV 저장: {csv_path}")


if __name__ == "__main__":
    main()
