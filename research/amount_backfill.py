"""stocks 테이블 amount 컬럼 백필 (2019~현재).

키움 REST API ka10081 (주식일봉차트) 의 trde_prica 를 stocks.amount 에 저장.
- 단위: 백만원 (stock_all 일관성)
- 페이지네이션: base_dt 기준 600일/회, 과거 방향
- 진행 상황 저장: progress.json (resume 가능)
- rate limit: 0.3s/호출
"""
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import text

# .env 두 곳 모두 시도 (data_center 의 키움 키 + research 의 DB)
load_dotenv("/Volumes/SSD/project/py/invest/data_center/.env")
load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env", override=False)

import backtest_top3_indicators as bt  # noqa: E402

KIWOOM_APPKEY = os.getenv("KIWOOM_APPKEY", "")
KIWOOM_SECRETKEY = os.getenv("KIWOOM_SECRETKEY", "")
KIWOOM_DOMAIN = "https://api.kiwoom.com"

START_DATE = "2019-01-02"
PROGRESS_FILE = "/Volumes/SSD/project/py/invest/data_center/research/report/amount_backfill/progress.json"
LOG_FILE = "/Volumes/SSD/project/py/invest/data_center/research/report/amount_backfill/backfill.log"


# ──────────────────────────────────────────────
# 헬퍼 (단위 테스트 대상)
# ──────────────────────────────────────────────
def parse_amount(raw_value) -> int | None:
    """ka10081 trde_prica 파싱 — 콤마/부호 제거, 정수 변환. 실패 시 None."""
    if raw_value is None or raw_value == "":
        return None
    try:
        s = str(raw_value).replace("+", "").replace(",", "")
        return int(s)
    except (ValueError, TypeError):
        return None


def build_amount_records(api_rows: list[dict], ticker: str
                         ) -> list[tuple[str, pd.Timestamp, int | None]]:
    """ka10081 응답 rows 를 (ticker, Timestamp, amount) 리스트로.

    - dt 필드가 비어있으면 skip
    - amount 누락 시 None
    """
    records = []
    for row in api_rows:
        dt_str = row.get("dt", "")
        if not dt_str:
            continue
        try:
            dt = pd.Timestamp(f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}")
        except Exception:
            continue
        amount = parse_amount(row.get("trde_prica"))
        records.append((ticker, dt, amount))
    return records


def next_base_dt(oldest_dt: str) -> str:
    """페이지네이션 다음 기준일 = oldest_dt - 1 day. YYYYMMDD."""
    dt = pd.Timestamp(f"{oldest_dt[:4]}-{oldest_dt[4:6]}-{oldest_dt[6:8]}")
    prev = dt - pd.Timedelta(days=1)
    return prev.strftime("%Y%m%d")


def load_progress(path: str) -> dict:
    """진행 상황 로드. 파일 없으면 빈 progress."""
    if not os.path.exists(path):
        return {"completed": [], "failed": []}
    try:
        with open(path) as f:
            data = json.load(f)
        if "completed" not in data:
            data["completed"] = []
        if "failed" not in data:
            data["failed"] = []
        return data
    except Exception:
        return {"completed": [], "failed": []}


def save_progress(path: str, progress: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def is_done(progress: dict, ticker: str) -> bool:
    return ticker in progress.get("completed", []) or ticker in progress.get("failed", [])


# ──────────────────────────────────────────────
# 키움 API
# ──────────────────────────────────────────────
def get_kiwoom_token() -> str:
    if not KIWOOM_APPKEY or not KIWOOM_SECRETKEY:
        raise RuntimeError("KIWOOM_APPKEY / KIWOOM_SECRETKEY 미설정 (data_center/.env)")
    url = f"{KIWOOM_DOMAIN}/oauth2/token"
    headers = {"Content-Type": "application/json;charset=UTF-8", "api-id": "au10001"}
    body = {
        "grant_type": "client_credentials",
        "appkey": KIWOOM_APPKEY,
        "secretkey": KIWOOM_SECRETKEY,
    }
    r = requests.post(url, json=body, headers=headers, timeout=10)
    data = r.json()
    if data.get("return_code") != 0:
        raise RuntimeError(f"키움 토큰 발급 실패: {data.get('return_msg')}")
    return data["token"]


def fetch_ka10081(token: str, ticker: str, base_dt: str) -> list[dict]:
    """ka10081 주식일봉차트조회. base_dt 기준 과거 600일치."""
    url = f"{KIWOOM_DOMAIN}/api/dostk/chart"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "ka10081",
        "authorization": f"Bearer {token}",
    }
    body = {"stk_cd": ticker, "base_dt": base_dt, "upd_stkpc_tp": "1"}
    time.sleep(0.3)
    try:
        r = requests.post(url, json=body, headers=headers, timeout=15)
        data = r.json()
    except Exception as e:
        print(f"  API 실패 {ticker} base={base_dt}: {e}")
        return []
    if data.get("return_code") != 0:
        return []
    return data.get("stk_dt_pole_chart_qry", [])


# ──────────────────────────────────────────────
# DB 스키마 변경
# ──────────────────────────────────────────────
def ensure_schema() -> None:
    """amount 컬럼 + ticker_date 인덱스 생성 (idempotent)."""
    with bt.ENGINE.begin() as conn:
        conn.execute(text("ALTER TABLE stocks ADD COLUMN IF NOT EXISTS amount BIGINT"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_stocks_ticker_date ON stocks(ticker, date)"
        ))
    print("✅ stocks.amount 컬럼 + idx_stocks_ticker_date 인덱스 준비 완료")


# ──────────────────────────────────────────────
# 종목별 백필
# ──────────────────────────────────────────────
def backfill_one_ticker(token: str, ticker: str, start_date: str = START_DATE,
                        end_date: str = None) -> tuple[int, int]:
    """한 종목 amount 백필. (n_rows_saved, n_api_calls) 반환."""
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    base_dt = end_date.replace("-", "")
    start_dt_str = start_date.replace("-", "")
    n_calls = 0
    all_records = []
    seen_dates = set()

    while base_dt >= start_dt_str:
        rows = fetch_ka10081(token, ticker, base_dt)
        n_calls += 1
        if not rows:
            break

        records = build_amount_records(rows, ticker)
        # 신규 기록만
        new_recs = [r for r in records if r[1] not in seen_dates]
        if not new_recs:
            break
        all_records.extend(new_recs)
        seen_dates.update(r[1] for r in new_recs)

        # 가장 오래된 날짜
        oldest = min(r[1] for r in new_recs)
        if oldest <= pd.Timestamp(start_date):
            break
        base_dt = next_base_dt(oldest.strftime("%Y%m%d"))

    if not all_records:
        return 0, n_calls

    # ka10081 trde_prica 단위 = 백만원 (Smoke test 005930 으로 확정)
    # → 그대로 저장 (stock_all 일관성)
    df = pd.DataFrame(all_records, columns=["ticker", "date", "amount"])
    df["amount_million"] = df["amount"]  # 변환 없이 그대로
    # start_date 이후 데이터만
    df = df[df["date"] >= pd.Timestamp(start_date)]

    if df.empty:
        return 0, n_calls

    # DB UPDATE: 일괄 (psycopg execute_values 같은 효율적 방식)
    n_saved = bulk_update_amount(df)
    return n_saved, n_calls


def bulk_update_amount(df: pd.DataFrame) -> int:
    """DataFrame (ticker, date, amount_million) 으로 stocks.amount 일괄 업데이트."""
    if df.empty:
        return 0

    n_total = 0
    chunk_size = 500
    with bt.ENGINE.begin() as conn:
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            for _, row in chunk.iterrows():
                amt = row["amount_million"]
                if amt is None or pd.isna(amt):
                    continue
                conn.execute(
                    text("UPDATE stocks SET amount = :a WHERE ticker = :t AND date = :d"),
                    {"a": int(amt), "t": row["ticker"], "d": row["date"]},
                )
                n_total += 1
    return n_total


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def get_all_tickers() -> list[str]:
    with bt.ENGINE.connect() as conn:
        r = conn.execute(text("SELECT DISTINCT ticker FROM stocks ORDER BY ticker")).fetchall()
    return [row[0] for row in r]


def log_line(msg: str) -> None:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)


def main(limit: int = None, single_ticker: str = None):
    """전체 또는 일부 종목 백필.

    - limit: None=전체, 정수=상위 N 종목
    - single_ticker: 단일 종목 (smoke test 용)
    """
    print("=" * 60)
    print("stocks.amount 백필 — 키움 ka10081")
    print("=" * 60)

    print("[1] DB 스키마 준비...")
    ensure_schema()

    print("[2] 키움 토큰 발급...")
    token = get_kiwoom_token()
    print(f"    ✅ 토큰 발급 완료")

    print("[3] 진행 상황 로드...")
    progress = load_progress(PROGRESS_FILE)
    print(f"    완료: {len(progress['completed'])}, 실패: {len(progress['failed'])}")

    if single_ticker:
        tickers = [single_ticker]
    else:
        tickers = get_all_tickers()
        if limit:
            tickers = tickers[:limit]
    print(f"[4] 대상: {len(tickers)} 종목")

    t_start = time.time()
    success_n = 0
    fail_n = 0
    for i, ticker in enumerate(tickers, 1):
        if is_done(progress, ticker) and not single_ticker:
            continue
        try:
            n_saved, n_calls = backfill_one_ticker(token, ticker)
            if n_saved > 0:
                progress["completed"].append(ticker)
                success_n += 1
                log_line(f"  [{i:4}/{len(tickers)}] {ticker} ✅ {n_saved}행 ({n_calls}호출)")
            else:
                progress["failed"].append(ticker)
                fail_n += 1
                log_line(f"  [{i:4}/{len(tickers)}] {ticker} ❌ 데이터 0 ({n_calls}호출)")
        except Exception as e:
            progress["failed"].append(ticker)
            fail_n += 1
            log_line(f"  [{i:4}/{len(tickers)}] {ticker} 💥 {type(e).__name__}: {e}")

        # 10 종목마다 progress 저장
        if i % 10 == 0:
            save_progress(PROGRESS_FILE, progress)

    save_progress(PROGRESS_FILE, progress)
    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    log_line(f"완료: 성공 {success_n} / 실패 {fail_n} / {elapsed/60:.1f}분")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", help="단일 종목 (smoke test)")
    p.add_argument("--limit", type=int, help="처음 N 종목만")
    args = p.parse_args()
    main(limit=args.limit, single_ticker=args.ticker)
