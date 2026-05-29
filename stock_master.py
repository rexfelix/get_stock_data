"""상장 종목 마스터 테이블(stock_master) 관리.

키움 ka10099(종목리스트)와 비교하여:
  - 신규 편입 종목 → INSERT (is_listed=TRUE)
  - 탈락(상장폐지) 종목 → 삭제하지 않고 is_listed=FALSE + delisted_date 기록 (보존)
  - 기존 상장 종목 → name/market/last_seen 갱신, is_listed=TRUE 복구

이 테이블이 일일 데이터 수집의 종목 유니버스(단일 source of truth)가 되며,
다음 작업(OHLCV/거래대금/재무 수집)은 get_listed_tickers()로 현재 상장 종목만 대상으로 한다.
"""

import os
import time
from datetime import date

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 환경변수 로드
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "stock_db")
DB_USER = os.getenv("DB_USER", "rexfelix")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

KIWOOM_APPKEY = os.getenv("KIWOOM_APPKEY", "")
KIWOOM_SECRETKEY = os.getenv("KIWOOM_SECRETKEY", "")
KIWOOM_DOMAIN = "https://api.kiwoom.com"

# ka10099 mrkt_tp → 시장 라벨
_MARKETS = {"KOSPI": "0", "KOSDAQ": "10"}


def get_db_engine():
    """PostgreSQL 연결 엔진 생성."""
    return create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


# ============================================================
# 테이블 관리
# ============================================================


def ensure_table(engine):
    """stock_master 테이블이 없으면 생성."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stock_master (
                ticker        VARCHAR(20) PRIMARY KEY,
                name          VARCHAR(100) NOT NULL,
                market        VARCHAR(20),
                is_listed     BOOLEAN NOT NULL DEFAULT TRUE,
                first_seen    DATE,
                last_seen     DATE,
                delisted_date DATE,
                updated_at    TIMESTAMP DEFAULT now()
            )
        """))


def is_empty(engine):
    """마스터 테이블이 비어있는지 확인."""
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM stock_master")).scalar() == 0


def seed_from_stocks_if_empty(engine):
    """마스터가 비어있으면 기존 stocks 테이블로 초기 시딩.

    시딩 시 모든 종목을 is_listed=TRUE 로 넣고, 이후 sync()가
    ka10099와 비교하여 현재 상장되지 않은 종목을 상폐 처리한다.
    name 은 가장 최근 날짜의 값을, first_seen/last_seen 은 가격이력의 MIN/MAX 를 사용.
    """
    if not is_empty(engine):
        return 0
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO stock_master (ticker, name, is_listed, first_seen, last_seen, updated_at)
            SELECT ticker, name, TRUE, first_dt, last_dt, now()
            FROM (
                SELECT ticker,
                       (array_agg(name ORDER BY date DESC))[1] AS name,
                       MIN(date) AS first_dt,
                       MAX(date) AS last_dt
                FROM stocks
                GROUP BY ticker
            ) t
            ON CONFLICT (ticker) DO NOTHING
        """))
        n = result.rowcount
    print(f"  [stock_master] 초기 시딩: stocks 에서 {n}개 종목 적재 (이후 API로 재조정)")
    return n


# ============================================================
# 키움 ka10099 종목 리스트 조회
# ============================================================


def get_kiwoom_token():
    """키움 REST API OAuth2 토큰 발급 (au10001)."""
    url = f"{KIWOOM_DOMAIN}/oauth2/token"
    headers = {"Content-Type": "application/json;charset=UTF-8", "api-id": "au10001"}
    body = {
        "grant_type": "client_credentials",
        "appkey": KIWOOM_APPKEY,
        "secretkey": KIWOOM_SECRETKEY,
    }
    response = requests.post(url, json=body, headers=headers)
    data = response.json()
    if data.get("return_code") != 0:
        raise Exception(f"토큰 발급 실패: {data.get('return_msg')}")
    return data["token"]


def get_stock_list_kiwoom(token, mrkt_tp):
    """ka10099 종목 리스트 조회 → [(code, name), ...].

    mrkt_tp: "0"=KOSPI, "10"=KOSDAQ. 페이지네이션은 cont-yn/next-key로 처리.
    """
    url = f"{KIWOOM_DOMAIN}/api/dostk/stkinfo"
    result = []
    req_headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "ka10099",
        "authorization": f"Bearer {token}",
    }
    body = {"mrkt_tp": mrkt_tp}

    while True:
        time.sleep(0.5)
        response = requests.post(url, json=body, headers=req_headers)
        data = response.json()
        if data.get("return_code") != 0:
            break
        stocks = data.get("list", [])
        if not stocks:
            break
        for s in stocks:
            code = s.get("code", "")
            name = s.get("name", "")
            if code and name:
                result.append((code, name))
        if response.headers.get("cont-yn", "N") != "Y":
            break
        req_headers["cont-yn"] = "Y"
        req_headers["next-key"] = response.headers.get("next-key", "")

    return result


def fetch_market_lists(token):
    """KOSPI/KOSDAQ을 병렬로 조회 → {"KOSPI": [(code,name),...], "KOSDAQ": [...]}."""
    out = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(get_stock_list_kiwoom, token, code): label
            for label, code in _MARKETS.items()
        }
        for fut in futures:
            out[futures[fut]] = fut.result()
    return out


# ============================================================
# 동기화
# ============================================================


def sync(engine, market_map, today=None):
    """ka10099 결과(market_map)와 마스터를 비교하여 변경 처리.

    market_map: {"KOSPI": [(code, name), ...], "KOSDAQ": [...]}
    반환: {"new": 신규편입, "delisted": 이번에상폐처리, "listed": 현재상장총수}
    """
    if today is None:
        today = date.today()

    # API 결과 평탄화 — 동일 ticker 가 여러 시장에 중복되면 마지막 값 사용
    api = {}  # ticker -> (name, market)
    for market, lst in market_map.items():
        for code, name in lst:
            api[code] = (name, market)
    api_tickers = list(api.keys())

    new_count = 0
    with engine.begin() as conn:
        existing = {
            r[0]
            for r in conn.execute(text("SELECT ticker FROM stock_master")).fetchall()
        }

        for ticker, (name, market) in api.items():
            if ticker in existing:
                conn.execute(text("""
                    UPDATE stock_master
                       SET name = :name,
                           market = :market,
                           last_seen = :today,
                           is_listed = TRUE,
                           delisted_date = NULL,
                           updated_at = now()
                     WHERE ticker = :ticker
                """), {"name": name, "market": market, "today": today, "ticker": ticker})
            else:
                conn.execute(text("""
                    INSERT INTO stock_master
                        (ticker, name, market, is_listed, first_seen, last_seen, updated_at)
                    VALUES (:ticker, :name, :market, TRUE, :today, :today, now())
                """), {"ticker": ticker, "name": name, "market": market, "today": today})
                new_count += 1

        # 탈락(상폐) 처리 — API에 없는데 현재 상장으로 표시된 종목. 삭제하지 않고 플래그만 변경.
        del_result = conn.execute(text("""
            UPDATE stock_master
               SET is_listed = FALSE,
                   delisted_date = :today,
                   updated_at = now()
             WHERE is_listed = TRUE
               AND NOT (ticker = ANY(:api_tickers))
        """), {"today": today, "api_tickers": api_tickers})
        del_count = del_result.rowcount

        listed = conn.execute(
            text("SELECT COUNT(*) FROM stock_master WHERE is_listed = TRUE")
        ).scalar()

    return {"new": new_count, "delisted": del_count, "listed": listed}


# ============================================================
# 소비처용 헬퍼
# ============================================================


def get_listed_tickers(engine):
    """현재 상장 종목 (ticker, name) 리스트 반환.

    stock_master 가 없거나 비어있으면 기존 stocks 테이블로 fallback.
    """
    try:
        df = pd.read_sql(
            "SELECT ticker, name FROM stock_master WHERE is_listed = TRUE ORDER BY ticker",
            engine,
        )
        if len(df) > 0:
            return list(zip(df["ticker"], df["name"]))
    except Exception:
        pass
    # fallback: stocks 테이블
    df = pd.read_sql(
        "SELECT DISTINCT ON (ticker) ticker, name FROM stocks ORDER BY ticker, date DESC",
        engine,
    )
    return list(zip(df["ticker"], df["name"]))


def resolve_universe(engine, verbose=True):
    """일일 수집 진입점: 테이블 보장 → 시딩 → 키움 동기화 → 현재 상장 종목 반환.

    키움 API 실패/0건 시 기존 마스터(또는 stocks)로 안전하게 fallback.
    """
    ensure_table(engine)
    seed_from_stocks_if_empty(engine)
    try:
        token = get_kiwoom_token()
        market_map = fetch_market_lists(token)
        total = sum(len(v) for v in market_map.values())
        if total > 0:
            summary = sync(engine, market_map)
            if verbose:
                print(
                    f"  [stock_master] 동기화 완료 — 신규편입 {summary['new']}건 / "
                    f"상폐처리 {summary['delisted']}건 / 현재상장 {summary['listed']}건"
                )
        elif verbose:
            print("  [stock_master] 키움 API 0건 반환 → 기존 마스터 종목으로 진행")
    except Exception as e:
        if verbose:
            print(f"  [stock_master] 키움 동기화 실패: {e} → 기존 마스터 종목으로 진행")
    return get_listed_tickers(engine)


if __name__ == "__main__":
    # 단독 실행: 마스터 테이블을 키움 ka10099 기준으로 동기화만 수행
    eng = get_db_engine()
    tickers = resolve_universe(eng)
    print(f"현재 상장 종목 수: {len(tickers)}")
