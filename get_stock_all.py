import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from tqdm import tqdm
import os
from dotenv import load_dotenv

# 환경변수 로드
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(env_path)

# 데이터베이스 연결 정보
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "stock_db")
DB_USER = os.getenv("DB_USER", "rexfelix")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# 키움 API 설정
KIWOOM_APPKEY = os.getenv("KIWOOM_APPKEY", "")
KIWOOM_SECRETKEY = os.getenv("KIWOOM_SECRETKEY", "")
KIWOOM_DOMAIN = "https://api.kiwoom.com"


def get_db_engine():
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)


def get_kiwoom_token():
    url = f"{KIWOOM_DOMAIN}/oauth2/token"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "au10001",
    }
    body = {
        "grant_type": "client_credentials",
        "appkey": KIWOOM_APPKEY,
        "secretkey": KIWOOM_SECRETKEY,
    }
    response = requests.post(url, json=body, headers=headers)
    data = response.json()
    if data.get("return_code") != 0:
        raise Exception(f"토큰 발급 실패: {data.get('return_msg')}")
    print(f"토큰 발급 성공 (만료: {data.get('expires_dt')})")
    return data["token"]


def get_tickers_from_db(engine):
    """stocks 테이블에서 유니크한 (ticker, name) 목록 가져오기"""
    query = """
        SELECT DISTINCT ON (ticker) ticker, name
        FROM stocks
        ORDER BY ticker, date DESC
    """
    df = pd.read_sql(query, engine)
    return list(zip(df["ticker"], df["name"]))


def parse_int(val):
    """부호/콤마 포함 문자열을 정수로 변환"""
    if not val:
        return None
    try:
        return int(val.replace("+", "").replace(",", ""))
    except (ValueError, TypeError):
        return None


# ============================================================
# ka10081: 주식일봉차트조회요청 (OHLCV + 거래대금)
# ============================================================

def fetch_ka10081(token, ticker, base_dt):
    """ka10081: 주식일봉차트조회요청
    - base_dt 기준으로 과거 방향 600일치 데이터 반환 (1회 호출)
    - 수정주가 적용 (upd_stkpc_tp=1)
    """
    url = f"{KIWOOM_DOMAIN}/api/dostk/chart"

    req_headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "ka10081",
        "authorization": f"Bearer {token}",
    }
    body = {
        "stk_cd": ticker,
        "base_dt": base_dt,
        "upd_stkpc_tp": "1",
    }

    time.sleep(0.3)
    response = requests.post(url, json=body, headers=req_headers, timeout=15)
    data = response.json()

    if data.get("return_code") != 0:
        return []

    return data.get("stk_dt_pole_chart_qry", [])


# ============================================================
# ka10060: 종목별투자자기관별차트요청 (투자자별 순매수)
# ============================================================

def fetch_ka10060(token, ticker, dt):
    """ka10060: 종목별투자자기관별차트요청
    - dt 기준으로 과거 방향 600일치 투자자별 순매수 데이터 반환 (1회 호출)
    - amt_qty_tp=2 (수량), trde_tp=0 (순매수), unit_tp=1 (단주)
    """
    url = f"{KIWOOM_DOMAIN}/api/dostk/chart"

    req_headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "ka10060",
        "authorization": f"Bearer {token}",
    }
    body = {
        "dt": dt,
        "stk_cd": ticker,
        "amt_qty_tp": "2",
        "trde_tp": "0",
        "unit_tp": "1",
    }

    time.sleep(0.3)
    response = requests.post(url, json=body, headers=req_headers, timeout=15)
    data = response.json()

    if data.get("return_code") != 0:
        return []

    return data.get("stk_invsr_orgn_chart", [])


# ============================================================
# 두 API 결과를 결합
# ============================================================

def build_dataframe(ticker, name, ka10081_rows, ka10060_rows):
    """ka10081(일봉 OHLCV) + ka10060(투자자별 순매수) 데이터를 결합"""
    # ka10081 → {date: {open, high, low, close, volume, amount}}
    ohlcv_map = {}
    for row in ka10081_rows:
        dt = row.get("dt", "")
        if not dt:
            continue
        ohlcv_map[dt] = {
            "open": abs(parse_int(row.get("open_pric")) or 0) or None,
            "high": abs(parse_int(row.get("high_pric")) or 0) or None,
            "low": abs(parse_int(row.get("low_pric")) or 0) or None,
            "close": abs(parse_int(row.get("cur_prc")) or 0) or None,
            "volume": parse_int(row.get("trde_qty")),
            "amount": parse_int(row.get("trde_prica")),
        }

    # ka10060 → {date: {외국인, 개인, 기관계}}
    investor_map = {}
    for row in ka10060_rows:
        dt = row.get("dt", "")
        if not dt:
            continue
        investor_map[dt] = {
            "외국인": parse_int(row.get("frgnr_invsr")),
            "개인": parse_int(row.get("ind_invsr")),
            "기관계": parse_int(row.get("orgn")),
        }

    # 합치기 (ka10081 기준, ka10060로 보강)
    all_dates = set(ohlcv_map.keys()) | set(investor_map.keys())
    if not all_dates:
        return None

    records = []
    for dt in sorted(all_dates, reverse=True):
        ohlcv = ohlcv_map.get(dt, {})
        inv = investor_map.get(dt, {})

        # OHLCV 데이터가 없는 날짜는 건너뛰기
        if not ohlcv:
            continue

        records.append({
            "ticker": ticker,
            "name": name,
            "date": dt,
            "open": ohlcv.get("open"),
            "high": ohlcv.get("high"),
            "low": ohlcv.get("low"),
            "close": ohlcv.get("close"),
            "amount": ohlcv.get("amount"),
            "volume": ohlcv.get("volume"),
            "외국인": inv.get("외국인"),
            "개인": inv.get("개인"),
            "기관계": inv.get("기관계"),
        })

    if not records:
        return None

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def save_batch_to_db(data_list, engine):
    """DataFrame 리스트를 병합하여 DB에 저장"""
    try:
        combined_df = pd.concat(data_list, ignore_index=True)
        combined_df.to_sql("stock_all", engine, if_exists="append", index=False)
        return len(combined_df)
    except Exception as e:
        print(f"\nError saving batch: {e}")
        return 0


def ensure_table(engine):
    """stock_all 테이블이 없으면 생성"""
    try:
        pd.read_sql("SELECT 1 FROM stock_all LIMIT 1", engine)
    except Exception:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE stock_all (
                    ticker VARCHAR(20) NOT NULL,
                    name VARCHAR(100),
                    date DATE NOT NULL,
                    open BIGINT,
                    high BIGINT,
                    low BIGINT,
                    close BIGINT,
                    amount BIGINT,
                    volume BIGINT,
                    "외국인" BIGINT,
                    "개인" BIGINT,
                    "기관계" BIGINT,
                    PRIMARY KEY (ticker, date)
                )
            """))
        print("stock_all 테이블 생성 완료")


def get_last_date(engine):
    """stock_all 테이블의 가장 최근 날짜 조회"""
    try:
        query = "SELECT MAX(date) FROM stock_all"
        last_date = pd.read_sql(query, engine).iloc[0, 0]
        if last_date:
            return pd.to_datetime(last_date)
        return None
    except Exception:
        return None


def delete_data_from_date(engine, from_date):
    """특정 날짜 이후의 데이터를 삭제 (해당 날짜 포함)"""
    try:
        date_str = from_date.strftime("%Y-%m-%d")
        with engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM stock_all WHERE date >= :target_date"),
                {"target_date": date_str},
            )
            deleted = result.rowcount
            print(f"  [stock_all] {date_str} 이후 기존 데이터 {deleted}건 삭제")
    except Exception as e:
        print(f"  삭제 중 오류: {e}")


def main():
    print("=== stock_all 수집 시작 (ka10081 + ka10060) ===")
    print("  ka10081: 주식일봉차트 (OHLCV + 거래대금)")
    print("  ka10060: 종목별투자자기관별차트 (외국인/개인/기관계)")

    engine = get_db_engine()

    # 1. 테이블 확인/생성
    ensure_table(engine)

    # 2. 마지막 업데이트 날짜 확인
    last_date = get_last_date(engine)
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    today_date = today.date()

    cutoff_date_str = None  # ka10081 조기 중단용

    if last_date is None:
        # 최초 실행: 전체 600일치 수집
        print("최초 실행: 전체 600일치 수집")
    else:
        print(f"Last stock_all date in DB: {last_date.strftime('%Y-%m-%d')}")

        if last_date.date() >= today_date:
            # 오늘 데이터가 이미 존재 → 삭제 후 재수집
            print(f"\n[REFRESH] 오늘({today_date}) 데이터가 이미 존재 → 삭제 후 재수집")
            delete_data_from_date(engine, datetime.combine(today_date, datetime.min.time()))
            cutoff_date_str = today_str
        else:
            next_date = last_date + timedelta(days=1)
            if next_date.date() > today_date:
                print("stock_all 데이터가 이미 최신 상태입니다.")
                return
            cutoff_date_str = next_date.strftime("%Y%m%d")

        print(f"수집 기간: {cutoff_date_str} ~ {today_str}")

    # 3. 종목 목록
    ticker_names = get_tickers_from_db(engine)
    print(f"대상 종목 수: {len(ticker_names)}")

    if not ticker_names:
        print("[ERROR] stocks 테이블에서 종목을 찾을 수 없습니다. get_stocks.py를 먼저 실행하세요.")
        return

    # 4. 토큰 발급
    token = get_kiwoom_token()

    # 5. 수집 시작 (ThreadPoolExecutor 병렬)
    start_time = time.time()
    BATCH_SIZE = 50
    MAX_WORKERS = 3  # 키움 rate limit 안전 마진 (add_daily_stocks.py와 동일)
    buffer = []
    total_saved = 0
    errors = 0

    cutoff_dt = (
        pd.to_datetime(cutoff_date_str, format="%Y%m%d") if cutoff_date_str else None
    )

    def process_one(item):
        """종목 1개의 ka10081 + ka10060 을 수집해 DataFrame 반환 (워커 스레드)."""
        ticker, name = item
        try:
            ka81_rows = fetch_ka10081(token, ticker, today_str)
            ka60_rows = fetch_ka10060(token, ticker, today_str)
            df = build_dataframe(ticker, name, ka81_rows, ka60_rows)
            if df is not None and not df.empty and cutoff_dt is not None:
                df = df[df["date"] >= cutoff_dt]
            return ("ok", ticker, df)
        except Exception as e:
            return ("error", ticker, e)

    # 워커 스레드는 fetch만 수행하고, DB 저장은 메인 스레드에서만 처리 (동시 쓰기 방지)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for status, ticker, payload in tqdm(
            executor.map(process_one, ticker_names),
            total=len(ticker_names),
            desc="stock_all 수집",
            unit="stock",
        ):
            if status == "error":
                errors += 1
                if errors <= 5:
                    print(f"\n  [{ticker}] 오류: {payload}")
                elif errors == 6:
                    print("\n  (이후 오류는 생략합니다)")
                continue

            df = payload
            if df is not None and not df.empty:
                buffer.append(df)

            if len(buffer) >= BATCH_SIZE:
                total_saved += save_batch_to_db(buffer, engine)
                buffer = []

    # 남은 데이터 저장
    if buffer:
        total_saved += save_batch_to_db(buffer, engine)

    elapsed = time.time() - start_time
    print(f"\n=== 완료 ===")
    print(f"  저장된 행: {total_saved:,}")
    print(f"  오류 종목: {errors}")
    print(f"  소요 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
