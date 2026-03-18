import time
import requests
import pandas as pd
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


def get_last_trading_date(engine):
    """trading_details 테이블의 가장 최근 날짜 조회"""
    try:
        query = "SELECT MAX(date) FROM trading_details"
        last_date = pd.read_sql(query, engine).iloc[0, 0]
        if last_date:
            return pd.to_datetime(last_date)
        return None
    except Exception:
        return None


def get_tickers_from_db(engine):
    """stocks 테이블에서 유니크한 티커 목록 가져오기"""
    query = "SELECT DISTINCT ticker FROM stocks ORDER BY ticker"
    df = pd.read_sql(query, engine)
    return df["ticker"].tolist()


def parse_int(val):
    """부호/콤마 포함 문자열을 정수로 변환"""
    if not val:
        return None
    try:
        return int(val.replace("+", "").replace(",", ""))
    except (ValueError, TypeError):
        return None


def parse_float(val):
    """문자열을 실수로 변환"""
    if not val:
        return None
    try:
        return float(val.replace("+", ""))
    except (ValueError, TypeError):
        return None


# ============================================================
# ka10015: 일별거래상세요청 (거래대금)
# ============================================================

def fetch_ka10015(token, ticker, today_str, cutoff_date_str=None):
    """ka10015: 일별거래상세요청
    - strt_dt는 해당 날짜부터 과거 방향으로 내림차순 반환
    - cutoff_date_str 미만 데이터가 나오면 조기 중단
    """
    url = f"{KIWOOM_DOMAIN}/api/dostk/stkinfo"
    all_rows = []
    reached_cutoff = False

    req_headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "ka10015",
        "authorization": f"Bearer {token}",
    }
    body = {
        "stk_cd": ticker,
        "strt_dt": today_str,
    }

    while True:
        time.sleep(0.5)
        response = requests.post(url, json=body, headers=req_headers)
        data = response.json()

        if data.get("return_code") != 0:
            break

        rows = data.get("daly_trde_dtl", [])
        if not rows:
            break

        for row in rows:
            dt = row.get("dt", "")
            if cutoff_date_str and dt < cutoff_date_str:
                reached_cutoff = True
                break
            all_rows.append(row)

        if reached_cutoff:
            break

        cont_yn = response.headers.get("cont-yn", "N")
        if cont_yn != "Y":
            break

        req_headers["cont-yn"] = "Y"
        req_headers["next-key"] = response.headers.get("next-key", "")

    return all_rows


# ============================================================
# ka10045: 종목별기관매매추이요청 (기관/외인 순매수)
# ============================================================

def fetch_ka10045(token, ticker, start_dt, end_dt):
    """ka10045: 종목별기관매매추이요청
    - strt_dt ~ end_dt 범위의 기관/외인 일별순매수 반환
    """
    url = f"{KIWOOM_DOMAIN}/api/dostk/mrkcond"
    all_rows = []

    req_headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "ka10045",
        "authorization": f"Bearer {token}",
    }
    body = {
        "stk_cd": ticker,
        "strt_dt": start_dt,
        "end_dt": end_dt,
        "orgn_prsm_unp_tp": "1",
        "for_prsm_unp_tp": "1",
    }

    while True:
        time.sleep(0.5)
        response = requests.post(url, json=body, headers=req_headers)
        data = response.json()

        if data.get("return_code") != 0:
            break

        rows = data.get("stk_orgn_trde_trnsn", [])
        if not rows:
            break

        all_rows.extend(rows)

        cont_yn = response.headers.get("cont-yn", "N")
        if cont_yn != "Y":
            break

        req_headers["cont-yn"] = "Y"
        req_headers["next-key"] = response.headers.get("next-key", "")

    return all_rows


# ============================================================
# 두 API 결과를 결합
# ============================================================

def build_dataframe(ticker, ka10015_rows, ka10045_rows):
    """ka10015(거래대금) + ka10045(기관/외인 순매수) 데이터를 결합"""
    # ka10015 → {date: {trading_value, volume, close}}
    trade_map = {}
    for row in ka10015_rows:
        dt = row.get("dt", "")
        qty = row.get("trde_qty", "0")
        prica = row.get("trde_prica", "0")
        if qty == "0" and prica == "0":
            continue
        trade_map[dt] = {
            "close": abs(parse_int(row.get("close_pric")) or 0) or None,
            "volume": parse_int(qty),
            "trading_value": parse_int(prica),
        }

    # ka10045 → {date: {inst_net, foreign_net, foreign_ratio}}
    investor_map = {}
    for row in ka10045_rows:
        dt = row.get("dt", "")
        investor_map[dt] = {
            "inst_net": parse_int(row.get("orgn_daly_nettrde_qty")),
            "foreign_net": parse_int(row.get("for_daly_nettrde_qty")),
            "foreign_ratio": parse_float(row.get("limit_exh_rt")),
        }

    # 합치기 (ka10015 기준, ka10045로 보강)
    all_dates = set(trade_map.keys()) | set(investor_map.keys())
    if not all_dates:
        return None

    records = []
    for dt in sorted(all_dates, reverse=True):
        record = {"ticker": ticker, "date": dt}
        trade = trade_map.get(dt, {})
        inv = investor_map.get(dt, {})

        record["close"] = trade.get("close")
        record["volume"] = trade.get("volume")
        record["trading_value"] = trade.get("trading_value")
        record["inst_net"] = inv.get("inst_net")
        record["foreign_net"] = inv.get("foreign_net")
        record["foreign_ratio"] = inv.get("foreign_ratio")

        # 거래대금이 없고 투자자 데이터도 없으면 건너뛰기
        if record["trading_value"] is None and record["inst_net"] is None:
            continue

        records.append(record)

    if not records:
        return None

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def delete_data_from_date(engine, from_date):
    """특정 날짜 이후의 데이터를 삭제 (해당 날짜 포함)"""
    try:
        date_str = from_date.strftime("%Y-%m-%d")
        with engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM trading_details WHERE date >= :target_date"),
                {"target_date": date_str},
            )
            deleted = result.rowcount
            print(f"  [trading_details] {date_str} 이후 기존 데이터 {deleted}건 삭제")
    except Exception as e:
        print(f"  삭제 중 오류 (테이블 미존재 가능): {e}")


def save_batch_to_db(data_list, engine):
    """DataFrame 리스트를 병합하여 DB에 저장"""
    try:
        combined_df = pd.concat(data_list, ignore_index=True)
        combined_df.to_sql("trading_details", engine, if_exists="append", index=False)
        return len(combined_df)
    except Exception as e:
        print(f"\nError saving batch: {e}")
        return 0


def create_table_if_not_exists(engine):
    """trading_details 테이블이 없으면 생성 (기존 테이블이 있으면 재생성)"""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS trading_details"))
        conn.execute(text("""
            CREATE TABLE trading_details (
                ticker VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                close BIGINT,
                volume BIGINT,
                trading_value BIGINT,
                foreign_net BIGINT,
                inst_net BIGINT,
                foreign_ratio FLOAT,
                PRIMARY KEY (ticker, date)
            )
        """))


def ensure_table(engine):
    """테이블 존재 여부 확인, 없으면 생성"""
    try:
        pd.read_sql("SELECT 1 FROM trading_details LIMIT 1", engine)
    except Exception:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE trading_details (
                    ticker VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    close BIGINT,
                    volume BIGINT,
                    trading_value BIGINT,
                    foreign_net BIGINT,
                    inst_net BIGINT,
                    foreign_ratio FLOAT,
                    PRIMARY KEY (ticker, date)
                )
            """))
            print("trading_details 테이블 생성 완료")


def main():
    print("=== 일별거래상세 수집 시작 (ka10015 + ka10045) ===")

    engine = get_db_engine()

    # 1. 테이블 확인/생성
    ensure_table(engine)

    # 2. 마지막 업데이트 날짜 확인
    last_date = get_last_trading_date(engine)
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    today_date = today.date()

    cutoff_date_str = None  # ka10015 조기 중단용
    start_dt = None         # ka10045 시작일

    if last_date is None:
        # 최초 실행
        default_start = (today - timedelta(days=60)).strftime("%Y%m%d")
        user_input = input(
            f"최초 실행입니다. 수집 시작 날짜를 입력하세요 (YYYYMMDD, 기본값: {default_start}): "
        ).strip()
        cutoff_date_str = user_input if user_input else default_start
        start_dt = cutoff_date_str
        print(f"수집 기간: {cutoff_date_str} ~ {today_str}")
    else:
        print(f"Last trading_details date in DB: {last_date.strftime('%Y-%m-%d')}")

        if last_date.date() >= today_date:
            print(f"\n[REFRESH] 오늘({today_date}) 데이터가 이미 존재 → 삭제 후 재수집")
            delete_data_from_date(engine, datetime.combine(today_date, datetime.min.time()))
            cutoff_date_str = today_str
            start_dt = today_str
        else:
            next_date = last_date + timedelta(days=1)
            if next_date.date() > today_date:
                print("Trading details data is already up to date.")
                return
            cutoff_date_str = next_date.strftime("%Y%m%d")
            start_dt = cutoff_date_str

        print(f"수집 기간: {cutoff_date_str} ~ {today_str}")

    # 3. 종목 목록
    tickers = get_tickers_from_db(engine)
    print(f"대상 종목 수: {len(tickers)}")

    if not tickers:
        print("[ERROR] stocks 테이블에서 종목을 찾을 수 없습니다. get_stocks.py를 먼저 실행하세요.")
        return

    # 4. 토큰 발급
    token = get_kiwoom_token()

    # 5. 수집 시작
    start_time = time.time()
    BATCH_SIZE = 50
    buffer = []
    total_saved = 0
    errors = 0

    for ticker in tqdm(tickers, desc="일별거래상세 수집", unit="stock"):
        try:
            # ka10015: 거래대금 (strt_dt=오늘, cutoff로 조기 중단)
            ka15_rows = fetch_ka10015(token, ticker, today_str, cutoff_date_str)

            # ka10045: 기관/외인 순매수 (날짜 범위 지정)
            ka45_rows = fetch_ka10045(token, ticker, start_dt, today_str)

            df = build_dataframe(ticker, ka15_rows, ka45_rows)
            if df is not None and not df.empty:
                buffer.append(df)

            if len(buffer) >= BATCH_SIZE:
                total_saved += save_batch_to_db(buffer, engine)
                buffer = []

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  [{ticker}] 오류: {e}")
            elif errors == 6:
                print("\n  (이후 오류는 생략합니다)")

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
