import time
import random
import pandas as pd
import FinanceDataReader as fdr
from sqlalchemy import create_engine, text
import multiprocessing
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


def get_db_engine():
    """PostgreSQL 연결 엔진 생성"""
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(connection_string)
    return engine


def get_last_update_date(engine):
    """DB에서 가장 최근 데이터의 날짜 조회"""
    try:
        query = "SELECT MAX(date) FROM stocks"
        last_date = pd.read_sql(query, engine).iloc[0, 0]
        if last_date:
            return pd.to_datetime(last_date)
        return None
    except Exception as e:
        print(f"Error checking last date: {e}")
        return None


def get_stock_data_wrapper(args):
    """multiprocessing.imap을 위한 래퍼 함수"""
    return get_stock_data(*args)


def get_stock_data(ticker, name, start_date, end_date):
    """특정 종목의 OHLCV 데이터 수집"""
    try:
        # 요청 과부하 방지를 위한 랜덤 지연
        time.sleep(random.uniform(0.1, 0.5))

        # FinanceDataReader를 사용하여 일봉 데이터 가져오기
        df = fdr.DataReader(ticker, start_date, end_date)

        # 데이터가 없는 경우 처리
        if df.empty:
            return None

        # 인덱스(날짜) 이름 설정 및 초기화
        df.index.name = "date"
        df = df.reset_index()

        # 컬럼 이름 변경 (FDR: Open, High, Low, Close, Volume)
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)

        # 필요한 컬럼만 선택
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in required_cols if c in df.columns]]

        # 티커 및 종목명 컬럼 추가
        df["ticker"] = ticker
        df["name"] = name

        return df
    except Exception as e:
        return None


def fetch_and_save_data(ticker_name_list, start_date, end_date):
    """멀티프로세싱 + Progress Bar + Batch Save"""
    print(f"Total tickers to fetch: {len(ticker_name_list)}")
    print(f"Fetching period: {start_date} ~ {end_date}")

    # 프로세스 풀 생성 (과부하 방지를 위해 4개로 제한)
    pool_size = 4
    print(f"Starting multiprocessing with {pool_size} processes...")

    # 인자 준비
    args = [(ticker, name, start_date, end_date) for ticker, name in ticker_name_list]

    # DB Engine
    engine = get_db_engine()

    # 배치 설정
    BATCH_SIZE = 50
    buffer = []

    with multiprocessing.Pool(processes=pool_size) as pool:
        # imap을 사용하여 순서대로 결과 반환 + tqdm으로 진행률 표시
        for result in tqdm(
            pool.imap(get_stock_data_wrapper, args), total=len(args), unit="stock"
        ):
            if result is not None:
                buffer.append(result)

            # 버퍼가 꽉 차면 저장 (항상 append 모드)
            if len(buffer) >= BATCH_SIZE:
                save_batch_to_db(buffer, engine)
                buffer = []

        # 남은 데이터 저장
        if buffer:
            save_batch_to_db(buffer, engine)


def save_batch_to_db(data_list, engine):
    """데이터프레임 리스트를 병합하여 DB에 저장 (Append Only)"""
    try:
        combined_df = pd.concat(data_list, ignore_index=True)
        # 이미 데이터가 있으므로 무조건 append
        combined_df.to_sql("stocks", engine, if_exists="append", index=False)
    except Exception as e:
        print(f"\nError saving batch: {e}")


def delete_data_from_date(engine, table_name, from_date):
    """특정 날짜 이후의 데이터를 삭제 (해당 날짜 포함)"""
    try:
        date_str = from_date.strftime("%Y-%m-%d")
        with engine.begin() as conn:
            result = conn.execute(
                text(f"DELETE FROM {table_name} WHERE date >= :target_date"),
                {"target_date": date_str},
            )
            deleted = result.rowcount
            print(f"  [{table_name}] {date_str} 이후 기존 데이터 {deleted}건 삭제")
    except Exception as e:
        print(f"  [{table_name}] 삭제 중 오류 (테이블 미존재 가능): {e}")


def get_tickers_from_db(engine):
    """DB에서 유니크한 티커와 종목명 가져오기 (Fallback)"""
    try:
        query = "SELECT DISTINCT ticker, name FROM stocks"
        df = pd.read_sql(query, engine)
        return list(zip(df["ticker"], df["name"]))
    except Exception as e:
        print(f"Error fetching tickers from DB: {e}")
        return []


# ============================================================
# 시장 지수 (KOSPI / KOSDAQ) 수집
# ============================================================

INDEX_SYMBOLS = {
    "KS11": "KOSPI",
    "KQ11": "KOSDAQ",
}


def get_last_index_update_date(engine):
    """market_indices 테이블에서 가장 최근 날짜 조회"""
    try:
        query = "SELECT MAX(date) FROM market_indices"
        last_date = pd.read_sql(query, engine).iloc[0, 0]
        if last_date:
            return pd.to_datetime(last_date)
        return None
    except Exception:
        # 테이블이 아직 없는 경우
        return None


def fetch_index_data(symbol, name, start_date, end_date):
    """단일 시장 지수의 OHLCV 데이터 수집"""
    try:
        df = fdr.DataReader(symbol, start_date, end_date)
        if df.empty:
            return None

        df.index.name = "date"
        df = df.reset_index()

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)

        required_cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in required_cols if c in df.columns]]

        df["symbol"] = symbol
        df["name"] = name

        return df
    except Exception as e:
        print(f"Error fetching index {symbol}: {e}")
        return None


def fetch_and_save_indices(start_date, end_date, engine):
    """모든 시장 지수를 수집하여 market_indices 테이블에 저장"""
    print("\n" + "=" * 50)
    print("시장 지수 데이터 수집 시작 (KOSPI / KOSDAQ)")
    print("=" * 50)

    all_dfs = []
    for symbol, name in INDEX_SYMBOLS.items():
        print(f"  Fetching {name} ({symbol})...")
        df = fetch_index_data(symbol, name, start_date, end_date)
        if df is not None:
            all_dfs.append(df)
            print(f"    -> {len(df)} rows")
        else:
            print(f"    -> No data")

    if not all_dfs:
        print("지수 데이터가 없습니다.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    try:
        combined_df.to_sql("market_indices", engine, if_exists="append", index=False)
        print(f"market_indices 테이블에 {len(combined_df)}건 저장 완료")
    except Exception as e:
        print(f"Error saving index data: {e}")


def main():
    engine = get_db_engine()

    # ========================================
    # Part 1: 개별 종목 데이터 수집
    # ========================================

    # 1. 마지막 업데이트 날짜 확인
    last_date = get_last_update_date(engine)

    if last_date is None:
        print("No existing data found in DB. Please run get_stocks.py first.")
        return

    print(f"Last stock data date in DB: {last_date.strftime('%Y-%m-%d')}")

    # 종료 날짜 = 오늘
    end_date_obj = datetime.now()
    end_date_str = end_date_obj.strftime("%Y%m%d")
    today_date = end_date_obj.date()

    # 2. 수집 시작 날짜 결정
    #    - 마지막 데이터가 오늘이면: 오늘 데이터 삭제 후 오늘부터 재수집 (장중 갱신 대응)
    #    - 그 외: 마지막 날짜 + 1일부터 수집
    if last_date.date() >= today_date:
        start_date_obj = datetime.combine(today_date, datetime.min.time())
        print(f"\n[REFRESH] 오늘({today_date}) 데이터가 이미 존재 → 삭제 후 재수집")
        delete_data_from_date(engine, "stocks", start_date_obj)
    else:
        start_date_obj = last_date + timedelta(days=1)

    if start_date_obj.date() > today_date:
        print("Stock data is already up to date.")
    else:
        start_date_str = start_date_obj.strftime("%Y%m%d")
        print(f"New stock data range: {start_date_str} ~ {end_date_str}")

        # 3. 대상 종목 가져오기 (Hybrid Mechanism)
        print("Fetching ticker list and names via FinanceDataReader(KRX)...")

        all_ticker_names = []

        try:
            df_krx = fdr.StockListing("KRX")
            # KOSPI, KOSDAQ 만 필터링
            df_filtered = df_krx[df_krx["Market"].isin(["KOSPI", "KOSDAQ"])]
            all_ticker_names = list(zip(df_filtered["Code"], df_filtered["Name"]))
            print(f"Tickers found via FDR: {len(all_ticker_names)}")

        except Exception as e:
            print(f"FDR Ticker Fetch Error: {e}")
            # 실패 시 빈 리스트 유지 -> Fallback으로 넘어감

        # Fallback: FDR이 0개를 반환하면 DB에서 가져옴
        if len(all_ticker_names) == 0:
            print(
                "\n[WARNING] FDR returned 0 tickers (or failed). Switching to Fallback Mode (DB)..."
            )
            all_ticker_names = get_tickers_from_db(engine)
            print(f"Tickers found via DB Fallback: {len(all_ticker_names)}")

        if len(all_ticker_names) == 0:
            print("\n[ERROR] No tickers found even from DB. Aborting stock fetch.")
        else:
            # 4. 실행
            start_time = time.time()
            fetch_and_save_data(all_ticker_names, start_date_str, end_date_str)
            end_time = time.time()
            print(f"Stock data execution time: {end_time - start_time:.2f} seconds")

    # ========================================
    # Part 2: 시장 지수 데이터 수집
    # ========================================

    last_index_date = get_last_index_update_date(engine)

    if last_index_date is not None:
        print(f"\nLast index data date in DB: {last_index_date.strftime('%Y-%m-%d')}")
        if last_index_date.date() >= today_date:
            index_start_obj = datetime.combine(today_date, datetime.min.time())
            print(
                f"[REFRESH] 오늘({today_date}) 지수 데이터가 이미 존재 → 삭제 후 재수집"
            )
            delete_data_from_date(engine, "market_indices", index_start_obj)
        else:
            index_start_obj = last_index_date + timedelta(days=1)
    else:
        # 지수 테이블이 없거나 비어있으면 stocks 테이블의 최초 날짜부터 수집
        index_start_obj = last_date - timedelta(days=365)  # 넉넉히 1년 전부터
        print(
            "\nNo existing index data. Collecting from 1 year before last stock date."
        )

    if index_start_obj.date() > today_date:
        print("Index data is already up to date.")
    else:
        index_start_str = index_start_obj.strftime("%Y%m%d")
        start_time = time.time()
        fetch_and_save_indices(index_start_str, end_date_str, engine)
        end_time = time.time()
        print(f"Index data execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
