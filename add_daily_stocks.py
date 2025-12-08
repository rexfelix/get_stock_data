import time
import random
import pandas as pd
from pykrx import stock
from sqlalchemy import create_engine
import multiprocessing
from datetime import datetime, timedelta
from tqdm import tqdm

# 데이터베이스 연결 정보
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 데이터베이스 연결 정보
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
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

        # pykrx를 사용하여 일봉 데이터 가져오기
        df = stock.get_market_ohlcv(start_date, end_date, ticker)

        # 데이터가 없는 경우 처리
        if df.empty:
            return None

        # 인덱스(날짜) 이름 설정 및 초기화
        df.index.name = "date"
        df = df.reset_index()

        # 컬럼 이름 변경 (한글 -> 영문)
        rename_map = {
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
            "거래대금": "trading_value",
            "등락률": "fluctuation_rate",
        }
        df = df.rename(columns=rename_map)

        # 불필요한 컬럼 제거
        cols_to_drop = ["trading_value", "fluctuation_rate"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # 티커 및 종목명 컬럼 추가
        df["ticker"] = ticker
        df["name"] = name

        # 컬럼 이름 소문자로 변경
        df.columns = [c.lower() for c in df.columns]

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


def main():
    engine = get_db_engine()

    # 1. 마지막 업데이트 날짜 확인
    last_date = get_last_update_date(engine)

    if last_date is None:
        print("No existing data found in DB. Please run get_stocks.py first.")
        return

    print(f"Last data date in DB: {last_date.strftime('%Y-%m-%d')}")

    # 2. 수집 시작 날짜 = 마지막 날짜 + 1일
    start_date_obj = last_date + timedelta(days=1)

    # 종료 날짜 = 오늘
    end_date_obj = datetime.now()

    if start_date_obj > end_date_obj:
        print("Stock data is already up to date.")
        return

    start_date_str = start_date_obj.strftime("%Y%m%d")
    end_date_str = end_date_obj.strftime("%Y%m%d")

    print(f"New data range: {start_date_str} ~ {end_date_str}")

    # 3. 대상 종목 가져오기
    print("Fetching ticker list and names...")
    kospi_tickers = stock.get_market_ticker_list(end_date_str, market="KOSPI")
    kospi_list = [(t, stock.get_market_ticker_name(t)) for t in kospi_tickers]

    kosdaq_tickers = stock.get_market_ticker_list(end_date_str, market="KOSDAQ")
    kosdaq_list = [(t, stock.get_market_ticker_name(t)) for t in kosdaq_tickers]

    all_ticker_names = kospi_list + kosdaq_list

    # 4. 실행
    start_time = time.time()
    fetch_and_save_data(all_ticker_names, start_date_str, end_date_str)
    end_time = time.time()

    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
