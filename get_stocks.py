import time
import random
import pandas as pd
import FinanceDataReader as fdr
from sqlalchemy import create_engine
import multiprocessing
from datetime import datetime
from tqdm import tqdm
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

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


def get_stock_data_wrapper(args):
    """multiprocessing.imap을 위한 래퍼 함수"""
    return get_stock_data(*args)


def get_stock_data(ticker, name, start_date, end_date):
    """특정 종목의 OHLCV 데이터 수집"""
    try:
        # 요청 과부하 방지를 위한 랜덤 지연
        time.sleep(random.uniform(0.1, 0.5))

        # FinanceDataReader를 사용하여 일봉 데이터 가져오기
        # FDR은 날짜 형식을 'YYYY-MM-DD' 또는 'YYYYMMDD' 모두 지원
        df = fdr.DataReader(ticker, start_date, end_date)

        # 데이터가 없는 경우 처리
        if df.empty:
            return None

        # 인덱스(날짜) 이름 설정 및 초기화
        df.index.name = "date"
        df = df.reset_index()

        # 컬럼 이름 변경 (FDR: Open, High, Low, Close, Volume, Change 등) -> DB 스키마에 맞춤
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
        # print(f"Error fetching {ticker}: {e}")
        return None


def fetch_and_save_data(ticker_name_list, start_date, end_date):
    """멀티프로세싱 + Progress Bar + Batch Save"""
    print(f"Total tickers to fetch: {len(ticker_name_list)}")

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
    is_first_batch = True

    with multiprocessing.Pool(processes=pool_size) as pool:
        # imap을 사용하여 순서대로 결과 반환 + tqdm으로 진행률 표시
        for result in tqdm(
            pool.imap(get_stock_data_wrapper, args), total=len(args), unit="stock"
        ):
            if result is not None:
                buffer.append(result)

            # 버퍼가 꽉 차면 저장
            if len(buffer) >= BATCH_SIZE:
                save_batch_to_db(buffer, engine, is_first_batch)
                buffer = []  # 버퍼 초기화
                is_first_batch = False  # 첫 배치가 끝났으므로 이후부터는 append

        # 남은 데이터 저장
        if buffer:
            save_batch_to_db(buffer, engine, is_first_batch)


def save_batch_to_db(data_list, engine, is_first_batch):
    """데이터프레임 리스트를 병합하여 DB에 저장"""
    try:
        combined_df = pd.concat(data_list, ignore_index=True)

        # 첫 번째 배치면 'replace' (테이블 초기화), 그 이후는 'append' (데이터 추가)
        if_exists_mode = "replace" if is_first_batch else "append"

        combined_df.to_sql("stocks", engine, if_exists=if_exists_mode, index=False)
        # print(f"Saved {len(combined_df)} rows to DB.")
    except Exception as e:
        print(f"\nError saving batch: {e}")


def main():
    # 1. 대상 종목 가져오기
    date_str = datetime.now().strftime("%Y%m%d")
    print("Fetching ticker list and names via FinanceDataReader(KRX)...")

    # FinanceDataReader는 통합 리스트 제공 (KRX = KOSPI + KOSDAQ + KONEX)
    # 필요한 경우 시장별로 필터링 할 수 있으나, 기존 코드 흐름상 전체를 가져옴.
    # 단, KONEX 제외 여부는 유저 요구사항에 없었으나, 기존 로직이 kospi+kosdaq이었으므로
    # KRX 전체에서 Market 컬럼을 보고 필터링하는 것이 안전함.

    try:
        df_krx = fdr.StockListing("KRX")

        # KOSPI, KOSDAQ 만 필터링
        df_filtered = df_krx[df_krx["Market"].isin(["KOSPI", "KOSDAQ"])]

        # (Code, Name) 튜플 리스트 생성
        all_ticker_names = list(zip(df_filtered["Code"], df_filtered["Name"]))
        print(f"Total KOSPI+KOSDAQ tickers: {len(all_ticker_names)}")

    except Exception as e:
        print(f"Error fetching ticker list: {e}")
        return

    # 테스트용 (주석 처리)
    # all_ticker_names = all_ticker_names[:30]

    # 2. 수집 기간 설정
    # 사용자가 시작 날짜를 입력하도록 변경 (기본값: 20190101)
    default_start_date = "20190101"
    user_input = input(
        f"수집 시작 날짜를 입력하세요 (YYYYMMDD, 기본값: {default_start_date}): "
    ).strip()
    start_date = user_input if user_input else default_start_date

    # 사용자가 종료 날짜를 입력하도록 변경 (기본값: 오늘)
    default_end_date = date_str
    user_input_end = input(
        f"수집 종료 날짜를 입력하세요 (YYYYMMDD, 기본값: {default_end_date}): "
    ).strip()
    end_date = user_input_end if user_input_end else default_end_date

    print(f"Data collection period: {start_date} ~ {end_date}")

    # 3. 실행
    start_time = time.time()
    fetch_and_save_data(all_ticker_names, start_date, end_date)
    end_time = time.time()

    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
