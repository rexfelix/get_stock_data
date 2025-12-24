import pandas as pd
from sqlalchemy import create_engine

# 데이터베이스 연결 정보 (get_stocks.py와 동일)
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 데이터베이스 연결 정보 (get_stocks.py와 동일)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "stock_db")
DB_USER = os.getenv("DB_USER", "rexfelix")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")


def get_db_engine():
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(connection_string)
    return engine


def verify_data():
    engine = get_db_engine()

    try:
        # 1. Total row count
        query_count = "SELECT COUNT(*) FROM stocks;"
        count = pd.read_sql(query_count, engine).iloc[0, 0]
        print(f"Total rows in 'stocks' table: {count}")

        # 2. Unique tickers count
        query_tickers = "SELECT COUNT(DISTINCT ticker) FROM stocks;"
        ticker_count = pd.read_sql(query_tickers, engine).iloc[0, 0]
        print(f"Unique tickers in 'stocks' table: {ticker_count}")

        # 3. Sample data
        query_sample = "SELECT * FROM stocks LIMIT 5;"
        df_sample = pd.read_sql(query_sample, engine)
        print("\nSample data:")
        print(df_sample)

        # 4. Check for new columns and missing columns
        columns = df_sample.columns.tolist()
        if "name" in columns:
            print("\n[SUCCESS] 'name' column exists.")
        else:
            print("\n[FAIL] 'name' column missing.")

        if "trading_value" not in columns and "fluctuation_rate" not in columns:
            print("[SUCCESS] 'trading_value' and 'fluctuation_rate' columns removed.")
        else:
            print(f"[FAIL] Unwanted columns present: {columns}")

        # 4. Check for nulls (optional quick check)
        # query_nulls = "SELECT COUNT(*) FROM stocks WHERE close IS NULL;"
        # null_count = pd.read_sql(query_nulls, engine).iloc[0, 0]
        # print(f"Rows with null close price: {null_count}")

    except Exception as e:
        print(f"Error during verification: {e}")


if __name__ == "__main__":
    verify_data()
