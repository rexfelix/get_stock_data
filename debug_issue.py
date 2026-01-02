import pandas as pd
from pykrx import stock
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(env_path)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "stock_db")
DB_USER = os.getenv("DB_USER", "rexfelix")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")


def get_db_engine():
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)


def check_system():
    # 1. Check DB Max Date
    try:
        engine = get_db_engine()
        query = "SELECT MAX(date) FROM stocks"
        last_date = pd.read_sql(query, engine).iloc[0, 0]
        print(f"Last Date in DB: {last_date}")
    except Exception as e:
        print(f"DB Error: {e}")

    # 2. Check Pykrx Data for 20260102
    try:
        ticker = "005930"  # Samsung Electronics
        start = "20260102"
        end = "20260102"
        print(f"Fetching data for {ticker} from {start} to {end}...")
        df = stock.get_market_ohlcv(start, end, ticker)
        if df.empty:
            print("Pykrx returned empty DataFrame.")
        else:
            print("Pykrx returned data:")
            print(df)

        # Check if 20260102 is considered a holiday?
        # Maybe check market ticker list for that day
        tickers = stock.get_market_ticker_list(end, market="KOSPI")
        print(f"Number of KOSPI tickers on {end}: {len(tickers)}")

        # Check previous dates
        for d in ["20260101", "20251230", "20240102"]:
            t = stock.get_market_ticker_list(d, market="KOSPI")
            print(f"Number of KOSPI tickers on {d}: {len(t)}")

    except Exception as e:
        print(f"Pykrx Error: {e}")


def check_db_tickers():
    print("\nChecking DB tickers...")
    try:
        engine = get_db_engine()
        query = "SELECT DISTINCT ticker, name FROM stocks"
        df = pd.read_sql(query, engine)
        print(f"Found {len(df)} unique tickers in DB.")
        print(df.head())
    except Exception as e:
        print(f"DB Ticker Error: {e}")


if __name__ == "__main__":
    check_system()
    check_db_tickers()
