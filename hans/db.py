"""
hans 프로젝트 독립 DB 연결 모듈
================================
data_center 의 backtest_crash 의존성을 끊고, 이 폴더만으로 동작하도록
PostgreSQL ENGINE 을 자체 구성한다.

.env 탐색 순서:
  1) 환경변수 HANS_ENV_PATH
  2) 이 파일의 상위(data_center/.env)
  3) find_dotenv() (cwd 기준 상향 탐색)
기본값: postgresql://rexfelix:@localhost:5432/stock_db
"""

import os

from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_env():
    cand = os.getenv("HANS_ENV_PATH")
    if cand and os.path.exists(cand):
        load_dotenv(cand)
        return cand
    parent_env = os.path.join(os.path.dirname(_HERE), ".env")  # data_center/.env
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
        return parent_env
    found = find_dotenv()
    if found:
        load_dotenv(found)
        return found
    return None


_ENV_PATH = _load_env()

DB_URL = "postgresql://{}:{}@{}:{}/{}".format(
    os.getenv("DB_USER", "rexfelix"),
    os.getenv("DB_PASSWORD", ""),
    os.getenv("DB_HOST", "localhost"),
    os.getenv("DB_PORT", "5432"),
    os.getenv("DB_NAME", "stock_db"),
)

ENGINE = create_engine(DB_URL)


if __name__ == "__main__":
    from sqlalchemy import text
    print(f".env: {_ENV_PATH}")
    with ENGINE.connect() as c:
        n = c.execute(text("SELECT COUNT(*) FROM kospi200_members")).scalar()
        print(f"kospi200_members rows: {n}")
        rng = c.execute(text("SELECT MIN(date), MAX(date) FROM stocks")).fetchone()
        print(f"stocks date range: {rng[0]} ~ {rng[1]}")
