import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

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


def main():
    engine = get_db_engine()

    # 1. summary 컬럼 추가 (이미 있으면 무시)
    with engine.connect() as conn:
        conn.execute(text(
            "ALTER TABLE themes ADD COLUMN IF NOT EXISTS summary TEXT DEFAULT ''"
        ))
        conn.commit()
    print("summary 컬럼 추가 완료")

    # 2. CSV 로드
    csv_path = os.path.join(os.path.dirname(__file__), "korean_stock_company_summaries.csv")
    df_csv = pd.read_csv(csv_path, dtype={"ticker": str})
    df_csv["ticker"] = df_csv["ticker"].str.strip()
    df_csv["summary"] = df_csv["summary"].fillna("").str.strip()
    print(f"CSV 로드: {len(df_csv)}개 종목")

    # 3. ticker 기준으로 summary 업데이트
    updated = 0
    with engine.connect() as conn:
        for _, row in df_csv.iterrows():
            result = conn.execute(
                text("UPDATE themes SET summary = :summary WHERE ticker = :ticker"),
                {"summary": row["summary"], "ticker": row["ticker"]}
            )
            updated += result.rowcount
        conn.commit()

    print(f"summary 업데이트 완료: {updated}개 종목")

    # 4. 결과 확인
    df_check = pd.read_sql("SELECT count(*) as total, count(nullif(summary, '')) as has_summary FROM themes", engine)
    print(f"themes 테이블: 전체 {df_check['total'].iloc[0]}개, summary 보유 {df_check['has_summary'].iloc[0]}개")


if __name__ == "__main__":
    main()
