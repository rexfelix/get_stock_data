"""
네이버 증권(navercomp) Financial Summary 크롤링
- 소스: navercomp.wisereport.co.kr/v2/company/cF1002.aspx
- 수집 항목: 매출액, 영업이익, 당기순이익, YoY, EPS, PER, PBR, ROE 등
- 연간 데이터 (실적 + 컨센서스)
- 전 종목 대상
- UPSERT 방식 (ON CONFLICT)
"""

import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB = os.getenv("DB_NAME")
USER = os.getenv("DB_USER")
PW = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
ENGINE = create_engine(f"postgresql://{USER}:{PW}@{HOST}:{PORT}/{DB}")

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

COLUMNS = [
    "ticker", "year", "is_estimate",
    "revenue", "revenue_yoy", "operating_income",
    "net_income", "eps", "per", "pbr",
    "roe", "ev_ebitda", "net_debt_ratio",
    "accounting_std",
    "pcr", "bps", "dps", "dividend_yield",
]


def get_all_tickers() -> pd.DataFrame:
    """현재 상장 종목(stock_master)만 대상. 없으면 stocks 로 fallback."""
    import stock_master
    rows = stock_master.get_listed_tickers(ENGINE)
    return pd.DataFrame(rows, columns=["ticker", "name"])


def ensure_table():
    with ENGINE.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS financial_summary (
                ticker VARCHAR(20) NOT NULL,
                year VARCHAR(10) NOT NULL,
                is_estimate BOOLEAN,
                revenue FLOAT,
                revenue_yoy FLOAT,
                operating_income FLOAT,
                net_income FLOAT,
                eps FLOAT,
                per FLOAT,
                pbr FLOAT,
                roe FLOAT,
                ev_ebitda FLOAT,
                net_debt_ratio FLOAT,
                accounting_std VARCHAR(20),
                pcr FLOAT,
                bps FLOAT,
                dps FLOAT,
                dividend_yield FLOAT,
                PRIMARY KEY (ticker, year)
            )
        """))
        # 기존 테이블에 새 컬럼이 없으면 추가
        for col in ["pcr", "bps", "dps", "dividend_yield"]:
            conn.execute(text(
                f"ALTER TABLE financial_summary ADD COLUMN IF NOT EXISTS {col} FLOAT"
            ))
        # 기존 테이블에 PK가 없으면 추가
        has_pk = conn.execute(text("""
            SELECT 1 FROM information_schema.table_constraints
            WHERE table_name = 'financial_summary' AND constraint_type = 'PRIMARY KEY'
        """)).fetchone()
        if not has_pk:
            # 중복 행 제거 후 PK 추가
            conn.execute(text("""
                DELETE FROM financial_summary a USING financial_summary b
                WHERE a.ctid < b.ctid AND a.ticker = b.ticker AND a.year = b.year
            """))
            conn.execute(text(
                "ALTER TABLE financial_summary ADD PRIMARY KEY (ticker, year)"
            ))


def fetch_financial(ticker: str) -> list[dict]:
    url = f"https://navercomp.wisereport.co.kr/v2/company/cF1002.aspx?cmp_cd={ticker}&finGubun=MAIN"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200 or not r.text.strip():
            return []
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    tbl = soup.find("table")
    if not tbl:
        return []

    rows = tbl.find_all("tr")
    if len(rows) < 3:
        return []

    results = []
    for row in rows[2:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if len(cells) < 11:
            continue

        year_str = cells[0]
        if not year_str:
            continue

        is_estimate = "(E)" in year_str
        year = year_str.replace("(A)", "").replace("(E)", "").strip()

        def parse_num(s):
            if not s or s in ("N/A", "", "-"):
                return None
            try:
                return float(s.replace(",", ""))
            except ValueError:
                return None

        results.append({
            "ticker": ticker,
            "year": year,
            "is_estimate": is_estimate,
            "revenue": parse_num(cells[1]),
            "revenue_yoy": parse_num(cells[2]),
            "operating_income": parse_num(cells[3]),
            "net_income": parse_num(cells[4]),
            "eps": parse_num(cells[5]),
            "per": parse_num(cells[6]),
            "pbr": parse_num(cells[7]),
            "roe": parse_num(cells[8]),
            "ev_ebitda": parse_num(cells[9]),
            "net_debt_ratio": parse_num(cells[10]),
            "accounting_std": cells[11] if len(cells) > 11 else None,
        })

    return results


def save_batch(records: list[dict]):
    if not records:
        return

    df = pd.DataFrame(records)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS]

    with ENGINE.connect() as conn:
        for _, row in df.iterrows():
            vals = row.to_dict()
            set_clause = ", ".join(
                f"{k} = :{k}" for k in vals if k not in ("ticker", "year")
            )
            conn.execute(text(f"""
                INSERT INTO financial_summary ({', '.join(vals.keys())})
                VALUES ({', '.join(':' + k for k in vals.keys())})
                ON CONFLICT (ticker, year)
                DO UPDATE SET {set_clause}
            """), vals)
        conn.commit()


def main():
    ensure_table()

    tickers = get_all_tickers()
    print(f"전체 종목 수: {len(tickers)}")

    all_data = []
    errors = []
    start = time.time()
    BATCH_SIZE = 50

    for i, row in tickers.iterrows():
        ticker = row["ticker"]
        name = row["name"]

        data = fetch_financial(ticker)
        if data:
            all_data.extend(data)
        else:
            errors.append(ticker)

        # 배치 저장
        if len(all_data) >= BATCH_SIZE * 5:
            save_batch(all_data)
            all_data = []

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(tickers) - i - 1) / rate
            print(f"  [{i+1}/{len(tickers)}] {ticker} {name} | {rate:.1f}종목/초 | 남은시간: {remaining/60:.1f}분")

        time.sleep(0.15)

    # 마지막 배치 저장
    if all_data:
        save_batch(all_data)

    elapsed = time.time() - start
    print(f"\n수집 완료")
    print(f"  실패: {len(errors)}건")
    print(f"  소요시간: {elapsed/60:.1f}분")

    # 결과 확인
    with ENGINE.connect() as conn:
        r = conn.execute(text("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM financial_summary"))
        cnt, tk_cnt = r.fetchone()
        print(f"\nDB 확인: {cnt}건, {tk_cnt}종목")


if __name__ == "__main__":
    main()
