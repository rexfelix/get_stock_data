"""
네이버 증권(navercomp) Financial Summary 크롤링
- 소스: navercomp.wisereport.co.kr/v2/company/cF1002.aspx
- 수집 항목: 매출액, 영업이익, 당기순이익, YoY, EPS, PER, PBR, ROE 등
- 연간 데이터 (실적 + 컨센서스)
- 전 종목 대상
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


def get_all_tickers() -> pd.DataFrame:
    """DB에서 전 종목 ticker/name 조회"""
    query = text("SELECT DISTINCT ticker, name FROM stocks ORDER BY ticker")
    with ENGINE.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def fetch_financial(ticker: str) -> list[dict]:
    """navercomp에서 종목의 Financial Summary 크롤링"""
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
    for row in rows[2:]:  # skip header rows
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if len(cells) < 11:
            continue

        year_str = cells[0]  # e.g. "2023(A)", "2026(E)"
        if not year_str:
            continue

        # 연도와 실적/추정 구분 파싱
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


def main():
    tickers = get_all_tickers()
    print(f"전체 종목 수: {len(tickers)}")

    all_data = []
    errors = []
    start = time.time()

    for i, row in tickers.iterrows():
        ticker = row["ticker"]
        name = row["name"]

        data = fetch_financial(ticker)
        if data:
            all_data.extend(data)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(tickers) - i - 1) / rate
            print(f"  [{i+1}/{len(tickers)}] {ticker} {name} | 수집: {len(all_data)}건 | {rate:.1f}종목/초 | 남은시간: {remaining/60:.1f}분")

        if not data:
            errors.append(ticker)

        # rate limit 방지
        time.sleep(0.15)

    print(f"\n수집 완료: {len(all_data)}건, 실패: {len(errors)}건")
    print(f"소요시간: {(time.time()-start)/60:.1f}분")

    if not all_data:
        print("수집된 데이터 없음")
        return

    df = pd.DataFrame(all_data)
    print(f"\nDataFrame: {df.shape}")
    print(df.head(10))

    # DB 저장
    df.to_sql("financial_summary", ENGINE, if_exists="replace", index=False)
    print("\nDB 저장 완료: financial_summary 테이블")

    # CSV 백업
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/financial_summary.csv", index=False, encoding="utf-8-sig")
    print("CSV 저장 완료: results/financial_summary.csv")


if __name__ == "__main__":
    main()
