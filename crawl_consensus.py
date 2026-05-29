"""
네이버 증권(navercomp) 컨센서스 + 주요지표 크롤링
- 소스: navercomp.wisereport.co.kr/v2/company/c1010001.aspx
- 수집: 투자의견컨센서스, 제공처별 투자의견, 주요지표(PCR/BPS/DPS/배당수익률)
- 저장: consensus_summary, consensus_provider 테이블 + financial_summary UPDATE
"""

import os
import re
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
    """현재 상장 종목(stock_master)만 대상. 없으면 stocks 로 fallback."""
    import stock_master
    rows = stock_master.get_listed_tickers(ENGINE)
    return pd.DataFrame(rows, columns=["ticker", "name"])


def ensure_tables():
    with ENGINE.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS consensus_summary (
                ticker VARCHAR(20) NOT NULL,
                rating FLOAT,
                target_price BIGINT,
                eps FLOAT,
                per FLOAT,
                analyst_count INTEGER,
                crawled_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (ticker)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS consensus_provider (
                ticker VARCHAR(20) NOT NULL,
                provider VARCHAR(100) NOT NULL,
                report_date DATE NOT NULL,
                target_price BIGINT,
                prev_target BIGINT,
                change_pct FLOAT,
                opinion VARCHAR(50),
                crawled_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (ticker, provider, report_date)
            )
        """))
        # financial_summary에 새 컬럼 추가 (이미 있으면 무시)
        for col in ["pcr", "bps", "dps", "dividend_yield"]:
            conn.execute(text(
                f"ALTER TABLE financial_summary ADD COLUMN IF NOT EXISTS {col} FLOAT"
            ))


def parse_num(s: str) -> float | None:
    if not s or s in ("N/A", "", "-"):
        return None
    s = s.replace(",", "").replace("원", "").replace("억", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_int(s: str) -> int | None:
    v = parse_num(s)
    return int(v) if v is not None else None


def fetch_page(ticker: str) -> BeautifulSoup | None:
    url = f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={ticker}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200 or not r.text.strip():
            return None
        return BeautifulSoup(r.text, "html.parser")
    except Exception:
        return None


def find_table_by_caption(tables: list, keyword: str):
    for tbl in tables:
        caption = tbl.find("caption")
        if caption and keyword in caption.get_text(strip=True):
            return tbl
    return None


def parse_key_indicators(tables: list) -> list[dict]:
    """주요지표 테이블 파싱 → [{year, pcr, bps, dps, dividend_yield}, ...]"""
    tbl = find_table_by_caption(tables, "주요지표")
    if not tbl:
        return []

    rows = tbl.find_all("tr")
    if len(rows) < 2:
        return []

    # 헤더에서 연도 추출: "2025/12(A)", "2026/12(E)"
    header_cells = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    years = []
    for h in header_cells[1:]:
        m = re.match(r"(\d{4})/\d+\(([AE])\)", h)
        if m:
            years.append({"year": m.group(1), "is_estimate": m.group(2) == "E"})

    if not years:
        return []

    # 데이터 행에서 값 추출
    field_map = {
        "PCR": "pcr",
        "BPS": "bps",
        "현금DPS": "dps",
        "현금배당수익률": "dividend_yield",
    }

    data = [{**y} for y in years]

    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if len(cells) < 2:
            continue
        label = cells[0]
        col = field_map.get(label)
        if not col:
            continue
        for j, val_str in enumerate(cells[1:]):
            if j < len(data):
                data[j][col] = parse_num(val_str)

    return data


def parse_consensus_summary(tables: list) -> dict | None:
    """투자의견컨센서스 파싱"""
    tbl = find_table_by_caption(tables, "투자의견컨센서스")
    if not tbl:
        return None

    rows = tbl.find_all("tr")
    # row[0]: 헤더 [rating_display, "투자의견", "목표주가(원)", "EPS(원)", "PER(배)", "추정기관수"]
    # row[1]: 데이터 [rating, target_price, eps, per, analyst_count]
    if len(rows) < 2:
        return None

    header_cells = [td.get_text(strip=True) for td in rows[0].find_all(["th", "td"])]
    data_cells = [td.get_text(strip=True) for td in rows[1].find_all(["th", "td"])]

    if len(data_cells) < 5:
        return None

    # 첫 번째 행의 첫 번째 셀이 rating (헤더에도 있고 데이터에도 있음)
    rating = parse_num(header_cells[0]) if header_cells else None

    return {
        "rating": rating,
        "target_price": parse_int(data_cells[1]),
        "eps": parse_num(data_cells[2]),
        "per": parse_num(data_cells[3]),
        "analyst_count": parse_int(data_cells[4]),
    }


def parse_consensus_providers(tables: list) -> list[dict]:
    """제공처별 투자의견 파싱"""
    tbl = find_table_by_caption(tables, "제공처별")
    if not tbl:
        return []

    rows = tbl.find_all("tr")
    results = []
    for row in rows[1:]:  # skip header
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if len(cells) < 6:
            continue

        provider = cells[0]
        if not provider:
            continue

        # 날짜: "26/04/01" → "2026-04-01"
        date_str = cells[1]
        report_date = None
        if date_str:
            m = re.match(r"(\d{2})/(\d{2})/(\d{2})", date_str)
            if m:
                report_date = f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"

        results.append({
            "provider": provider,
            "report_date": report_date,
            "target_price": parse_int(cells[2]),
            "prev_target": parse_int(cells[3]),
            "change_pct": parse_num(cells[4]),
            "opinion": cells[5] if len(cells) > 5 else None,
        })

    return results


def save_batch(ticker_data_list: list[dict]):
    """배치 저장: consensus_summary, consensus_provider, financial_summary UPDATE"""
    if not ticker_data_list:
        return

    with ENGINE.connect() as conn:
        for item in ticker_data_list:
            ticker = item["ticker"]

            # 1. consensus_summary UPSERT
            cs = item.get("consensus_summary")
            if cs:
                conn.execute(text("""
                    INSERT INTO consensus_summary (ticker, rating, target_price, eps, per, analyst_count, crawled_at)
                    VALUES (:ticker, :rating, :target_price, :eps, :per, :analyst_count, NOW())
                    ON CONFLICT (ticker)
                    DO UPDATE SET rating = :rating, target_price = :target_price,
                                  eps = :eps, per = :per, analyst_count = :analyst_count,
                                  crawled_at = NOW()
                """), {"ticker": ticker, **cs})

            # 2. consensus_provider UPSERT
            for cp in item.get("providers", []):
                if not cp.get("report_date"):
                    continue
                conn.execute(text("""
                    INSERT INTO consensus_provider
                        (ticker, provider, report_date, target_price, prev_target, change_pct, opinion, crawled_at)
                    VALUES (:ticker, :provider, :report_date, :target_price, :prev_target, :change_pct, :opinion, NOW())
                    ON CONFLICT (ticker, provider, report_date)
                    DO UPDATE SET target_price = :target_price, prev_target = :prev_target,
                                  change_pct = :change_pct, opinion = :opinion, crawled_at = NOW()
                """), {"ticker": ticker, **cp})

            # 3. financial_summary UPDATE (pcr, bps, dps, dividend_yield)
            for ind in item.get("indicators", []):
                if not any(ind.get(k) is not None for k in ["pcr", "bps", "dps", "dividend_yield"]):
                    continue
                conn.execute(text("""
                    UPDATE financial_summary
                    SET pcr = COALESCE(:pcr, pcr),
                        bps = COALESCE(:bps, bps),
                        dps = COALESCE(:dps, dps),
                        dividend_yield = COALESCE(:dividend_yield, dividend_yield)
                    WHERE ticker = :ticker AND year = :year
                """), {
                    "ticker": ticker,
                    "year": ind["year"],
                    "pcr": ind.get("pcr"),
                    "bps": ind.get("bps"),
                    "dps": ind.get("dps"),
                    "dividend_yield": ind.get("dividend_yield"),
                })

        conn.commit()


def main():
    ensure_tables()

    tickers = get_all_tickers()
    print(f"전체 종목: {len(tickers)}")

    total_consensus = 0
    total_providers = 0
    total_indicators = 0
    errors = []
    start = time.time()
    batch = []
    BATCH_SIZE = 50

    for i, row in tickers.iterrows():
        ticker = row["ticker"]
        name = row["name"]

        soup = fetch_page(ticker)
        if not soup:
            errors.append(ticker)
            time.sleep(0.15)
            continue

        tables = soup.find_all("table")

        cs = parse_consensus_summary(tables)
        providers = parse_consensus_providers(tables)
        indicators = parse_key_indicators(tables)

        item = {
            "ticker": ticker,
            "consensus_summary": cs,
            "providers": providers,
            "indicators": indicators,
        }

        if cs:
            total_consensus += 1
        total_providers += len(providers)
        total_indicators += len(indicators)

        batch.append(item)

        if len(batch) >= BATCH_SIZE:
            save_batch(batch)
            batch = []

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(tickers) - i - 1) / rate
            print(
                f"  [{i+1}/{len(tickers)}] {ticker} {name}"
                f" | 컨센: {total_consensus} | 제공처: {total_providers} | 지표: {total_indicators}"
                f" | {rate:.1f}/초 | 남은: {remaining/60:.1f}분"
            )

        if not cs and not providers and not indicators:
            errors.append(ticker)

        time.sleep(0.15)

    if batch:
        save_batch(batch)

    elapsed = time.time() - start
    print(f"\n수집 완료")
    print(f"  컨센서스: {total_consensus}건")
    print(f"  제공처별: {total_providers}건")
    print(f"  주요지표: {total_indicators}건")
    print(f"  실패/무데이터: {len(errors)}건")
    print(f"  소요시간: {elapsed/60:.1f}분")

    with ENGINE.connect() as conn:
        r1 = conn.execute(text("SELECT COUNT(*) FROM consensus_summary"))
        r2 = conn.execute(text("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM consensus_provider"))
        r3 = conn.execute(text("SELECT COUNT(*) FROM financial_summary WHERE pcr IS NOT NULL"))
        print(f"\nDB 확인:")
        print(f"  consensus_summary: {r1.fetchone()[0]}건")
        cnt, tk = r2.fetchone()
        print(f"  consensus_provider: {cnt}건 ({tk}종목)")
        print(f"  financial_summary (pcr 있음): {r3.fetchone()[0]}건")


if __name__ == "__main__":
    main()
