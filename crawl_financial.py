"""
네이버 증권(navercomp) 연간+분기 재무데이터 수집
- cF3002: 손익계산서 상세
- cF4002: 재무비율 (ROE, ROA, ROIC, 각종 마진율)
- 저장: financial_annual 테이블 (quarter=0: 연간, 1~4: 분기)
"""

import os
import re
import json
import time
import requests
import pandas as pd
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

INCOME_STMT_MAP = {
    "매출액(수익)": "revenue",
    "매출원가": "cost_of_sales",
    "매출총이익": "gross_profit",
    "판매비와관리비": "sga_expense",
    "영업이익": "operating_income",
    "금융수익": "financial_income",
    "금융원가": "financial_cost",
    "기타영업외손익": "other_non_op",
    "법인세비용차감전계속사업이익": "pretax_income",
    "법인세비용": "income_tax",
    "당기순이익": "net_income",
    "(지배주주지분)당기순이익": "net_income_ctrl",
    "(지배주주지분)주당순이익": "eps",
    "*(지배주주지분)주당순이익": "eps",
}

INCOME_STMT_DETAIL = {
    "연구개발비": "rd_expense",
}

RATIO_MAP = {
    "매출총이익률": "gross_margin",
    "영업이익률": "operating_margin",
    "순이익률": "net_margin",
    "EBITDA마진율": "ebitda_margin",
    "ROE": "roe",
    "ROA": "roa",
    "ROIC": "roic",
}

# 월 → 분기 매핑
MONTH_TO_QUARTER = {"03": 1, "06": 2, "09": 3, "12": 4}


def get_all_tickers() -> pd.DataFrame:
    """현재 상장 종목(stock_master)만 대상. 없으면 stocks 로 fallback."""
    import stock_master
    rows = stock_master.get_listed_tickers(ENGINE)
    return pd.DataFrame(rows, columns=["ticker", "name"])


def ensure_table():
    with ENGINE.begin() as conn:
        # 새 테이블이면 quarter 포함해서 생성
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS financial_annual (
                ticker VARCHAR(20) NOT NULL,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL DEFAULT 0,
                is_estimate BOOLEAN,
                accounting_std VARCHAR(20),
                revenue FLOAT, cost_of_sales FLOAT, gross_profit FLOAT,
                sga_expense FLOAT, rd_expense FLOAT, operating_income FLOAT,
                financial_income FLOAT, financial_cost FLOAT, other_non_op FLOAT,
                pretax_income FLOAT, income_tax FLOAT, net_income FLOAT,
                net_income_ctrl FLOAT, ebitda FLOAT, eps FLOAT,
                gross_margin FLOAT, operating_margin FLOAT, net_margin FLOAT,
                ebitda_margin FLOAT, roe FLOAT, roa FLOAT, roic FLOAT,
                PRIMARY KEY (ticker, year, quarter)
            )
        """))
        # 기존 테이블에 quarter 컬럼이 없으면 마이그레이션
        has_quarter = conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'financial_annual' AND column_name = 'quarter'
        """)).fetchone()
        if not has_quarter:
            conn.execute(text("ALTER TABLE financial_annual ADD COLUMN quarter INTEGER NOT NULL DEFAULT 0"))
            conn.execute(text("ALTER TABLE financial_annual DROP CONSTRAINT IF EXISTS financial_annual_pkey"))
            conn.execute(text("ALTER TABLE financial_annual ADD PRIMARY KEY (ticker, year, quarter)"))
        # (ticker, year) 레거시 UNIQUE 제약조건 제거
        conn.execute(text("ALTER TABLE financial_annual DROP CONSTRAINT IF EXISTS financial_annual_ticker_year_key"))


def get_encparam(code: str, page: str) -> str | None:
    url = f"https://navercomp.wisereport.co.kr/v2/company/{page}.aspx?cmp_cd={code}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        m = re.search(r"encparam:\s*'([^']+)'", r.text)
        return m.group(1) if m else None
    except Exception:
        return None


def fetch_cf_data(code: str, encparam: str, page: str, frq_typ: str) -> dict | None:
    """cF3002 또는 cF4002 데이터 가져오기 (frq_typ: 0=연간, 1=분기)"""
    url = f"https://navercomp.wisereport.co.kr/v2/company/{page}.aspx"
    referer_page = "c1030001" if page == "cF3002" else "c1040001"
    try:
        r = requests.get(url, params={
            "cmp_cd": code, "frqTyp": frq_typ, "finGubun": "MAIN", "encparam": encparam,
        }, headers={
            **HEADERS,
            "Referer": f"https://navercomp.wisereport.co.kr/v2/company/{referer_page}.aspx?cmp_cd={code}",
        }, timeout=10)
        return json.loads(r.text) if r.text else None
    except Exception:
        return None


def parse_periods(yymm_list: list, is_quarterly: bool) -> list[dict]:
    """YYMM 리스트에서 연도/분기 파싱"""
    results = []
    for i, y in enumerate(yymm_list):
        y_clean = y.split("<")[0].strip()
        if "전년" in y_clean or "전분기" in y_clean:
            break
        is_estimate = "(E)" in y_clean
        y_clean = y_clean.replace("(E)", "").strip()

        if is_quarterly:
            # "2025/03", "2025/06" 등
            m = re.match(r"(\d{4})/(\d{2})", y_clean)
            if not m:
                continue
            year = int(m.group(1))
            quarter = MONTH_TO_QUARTER.get(m.group(2), 0)
            if quarter == 0:
                continue
        else:
            # "2023/12" or "2023"
            year_str = re.sub(r"/\d+", "", y_clean).strip()
            try:
                year = int(year_str)
            except ValueError:
                continue
            quarter = 0

        acct_std = ""
        if "IFRS연결" in y:
            acct_std = "IFRS연결"
        elif "IFRS별도" in y:
            acct_std = "IFRS별도"
        elif "GAAP" in y:
            acct_std = "GAAP"

        results.append({
            "data_key": f"DATA{i+1}",
            "year": year,
            "quarter": quarter,
            "is_estimate": is_estimate,
            "accounting_std": acct_std,
        })
    return results


def extract_stock_data(code: str, cf3_data: dict, cf4_data: dict | None,
                       is_quarterly: bool) -> list[dict]:
    if not cf3_data or "YYMM" not in cf3_data or not cf3_data["YYMM"]:
        return []

    periods = parse_periods(cf3_data["YYMM"], is_quarterly)
    if not periods:
        return []

    rows = {}
    for p in periods:
        rows[p["data_key"]] = {
            "ticker": code,
            "year": p["year"],
            "quarter": p["quarter"],
            "is_estimate": p["is_estimate"],
            "accounting_std": p["accounting_std"],
        }

    for item in cf3_data.get("DATA", []):
        nm = item["ACC_NM"]
        col = INCOME_STMT_MAP.get(nm) or INCOME_STMT_DETAIL.get(nm)
        if not col:
            continue
        for dk, row in rows.items():
            val = item.get(dk)
            if val is not None:
                row[col] = val

    if cf4_data and "DATA" in cf4_data:
        for item in cf4_data["DATA"]:
            nm = item["ACC_NM"]
            col = RATIO_MAP.get(nm)
            if not col:
                continue

            if nm == "EBITDA마진율":
                for dk, row in rows.items():
                    val = item.get(dk)
                    if val is not None:
                        row[col] = val
                continue

            for dk, row in rows.items():
                val = item.get(dk)
                if val is not None:
                    row[col] = val

        for item in cf4_data["DATA"]:
            if item["ACC_NM"] == "EBITDA＜당기＞":
                for dk, row in rows.items():
                    val = item.get(dk)
                    if val is not None:
                        row["ebitda"] = val

    return list(rows.values())


def save_batch(records: list[dict]):
    if not records:
        return

    df = pd.DataFrame(records)

    cols = [
        "ticker", "year", "quarter", "is_estimate", "accounting_std",
        "revenue", "cost_of_sales", "gross_profit", "sga_expense", "rd_expense",
        "operating_income", "financial_income", "financial_cost", "other_non_op",
        "pretax_income", "income_tax", "net_income", "net_income_ctrl", "ebitda", "eps",
        "gross_margin", "operating_margin", "net_margin", "ebitda_margin",
        "roe", "roa", "roic",
    ]

    for c in cols:
        if c not in df.columns:
            df[c] = None

    df = df[cols]

    with ENGINE.connect() as conn:
        for _, row in df.iterrows():
            vals = row.to_dict()
            set_clause = ", ".join(
                f"{k} = :{k}" for k in vals if k not in ("ticker", "year", "quarter")
            )
            conn.execute(text(f"""
                INSERT INTO financial_annual ({', '.join(vals.keys())})
                VALUES ({', '.join(':' + k for k in vals.keys())})
                ON CONFLICT (ticker, year, quarter)
                DO UPDATE SET {set_clause}
            """), vals)
        conn.commit()


def main():
    ensure_table()

    tickers = get_all_tickers()
    print(f"전체 종목: {len(tickers)}")

    total_records = 0
    errors = []
    start = time.time()
    batch = []
    BATCH_SIZE = 50

    for i, row in tickers.iterrows():
        ticker = row["ticker"]
        name = row["name"]

        encparam = get_encparam(ticker, "c1030001")
        if not encparam:
            encparam = get_encparam(ticker, "c1040001")

        if not encparam:
            errors.append(ticker)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                print(f"  [{i+1}/{len(tickers)}] {ticker} {name} | 저장: {total_records}건 | 실패: {len(errors)} | {(i+1)/elapsed:.1f}/초")
            time.sleep(0.1)
            continue

        # 연간 데이터 (frqTyp=0)
        cf3_y = fetch_cf_data(ticker, encparam, "cF3002", "0")
        time.sleep(0.05)
        cf4_y = fetch_cf_data(ticker, encparam, "cF4002", "0")
        time.sleep(0.05)

        records_y = extract_stock_data(ticker, cf3_y, cf4_y, is_quarterly=False)

        # 분기 데이터 (frqTyp=1)
        cf3_q = fetch_cf_data(ticker, encparam, "cF3002", "1")
        time.sleep(0.05)
        cf4_q = fetch_cf_data(ticker, encparam, "cF4002", "1")
        time.sleep(0.05)

        records_q = extract_stock_data(ticker, cf3_q, cf4_q, is_quarterly=True)

        records = records_y + records_q
        if records:
            batch.extend(records)
            total_records += len(records)
        else:
            errors.append(ticker)

        if len(batch) >= BATCH_SIZE * 5:
            save_batch(batch)
            batch = []

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(tickers) - i - 1) / rate
            print(f"  [{i+1}/{len(tickers)}] {ticker} {name} | 저장: {total_records}건 | 실패: {len(errors)} | {rate:.1f}/초 | 남은: {remaining/60:.1f}분")

    if batch:
        save_batch(batch)

    elapsed = time.time() - start
    print(f"\n수집 완료")
    print(f"  총 저장: {total_records}건")
    print(f"  실패: {len(errors)}건")
    print(f"  소요시간: {elapsed/60:.1f}분")

    with ENGINE.connect() as conn:
        r = conn.execute(text("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM financial_annual"))
        cnt, tk_cnt = r.fetchone()
        print(f"\nDB 확인: {cnt}건, {tk_cnt}종목")

        r2 = conn.execute(text("""
            SELECT quarter, year, is_estimate, COUNT(*),
                   COUNT(operating_income), COUNT(roe)
            FROM financial_annual
            GROUP BY quarter, year, is_estimate
            ORDER BY quarter, year, is_estimate
        """))
        print("\n기간별 현황:")
        print(f"{'구간':>6} {'연도':>6} {'구분':>4} {'건수':>6} {'영업이익':>8} {'ROE':>6}")
        for row in r2:
            q_label = "연간" if row[0] == 0 else f"{row[0]}Q"
            e_label = "추정" if row[2] else "실적"
            print(f"{q_label:>6} {row[1]:>6} {e_label:>4} {row[3]:>6} {row[4]:>8} {row[5]:>6}")


if __name__ == "__main__":
    main()
