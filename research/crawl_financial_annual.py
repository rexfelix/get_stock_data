"""
네이버 증권(navercomp) 연간 재무데이터 수집
- cF3002: 손익계산서 상세 (2021~2025 실적 + 2026 추정)
- cF4002: 재무비율 (ROE, ROA, ROIC, 각종 마진율)
- 저장: financial_annual 테이블
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

# cF3002에서 추출할 항목 (ACC_NM → DB 컬럼)
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
}

# cF3002에서 세부 항목
INCOME_STMT_DETAIL = {
    "연구개발비": "rd_expense",
}

# cF4002에서 추출할 항목
RATIO_MAP = {
    "매출총이익률": "gross_margin",
    "영업이익률": "operating_margin",
    "순이익률": "net_margin",
    "EBITDA마진율": "ebitda_margin",
    "ROE": "roe",
    "ROA": "roa",
    "ROIC": "roic",
}


def get_all_tickers() -> pd.DataFrame:
    query = text("SELECT DISTINCT ticker, name FROM stocks ORDER BY ticker")
    with ENGINE.connect() as conn:
        return pd.read_sql(query, conn)


def get_encparam(code: str, page: str) -> str | None:
    """페이지에서 encparam 추출"""
    url = f"https://navercomp.wisereport.co.kr/v2/company/{page}.aspx?cmp_cd={code}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        m = re.search(r"encparam:\s*'([^']+)'", r.text)
        return m.group(1) if m else None
    except Exception:
        return None


def fetch_cF3002(code: str, encparam: str) -> dict | None:
    """손익계산서 데이터"""
    url = "https://navercomp.wisereport.co.kr/v2/company/cF3002.aspx"
    try:
        r = requests.get(url, params={
            "cmp_cd": code, "frqTyp": "0", "finGubun": "MAIN", "encparam": encparam,
        }, headers={
            **HEADERS,
            "Referer": f"https://navercomp.wisereport.co.kr/v2/company/c1030001.aspx?cmp_cd={code}",
        }, timeout=10)
        return json.loads(r.text) if r.text else None
    except Exception:
        return None


def fetch_cF4002(code: str, encparam: str) -> dict | None:
    """재무비율 데이터"""
    url = "https://navercomp.wisereport.co.kr/v2/company/cF4002.aspx"
    try:
        r = requests.get(url, params={
            "cmp_cd": code, "frqTyp": "0", "finGubun": "MAIN", "encparam": encparam,
        }, headers={
            **HEADERS,
            "Referer": f"https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd={code}",
        }, timeout=10)
        return json.loads(r.text) if r.text else None
    except Exception:
        return None


def parse_years(yymm_list: list) -> list[dict]:
    """YYMM 리스트에서 연도와 실적/추정 여부 파싱"""
    results = []
    for i, y in enumerate(yymm_list):
        y_clean = y.split("<")[0].strip()
        if "전년" in y_clean:
            break
        is_estimate = "(E)" in y_clean
        year_str = y_clean.replace("(E)", "").replace("/12", "").strip()
        try:
            year = int(year_str)
        except ValueError:
            continue

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
            "is_estimate": is_estimate,
            "accounting_std": acct_std,
        })
    return results


def extract_stock_data(code: str, cf3_data: dict, cf4_data: dict | None) -> list[dict]:
    """종목 1개의 연도별 재무데이터 추출"""
    if not cf3_data or "YYMM" not in cf3_data or not cf3_data["YYMM"]:
        return []

    years = parse_years(cf3_data["YYMM"])
    if not years:
        return []

    # 연도별 초기화
    rows = {}
    for y_info in years:
        rows[y_info["data_key"]] = {
            "ticker": code,
            "year": y_info["year"],
            "is_estimate": y_info["is_estimate"],
            "accounting_std": y_info["accounting_std"],
        }

    # cF3002 파싱
    for item in cf3_data.get("DATA", []):
        nm = item["ACC_NM"]
        col = INCOME_STMT_MAP.get(nm) or INCOME_STMT_DETAIL.get(nm)
        if not col:
            continue
        for dk, row in rows.items():
            val = item.get(dk)
            if val is not None:
                row[col] = val

    # cF4002 파싱
    if cf4_data and "DATA" in cf4_data:
        for item in cf4_data["DATA"]:
            nm = item["ACC_NM"]
            col = RATIO_MAP.get(nm)
            if not col:
                continue

            # EBITDA마진율에서 EBITDA 값도 추출
            if nm == "EBITDA마진율":
                for dk, row in rows.items():
                    val = item.get(dk)
                    if val is not None:
                        row[col] = val
                # EBITDA 값은 별도 항목에서 가져옴
                continue

            for dk, row in rows.items():
                val = item.get(dk)
                if val is not None:
                    row[col] = val

        # EBITDA 값 직접 추출
        for item in cf4_data["DATA"]:
            if item["ACC_NM"] == "EBITDA＜당기＞":
                for dk, row in rows.items():
                    val = item.get(dk)
                    if val is not None:
                        row["ebitda"] = val

    return list(rows.values())


def save_batch(records: list[dict]):
    """배치 단위로 DB 저장 (UPSERT)"""
    if not records:
        return

    df = pd.DataFrame(records)

    # 필요한 컬럼만
    cols = [
        "ticker", "year", "is_estimate", "accounting_std",
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
                f"{k} = :{k}" for k in vals if k not in ("ticker", "year")
            )
            conn.execute(text(f"""
                INSERT INTO financial_annual ({', '.join(vals.keys())})
                VALUES ({', '.join(':' + k for k in vals.keys())})
                ON CONFLICT (ticker, year)
                DO UPDATE SET {set_clause}
            """), vals)
        conn.commit()


def main():
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

        # c1030001 페이지에서 encparam 추출 (cF3002, cF4002 공용)
        encparam = get_encparam(ticker, "c1030001")
        if not encparam:
            # c1040001에서 시도
            encparam = get_encparam(ticker, "c1040001")

        if not encparam:
            errors.append(ticker)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                print(f"  [{i+1}/{len(tickers)}] {ticker} {name} | 저장: {total_records}건 | 실패: {len(errors)} | {(i+1)/elapsed:.1f}/초")
            time.sleep(0.1)
            continue

        cf3 = fetch_cF3002(ticker, encparam)
        time.sleep(0.05)
        cf4 = fetch_cF4002(ticker, encparam)
        time.sleep(0.05)

        records = extract_stock_data(ticker, cf3, cf4)
        if records:
            batch.extend(records)
            total_records += len(records)

        # 배치 저장
        if len(batch) >= BATCH_SIZE * 5:
            save_batch(batch)
            batch = []

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(tickers) - i - 1) / rate
            print(f"  [{i+1}/{len(tickers)}] {ticker} {name} | 저장: {total_records}건 | 실패: {len(errors)} | {rate:.1f}/초 | 남은: {remaining/60:.1f}분")

        if not records:
            errors.append(ticker)

    # 마지막 배치 저장
    if batch:
        save_batch(batch)

    elapsed = time.time() - start
    print(f"\n수집 완료")
    print(f"  총 저장: {total_records}건")
    print(f"  실패: {len(errors)}건")
    print(f"  소요시간: {elapsed/60:.1f}분")

    # 결과 확인
    with ENGINE.connect() as conn:
        r = conn.execute(text("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM financial_annual"))
        cnt, tickers = r.fetchone()
        print(f"\nDB 확인: {cnt}건, {tickers}종목")

        r2 = conn.execute(text("""
            SELECT year, is_estimate, COUNT(*),
                   COUNT(operating_income), COUNT(roe)
            FROM financial_annual
            GROUP BY year, is_estimate
            ORDER BY year, is_estimate
        """))
        print("\n연도별 현황:")
        print(f"{'연도':>6} {'구분':>4} {'건수':>6} {'영업이익':>8} {'ROE':>6}")
        for row in r2:
            label = "추정" if row[1] else "실적"
            print(f"{row[0]:>6} {label:>4} {row[2]:>6} {row[3]:>8} {row[4]:>6}")


if __name__ == "__main__":
    main()
