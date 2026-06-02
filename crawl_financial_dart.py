"""
Open DART 기반 재무제표 수집 (네이버 크롤링 대체)
- 소스: OpenDartReader.finstate_all (단일회사 전체 재무제표)
- 수집: 손익계산서 + 재무상태표 + 계산지표 (ROE/ROA/순부채비율/BPS/EBITDA/각 마진)
- 제외: 추정치(E)/목표주가/컨센서스/PER·PBR·EV·EBITDA 등 주가기반 valuation
         → DART에 없음. (PER/PBR류는 키움 주가 join, 컨센서스는 네이버 유지)
- 저장: financial_dart 테이블 (quarter=0:연간(사업보고서), 1/2/3:단일분기)
        금액 단위 = 원(KRW raw). ※네이버 financial_annual(억원)과 단위 다름 주의
- 증분 설계:
    * 2023 ~ 현재까지, 공시 마감일이 지난 "이용가능 기간"만 타깃
    * 이미 저장된 (ticker,year,quarter) 또는 과거기간 무데이터로 기록된 건은 skip
    * 따라서 최초 1회는 전체 backfill, 이후엔 신규 공시분만 추가
    * DART 일일 한도(2만건) 초과 시 graceful 중단 → 다음날 재실행하면 이어받음

reprt_code: 11013=1Q, 11012=반기(2Q), 11014=3Q, 11011=사업보고서(연간)
값 의미: thstrm_amount = 손익은 단일 3개월/연간, 재무상태표는 기말 시점값
"""

import os
import time
import argparse
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import OpenDartReader

import stock_master

load_dotenv()

DB = os.getenv("DB_NAME")
USER = os.getenv("DB_USER")
PW = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
ENGINE = create_engine(f"postgresql://{USER}:{PW}@{HOST}:{PORT}/{DB}")

DART_API_KEY = os.getenv("DART_API_KEY", "cedf1d5e1e536090e6aa1b8c2c51e3ba96775974")
DART = OpenDartReader(DART_API_KEY)

START_YEAR = 2023

# reprt_code → (quarter, 라벨). quarter 0 = 연간(사업보고서)
REPRT = [
    ("11013", 1, "1Q"),
    ("11012", 2, "2Q"),
    ("11014", 3, "3Q"),
    ("11011", 0, "연간"),
]
QUARTER_BY_REPRT = {rc: q for rc, q, _ in REPRT}

# ── 계정 매핑: (account_id 우선, account_nm 부분일치 fallback) ──────────────
# 손익계산서 (sj_div: IS=손익계산서, CIS=포괄손익계산서)
IS_FIELDS = {
    "revenue":          (["ifrs-full_Revenue"], ["매출액", "영업수익", "수익(매출액)"]),
    "cost_of_sales":    (["ifrs-full_CostOfSales"], ["매출원가"]),
    "gross_profit":     (["ifrs-full_GrossProfit"], ["매출총이익"]),
    "sga_expense":      (["dart_TotalSellingGeneralAdministrativeExpenses"], ["판매비와관리비", "판매비와 관리비"]),
    "operating_income": (["dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"], ["영업이익"]),
    "financial_income": (["ifrs-full_FinanceIncome"], ["금융수익"]),
    "financial_cost":   (["ifrs-full_FinanceCosts"], ["금융원가", "금융비용"]),
    "pretax_income":    (["ifrs-full_ProfitLossBeforeTax"], ["법인세비용차감전", "법인세차감전"]),
    "income_tax":       (["ifrs-full_IncomeTaxExpenseContinuingOperations", "ifrs-full_IncomeTaxExpense"], ["법인세비용"]),
    "net_income":       (["ifrs-full_ProfitLoss"], ["당기순이익"]),
    "net_income_ctrl":  (["ifrs-full_ProfitLossAttributableToOwnersOfParent"], ["지배기업의 소유주", "지배주주지분"]),
    "eps":              (["ifrs-full_BasicEarningsLossPerShare", "ifrs-full_BasicEarningsLossPerShareFromContinuingOperations"], ["기본주당이익", "기본주당순이익"]),
}
# 재무상태표 (sj_div: BS)
BS_FIELDS = {
    "total_assets":        (["ifrs-full_Assets"], ["자산총계"]),
    "current_assets":      (["ifrs-full_CurrentAssets"], ["유동자산"]),
    "total_liabilities":   (["ifrs-full_Liabilities"], ["부채총계"]),
    "current_liabilities": (["ifrs-full_CurrentLiabilities"], ["유동부채"]),
    "total_equity":        (["ifrs-full_Equity"], ["자본총계"]),
    "equity_ctrl":         (["ifrs-full_EquityAttributableToOwnersOfParent"], ["지배기업의 소유주", "지배주주지분"]),
    "retained_earnings":   (["ifrs-full_RetainedEarnings"], ["이익잉여금"]),
    "cash":                (["ifrs-full_CashAndCashEquivalents"], ["현금및현금성자산"]),
}
# 이자부부채(차입금성) 합산 대상 account_nm 키워드
DEBT_KEYWORDS = ["단기차입금", "장기차입금", "사채", "유동성장기", "리스부채"]
# 감가상각비(EBITDA용) account_nm 키워드 (현금흐름표 등)
DEP_KEYWORDS = ["감가상각", "무형자산상각"]

AMOUNT_COLS = [
    "revenue", "cost_of_sales", "gross_profit", "sga_expense", "operating_income",
    "financial_income", "financial_cost", "pretax_income", "income_tax",
    "net_income", "net_income_ctrl", "ebitda",
    "total_assets", "current_assets", "total_liabilities", "current_liabilities",
    "total_equity", "equity_ctrl", "retained_earnings", "cash", "total_debt", "net_debt",
]
RATIO_COLS = [
    "eps", "bps", "gross_margin", "operating_margin", "net_margin",
    "ebitda_margin", "roe", "roa", "net_debt_ratio",
]
META_COLS = ["ticker", "year", "quarter", "fs_type"]
ALL_COLS = META_COLS + AMOUNT_COLS + RATIO_COLS


# ── DB 스키마 ──────────────────────────────────────────────────────────────
def ensure_tables():
    amt_defs = ",\n                ".join(f"{c} DOUBLE PRECISION" for c in AMOUNT_COLS)
    ratio_defs = ",\n                ".join(f"{c} DOUBLE PRECISION" for c in RATIO_COLS)
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS financial_dart (
                ticker VARCHAR(20) NOT NULL,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                fs_type VARCHAR(4),
                {amt_defs},
                {ratio_defs},
                updated_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (ticker, year, quarter)
            )
        """))
        # 무데이터(과거 미공시) 기록 → 다음 실행에서 재조회 skip
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS financial_dart_log (
                ticker VARCHAR(20) NOT NULL,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                status VARCHAR(20) NOT NULL,
                checked_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (ticker, year, quarter)
            )
        """))


# ── 증분 대상 산출 ───────────────────────────────────────────────────────────
def available_periods(today: date) -> list[tuple[int, str, int]]:
    """공시 마감일이 지난 (year, reprt_code, quarter) 목록 (2023~현재).

    제출기한(상장사 기준 대략):
      1Q 11013 ~ 5/15, 반기 11012 ~ 8/14, 3Q 11014 ~ 11/14,
      사업보고서 11011 ~ 익년 3/31 (버퍼 4/10).
    기한 + 여유일이 지난 기간만 포함 → 미공시 미래기간은 매번 조회하지 않음.
    """
    periods = []
    for y in range(START_YEAR, today.year + 1):
        if today >= date(y, 5, 16):
            periods.append((y, "11013", 1))
        if today >= date(y, 8, 15):
            periods.append((y, "11012", 2))
        if today >= date(y, 11, 15):
            periods.append((y, "11014", 3))
        # 연간(사업보고서)은 익년 4/10 이후
        if today >= date(y + 1, 4, 10):
            periods.append((y, "11011", 0))
    return periods


def load_done_and_skip():
    with ENGINE.connect() as conn:
        done = set(conn.execute(text(
            "SELECT ticker, year, quarter FROM financial_dart")).fetchall())
        skip = set(conn.execute(text(
            "SELECT ticker, year, quarter FROM financial_dart_log WHERE status = 'empty'")).fetchall())
    return done, skip


# ── 값 추출 helper ───────────────────────────────────────────────────────────
def _to_float(s):
    if s is None:
        return None
    s = str(s).strip()
    if s in ("", "-", "N/A"):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _pick(df, ids, nms):
    """account_id 우선 매칭, 없으면 account_nm 부분일치. thstrm_amount 반환."""
    for aid in ids:
        hit = df[df["account_id"] == aid]
        if len(hit):
            v = _to_float(hit["thstrm_amount"].iloc[0])
            if v is not None:
                return v
    for nm in nms:
        hit = df[df["account_nm"].astype(str).str.contains(nm, na=False, regex=False)]
        if len(hit):
            v = _to_float(hit["thstrm_amount"].iloc[0])
            if v is not None:
                return v
    return None


def _sum_by_nm(df, keywords):
    total, found = 0.0, False
    for kw in keywords:
        for _, row in df[df["account_nm"].astype(str).str.contains(kw, na=False, regex=False)].iterrows():
            v = _to_float(row["thstrm_amount"])
            if v is not None:
                total += v
                found = True
    return total if found else None


def _safe_div(a, b, pct=True):
    if a is None or b in (None, 0):
        return None
    r = a / b
    return r * 100 if pct else r


# ── 단일 (ticker, year, reprt) 조회 + 계산 ───────────────────────────────────
class RateLimit(Exception):
    pass


def fetch_statement(ticker, year, reprt_code):
    """연결(CFS) 우선, 없으면 별도(OFS). (df, fs_type) 또는 (None, None)."""
    for fs_div in ("CFS", "OFS"):
        for attempt in range(2):
            try:
                df = DART.finstate_all(ticker, year, reprt_code=reprt_code, fs_div=fs_div)
                if df is not None and len(df) > 0:
                    return df, fs_div
                break  # 데이터 없음 → 다음 fs_div
            except Exception as e:
                msg = str(e)
                if "020" in msg or "초과" in msg or "한도" in msg.lower() or "limit" in msg.lower():
                    if "021" in msg:  # 분당 한도 → 대기 후 재시도
                        time.sleep(61)
                        continue
                    raise RateLimit(msg)
                break  # 기타 오류 → 다음 fs_div
    return None, None


def compute_row(ticker, year, quarter, df, fs_type):
    is_df = df[df["sj_div"].isin(["IS", "CIS"])]
    bs_df = df[df["sj_div"] == "BS"]
    cf_df = df[df["sj_div"] == "CF"]

    row = {"ticker": ticker, "year": year, "quarter": quarter, "fs_type": fs_type}
    for col, (ids, nms) in IS_FIELDS.items():
        row[col] = _pick(is_df, ids, nms)
    for col, (ids, nms) in BS_FIELDS.items():
        row[col] = _pick(bs_df, ids, nms)

    # 이자부부채 / 순부채
    row["total_debt"] = _sum_by_nm(bs_df, DEBT_KEYWORDS)
    if row["total_debt"] is not None:
        row["net_debt"] = row["total_debt"] - (row.get("cash") or 0)
    else:
        row["net_debt"] = None

    # EBITDA = 영업이익 + 감가상각비(현금흐름표)
    dep = _sum_by_nm(cf_df, DEP_KEYWORDS)
    if row.get("operating_income") is not None and dep is not None:
        row["ebitda"] = row["operating_income"] + dep
    else:
        row["ebitda"] = None

    # 계산지표 (분기행은 비연율화 기간 기준)
    rev = row.get("revenue")
    row["gross_margin"]     = _safe_div(row.get("gross_profit"), rev)
    row["operating_margin"] = _safe_div(row.get("operating_income"), rev)
    row["net_margin"]       = _safe_div(row.get("net_income"), rev)
    row["ebitda_margin"]    = _safe_div(row.get("ebitda"), rev)
    row["roe"]              = _safe_div(row.get("net_income"), row.get("total_equity"))
    row["roa"]              = _safe_div(row.get("net_income"), row.get("total_assets"))
    row["net_debt_ratio"]   = _safe_div(row.get("net_debt"), row.get("total_equity"))

    # BPS = 지배주주지분 자본 / 주식수(순이익_지배/EPS 추정)
    eps, nic = row.get("eps"), row.get("net_income_ctrl")
    eq_ctrl = row.get("equity_ctrl") or row.get("total_equity")
    if eps and nic and eq_ctrl:
        shares = nic / eps
        row["bps"] = eq_ctrl / shares if shares else None
    else:
        row["bps"] = None

    # 최소 유효성: 손익 또는 재무상태표 핵심값이 하나도 없으면 무효
    if row.get("revenue") is None and row.get("operating_income") is None \
            and row.get("total_assets") is None:
        return None
    return row


# ── 저장 ─────────────────────────────────────────────────────────────────────
def save_rows(rows):
    if not rows:
        return
    cols = ALL_COLS
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("ticker", "year", "quarter"))
    sql = text(f"""
        INSERT INTO financial_dart ({', '.join(cols)}, updated_at)
        VALUES ({', '.join(':' + c for c in cols)}, NOW())
        ON CONFLICT (ticker, year, quarter)
        DO UPDATE SET {set_clause}, updated_at = NOW()
    """)
    with ENGINE.begin() as conn:
        for r in rows:
            conn.execute(sql, {c: r.get(c) for c in cols})


def log_empty(items):
    """과거기간(올해 미만) 무데이터만 기록 → 향후 재조회 skip."""
    if not items:
        return
    sql = text("""
        INSERT INTO financial_dart_log (ticker, year, quarter, status, checked_at)
        VALUES (:ticker, :year, :quarter, 'empty', NOW())
        ON CONFLICT (ticker, year, quarter) DO UPDATE SET checked_at = NOW()
    """)
    with ENGINE.begin() as conn:
        for it in items:
            conn.execute(sql, it)


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3, help="동시 요청 수 (DART 분당한도 주의)")
    ap.add_argument("--limit", type=int, default=0, help="테스트용 종목 수 제한 (0=전체)")
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    args = ap.parse_args()

    ensure_tables()

    tickers = stock_master.get_listed_tickers(ENGINE)
    if args.limit:
        tickers = tickers[:args.limit]
    name_of = dict(tickers)
    today = date.today()

    periods = [p for p in available_periods(today) if p[0] >= args.start_year]
    done, skip = load_done_and_skip()

    # 조회 대상 작업목록 구성 (이미 저장됐거나 무데이터 기록된 건 제외)
    work = []
    for ticker, _name in tickers:
        for year, reprt_code, quarter in periods:
            key = (ticker, year, quarter)
            if key in done or key in skip:
                continue
            work.append((ticker, year, reprt_code, quarter))

    total = len(work)
    print(f"대상 종목: {len(tickers)} | 이용가능 기간: {len(periods)} | 신규 조회 작업: {total}건")
    if total == 0:
        print("추가할 신규 데이터 없음 (모두 최신).")
        return

    saved, empties, errors = 0, [], 0
    rows_buf, empty_buf = [], []
    start = time.time()
    stop = False

    def task(w):
        ticker, year, reprt_code, quarter = w
        df, fs_type = fetch_statement(ticker, year, reprt_code)
        if df is None:
            return w, None
        return w, compute_row(ticker, year, quarter, df, fs_type)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(task, w): w for w in work}
        for i, fut in enumerate(as_completed(futures), 1):
            if stop:
                break
            w = futures[fut]
            ticker, year, reprt_code, quarter = w
            try:
                _, row = fut.result()
            except RateLimit as e:
                print(f"\n⚠️ DART 일일 한도 초과 추정 → 중단. 다음날 재실행하면 이어받습니다. ({e})")
                stop = True
                break
            except Exception:
                errors += 1
                continue

            if row:
                rows_buf.append(row)
            else:
                # 과거기간(올해 미만)만 영구 skip 기록, 올해분은 미공시일 수 있어 재시도 허용
                if year < today.year:
                    empty_buf.append({"ticker": ticker, "year": year, "quarter": quarter})
                empties.append(w)

            if len(rows_buf) >= 200:
                save_rows(rows_buf); saved += len(rows_buf); rows_buf = []
            if len(empty_buf) >= 200:
                log_empty(empty_buf); empty_buf = []

            if i % 500 == 0:
                el = time.time() - start
                rate = i / el
                print(f"  [{i}/{total}] {ticker} {name_of.get(ticker,'')} {year}Q{quarter} "
                      f"| 저장 {saved+len(rows_buf)} | 무데이터 {len(empties)} | 오류 {errors} "
                      f"| {rate:.1f}/s | 남은 {(total-i)/rate/60:.1f}분")

    save_rows(rows_buf); saved += len(rows_buf)
    log_empty(empty_buf)

    el = time.time() - start
    print(f"\n수집 완료 ({'중단됨' if stop else '정상'})")
    print(f"  저장: {saved}건 | 무데이터: {len(empties)}건 | 오류: {errors}건 | {el/60:.1f}분")

    with ENGINE.connect() as conn:
        cnt, tk = conn.execute(text(
            "SELECT COUNT(*), COUNT(DISTINCT ticker) FROM financial_dart")).fetchone()
        print(f"  DB: financial_dart {cnt}건, {tk}종목")
        rows = conn.execute(text("""
            SELECT year, quarter, COUNT(*), COUNT(operating_income), COUNT(roe)
            FROM financial_dart GROUP BY year, quarter ORDER BY year, quarter
        """)).fetchall()
        print(f"\n  {'연도':>6} {'분기':>4} {'건수':>7} {'영업이익':>8} {'ROE':>6}")
        for y, q, c, oc, rc in rows:
            print(f"  {y:>6} {('연간' if q==0 else str(q)+'Q'):>4} {c:>7} {oc:>8} {rc:>6}")


if __name__ == "__main__":
    main()
