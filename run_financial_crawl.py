"""
재무데이터 크롤링 통합 실행
1. crawl_financial.py       - Financial Summary (cF1002)
2. crawl_financial_annual.py - 손익계산서 + 재무비율 (cF3002 + cF4002)
3. crawl_consensus.py       - 컨센서스 + 주요지표 (c1010001)
4. PEG 계산                 - consensus_summary.per ÷ EPS 연평균성장률(CAGR) → consensus_summary.peg
"""

import time
import subprocess
import sys
import os

SCRIPTS = [
    ("crawl_financial_summary.py", "Financial Summary (cF1002 → financial_summary)"),
    ("crawl_financial.py", "연간+분기 재무데이터 (cF3002+cF4002 → financial_annual)"),
    ("crawl_consensus.py", "컨센서스 + 주요지표 (c1010001 → consensus)"),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PEG = PER ÷ EPS 연평균성장률(CAGR).
#   CAGR(%) = (EPS_최종연도 / EPS_시작연도)^(1/연수) - 1, ×100
#   대상: financial_annual quarter=0, 2024~2027 컨센서스(가용 연도만, 종목별 첫·끝 연도)
#   PER : consensus_summary.per, PEG 은 CAGR>0 인 경우만 산출
PEG_YEAR_FROM = 2024
PEG_YEAR_TO = 2027

PEG_UPDATE_SQL = f"""
WITH e AS (
    SELECT ticker, year, eps
    FROM financial_annual
    WHERE quarter = 0 AND year BETWEEN {PEG_YEAR_FROM} AND {PEG_YEAR_TO}
      AND eps IS NOT NULL AND eps <> 'NaN'::double precision
),
agg AS (
    SELECT ticker,
           count(*) AS n_years,
           (array_agg(year ORDER BY year))[1]      AS y0,
           (array_agg(year ORDER BY year DESC))[1] AS y1,
           (array_agg(eps  ORDER BY year))[1]      AS eps0,
           (array_agg(eps  ORDER BY year DESC))[1] AS eps1
    FROM e GROUP BY ticker
),
calc AS (
    SELECT ticker,
           CASE WHEN n_years >= 2 AND (y1 - y0) >= 1 AND eps0 > 0 AND eps1 > 0
                THEN (power(eps1 / eps0, 1.0 / (y1 - y0)) - 1) * 100
           END AS cagr_pct
    FROM agg
)
UPDATE consensus_summary cs
SET eps_cagr = c.cagr_pct,
    peg      = CASE WHEN c.cagr_pct > 0 THEN cs.per / c.cagr_pct END
FROM calc c
WHERE cs.ticker = c.ticker
"""


def compute_and_store_peg():
    """크롤·저장 완료 후 consensus_summary 에 eps_cagr / peg 컬럼을 채운다."""
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv

    load_dotenv()
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE consensus_summary ADD COLUMN IF NOT EXISTS eps_cagr double precision"
        ))
        conn.execute(text(
            "ALTER TABLE consensus_summary ADD COLUMN IF NOT EXISTS peg double precision"
        ))
        conn.execute(text(
            "COMMENT ON COLUMN consensus_summary.eps_cagr IS "
            "'EPS 연평균성장률(%) = (EPS_최종연도/EPS_시작연도)^(1/연수)-1, "
            f"{PEG_YEAR_FROM}~{PEG_YEAR_TO} quarter=0 컨센서스 가용연도'"
        ))
        conn.execute(text(
            "COMMENT ON COLUMN consensus_summary.peg IS "
            "'PEG = per / eps_cagr(%). eps_cagr>0 인 경우만 산출'"
        ))
        # 재계산 대비 초기화 후 UPDATE
        conn.execute(text("UPDATE consensus_summary SET eps_cagr = NULL, peg = NULL"))
        conn.execute(text(PEG_UPDATE_SQL))
        row = conn.execute(text(
            "SELECT count(*) FILTER (WHERE peg IS NOT NULL), "
            "count(*) FILTER (WHERE eps_cagr IS NOT NULL) FROM consensus_summary"
        )).fetchone()
    engine.dispose()
    print(f"  → PEG {row[0]}건 / EPS CAGR {row[1]}건 저장 (consensus_summary.peg, .eps_cagr)")


def main():
    print("=" * 60)
    print("  재무데이터 크롤링 통합 실행")
    print("=" * 60)

    total_start = time.time()
    results = []

    for i, (script, desc) in enumerate(SCRIPTS, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(SCRIPTS)}] {desc}")
        print(f"  실행: {script}")
        print(f"{'─' * 60}\n")

        start = time.time()
        path = os.path.join(BASE_DIR, script)
        ret = subprocess.run([sys.executable, path], cwd=BASE_DIR)
        elapsed = time.time() - start

        status = "성공" if ret.returncode == 0 else "실패"
        results.append((script, status, elapsed))
        print(f"\n  → {status} ({elapsed/60:.1f}분)")

    # 크롤·저장 완료 후 PEG 계산·저장
    print(f"\n{'─' * 60}")
    print(f"  [{len(SCRIPTS) + 1}/{len(SCRIPTS) + 1}] PEG 계산 (per ÷ EPS CAGR → consensus_summary.peg)")
    print(f"{'─' * 60}\n")
    start = time.time()
    try:
        compute_and_store_peg()
        status = "성공"
    except Exception as exc:
        status = "실패"
        print(f"  → PEG 계산 실패: {exc}")
    elapsed = time.time() - start
    results.append(("PEG 계산", status, elapsed))
    print(f"\n  → {status} ({elapsed/60:.1f}분)")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  전체 완료 (총 {total_elapsed/60:.1f}분)")
    print(f"{'=' * 60}")
    for script, status, elapsed in results:
        print(f"  {status}  {script} ({elapsed/60:.1f}분)")


if __name__ == "__main__":
    main()
