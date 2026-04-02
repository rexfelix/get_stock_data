"""
재무데이터 크롤링 통합 실행
1. crawl_financial.py       - Financial Summary (cF1002)
2. crawl_financial_annual.py - 손익계산서 + 재무비율 (cF3002 + cF4002)
3. crawl_consensus.py       - 컨센서스 + 주요지표 (c1010001)
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

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  전체 완료 (총 {total_elapsed/60:.1f}분)")
    print(f"{'=' * 60}")
    for script, status, elapsed in results:
        print(f"  {status}  {script} ({elapsed/60:.1f}분)")


if __name__ == "__main__":
    main()
