"""stocks.amount 백필 단위 테스트 (TDD Red).

검증:
- T1: 모듈 임포트
- T2: parse_amount(raw) — 콤마/부호 처리, NULL 처리
- T3: build_amount_records(api_rows, ticker) — API 응답을 (ticker, date, amount) 리스트로
- T4: next_base_dt(oldest_dt) — 페이지네이션 다음 기준일
- T5: 진행 상황 (progress) load/save
"""
import os
import sys
import json

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

def test_module_can_be_imported():
    import amount_backfill  # noqa: F401


# ---------------------------------------------------------------------------
# T2: parse_amount
# ---------------------------------------------------------------------------

def test_parse_amount_plain():
    import amount_backfill as m
    assert m.parse_amount("123456") == 123456


def test_parse_amount_with_comma():
    import amount_backfill as m
    assert m.parse_amount("1,234,567") == 1234567


def test_parse_amount_with_sign():
    import amount_backfill as m
    assert m.parse_amount("+1234") == 1234


def test_parse_amount_none_or_empty():
    import amount_backfill as m
    assert m.parse_amount(None) is None
    assert m.parse_amount("") is None


def test_parse_amount_invalid():
    import amount_backfill as m
    assert m.parse_amount("abc") is None


# ---------------------------------------------------------------------------
# T3: build_amount_records
# ---------------------------------------------------------------------------

def test_build_amount_records_basic():
    import amount_backfill as m
    api_rows = [
        {"dt": "20240105", "trde_prica": "1000"},
        {"dt": "20240104", "trde_prica": "2000"},
        {"dt": "20240103", "trde_prica": "1500"},
    ]
    records = m.build_amount_records(api_rows, ticker="005930")
    assert len(records) == 3
    assert records[0] == ("005930", pd.Timestamp("2024-01-05"), 1000)
    assert records[1] == ("005930", pd.Timestamp("2024-01-04"), 2000)
    assert records[2] == ("005930", pd.Timestamp("2024-01-03"), 1500)


def test_build_amount_records_skips_invalid_dt():
    import amount_backfill as m
    api_rows = [
        {"dt": "20240105", "trde_prica": "1000"},
        {"dt": "", "trde_prica": "2000"},  # invalid dt
        {"dt": "20240103", "trde_prica": ""},  # missing amount → None
    ]
    records = m.build_amount_records(api_rows, ticker="A")
    # invalid dt skipped, missing amount → None
    assert len(records) == 2
    assert records[0] == ("A", pd.Timestamp("2024-01-05"), 1000)
    assert records[1] == ("A", pd.Timestamp("2024-01-03"), None)


def test_build_amount_records_empty():
    import amount_backfill as m
    assert m.build_amount_records([], ticker="A") == []


# ---------------------------------------------------------------------------
# T4: next_base_dt
# ---------------------------------------------------------------------------

def test_next_base_dt_returns_day_before():
    """oldest_dt - 1 day. 형식 YYYYMMDD."""
    import amount_backfill as m
    assert m.next_base_dt("20240105") == "20240104"
    assert m.next_base_dt("20240101") == "20231231"


def test_next_base_dt_handles_string():
    import amount_backfill as m
    assert m.next_base_dt("20190102") == "20190101"


# ---------------------------------------------------------------------------
# T5: progress save/load
# ---------------------------------------------------------------------------

def test_progress_save_and_load(tmp_path):
    import amount_backfill as m
    progress_file = tmp_path / "progress.json"

    # 빈 progress 로드 → 빈 dict
    p = m.load_progress(str(progress_file))
    assert p == {"completed": [], "failed": []}

    # 완료 표시
    p["completed"] = ["005930", "000660"]
    p["failed"] = ["999999"]
    m.save_progress(str(progress_file), p)

    # 재로드
    p2 = m.load_progress(str(progress_file))
    assert "005930" in p2["completed"]
    assert "000660" in p2["completed"]
    assert "999999" in p2["failed"]


def test_is_done_check():
    import amount_backfill as m
    p = {"completed": ["005930", "000660"], "failed": ["999999"]}
    assert m.is_done(p, "005930") is True
    assert m.is_done(p, "000660") is True
    assert m.is_done(p, "999999") is True  # failed 도 done 처리
    assert m.is_done(p, "035720") is False
