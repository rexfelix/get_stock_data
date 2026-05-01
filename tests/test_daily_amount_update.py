"""add_daily_stocks.py 에 추가될 amount 일일 업데이트 단위 테스트 (TDD Red).

검증:
- T1: 모듈 임포트
- T2: compute_start_date(last_date, today) — last_date 그대로 반환 (포함 재수집)
- T3: filter_records_by_date_range — 구간 필터
- T4: update_amount_for_period 흐름 (mock)
"""
import os
import sys
from unittest.mock import patch, MagicMock

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
    import add_daily_stocks as ads  # noqa: F401
    # amount 신규 함수 존재 확인
    assert hasattr(ads, "compute_start_date")
    assert hasattr(ads, "filter_records_by_date_range")
    assert hasattr(ads, "update_amount_for_period")


# ---------------------------------------------------------------------------
# T2: compute_start_date — last_date 그대로 (포함 재수집)
# ---------------------------------------------------------------------------

def test_compute_start_date_returns_last_date():
    """last_date 가 어제든 오늘이든 last_date 그대로 (포함 재수집)."""
    import add_daily_stocks as ads
    last = pd.Timestamp("2026-04-29")
    today = pd.Timestamp("2026-04-30")
    assert ads.compute_start_date(last, today) == last


def test_compute_start_date_when_last_is_today():
    """오늘이 last_date 여도 동일하게 last_date 반환 (오늘 데이터 재수집)."""
    import add_daily_stocks as ads
    today = pd.Timestamp("2026-04-30")
    assert ads.compute_start_date(today, today) == today


def test_compute_start_date_when_last_is_in_future():
    """비정상 케이스 (last > today) — last_date 반환 (no-op 구간)."""
    import add_daily_stocks as ads
    last = pd.Timestamp("2026-05-05")
    today = pd.Timestamp("2026-05-01")
    # 함수는 단순히 last 그대로 반환 (호출자에서 더 이상 작업 안 함)
    assert ads.compute_start_date(last, today) == last


# ---------------------------------------------------------------------------
# T3: filter_records_by_date_range
# ---------------------------------------------------------------------------

def test_filter_records_inclusive_range():
    """[start, end] 양 끝 포함."""
    import add_daily_stocks as ads
    records = [
        ("A", pd.Timestamp("2024-01-01"), 100),
        ("A", pd.Timestamp("2024-01-15"), 200),
        ("A", pd.Timestamp("2024-01-31"), 300),
        ("A", pd.Timestamp("2024-02-15"), 400),
    ]
    out = ads.filter_records_by_date_range(
        records, pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-31"),
    )
    assert len(out) == 2
    assert out[0][1] == pd.Timestamp("2024-01-15")
    assert out[1][1] == pd.Timestamp("2024-01-31")


def test_filter_records_empty_range():
    import add_daily_stocks as ads
    records = [("A", pd.Timestamp("2024-01-01"), 100)]
    out = ads.filter_records_by_date_range(
        records, pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-28"),
    )
    assert out == []


def test_filter_records_empty_input():
    import add_daily_stocks as ads
    out = ads.filter_records_by_date_range(
        [], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
    )
    assert out == []


# ---------------------------------------------------------------------------
# T4: update_amount_for_period — 호출 흐름 (mock)
# ---------------------------------------------------------------------------

def test_update_amount_for_period_calls_kiwoom_per_ticker():
    """각 ticker 별로 ka10081 호출하고 결과를 stocks.amount UPDATE."""
    import add_daily_stocks as ads

    fake_token = "TKN"
    tickers = ["005930", "000660"]
    start_date = "2026-04-28"
    end_date = "2026-04-30"

    fake_ka_response = [
        {"dt": "20260430", "trde_prica": "5000000"},
        {"dt": "20260429", "trde_prica": "4000000"},
        {"dt": "20260428", "trde_prica": "3000000"},
        {"dt": "20260427", "trde_prica": "2000000"},  # 범위 밖
    ]

    with patch.object(ads, "fetch_ka10081", return_value=fake_ka_response) as mock_fetch, \
         patch.object(ads, "bulk_update_stocks_amount", return_value=3) as mock_update:
        n_updated = ads.update_amount_for_period(
            fake_token, tickers, start_date, end_date,
        )

    # 종목 수만큼 fetch 호출
    assert mock_fetch.call_count == len(tickers)
    # bulk_update 도 종목 수만큼 호출
    assert mock_update.call_count == len(tickers)
    # 각 호출에서 4월 27일은 빠지고 28~30 만 (3건 × 2종목 = 6)
    assert n_updated == 6
