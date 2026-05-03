"""K-Tide 10 슬롯 선택 기준 비교 단위 테스트 (TDD Red).

검증:
- T-1: entry_date_to_annual_year — 4월 1일 lookahead 컷오프
- T-2: calc_yoy — abs 분모 + 결측/0 처리 + 흑자/적자전환
- T-3: rank_candidates — NaN 후순위 + amount fallback (filter 가 되지 않음)
- T-4: composite_zscore — 3개 컬럼 모두 가용 row 만 z-score
- T-5: count_ranking_changes — base picks vs scenario picks 차이 카운트

이 테스트는 backtest_finance_ranking.py 가 존재하지 않을 때 Red.
3단계 구현이 끝나면 Green.
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


# ---------------------------------------------------------------------------
# T-0: 모듈 임포트 + 핵심 함수 존재
# ---------------------------------------------------------------------------

def test_module_can_be_imported():
    import backtest_finance_ranking as m
    assert hasattr(m, "entry_date_to_annual_year")
    assert hasattr(m, "calc_yoy")
    assert hasattr(m, "rank_candidates")
    assert hasattr(m, "composite_zscore")
    assert hasattr(m, "count_ranking_changes")


# ---------------------------------------------------------------------------
# T-1: entry_date_to_annual_year — 4월 1일 컷오프
# ---------------------------------------------------------------------------

def test_entry_date_april_cutoff_before():
    """4월 1일 이전 진입 → year - 2 의 annual 사용 (보수적)."""
    import backtest_finance_ranking as m
    assert m.entry_date_to_annual_year(date(2024, 3, 31)) == 2022


def test_entry_date_april_cutoff_on():
    """4월 1일 진입 → year - 1 의 annual 사용."""
    import backtest_finance_ranking as m
    assert m.entry_date_to_annual_year(date(2024, 4, 1)) == 2023


def test_entry_date_year_end():
    """12월 31일 진입 → 같은 해 - 1 의 annual 사용."""
    import backtest_finance_ranking as m
    assert m.entry_date_to_annual_year(date(2024, 12, 31)) == 2023


def test_entry_date_next_year_april():
    import backtest_finance_ranking as m
    assert m.entry_date_to_annual_year(date(2025, 4, 1)) == 2024


def test_entry_date_accepts_pandas_timestamp():
    """pandas.Timestamp 입력도 동일하게 동작."""
    import backtest_finance_ranking as m
    assert m.entry_date_to_annual_year(pd.Timestamp("2025-04-01")) == 2024


# ---------------------------------------------------------------------------
# T-2: calc_yoy — abs 분모 + 결측/0 처리
# ---------------------------------------------------------------------------

def test_calc_yoy_basic_growth():
    import backtest_finance_ranking as m
    assert m.calc_yoy(110, 100) == pytest.approx(0.10)


def test_calc_yoy_negative_to_negative_improvement():
    """적자 호전: prev=-100, curr=-50 → abs 분모로 +0.5 (호전 = 양수)."""
    import backtest_finance_ranking as m
    assert m.calc_yoy(-50, -100) == pytest.approx(0.5)


def test_calc_yoy_turnaround_positive():
    """흑자전환: prev=-100, curr=50 → abs 분모로 +1.5 (큰 양수)."""
    import backtest_finance_ranking as m
    assert m.calc_yoy(50, -100) == pytest.approx(1.5)


def test_calc_yoy_turnaround_negative():
    """적자전환: prev=100, curr=-50 → -1.5 (큰 음수)."""
    import backtest_finance_ranking as m
    assert m.calc_yoy(-50, 100) == pytest.approx(-1.5)


def test_calc_yoy_zero_prev_returns_nan():
    import backtest_finance_ranking as m
    assert pd.isna(m.calc_yoy(50, 0))


def test_calc_yoy_nan_input_returns_nan():
    import backtest_finance_ranking as m
    assert pd.isna(m.calc_yoy(np.nan, 100))
    assert pd.isna(m.calc_yoy(50, np.nan))
    assert pd.isna(m.calc_yoy(np.nan, np.nan))


# ---------------------------------------------------------------------------
# T-3: rank_candidates — NaN 후순위 + fallback (filter 효과 없음)
# ---------------------------------------------------------------------------

def test_rank_candidates_normal_values_sorted_desc():
    """정상값 종목들은 key 컬럼 내림차순으로 정렬."""
    import backtest_finance_ranking as m
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "growth": [0.10, 0.30, 0.20],
        "amount": [1000, 2000, 3000],
    })
    out = m.rank_candidates(df, key_col="growth", fallback_col="amount")
    assert list(out["ticker"]) == ["B", "C", "A"]


def test_rank_candidates_nan_demoted_to_bottom():
    """NaN row 는 정상값 row 보다 항상 하위. NaN row 자체 수는 보존됨 (filter 아님)."""
    import backtest_finance_ranking as m
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "growth": [0.10, np.nan, 0.30, np.nan],
        "amount": [1000, 5000, 2000, 4000],
    })
    out = m.rank_candidates(df, key_col="growth", fallback_col="amount")
    # 정상값 (C: 0.30, A: 0.10) 우선, 그 다음 NaN (B/D 는 amount DESC: B=5000, D=4000)
    assert list(out["ticker"]) == ["C", "A", "B", "D"]
    # 후보 풀 자체 수는 4 그대로 유지 — filter 효과 없음
    assert len(out) == 4


def test_rank_candidates_tiebreaker_uses_fallback():
    """key 동률 시 fallback (amount) DESC 로 결정."""
    import backtest_finance_ranking as m
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "growth": [0.20, 0.20, 0.20],
        "amount": [1000, 3000, 2000],
    })
    out = m.rank_candidates(df, key_col="growth", fallback_col="amount")
    assert list(out["ticker"]) == ["B", "C", "A"]


def test_rank_candidates_all_nan_falls_back_to_amount():
    """모두 NaN → fallback (amount) DESC 단독 사용."""
    import backtest_finance_ranking as m
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "growth": [np.nan, np.nan, np.nan],
        "amount": [1000, 3000, 2000],
    })
    out = m.rank_candidates(df, key_col="growth", fallback_col="amount")
    assert list(out["ticker"]) == ["B", "C", "A"]


# ---------------------------------------------------------------------------
# T-4: composite_zscore — 3컬럼 모두 가용 row 만 z-score
# ---------------------------------------------------------------------------

def test_composite_zscore_all_present():
    """모든 컬럼이 가용한 row 는 z-score 평균을 받음."""
    import backtest_finance_ranking as m
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "rev": [0.10, 0.20, 0.30],
        "op": [0.10, 0.20, 0.30],
        "eps": [0.10, 0.20, 0.30],
    })
    out = m.composite_zscore(df, cols=["rev", "op", "eps"])
    # B 가 평균이므로 z=0, C 는 양수, A 는 음수
    assert "composite" in out.columns
    assert out.loc[out["ticker"] == "B", "composite"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out.loc[out["ticker"] == "C", "composite"].iloc[0] > 0
    assert out.loc[out["ticker"] == "A", "composite"].iloc[0] < 0


def test_composite_zscore_partial_nan_row_is_nan():
    """한 컬럼이라도 NaN 인 row 는 composite NaN."""
    import backtest_finance_ranking as m
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "rev": [0.10, 0.20, np.nan],
        "op": [0.10, 0.20, 0.30],
        "eps": [0.10, 0.20, 0.30],
    })
    out = m.composite_zscore(df, cols=["rev", "op", "eps"])
    assert pd.isna(out.loc[out["ticker"] == "C", "composite"].iloc[0])
    assert not pd.isna(out.loc[out["ticker"] == "A", "composite"].iloc[0])
    assert not pd.isna(out.loc[out["ticker"] == "B", "composite"].iloc[0])


def test_composite_zscore_preserves_row_count():
    """입력 행 수 == 출력 행 수 (filter 가 아님)."""
    import backtest_finance_ranking as m
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "rev": [0.10, np.nan, 0.30, 0.40],
        "op": [0.10, 0.20, 0.30, 0.40],
        "eps": [0.10, 0.20, 0.30, np.nan],
    })
    out = m.composite_zscore(df, cols=["rev", "op", "eps"])
    assert len(out) == 4


# ---------------------------------------------------------------------------
# T-5: count_ranking_changes — base vs scenario picks 차이 카운트
# ---------------------------------------------------------------------------

def test_count_ranking_changes_identical_picks():
    """동일 선택 → 차이 0."""
    import backtest_finance_ranking as m
    base = [("2024-01-02", ("A", "B", "C"))]
    scen = [("2024-01-02", ("A", "B", "C"))]
    diff_days, total_days = m.count_ranking_changes(base, scen)
    assert diff_days == 0
    assert total_days == 1


def test_count_ranking_changes_one_swap():
    """하루의 picks 가 다르면 1로 카운트 (일 단위)."""
    import backtest_finance_ranking as m
    base = [("2024-01-02", ("A", "B", "C"))]
    scen = [("2024-01-02", ("A", "B", "D"))]
    diff_days, total_days = m.count_ranking_changes(base, scen)
    assert diff_days == 1
    assert total_days == 1


def test_count_ranking_changes_order_insensitive():
    """같은 종목 집합이면 순서가 달라도 동일로 간주 (집합 비교)."""
    import backtest_finance_ranking as m
    base = [("2024-01-02", ("A", "B", "C"))]
    scen = [("2024-01-02", ("C", "A", "B"))]
    diff_days, total_days = m.count_ranking_changes(base, scen)
    assert diff_days == 0


def test_count_ranking_changes_multiple_days():
    import backtest_finance_ranking as m
    base = [
        ("2024-01-02", ("A", "B", "C")),
        ("2024-01-03", ("A", "B", "C")),
        ("2024-01-04", ("A", "B", "C")),
    ]
    scen = [
        ("2024-01-02", ("A", "B", "C")),  # 동일
        ("2024-01-03", ("A", "B", "D")),  # 차이 1
        ("2024-01-04", ("A", "X", "Y")),  # 차이 1
    ]
    diff_days, total_days = m.count_ranking_changes(base, scen)
    assert diff_days == 2
    assert total_days == 3
