"""3000억 N/N + K슬롯 N×K 매트릭스 백테스트 단위 테스트 (TDD Red 단계).

검증 대상:
- T1: backtest_n_k_matrix_3000 모듈이 존재하고 임포트 가능
- T2: THRESHOLD_WON 상수가 정확히 3000억(300_000_000_000)
- T3: 동일 daily_data 입력에서 3000억 임계 신호 < 1500억 임계 신호 (감소 검증)
- T4: N×K 매트릭스 그리드가 N=[3,5,7,10,15] × K=[3,5,7,10,15] 25 조합인지
- T5: OUTPUT_MD 경로가 results/backtest_n_k_matrix_3000.md 로 설정됨

이 테스트들은 backtest_n_k_matrix_3000 모듈 구현 후 모두 통과해야 한다.
"""
import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


# ---------------------------------------------------------------------------
# T1: 모듈 임포트 가능 여부
# ---------------------------------------------------------------------------

def test_module_can_be_imported():
    """backtest_n_k_matrix_3000 모듈이 존재해야 한다."""
    import backtest_n_k_matrix_3000  # noqa: F401


# ---------------------------------------------------------------------------
# T2: 임계치 상수
# ---------------------------------------------------------------------------

def test_threshold_constant_is_3000eok():
    """THRESHOLD_WON == 300,000,000,000 (3000억원)."""
    import backtest_n_k_matrix_3000 as m
    assert m.THRESHOLD_WON == 300_000_000_000


# ---------------------------------------------------------------------------
# T3: 임계치 상승에 따른 신호 감소
# ---------------------------------------------------------------------------

def _make_daily_data() -> dict[str, pd.DataFrame]:
    """A: 매일 2000억(1500억 통과/3000억 실패),
       B: 매일 4000억(둘 다 통과),
       C: 매일 1000억(둘 다 실패)."""
    dates = pd.to_datetime([f"2026-01-{d:02d}" for d in [2, 3, 6, 7, 8, 9, 10, 13, 14, 15]])
    base = {"date": dates, "open": [100.0] * 10, "close": [100.0] * 10,
            "high": [101.0] * 10, "low": [99.0] * 10, "shares": [1000] * 10}
    df_a = pd.DataFrame({**base, "amount": [200_000_000_000] * 10})  # 2000억
    df_b = pd.DataFrame({**base, "amount": [400_000_000_000] * 10})  # 4000억
    df_c = pd.DataFrame({**base, "amount": [100_000_000_000] * 10})  # 1000억
    return {"A": df_a, "B": df_b, "C": df_c}


def test_3000eok_threshold_filters_more_strictly_than_1500eok():
    """동일 daily_data로 1500억 vs 3000억 신호 비교: 3000억이 더 엄격."""
    from backtest_5d_amount_filter import compute_5d_filter_signals

    daily_data = _make_daily_data()

    # 1500억 임계: A(2000억), B(4000억) 모두 통과 → 매일 2종목
    signals_1500 = compute_5d_filter_signals(
        daily_data, threshold_won=150_000_000_000, lookback=5, top_k=200,
    )
    # 3000억 임계: B(4000억)만 통과 → 매일 1종목
    signals_3000 = compute_5d_filter_signals(
        daily_data, threshold_won=300_000_000_000, lookback=5, top_k=200,
    )

    # 신호 발생 일수는 같을 수 있으나, 일별 신호 수는 줄어들어야 한다
    avg_1500 = sum(len(v) for v in signals_1500.values()) / max(len(signals_1500), 1)
    avg_3000 = sum(len(v) for v in signals_3000.values()) / max(len(signals_3000), 1)
    assert avg_3000 < avg_1500, (
        f"3000억 신호 평균({avg_3000})이 1500억({avg_1500})보다 작아야 한다"
    )

    # 3000억 신호 종목은 B 하나뿐이어야 한다
    unique_3000 = set(t for v in signals_3000.values() for t in v)
    assert unique_3000 == {"B"}, f"3000억 신호 종목 = {unique_3000}, 기대 = {{'B'}}"


# ---------------------------------------------------------------------------
# T4: 매트릭스 그리드
# ---------------------------------------------------------------------------

def test_matrix_grid_is_3_5_7_10_15_squared():
    """N_VALUES, K_VALUES 가 [3,5,7,10,15] 로 고정."""
    import backtest_n_k_matrix_3000 as m
    assert m.N_VALUES == [3, 5, 7, 10, 15]
    assert m.K_VALUES == [3, 5, 7, 10, 15]


# ---------------------------------------------------------------------------
# T5: 결과 출력 경로
# ---------------------------------------------------------------------------

def test_output_md_path():
    """OUTPUT_MD 가 results/backtest_n_k_matrix_3000.md 로 설정."""
    import backtest_n_k_matrix_3000 as m
    assert m.OUTPUT_MD.endswith("results/backtest_n_k_matrix_3000.md")
