# TASK - 1500억 N=15 + MA 정배열 필터 + K=3 백테스트

## 작업 우선순위

1. PRD 승인 (현재 단계)
2. 테스트 작성 (Red)
3. 분석 스크립트 구현 (Green)
4. 실행 + 리포트
5. 종합 결론

## 작업 단위 목록

### 단계 1. 계획 수립

- [x] PRD 작성 / TASK 작성 / ma5,ma20 컬럼 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 2. 테스트 작성 (Red)

- [x] T1~T5 (6 테스트) 작성 + Red 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 3. 분석 스크립트 구현 (Green)

- [x] `backtest_n15_k3_ma_filter.py` 작성 + 6 테스트 Green + 4 시나리오 산출
- [x] Reviewer 승인 (APPROVED)

### 단계 4. 종합 결론 (단계 3 리포트 7장에 통합)

- [x] MA 필터 효과 판정 — **악화** (Calmar 41.88 → 2.17, 95% 폭락)
- [x] 매도 규칙 비교 — LIST_EXIT 미세 우위 (2.17 vs 1.93)
- [x] feedback_simple_is_better 와 일관 — whipsaw 폭락 재현
- [x] Reviewer 승인 (APPROVED)

### 전체 작업 완료 ✅

## 의존 관계

- 기존 모듈 재사용:
  - `backtest_top3_indicators.{run_backtest, build_daily_data}`
  - `backtest_5d_amount_filter.compute_5d_filter_signals`
  - `backtest_5d_realistic_k.equity_real_k`

## 현재 진행 상태

- 단계 1 작성 완료, Reviewer 승인 대기
- 사전 위험 고지 메모리: feedback_simple_is_better (Calmar 4.96→1.05 폭락 사례)

## 완료 여부 체크박스 합계

- 단계 1: 3/4 (Reviewer 승인 대기)
- 단계 2: 0/6
- 단계 3: 0/4
- 단계 4: 0/4
