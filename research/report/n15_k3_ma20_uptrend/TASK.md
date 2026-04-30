# TASK - 1500억 N=15 + MA20 5일 상승 필터 + K=3

## 작업 우선순위

1. PRD 승인 (현재 단계)
2. 테스트 작성 (Red)
3. 분석 스크립트 구현 (Green)
4. 실행 + 결론

## 작업 단위 목록

### 단계 1. 계획 수립

- [x] PRD / TASK 작성
- [x] Reviewer 승인 (APPROVED)

### 단계 2. 테스트 작성 (Red)

- [x] T1~T5 (10 테스트) 작성 + Red 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 3. 분석 스크립트 구현 (Green)

- [x] `backtest_n15_k3_ma20_uptrend.py` 작성 + 10/10 Green + 결과 산출
- [x] Reviewer 승인 (APPROVED)

### 단계 4. 종합 결론 (단계 3 리포트에 통합)

- [x] MA20 추세 필터 효과 — **악화** (Calmar 41.88→6.72, -84%)
- [x] 정배열 필터(-95%) 와 비교 — 본 필터 -84%
- [x] feedback memory 연결 — whipsaw 3회 검증 일관
- [x] Reviewer 승인 (APPROVED)

### 전체 작업 완료 ✅

## 의존 관계

- `backtest_top3_indicators.{run_backtest, build_daily_data}` (ma20 컬럼 보장)
- `backtest_5d_amount_filter.compute_5d_filter_signals`
- `backtest_5d_realistic_k.equity_real_k`

## 현재 진행 상태

- 단계 1 작성 완료, Reviewer 승인 대기
- 사전 위험 고지 (memory feedback_simple_is_better) 확인됨

## 완료 여부 체크박스 합계

- 단계 1: 2/3 (Reviewer 승인 대기)
- 단계 2: 0/6
- 단계 3: 0/4
- 단계 4: 0/4
