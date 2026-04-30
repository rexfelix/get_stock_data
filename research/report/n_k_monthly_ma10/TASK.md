# TASK - 1500억 (N,K) + 월봉 MA10 위 필터 매트릭스

## 작업 우선순위

1. PRD 승인
2. 테스트 작성 (Red)
3. 분석 스크립트 구현 (Green)
4. 실행 + 결론

## 작업 단위 목록

### 단계 1. 계획 수립

- [x] PRD / TASK 작성 + DB 시작일 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 2. 테스트 작성 (Red)

- [x] T1~T5 (6 테스트) + Red 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 3. 분석 스크립트 구현 (Green)

- [x] `backtest_n_k_monthly_ma10.py` 작성 + 6/6 Green + 12 시나리오
- [x] Reviewer 승인 (APPROVED)

### 단계 4. 종합 결론

- [x] 시기 의존성 극복 실패 / 강세장도 악화 / feedback 4회 검증 / (15,3) 유지
- [x] Reviewer 승인 (APPROVED)

### 전체 작업 완료 ✅

## 의존 관계

- `backtest_top3_indicators.{run_backtest, build_daily_data}`
- `backtest_5d_amount_filter.compute_5d_filter_signals`
- `backtest_5d_realistic_k.equity_real_k`
- `backtest_n15_k3_2019_2023.{estimate_amount_column, load_price_data_estimated}` (재사용)

## 현재 진행 상태

- 단계 1 작성 완료, Reviewer 승인 대기
- 사전 위험: 월봉 MA10 의 stable 특성이 whipsaw 회피 효과 있을지 검증

## 완료 여부 체크박스 합계

- 단계 1: 2/3 (승인 대기)
- 단계 2: 0/6
- 단계 3: 0/4
- 단계 4: 0/5
