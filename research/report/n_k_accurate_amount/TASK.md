# TASK - 정확 amount (15,3)·(10,10)·(15,15) 2019~2023 재검증

## 작업 우선순위

1. PRD 승인
2. 테스트 작성 (Red)
3. 구현 + 6 시나리오 실행 (Green)
4. 결과 분석 + 결론

## 작업 단위 목록

### 단계 1. 계획 수립

- [x] PRD / TASK 작성
- [x] amount_backfill 결과 (99.0% 채움) 확인
- [ ] **Reviewer 승인 대기**

### 단계 2. 테스트 작성 (Red)

- [x] T1~T4 (4 테스트) + Red 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 3. 구현 + 실행 (Green)

- [x] `backtest_n_k_accurate_amount.py` 작성 + 4/4 Green
- [x] 6 시나리오 실행 완료
- [ ] **Reviewer 승인 대기**

### 단계 4. 종합 결론

- [x] (10,10) 시기 robust 1위 확정 (Calmar 1.18, 정확값)
- [x] (15,3) 시기 의존성 확정 (Calmar 0.49, 정확값에서도 1 미만)
- [x] (15,3) 채택 → (10,10) 변경 권고
- [ ] **결론 Reviewer 승인 대기**

## 의존 관계

- `backtest_top3_indicators.{run_backtest, build_daily_data}` (mcap)
- `backtest_5d_amount_filter.compute_5d_filter_signals`
- `backtest_5d_realistic_k.equity_real_k`
- DB stocks.amount (백만원 단위, 99.0% 채움)

## 현재 진행 상태

- 단계 1 작성 완료, Reviewer 승인 대기

## 완료 여부 체크박스 합계

- 단계 1: 2/3 (승인 대기)
- 단계 2: 0/5
- 단계 3: 0/3
- 단계 4: 0/3
