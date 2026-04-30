# TASK - (N=15, K=3) 실전 운영성 검증

## 작업 우선순위

1. PRD 승인 (현재 단계)
2. 테스트 작성 (Red)
3. 분석 스크립트 구현 (Green)
4. 실행 + 리포트
5. 종합 결론

## 작업 단위 목록

### 단계 1. 계획 수립

- [x] PRD 작성 (`report/n15_k3_validation/PRD.md`)
- [x] TASK 작성 (`report/n15_k3_validation/TASK.md`)
- [x] Reviewer 승인 (APPROVED)

### 단계 2. 테스트 작성 (Red)

- [x] T1~T4 (6 테스트) 작성 완료
- [x] Red 확인: 6 failed (ModuleNotFoundError)
- [x] Reviewer 승인 (APPROVED)

### 단계 3. 분석 스크립트 구현 (Green)

- [x] `backtest_n15_k3_validation.py` 작성
- [x] T1~T4 통과 확인 (6 passed)
- [x] 실제 실행으로 결과 산출 완료
- [x] Reviewer 승인 (APPROVED)

### 단계 4. 종합 결론 (단계 3 리포트 6장에 통합)

- [x] (15,3) 채택 가능성 정량 평가
- [x] 무거래 구간 / 종목 집중도 / Stress Test 결과로 risk profile 명시
- [x] (5,5) 와 trade-off 정량 비교
- [x] Reviewer 승인 (APPROVED)

### 전체 작업 완료 ✅

## 의존 관계

- F1 (데이터 추출) → F2~F5 (분석) → F6 (리포트)
- 기존 모듈 재사용:
  - `backtest_top3_indicators.{run_backtest, build_daily_data, ...}`
  - `backtest_5d_amount_filter.compute_5d_filter_signals(threshold_won=150e9)`
  - `backtest_5d_realistic_k.equity_real_k`

## 현재 진행 상태

- 단계 1 (계획 수립) PRD/TASK 작성 완료, Reviewer 승인 대기
- 단계 2~4 는 단계 1 승인 후 순차 진행

## 완료 여부 체크박스 합계

- 단계 1: 2/3 (Reviewer 승인 대기)
- 단계 2: 0/5
- 단계 3: 0/4
- 단계 4: 0/4
