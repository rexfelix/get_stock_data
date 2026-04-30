# TASK - 1500억 (15,3) 2019~2023 백테스트 (amount 추정)

## 작업 우선순위

1. PRD 승인 (현재 단계)
2. 테스트 작성 (Red)
3. 분석 스크립트 구현 (Green)
4. 실행 + 결론

## 작업 단위 목록

### 단계 1. 계획 수립

- [x] PRD / TASK 작성 + 데이터 범위 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 2. 테스트 작성 (Red)

- [x] T1~T5 (6 테스트) 작성 + Red 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 3. 분석 스크립트 구현 (Green)

- [x] `backtest_n15_k3_2019_2023.py` 작성 + 6/6 Green + 결과 산출
- [x] Reviewer 승인 (APPROVED)

### 단계 4. 종합 결론

- [x] 시기 의존성 판정 — **H2 강력 지지** (Calmar 41.88 → 0.29)
- [x] (15,3) 채택 결정 **재검토 필요** 명시
- [x] Reviewer 승인 (APPROVED)

### 전체 작업 완료 ✅

## 의존 관계

- `backtest_top3_indicators.{run_backtest, build_daily_data}` (mcap 컬럼 그대로)
- `backtest_5d_amount_filter.compute_5d_filter_signals`
- `backtest_5d_realistic_k.equity_real_k`
- 신규 데이터 로드: stocks 직접 query (stock_all join 제거)

## 현재 진행 상태

- 단계 1 작성 완료, Reviewer 승인 대기
- 데이터 한계: amount 추정 (close × volume), mcap 시계열 한계 명시

## 완료 여부 체크박스 합계

- 단계 1: 2/3 (승인 대기)
- 단계 2: 0/5
- 단계 3: 0/4
- 단계 4: 0/4
