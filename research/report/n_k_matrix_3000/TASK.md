# TASK - 3000억 N/N + K슬롯 N×K 매트릭스 백테스트

## 작업 우선순위

1. PRD 승인 (현재 단계)
2. 테스트 작성 (Red)
3. 최소 구현 (Green)
4. 리팩토링
5. 결과 검증 및 리포트

## 작업 단위 목록

### 단계 1. 계획 수립

- [x] PRD 작성 (`report/n_k_matrix_3000/PRD.md`)
- [x] TASK 작성 (`report/n_k_matrix_3000/TASK.md`)
- [x] Reviewer 승인 (APPROVED)

### 단계 2. 테스트 작성 (Red)

- [x] T1: `backtest_n_k_matrix_3000` 모듈 임포트 가능 검증 (현재 미존재 → Red)
- [x] T2: `THRESHOLD_WON == 300_000_000_000` 상수 검증 (현재 미존재 → Red)
- [x] T3: 1500억 vs 3000억 동일 daily_data 입력에서 신호 수 감소 검증 (기존 함수 회귀 → 작성 시점부터 Green)
- [x] T4: `N_VALUES == K_VALUES == [3,5,7,10,15]` 그리드 고정 검증 (현재 미존재 → Red)
- [x] T5: `OUTPUT_MD` 가 `results/backtest_n_k_matrix_3000.md` 로 설정 검증 (현재 미존재 → Red)
- [x] `tests/test_n_k_matrix_3000.py` 작성 완료
- [x] 실패 확인: T1/T2/T4/T5 → `ModuleNotFoundError`, T3 → 기존 함수 검증 통과
- [x] Reviewer 승인 (APPROVED)

### 단계 3. 최소 구현 (Green)

- [x] `backtest_n_k_matrix_3000.py` 작성
- [x] T1~T5 통과 확인 (5 passed)
- [x] 실제 실행으로 25 조합 결과 산출 완료
- [x] Reviewer 승인 (APPROVED)

### 단계 4. 리팩토링 (실행 결과 기반 판단)

- [x] 기존 `backtest_n_k_matrix.py` 와의 중복 코드 식별
- [x] 결론: 현재 범위에서는 리팩토링 보류
- [x] Reviewer 승인 (APPROVED — 다임계 비교 추가 시 재평가)

### 단계 5. 리포트 강화 (단계 3 구현에 통합 완료)

- [x] 1500억 vs 3000억 비교 표 추가 (Calmar/CAGR/MDD/거래수/미체결 Δ)
- [x] 종합 고찰 (가설 1/2/3 + Calmar 우월 영역) 추가
- [x] Reviewer 승인 (APPROVED)

## 의존 관계

- F1 (신호 생성) → F2 (백테스트) → F3 (리포트)
- 기존 모듈 의존:
  - `backtest_top3_indicators` (load_kospi200_tickers, load_market_cap_snapshot, load_price_data, build_daily_data, build_daily_indicator_panel, run_backtest, compute_stats)
  - `backtest_5d_amount_filter.compute_5d_filter_signals`
  - `backtest_5d_realistic_k.equity_real_k`
- 비교 대상: `results/backtest_n_k_matrix.md` (1500억 결과)

## 현재 진행 상태

- 단계 1 (계획 수립) 완료 — APPROVED
- 단계 2 (테스트 작성) 완료 — APPROVED
- 단계 3 (최소 구현) 완료 — APPROVED
- 단계 4 (리팩토링) 보류 결정 — APPROVED
- 단계 5 (리포트 강화) 단계 3에 통합 완료 — APPROVED
- 전체 작업 완료 ✅

## 완료 여부 체크박스 합계

- 단계 1: 3/3 ✅
- 단계 2: 8/8 ✅
- 단계 3: 4/4 ✅
- 단계 4: 3/3 ✅
- 단계 5: 3/3 ✅
