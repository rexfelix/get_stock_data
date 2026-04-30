# TASK - 1500억 (15,3) + 시총 캡 사전 제외 매트릭스

## 작업 우선순위

1. PRD 승인 (현재 단계)
2. 테스트 작성 (Red)
3. 분석 스크립트 구현 (Green)
4. 실행 + 결론

## 작업 단위 목록

### 단계 1. 계획 수립

- [x] PRD / TASK 작성
- [x] daily_data["mcap"] 컬럼 존재 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 2. 테스트 작성 (Red)

- [x] T1~T5 (7 테스트) 작성 + Red 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 3. 분석 스크립트 구현 (Green)

- [x] `backtest_n15_k3_mcap_cap.py` 작성 + 7/7 Green + 5 cap 매트릭스 산출
- [ ] **Reviewer 승인 대기**

### 단계 4. 종합 결론 (단계 3 리포트에 통합)

- [x] 최적 cap 임계 식별 — 모든 cap 시나리오가 베이스(∞) Calmar 41.88 미달
- [x] 005930 매수 변화 — 베이스 1회 → 50조 cap부터 0회
- [x] Trade-off 분석 — 005930 제외 자체가 시스템 약화
- [ ] **결론 Reviewer 승인 대기**

## 의존 관계

- `backtest_top3_indicators.{run_backtest, build_daily_data}` (mcap 컬럼)
- `backtest_5d_amount_filter.compute_5d_filter_signals`
- `backtest_5d_realistic_k.equity_real_k`

## 현재 진행 상태

- 단계 1 완료, Reviewer 승인 대기
- 사전 위험: 본 검증은 universe 변경 (whipsaw 위험 없음)

## 완료 여부 체크박스 합계

- 단계 1: 2/3 (승인 대기)
- 단계 2: 0/6
- 단계 3: 0/4
- 단계 4: 0/4
