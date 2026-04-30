# REVIEW - 3000억 N/N + K슬롯 N×K 매트릭스 백테스트

## 리뷰 이력

(Reviewer 승인 시 누적 기록)

---

### 리뷰 #001 — 단계 1 계획 수립

- **리뷰 일시**: 2026-05-01
- **리뷰 대상 단계**: 1단계 (계획 수립)
- **검토 범위**:
  - `report/n_k_matrix_3000/PRD.md`
  - `report/n_k_matrix_3000/TASK.md`
- **요구사항 충족 여부**: 충족 (3000억 임계, 동일 매트릭스 그리드, 신규 파일 산출)
- **TDD 준수 여부**: 단계 1은 계획 수립 단계, 다음 단계에서 테스트 우선 작성 약속됨
- **발견 사항**:
  - PRD 범위/비범위 명확
  - 1500억 vs 3000억 비교 표 추가 의도 적절
  - 가설(신호 빈도 감소, N=15 위험)을 정량 검증할 항목 T1~T5 제시됨
- **수정 요청 사항**: 없음
- **결정**: **APPROVED**

---

### 리뷰 #002 — 단계 2 테스트 작성 (Red)

- **리뷰 일시**: 2026-05-01
- **리뷰 대상 단계**: 2단계 (테스트 작성)
- **검토 범위**:
  - `tests/test_n_k_matrix_3000.py` (신규)
- **요구사항 충족 여부**: 충족 (T1~T5 PRD 검증 항목과 1:1 일치)
- **TDD 준수 여부**:
  - 모듈 미존재 상태에서 테스트 작성 → 임포트 의존 4개 테스트 `ModuleNotFoundError` 로 Red 확인
  - 기존 함수 회귀 검증(T3)은 작성 즉시 Green (1500억 vs 3000억 신호 감소 가설 직접 검증)
- **테스트 실행 결과 (Red)**:

  ```text
  test_module_can_be_imported FAILED
  test_threshold_constant_is_3000eok FAILED
  test_3000eok_threshold_filters_more_strictly_than_1500eok PASSED
  test_matrix_grid_is_3_5_7_10_15_squared FAILED
  test_output_md_path FAILED
  4 failed, 1 passed
  ```

- **결정**: **APPROVED**

---

### 리뷰 #003 — 단계 3 최소 구현 (Green) + 단계 4 리팩토링 + 단계 5 리포트

- **리뷰 일시**: (대기 중)
- **리뷰 대상 단계**: 3 (구현), 4 (리팩토링 판단), 5 (리포트 강화)
- **검토 범위**:
  - `backtest_n_k_matrix_3000.py` (신규)
  - `results/backtest_n_k_matrix_3000.md` (신규)
- **요구사항 충족 여부**: (Reviewer 판정)
- **TDD 준수 여부**:
  - Red→Green 전환 확인: 5/5 passed
- **테스트 실행 결과 (Green)**:

  ```text
  test_module_can_be_imported PASSED
  test_threshold_constant_is_3000eok PASSED
  test_3000eok_threshold_filters_more_strictly_than_1500eok PASSED
  test_matrix_grid_is_3_5_7_10_15_squared PASSED
  test_output_md_path PASSED
  5 passed
  ```

- **백테스트 산출 요약**:
  - 데이터: KOSPI200 199종목, 2023~현재
  - N별 신호 일수 / 일별 평균: N=3 → 628일 평균 3.27 / N=15 → 616일 평균 2.13
  - **Calmar 최고**: N=7/K=3 → 11.59 (CAGR +75.4% / MDD -6.5%)
  - **CAGR 최고**: N=3/K=3 → +89.2% (MDD -10.6%)
  - **MDD 최저**: N=15/K=15 → -1.93% (CAGR +16.4%)
  - **N=K 대각선**: 8.45 / 6.49 / 3.51 / 8.56 / 8.51
- **1500억 대비 비교 핵심**:
  - K=3 열 전체에서 1500억이 압도 우위 (3000억 K=3 Calmar 6~12 vs 1500억 23~42)
  - **K=10~15, N=10~15 구간에서 3000억 우위** (예: N=10/K=10 8.56 vs 7.18, N=10/K=15 8.98 vs 6.51)
  - 미체결 합계: 1500억 241 → 3000억 39 (cap 효율 ✅)
  - 거래수 합계 큰 폭 감소 (특히 N=10~15)
  - Calmar 우월 영역: 25개 중 9개에서 3000억 우위
- **리팩토링 판단**: 보류 (단순 추출 어려움 + 향후 다임계 비교 시 공통화 권장)
- **발견 사항**:
  - 임계 상승은 일률적 개선이 아니라 **(K, N) 영역별로 효과가 갈림**
  - K=3 열은 1500억이 압도, N=10~15 × K=10~15 영역에서 3000억 우위
  - 미체결 합계 241 → 39 로 cap 효율 ✅
  - N=15 거래수 합 78건은 매트릭스적 의미는 유지하되 일반화에는 한계
- **수정 요청 사항**: 없음
- **결정**: **APPROVED**
