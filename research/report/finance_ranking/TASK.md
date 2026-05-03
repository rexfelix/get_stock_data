# TASK - K-Tide 10 슬롯 선택 기준 비교 (거래대금 vs 재무 증가율)

## 작업 목록 (PDCA 5단계)

### 1단계. 계획 수립 (현재 단계, In Progress)

- [x] 데이터 source 사전 점검 (`financial_summary`, `financial_annual` 스키마/가용 연도)
- [x] K-Tide 10 trade list 생성 경로 확인 (`backtest_n_k_accurate_amount.py`)
- [x] 후보 풀 크기 분포 사전 측정 (강세장 7.1% / 약세장 2.5% > 10)
- [x] 메모리 위반 점검: 후보 풀 변경 없음 + 매도 규칙 변경 없음 → "매수 추가 필터" / "매도 규칙 변경" 함정 회피 확인
- [x] 메모리 선례 인지: "거래대금 turnover 정렬 → Calmar 1.86" 부진 가능성을 PRD 위험으로 명시
- [x] `report/PRD.md` 작성 (요구사항 정정 1회 반영)
- [x] `report/TASK.md` 작성 (본 문서)
- [x] `report/REVIEW.md` 초기화
- [ ] Reviewer 에게 1단계 리뷰 요청 (멈춤)

**1단계 종료 조건**: PRD/TASK/REVIEW 가 최신 상태 + Reviewer APPROVED.

---

### 2단계. 테스트 작성 (Red, In Progress)

- [x] `research/tests/test_finance_ranking.py` 생성 (23개 테스트)
- [x] T-0: 모듈 임포트 + 핵심 함수 5개 존재 확인 (1)
- [x] T-1: `entry_date_to_annual_year(D)` 단위 테스트 (5)
  - 2024-03-31 → 2022
  - 2024-04-01 → 2023
  - 2024-12-31 → 2023
  - 2025-04-01 → 2024
  - pandas.Timestamp 입력 호환
- [x] T-2: `calc_yoy(curr, prev)` 단위 테스트 abs 분모 (6)
  - (110, 100) → 0.10
  - (-50, -100) → +0.5 (적자 호전)
  - (50, -100) → +1.5 (흑자전환, 큰 양수)
  - (-50, 100) → -1.5 (적자전환, 큰 음수)
  - prev=0 → NaN
  - 입력 NaN → NaN (3 가지 패턴)
- [x] T-3: `rank_candidates(df, key_col, fallback_col)` 단위 테스트 (4)
  - 정상값 DESC
  - NaN 후순위 강등 + fallback (filter 효과 없음, len 보존)
  - 정상값 동률 → fallback DESC
  - 모두 NaN → fallback DESC
- [x] T-4: `composite_zscore(df, cols)` 단위 테스트 (3)
  - 정상 row 모두 z-score (B=평균=0)
  - 한 컬럼 NaN row → composite NaN
  - 입력 행수 == 출력 행수 (filter 아님)
- [x] T-5: `count_ranking_changes(base_picks, scenario_picks)` 단위 테스트 (4)
  - 동일 선택 → 0
  - 한 종목 교체 → 1
  - 순서만 다름 → 0 (집합 비교)
  - 다중 일자 정확 카운트
- [x] 모든 테스트 실패 확인 (Red): **23/23 failed** (`ModuleNotFoundError: No module named 'backtest_finance_ranking'`)
- [ ] Reviewer 에게 2단계 리뷰 요청 (멈춤)

**2단계 종료 조건**: 실패 테스트 존재 + 실패 확인 + Reviewer APPROVED.

---

### 3단계. 최소 구현 (Green, In Progress)

- [x] `research/backtest_finance_ranking.py` 생성
- [x] 헬퍼 함수 구현 (테스트 통과 최소 범위):
  - [x] `entry_date_to_annual_year(date)`
  - [x] `calc_yoy(curr, prev)`
  - [x] `rank_candidates(df, key_col, fallback_col)`
  - [x] `composite_zscore(df, cols)`
  - [x] `count_ranking_changes(base_picks, scenario_picks)`
- [x] DB 로드:
  - [x] `financial_summary` (is_estimate=False, revenue_yoy 포함)
  - [x] `financial_annual` (quarter=0, fallback)
  - [x] op_income / eps YoY 계산 (Y vs Y-1, abs 분모)
- [x] K-Tide 10 백테스트 엔진 ranking 주입:
  - [x] 기존 엔진 `bt.run_backtest` 가 `signals[date]` 리스트 순서로 매수 우선순위 결정 확인
  - [x] `build_signals(pool_with_fin, ranking_key, fallback_key)` 로 ranking 주입
  - [x] **BASE sanity check 통과**: 강세장 Calmar 5.05 / CAGR +44.77% / MDD -8.86%, 약세장 1.18 / +10.43% / -8.80% — 메모리 `reference_k_tide_10.md` 와 **정확히 일치**
- [x] 5개 시나리오 × 2 기간 실행:
  - [x] BASE (amount DESC), S1 (revenue), S2 (op_income), S3 (eps), S4 (composite)
  - [x] 강세장 (2023-01-01 ~ 2026-12-31) + 약세장 (2019-01-01 ~ 2023-12-31)
- [x] 시나리오별 산출:
  - [x] CAGR / MDD / Calmar
  - [x] 거래수 / 승률 / 평균 보유기간
  - [x] 정렬 변경 일수 / 비율
- [x] 리포트: `research/results/backtest_finance_ranking.md`
  - [x] 헤더 + 메타 (실행 일시, 후보 풀 통계, 매칭률)
  - [x] 강세장 / 약세장 표 (절대값 + Δ vs BASE)
  - [x] ranking 변경 비율 표
  - [x] 한국어 해석 + 결론
- [x] 단위 테스트 통과 확인 (23/23 Green)
- [x] 리포트 사후 정정 (실측 4.6% / 약세장 동일 분류 neutral 분기)
- [ ] Reviewer 에게 3단계 리뷰 요청 (멈춤)

**3단계 핵심 결과**:

| 기간 | BASE Calmar | 우월 시나리오 | 부진 시나리오 |
|---|---:|---|---|
| 2024~2026 강세장 | **5.05** | S4 composite +0.56 / S1 매출 +0.51 / S2 영업이익 +0.28 | S3 EPS -0.34 |
| 2019~2023 약세장 | **1.18** | (모두 동일, 매칭률 부족) | - |

**3단계 종료 조건**: 모든 단위 테스트 통과 + 리포트 생성 + 베이스 sanity check 일치 + Reviewer APPROVED.

---

### 4단계. 리팩토링 (Pending, 필요 시)

- [ ] 중복 코드 제거 (DB 로드 / ranking 주입 / 통계 산출)
- [ ] 변수명·함수명 일관성 점검
- [ ] 단위 테스트 계속 통과 확인
- [ ] Reviewer 에게 4단계 리뷰 요청 (멈춤)

**4단계 종료 조건**: 테스트 통과 유지 + 구조 개선 + Reviewer APPROVED.

---

### 5단계. 메모리 / 결과 정리 (Completed, 사용자 직접 지시 2026-05-03)

- [x] 메모리 갱신: `project_finance_ranking.md` 신규 작성
  - 결론: 강세장 composite/매출 약간 우월, EPS 단독 부진, 약세장 결론 도출 불가
  - 운영 적용은 보류 권고 (표본 작음, out-of-sample 부재)
- [x] `MEMORY.md` 인덱스 갱신
- [x] 4단계 (리팩토링) 생략 — 큰 중복/성능 문제 없음, 사용자 직접 5단계 지시
- [x] git 커밋 (사용자 직접 지시)

**5단계 종료**: 사용자 직접 지시로 PDCA 종료.

---

## 의존 관계

```text
[1] 계획 수립 (PRD/TASK/REVIEW)
        ↓ APPROVED
[2] 실패 테스트 작성 (Red)
        ↓ APPROVED
[3] 최소 구현 + sanity check + 리포트 (Green)
        ↓ APPROVED
[4] 리팩토링 (필요 시)
        ↓ APPROVED
[5] 메모리 정리
        ↓ APPROVED
[완료]
```

## 현재 진행 상태

- **현재 단계**: 1단계 (계획 수립) — 요구사항 정정 1회 반영 후 마무리
- **다음 액션**: Reviewer 에게 1단계 리뷰 요청 → 종료 대기

## 비고 (범위 통제)

- 본 PDCA 는 **ranking 함수만** 교체. 후보 풀(filter) / 매도 규칙 / N / K 모두 K-Tide 10 그대로 유지.
- 결과 적용은 별도 PDCA 로 분리 (`feedback_simple_is_better.md` 위반 위험 사전 차단).
- 후보 풀 > 10 인 날이 강세장 7.1% / 약세장 2.5% — 차이가 작을 수 있음을 사전 인지하고 진행.
