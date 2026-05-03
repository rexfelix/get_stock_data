# PRD - K-Tide 10 슬롯 선택 기준 비교 (거래대금 vs 재무 증가율)

## 프로젝트 목적

이미 채택 확정된 매매봇 **K-Tide 10**((N=10, K=10) + 1500억 거래대금 + LIST_EXIT, 2026-05-02 확정)의 **슬롯 선택 정렬 기준**을 변경했을 때 더 좋은 성과가 나오는지 백테스트로 비교한다.

- 베이스(현행): 후보 풀 → **거래대금(amount) 내림차순** 으로 K=10 매수
- 비교 후보:
  - S1: **매출액 증가율(revenue YoY)** 내림차순
  - S2: **영업이익 증가율(operating_income YoY)** 내림차순
  - S3: **EPS 증가율(eps YoY)** 내림차순
  - S4(선택): 3개 지표의 **z-score 평균** 내림차순 (composite)

후보 풀(filter, "10일 연속 거래대금 ≥ 1500억")과 매도 규칙(LIST_EXIT)은 **변경하지 않는다**. 슬롯 채울 때의 ranking 함수만 교체.

## 해결하려는 문제

- K-Tide 10 의 거래대금 정렬은 단순한 합리적 기본값일 뿐 최적화된 근거는 약함.
- 재무 펀더멘털(매출/영업이익/EPS 증가율) 기준으로 K개를 고르면 동일 후보 풀에서도 더 우월한 종목 조합이 만들어질 수 있다는 가설을 검증.
- 만약 재무 정렬이 **베이스 대비 동등하거나 우월**하면, K-Tide 10 의 ranking 을 교체할 만한 근거가 된다.
- 만약 **부진**하면, 메모리 `reference_k_tide_10.md` 의 "거래대금 turnover 정렬 → Calmar 1.86 부진" 선례와 동일한 결과로, 정렬 변경 함정 데이터베이스를 강화.

## 핵심 사전 측정값 (2026-05-03)

후보 풀 크기 분포 — 정렬 기준이 **선택 결과에 영향을 미치는 날의 비율**:

| 기간 | 풀 존재 일수 | 풀 > 10 일수 | 비율 | 평균 풀 | 최대 풀 |
|---|---:|---:|---:|---:|---:|
| 2024~2026 강세장 | 566 | 40 | **7.1%** | 4.67 | 17 |
| 2019~2023 약세장 | 1233 | 31 | **2.5%** | 3.73 | 15 |

**해석**: 풀 ≤ 10 인 날은 모든 후보를 매수하므로 정렬이 무의미. 정렬이 실제로 선택을 바꾸는 날이 강세장 7.1% / 약세장 2.5% 뿐. 따라서 실험 결과는 **베이스에 매우 가까울 가능성**이 큼. 그래도 차이가 발생하는 7.1% / 2.5% 날에서 발생하는 효과 자체가 의미 있는 정보.

## 범위 (In-Scope)

### 1. 베이스 대비 비교 시나리오 (5개)

| 코드 | 정렬 기준 | 동률 처리 |
|---|---|---|
| BASE | amount DESC (현행 K-Tide 10) | (없음) |
| S1 | revenue_yoy DESC | amount DESC |
| S2 | op_income_yoy DESC | amount DESC |
| S3 | eps_yoy DESC | amount DESC |
| S4 | composite z-score(rev,op,eps) DESC | amount DESC |

각 시나리오 × 2 기간 (강세장 + 약세장) = **10 백테스트 런**.

### 2. 데이터 source

- **가격/거래대금**: 현행 K-Tide 10 와 동일 (`stocks` 테이블, amount 정확값)
- **finance**: `financial_summary` 우선 (revenue_yoy 컬럼 직접 사용)
  - operating_income / eps YoY 는 별도 계산
  - `is_estimate = TRUE` row 는 사용 금지
  - fallback: `financial_annual` (quarter=0)

### 3. Lookahead bias 회피 규칙

- 진입 후보 평가일 D 에 대해 가용 annual:
  - D.month >= 4 → year = D.year - 1 의 annual 사용
  - D.month < 4 → year = D.year - 2 의 annual 사용
- 사유: 한국 상장사 사업보고서 법정 제출 기한 90일 (3월 말). 4월 이후 보수적 사용.

### 4. NaN / 결측 처리 정책 (Critical)

- 후보의 finance 가 NaN 인 경우 → **랭킹 최하위로 강등 후 amount DESC 로 fallback**
- 후보 풀 자체에서 NaN 종목을 **제외하지 않음** (filter 가 되어 메모리 함정 위반)
- 사례: 신규상장으로 직전년 annual 이 없는 종목 → 후보 풀엔 남되, 풀 > 10 시 amount 낮은 NaN 종목이 후순위가 될 뿐

### 5. 증가율 정의

- **revenue YoY** = `financial_summary.revenue_yoy` 직접 사용 (이미 계산된 값)
- **operating_income YoY** = `(op[Y] - op[Y-1]) / abs(op[Y-1])`
- **EPS YoY** = `(eps[Y] - eps[Y-1]) / abs(eps[Y-1])`
- 분모 `abs()` 사용: 직전년 적자 → 흑자전환 부호 뒤집힘 방지
- prev=0 → NaN
- prev<0 & curr>0 → "흑자전환" (분석에선 큰 양수로 해석되도록 abs 적용; 정렬에는 그대로 사용)
- prev>0 & curr<0 → 큰 음수
- 정렬 시 NaN 은 후순위 강등 (위 4번 규칙)

### 6. 산출물

- 백테스트 스크립트: `research/backtest_finance_ranking.py`
- 단위 테스트: `research/tests/test_finance_ranking.py`
- 결과 리포트: `research/results/backtest_finance_ranking.md`

### 7. 보고할 지표 (시나리오별)

- CAGR, MDD, Calmar
- 거래수, 승률, 평균 보유기간
- **정렬 기준이 선택을 실제로 바꾼 거래 비율** (베이스와 다른 매수가 발생한 % — 핵심 진단 지표)
- 1종목 사고 시 충격 (자본 분할 1/K)
- 강세장 / 약세장 / 통합

## 비범위 (Out-of-Scope)

- 후보 풀(filter) 변경: "10일 연속 1500억" 그대로 유지
- 매도 규칙(LIST_EXIT) 변경
- N, K 변경 (10, 10 고정)
- 분기 finance 사용 (annual 만)
- 자동매매 코드 변경
- 본 PDCA 결과를 K-Tide 10 운영봇에 즉시 반영 (별도 의사결정 PDCA 필요)

## 사용자 시나리오

1. PO 가 K-Tide 10 백테스트 엔진의 "ranking 함수" 부분만 교체할 수 있도록 추출/일반화한다.
2. finance 기반 ranking 5종을 정의한다.
3. 동일 후보 풀 / 동일 매도 규칙으로 5개 시나리오 × 2 기간을 실행한다.
4. 정렬이 실제로 선택을 바꾼 거래 비율과 성과 지표를 베이스와 비교한다.
5. 결과 리포트에 표 + 해석 + 결론을 한국어로 작성한다.

## 기능 요구사항

- FR-1: 기존 K-Tide 10 백테스트 엔진을 재사용하되, ranking 함수를 인자로 주입할 수 있어야 한다 (의존성 역전).
- FR-2: finance 매핑 함수가 lookahead bias 없이 동작해야 한다 (4월 컷오프).
- FR-3: NaN 정책이 일관되게 적용되어야 한다 (후순위 강등 + amount fallback).
- FR-4: 결과 리포트는 BASE 대비 각 시나리오의 절대값과 차이(Δ)를 모두 포함해야 한다.
- FR-5: 결과는 결정적(deterministic). 동일 입력 → 동일 결과.

## 비기능 요구사항

- 단위 테스트: ranking 함수 + lookahead 매핑 + NaN 정책 + composite z-score.
- 외부 의존: pandas, numpy 외 추가 도입 금지.
- 실행 시간: 5개 시나리오 × 2 기간 합산 10 분 이내.

## 완료 기준 (Definition of Done)

- [ ] PRD / TASK / REVIEW 가 최신 상태이며 각 단계마다 Reviewer 가 APPROVED 명시
- [ ] `backtest_finance_ranking.py` 가 결정적으로 동작하며 결과 리포트 생성
- [ ] `tests/test_finance_ranking.py` 의 단위 테스트가 모두 통과 (FR-1 ~ FR-5 검증)
- [ ] `results/backtest_finance_ranking.md` 가 한국어 표/해석/결론을 포함
- [ ] 결론에 다음 항목 명시:
  - 시나리오별 Calmar / CAGR / MDD / 거래수 / 승률 (강세장 + 약세장 + 통합)
  - **정렬 기준이 선택을 실제로 바꾼 거래 비율**
  - 베이스 대비 차이의 통계적 유의성 또는 표본 한계 설명
  - K-Tide 10 운영 적용 권장 여부 (별도 PDCA 로 분리될 후속 의사결정의 입력)

## 테스트 관점의 핵심 검증 항목

- T-1: `entry_date_to_annual_year(D)` — 4월 컷오프
  - 2024-03-31 → 2022 / 2024-04-01 → 2023 / 2025-04-01 → 2024
- T-2: `calc_yoy(curr, prev)` — abs 분모 + 결측/0 처리
  - (110, 100) → 0.10
  - (-50, -100) → +0.5 (abs 분모로 호전 방향이 양수)
  - (50, -100) → +1.5 (흑자전환, 큰 양수)
  - (-50, 100) → -1.5 (적자전환, 큰 음수)
  - prev=0 → NaN
  - 입력 NaN → NaN
- T-3: `rank_candidates(candidates, key_fn, fallback_fn)` — NaN 후순위 + fallback 적용
  - NaN 종목들 사이는 fallback (amount) 으로 정렬
  - 정상값 종목들 사이는 key_fn 내림차순
  - 정상값 vs NaN 비교 시 정상값이 항상 상위
- T-4: `composite_zscore(df, cols)` — z-score 평균 (NaN 안전)
  - 단일 컬럼만 NaN 인 row → 가용 컬럼 평균 사용 또는 NaN 으로 강등 (정책 결정 후 테스트)
  - 본 PDCA 의 결정: **3개 컬럼 모두 가용한 row 만 z-score 부여, 한 개라도 NaN 이면 NaN**
- T-5: `count_ranking_effective_trades(base_trades, scenario_trades)` — 정렬 변경이 선택을 바꾼 거래 카운트

## 제약사항

- 후보 풀 > 10 인 날의 비율이 강세장 7.1% / 약세장 2.5% — 효과가 작을 가능성 높음. 결과가 베이스와 거의 동일하면 그 자체로 결론.
- finance 데이터 가용 범위로 약세장 매칭률이 낮음 — 약세장 정렬 시나리오에서 NaN 비율이 높을 수 있음. 이 경우 NaN 후순위 정책으로 자연스럽게 amount fallback 발생.
- `revenue_yoy` 가 financial_summary 에 이미 있으나 부호·스케일 일치를 검증해야 함 (T-2 와 별도 sanity check).

## 가정사항

- K-Tide 10 백테스트 엔진(`backtest_n_k_accurate_amount.py` + `backtest_5d_realistic_k.py`)이 ranking 함수를 인자로 분리 가능한 구조이거나, 최소한의 변경으로 분리 가능
- 한국 상장사 annual 4월 가용 가정
- finance 데이터는 2026-04-03 PDCA 시점 이후 추가 갱신이 없어도 본 분석에 충분 (2025 annual 까지 가용)

## 위험 (메모리 선례)

- `reference_k_tide_10.md` "거래대금 turnover 정렬 → Calmar 1.86 부진" — 정렬 기준 변경이 부진한 선례. 본 결과도 부진할 가능성을 예상하고 진행.
- 약세장 NaN 비율 높을 경우 "amount fallback" 으로 베이스와 거의 동일한 결과가 나올 수 있음 — 이는 정상 작동.
