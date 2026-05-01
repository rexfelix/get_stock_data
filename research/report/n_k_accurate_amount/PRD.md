# PRD - 정확 amount 로 (15,3)·(10,10)·(15,15) 2019~2023 재검증

## 프로젝트 목적

직전 amount_backfill PDCA 로 **stocks.amount 정확값** (키움 ka10081, 99.71% stock_all 일치) 이 가용해졌다. 기존 (15,3) 2019~2023 백테스트는 **amount 추정값(close × volume)** 으로 Calmar 0.29 결과를 얻었는데, 정확값으로 재검증하여 **시기 의존성 결론을 확정**한다. 동시에 (10,10)·(15,15) 도 같은 정확값으로 재검증하여 최종 권장 (N,K) 를 확정한다.

## 해결하려는 문제

- 이전 결과 (`project_n15_k3_2019_2023.md`): (15,3) 2019~2023 amount 추정 → Calmar 0.29
- 추정값 한계: 거래수 42 vs 베이스 17건 — 추정값이 실제보다 신호를 더 많이 포착했을 가능성
- 정확값으로 재검증 시:
  - (15,3) Calmar 가 더 떨어질지 (추정값이 낙관적이었을 수도) 또는 회복할지 (추정값이 비관적이었을 수도)
  - 최종 채택 결정의 근거 확정
- (10,10) / (15,15) 도 같은 정확값으로 시기 robust 비교 필요

## 범위 (In-Scope)

### 시나리오 매트릭스 (6개)

| (N,K) | 기간 |
|---|---|
| (15,3) | 2019~2023 (정확 amount) |
| (15,3) | 2024~2026 (정확 amount, sanity check) |
| (10,10) | 2019~2023 (정확 amount) |
| (10,10) | 2024~2026 (정확 amount, sanity check) |
| (15,15) | 2019~2023 (정확 amount) |
| (15,15) | 2024~2026 (정확 amount, sanity check) |

### 데이터 source

- **stocks.amount 직접 사용** (stock_all join 제거)
- 단위: 백만원 → load 시 × 1_000_000 으로 원 환산 (기존 load_price_data 동일)
- 2024~2026 sanity check: 추정 + stock_all 결과와 동일해야 함 (99.71% 일치 가정)

### 결과

- `research/backtest_n_k_accurate_amount.py`
- `research/results/backtest_n_k_accurate_amount.md`

## 비범위 (Out-of-Scope)

- (5,5) 추가 (작업 단순화, 향후 확장 가능)
- N×K 매트릭스 전체 재실행 (3 (N,K) 만)
- 매수 추가 필터 (4회 검증 폭락 확정)
- 시총 캡 변경
- 36 amount 실패 종목 처리 (별도)

## 사용자 시나리오

투자자(PO)는:
- "(15,3) Calmar 0.29 가 추정값 한계 때문이었나? 정확값으로 다시 보면?"
- "(10,10) 가 정말 시기 robust 1위인가, (15,15) 와 비교?"
- "최종 채택 (N,K) 를 정확값 기반으로 확정"

## 가설

- **H1 (추정값과 비슷)**: 정확 (15,3) 2019~2023 Calmar < 1 → 시기 의존성 확정. 채택 재검토 권고 유지
- **H2 (회복)**: 정확값에서 Calmar 5+ → 추정값 한계가 결과 왜곡. (15,3) 채택 정당화 가능
- **H3 ((10,10) 시기 robust 확정)**: 정확값 기반 (10,10) 2019~2023 Calmar > (15,3), (15,15) 추정값 비교 일관

## 기능 요구사항

### F1. 정확 amount 데이터 로드

`load_price_data_accurate(tickers, start, end)`:
- stocks 테이블에서 OHLCV + amount 직접 query
- amount NULL 행은 NaN 처리 (이후 1500억 N/N 필터에서 자동 제외)
- 단위: 백만원 → × 1_000_000 으로 원 환산 (기존 일관성)

### F2. 6 시나리오 백테스트

각 시나리오에서:
- compute_5d_filter_signals (1500억, lookback=N, top_k=200)
- run_backtest (rule="LIST_EXIT", slots=K, max_concurrent=K)
- equity_real_k(K=K)

### F3. 비교 표

- 정확값 vs 추정값 결과 차이 (특히 (15,3) 2019~2023)
- 시기 (2019~2023 vs 2024~2026) Calmar 차이
- (N,K) 별 시기 robust 순위

### F4. 최종 권장

- 정확값 기반 시기 robust 1위 (N,K)
- 강세장 1위 vs 시기 robust 1위 trade-off
- 사용자 채택 결정 영향 (15,3 유지/변경)

## 비기능 요구사항

- 단일 스크립트, 5분 이내
- 기존 모듈 재사용 (run_backtest, equity_real_k)
- 정확값 사용 시 코드 변경 최소

## 완료 기준 (Definition of Done)

- [ ] 6 시나리오 결과 산출
- [ ] 추정값 vs 정확값 비교 표
- [ ] 시기 robust 순위 (2019~2023 정확값 기준)
- [ ] 사용자 채택 결정 영향 평가
- [ ] Reviewer 승인

## 테스트 관점 핵심 검증 항목

- **T1**: 모듈 임포트
- **T2**: `load_price_data_accurate` 가 stocks.amount 사용 (stock_all join 없음)
- **T3**: amount 단위 변환 (× 1_000_000)
- **T4**: 빈 입력 처리

## 제약사항

- amount 백필 99.0% 행 채움 → 0.9% NULL (실패 36 종목 + 일부 일자)
- mcap 시계열 한계 그대로 (snapshot shares 고정)
- 2019~2023 KOSPI200 멤버십 시계열 미보유 (forward-looking bias)

## 가정사항

- stocks.amount = 백만원 단위 (amount_backfill PDCA 검증)
- 99.0% 채움률로도 1500억 N/N 필터 정확도 충분 (NULL 은 자동 제외)
- run_backtest, equity_real_k 등 핵심 백테스트 엔진 변경 없음
