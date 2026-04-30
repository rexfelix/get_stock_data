# PRD - 1500억 (15,3) + 시총 캡 사전 제외 매트릭스

## 프로젝트 목적

(N=15, K=3) 베이스(Calmar 41.88)의 운영 우려 — **삼성전자 등 시총 거대주가 거래대금 1~2위에 거의 항상 포함되어 매수→장기 묶임 리스크** — 를 정량 검증한다. 매수 후보 풀에서 **시총 X조 초과 종목 사전 제외** 시 수익성과 안정성이 어떻게 변하는지 매트릭스로 측정.

## 해결하려는 문제

- (15,3) 17건 베이스에서 005930(삼성전자) 1회 매수 → 615일 보유 +220.8% (큰 winner)
- 그러나 다른 시기에 매수됐다면 박스권/하락에 자본 1/3 묶여 기회비용 발생 가능
- 시총 캡 사전 제외 = whipsaw 위험 없는 universe 변경 → 안전한 검증
- Trade-off: 큰 winner를 잃을 수 있음 vs 다양한 종목으로 분산

## ⚠️ 사전 위험 고지 (이전 검증 인용)

memory `feedback_simple_is_better` (3회 검증):
- 매수 추가 필터(MA 정배열·MA20 추세·이격도)는 -79~95% Calmar 폭락
- 본 검증은 **매수 필터가 아니라 universe 사전 변경** — whipsaw 위험 없음

## 범위 (In-Scope)

- **베이스라인**: (N=15, K=3) 1500억 LIST_EXIT — Calmar 41.88
- **변형 (시총 캡 사전 제외)**:
  - 5개 시총 cap 임계: **10조 / 20조 / 30조 / 50조 / ∞ (제외 없음 = 베이스)**
  - 매일 시총 cap 초과 종목을 universe에서 제외 (그 후 1500억 15/15 신호 + amount 정렬 + K=3 매수)
- **시총 정의**: daily_data 의 `mcap = close × shares_outstanding` (이미 build_daily_data 에서 계산)
- **제외 시점**: T일 기준 mcap > cap 종목 → 그 날 매수 후보에서 제외 (시계열 따라 동적 변경)
- **매도**: LIST_EXIT (1500억 15/15 깨지면 매도) — 변경 없음

### 매트릭스

| cap 임계 | 의미 |
|---|---|
| ∞ (제외 없음) | 베이스라인 = 기존 (15,3) |
| 50조 | 시총 1~3위 정도 제외 (예: 005930 ~ 350조 / 000660 SK하이닉스 ~ 200조 / 373220 LG에너지솔루션 등) |
| 30조 | 상위 ~10개 제외 |
| 20조 | 상위 ~20개 제외 |
| 10조 | 상위 ~30개 제외, 거의 모든 대형주 제외 |

### 결과

- `research/backtest_n15_k3_mcap_cap.py`
- `research/results/backtest_n15_k3_mcap_cap.md`

## 비범위 (Out-of-Scope)

- N, K 변경 (15, 3 고정)
- 임계 1500억 변경
- 시총 cap을 동적(평균 또는 백분위)으로 정의 (절대 cap 만 검증)
- 매도 규칙 변경
- KOSDAQ150 universe 추가

## 사용자 시나리오

투자자(PO)는:
- "005930 1회 +220.8%은 운이었나? 시총 거대주 제외 시 (15,3)이 망가지나?"
- "30조 cap 정도면 005930·SK하이닉스 등 압도적 시총만 제외, 그 외 대부분 종목 유지 → 합리적"
- "10조 cap은 너무 강해서 baseline (15,3) 자체가 무너질 수 있음 — 한계 검증"

## 기능 요구사항

### F1. 시총 캡 필터 헬퍼

`apply_mcap_cap_filter(signals, daily_data, cap_won)`
- 입력 signals 의 각 (date, ticker) 에서 daily_data[ticker] 의 해당 date 행의 mcap 조회
- mcap ≤ cap_won 인 종목만 통과
- mcap NaN 또는 데이터 부재 시 불통
- 입력 순서 (amount 내림차순) 유지

### F2. 매트릭스 백테스트

5개 cap (∞, 50조, 30조, 20조, 10조) 각각 대해:
- 1500억 15/15 신호 + 시총 캡 필터 적용
- run_backtest(rule="LIST_EXIT", slots=3, max_concurrent=3)
- equity_real_k(K=3) 자본 시뮬레이션

### F3. 분석

- cap 별 거래수 / CAGR / MDD / Calmar / 자본 비교 표
- 005930 매수 횟수 변화 (cap 별)
- 신호 종목 다양성 (cap 별 unique tickers)
- 거래 품질 (승률, 평균 수익률, 최대 단일 손실)

### F4. 결론

- 최적 cap 임계 식별 (Calmar 기준)
- 005930 등 거대주 제외가 (15,3) 운영 안정성에 미치는 영향 정량 평가
- 사용자 채택 결정 권고

## 비기능 요구사항

- 단일 스크립트, 5분 이내
- 기존 모듈 재사용

## 완료 기준 (Definition of Done)

- [ ] 5개 cap 시나리오 결과 산출
- [ ] cap 별 거래수·종목 분포 표
- [ ] 005930 매수 빈도 변화 추적
- [ ] 최적 cap 권고 + Reviewer 승인 (`APPROVED`)

## 테스트 관점 핵심 검증 항목

- **T1**: 모듈 임포트 가능
- **T2**: `apply_mcap_cap_filter` 인공 케이스 — cap 미만/초과/같음 분리
- **T3**: NaN mcap 시 불통
- **T4**: 입력 순서 보존
- **T5**: 빈 입력 시 빈 dict

## 제약사항

- mcap 계산 = close × shares_outstanding (snapshot 시점 상장주식수 고정)
- 즉 mcap 시계열은 close 변화만 반영, 상장주식수 변동 미반영 (limitation)
- 본 검증은 cap 사전 제외 효과만 보므로 mcap 정확도가 결정적이지 않음

## 가정사항

- daily_data[ticker]["mcap"] 컬럼 존재 (build_daily_data L103)
- mcap 단위 = 원 (close × shares, 둘 다 원/주 / 주)
- snapshot의 shares_outstanding 은 2026-04 기준 — 2023~현재 동안 일정 가정
