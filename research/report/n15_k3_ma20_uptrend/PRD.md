# PRD - 1500억 N=15 + MA20 5일 연속 상승 필터 + K=3

## 프로젝트 목적

(N=15, K=3) 베이스라인(Calmar 41.88) 위에 **MA20 5일 연속 상승** 매수 필터를 추가했을 때 효과를 검증한다. 이전 검증(MA 정배열 필터)에서 95% 폭락 사례가 있었으나, MA20 추세는 더 안정적이고 매도 규칙(LIST_EXIT 단독)이 매수 필터와 분리되어 있어 결과가 다를 가능성이 있음.

## 해결하려는 문제

- (15,3) 베이스가 압도적으로 좋지만 MA20 추세를 무시하므로 약세 추세 종목도 진입 가능
- MA20 5일 연속 상승은 **거시 추세 확인** 역할 — 단기 정배열보다 안정적
- 매도 규칙은 LIST_EXIT 단독(매수 필터와 분리)이므로 정배열 필터처럼 즉각 매도 트리거가 발생하지 않음 → whipsaw 위험 낮을 가능성

## ⚠️ 사전 위험 고지

memory `feedback_simple_is_better` (2회 검증):

- 5/5 + MA정배열 + 이격도 → Calmar 4.96→1.05 (79% 폭락)
- (15,3) + MA정배열 → Calmar 41.88→2.17 (95% 폭락)
- **공통 원인**: 매수 필터가 매도 트리거(LIST_EXIT)와 동기화되어 whipsaw

본 검증의 차별점:
- MA20 5일 연속 상승은 정배열보다 변동성 낮음 (느린 지표)
- 매도는 LIST_EXIT 단독 — 매수 필터(MA20)와 무관하게 1500억 15/15만 봄
- 가능성: whipsaw 덜 발생할 수도 / 또는 동일 패턴 폭락

## 범위 (In-Scope)

- **베이스라인**: 1500억 N=15, K=3, LIST_EXIT — 기존 (15,3) Calmar 41.88
- **변형 매수 조건** (T일 마감 시점):
  - 기존: 1500억 15/15 만족
  - **추가 필터**: MA20[T-4] < MA20[T-3] < MA20[T-2] < MA20[T-1] < MA20[T]
    (즉 최근 5거래일 동안 MA20이 매일 상승)
  - 통과 종목 중 amount 상위 K=3
- **매도 규칙**: LIST_EXIT 만 (매수 필터와 무관, 1500억 15/15 깨지면 매도)
- **자본**: 진짜 K슬롯 (K=3, equity_real_k)
- **결과 비교**:
  - 베이스 (필터 없음, LIST_EXIT) — 기존 (15,3)
  - 변형 (MA20 5일 상승 필터, LIST_EXIT)
- **출력**:
  - `research/backtest_n15_k3_ma20_uptrend.py`
  - `research/results/backtest_n15_k3_ma20_uptrend.md`

## 비범위 (Out-of-Scope)

- N, K 변경 (N=15, K=3 고정)
- 다른 추세 정의 (MA20 단순 차분 vs 5일 연속 등 변형)
- MA5 이탈 매도 추가 (이전 작업에서 LIST_EXIT 미세 우위 입증, 본 작업은 LIST_EXIT 만)
- 임계치 변경

## 사용자 시나리오

투자자(PO)는:
- "(15,3) Calmar 41.88 의 17건 거래 중 MA20 약세 종목이 있나? 거른다고 더 좋아질까?"
- "MA 정배열 필터는 95% 폭락이었는데, MA20 5일 상승은 더 안정적일까?"
- "거래수가 17 → 5건 이하로 떨어지면 의미 없으니 그것도 검증"

## 기능 요구사항

### F1. MA20 5일 연속 상승 필터 적용

기존 1500억 N=15 신호에서 추가 필터:
- T일 기준 MA20[T-4 ~ T] 5개 값이 모두 단조 증가 (strictly increasing)
- daily_data 의 ma20 컬럼 사용 (이미 build_daily_data 에서 계산)

### F2. 백테스트 실행

- 베이스: `compute_5d_filter_signals(threshold_won=1500억, lookback=15)` → `run_backtest(rule="LIST_EXIT", slots=3, max_concurrent=3)`
- 변형: 베이스 signals 에 `apply_ma20_uptrend_filter` 적용 후 동일 백테스트

### F3. 분석 + 비교

- 거래수 / CAGR / MDD / Calmar 비교 표
- 거래 품질 (승률, 평균 수익률, 최대 손실, 평균 보유일)
- 종목 다양성 (unique, 최다 거래 종목)
- 변형 거래 상세 (필터로 어떤 종목이 추가/제외됐는지)

### F4. 결론

- MA20 추세 필터의 효과 판정 (개선 / 무효 / 악화)
- 정배열 필터(폭락) vs MA20 추세 필터(?) 비교
- feedback memory 와의 연결

## 비기능 요구사항

- 단일 스크립트, 5분 이내
- 기존 모듈 재사용

## 완료 기준 (Definition of Done)

- [ ] 베이스 + 변형 결과 산출
- [ ] 거래 품질 비교 표 생성
- [ ] 결론: MA20 추세 필터 효과 판정
- [ ] feedback memory 와 연결 검토
- [ ] Reviewer 승인 (`APPROVED`)

## 테스트 관점 핵심 검증 항목

- **T1**: 모듈 임포트 가능
- **T2**: `is_ma20_uptrend_5d(ma20_series, idx)` — 5일 연속 상승 검증 (인공 케이스)
- **T3**: 단조 증가가 아닌 경우 (등락) 불통
- **T4**: NaN 포함 시 불통 (초기 데이터)
- **T5**: `apply_ma20_uptrend_filter(signals, daily_data)` — 통합 동작

## 제약사항

- KOSPI200 universe + 2023~현재
- daily_data 의 ma20 컬럼 의존
- MA20 미계산 구간(초기 19일)에서는 신호 없음

## 가정사항

- MA 정배열 필터와 다르게 MA20은 더 stable → whipsaw 덜 발생 기대 (검증 대상)
- LIST_EXIT 단독 매도 → 매수 필터가 매도 트리거에 영향 안 줌
