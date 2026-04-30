# PRD - 1500억 N=15 + MA 정배열 필터 + K=3 백테스트

## 프로젝트 목적

기존 (N=15, K=3) 베이스라인(Calmar 41.88)에 **MA 정배열 필터(close>MA5 AND MA5>MA20)** 를 매수 조건에 추가했을 때, 거래 품질이 개선되는지 정량 검증한다. 매도 규칙은 LIST_EXIT(1500억 15/15 깨지면 매도) 와 MA 이탈(close<MA5 시 매도) 두 가지를 함께 비교한다.

## 해결하려는 문제

- (15,3) 베이스 거래수 17건의 **표본 부족** 문제 — MA 필터로 더 줄어들 것이 우려됨
- 동시에 MA 필터로 진입 시점의 추세 방향이 양호한 종목만 선별되어 거래 품질이 개선될 가능성도 있음
- ⚠️ **사전 위험 고지** (memory feedback_simple_is_better): "1500억 + 정배열 + 이격도(5)>97" 조합에서 Calmar 4.96→1.05 폭락 사례 있음 (5/5 + 정배열). 본 검증은 5/5 가 아닌 (15,3) 베이스라는 점에서 다르지만, **whipsaw 폭락 위험을 사전 인지하고 진행**

## 범위 (In-Scope)

- **베이스라인**: 1500억 N=15, K=3, LIST_EXIT (이미 검증된 Calmar 41.88)
- **변형 매수 조건** (T일 마감 시점):
  - 기존: 최근 15일 거래대금 ≥ 1500억 인 날이 15일 모두
  - **추가 필터**: T일 종가 > T일 MA5 AND T일 MA5 > T일 MA20
  - 통과 종목 중 amount 상위 K=3 매수 (다음날 시가)
- **매도 규칙 (둘 다 테스트)**:
  - 변형 A: LIST_EXIT (보유 종목이 1500억 15/15 조건 깨지면 매도)
  - 변형 B: MA5 이탈 (close < MA5 시 다음날 시가 매도)
  - 기존 run_backtest 의 `rule="LIST_EXIT"` / `rule="MA5"` 사용
- **결과 비교 set**:
  - 베이스 (필터 없음, LIST_EXIT) — 기존 (15,3)
  - 변형 A: 필터 + LIST_EXIT
  - 변형 B: 필터 + MA5 이탈
- **자본 모델**: 진짜 K슬롯 (K=3, equity_real_k)
- **출력**:
  - `research/backtest_n15_k3_ma_filter.py`
  - `research/results/backtest_n15_k3_ma_filter.md`

## 비범위 (Out-of-Scope)

- N 또는 K 변경 (N=15, K=3 고정)
- 임계치 변경 (1500억 고정)
- 다른 MA 조합 (MA10, MA60 등)
- 이격도 / RSI / 거래량 추가 필터

## 사용자 시나리오

투자자(PO)는:
- "(15,3) 거래 17건 중 한두 건 손실은 MA 필터로 거를 수 있을까?"
- "필터로 거래수가 17 → 5건 이하가 되면 의미 없으니 그것도 검증 대상"
- "LIST_EXIT 와 MA 이탈 매도 어느 쪽이 필터와 궁합 좋은가?"

## 기능 요구사항

### F1. MA 필터 적용 신호 생성

기존 `compute_5d_filter_signals(threshold_won=1500억, lookback=15)` 결과에서 추가 필터:
- T일 종가 > T일 MA5
- T일 MA5 > T일 MA20

위 두 조건 모두 만족하는 종목만 남기고, amount 내림차순 정렬, top_k=200.

### F2. 백테스트 실행

- 변형 A: `run_backtest(rule="LIST_EXIT", slots=3, max_concurrent=3)` + 필터 적용 signals
- 변형 B: `run_backtest(rule="MA5", slots=3, max_concurrent=3)` + 필터 적용 signals
- 베이스: 기존 (15,3) LIST_EXIT 결과 그대로 (재실행 후 비교)

### F3. 분석

- 거래수, CAGR, MDD, Calmar, 자본
- 필터 적용 전후 거래수 차이 (얼마나 거를렸는가)
- 거래 품질 (승률, 평균 수익률, 최대 손실)
- 종목 다양성

### F4. 비교 표 + 결론

세 결과를 한 표에 정리, MA 필터의 효과 판정.

## 비기능 요구사항

- 단일 스크립트 실행, 5분 이내
- 기존 모듈 재사용 (run_backtest, equity_real_k, compute_5d_filter_signals)

## 완료 기준 (Definition of Done)

- [ ] 변형 A, 변형 B, 베이스 3개 결과 산출
- [ ] 거래수/CAGR/MDD/Calmar 비교 표 생성
- [ ] 거래 품질 비교 (승률, 평균 수익률, 최대 손실)
- [ ] 종합 결론: MA 필터 효과 판정
- [ ] Reviewer 승인 (`APPROVED`)

## 테스트 관점 핵심 검증 항목

- **T1**: `apply_ma_filter(signals, daily_data)` — 필터 적용 후 종목 수 ≤ 원본
- **T2**: 인공 데이터 — close>MA5 만족 / 불만족 종목 분리 검증
- **T3**: MA5>MA20 만족 / 불만족 종목 분리 검증
- **T4**: 두 조건 모두 만족하는 종목만 통과
- **T5**: 모듈 임포트 가능

## 제약사항

- KOSPI200 universe + 2023~현재
- daily_data 의 ma5/ma20 컬럼이 이미 build_daily_data 에서 계산됨

## 가정사항

- `build_daily_data` 가 ma5, ma20 컬럼을 추가함 (top3_indicators 코드 확인 필요)
- (15,3) 베이스라인 결과(17건, Calmar 41.88)는 재실행해도 동일 재현
- MA 미계산 구간(초기 19일)에서는 신호 없음 — 필터 통과 0종목 처리
