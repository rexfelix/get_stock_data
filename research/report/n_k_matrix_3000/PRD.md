# PRD - 3000억 거래대금 N/N + K슬롯 N×K 매트릭스 백테스트

## 프로젝트 목적

기존 1500억 N/N + K슬롯 매트릭스(`backtest_n_k_matrix.py`)와 동일한 구조로 **거래대금 임계치만 3000억으로 상향**한 매트릭스를 산출하여, 임계치 변화가 신호 빈도/체결률/Calmar/CAGR/MDD에 미치는 영향을 정량 비교한다.

## 해결하려는 문제

- 1500억 기준 매트릭스에서 N=K 대각선이 우위였고 자본별 N/K 매칭(5/5, 10/10, 15/15)이 추천 조합으로 채택됨 (memory: project_top3_strategies)
- 임계치를 두 배(3000억)로 올리면:
  - 신호 빈도가 줄어들어 미체결(missed)이 감소할 것이라는 가설
  - 더 큰 거래대금 종목으로 압축되어 모멘텀 강도가 높아질 것이라는 가설
  - 그러나 N=15 같은 긴 룩백에서 신호가 거의 사라져 거래수가 부족해질 위험
- 위 가설을 검증하고, 1500억 매트릭스 대비 어떤 (N,K) 조합이 우월/열위인지 한눈에 비교 가능한 리포트가 필요하다

## 범위 (In-Scope)

- **대상**: KOSPI200 (기존과 동일)
- **기간**: 2023-01-01 ~ 현재 (`backtest_top3_indicators.load_price_data` 로딩 범위와 동일)
- **매수 조건**: 최근 N일 거래대금 ≥ **3000억** 인 날이 N일 모두 (N/N 필터)
- **매도 규칙**: LIST_EXIT (다음날 N/N 깨지면 다다음날 시가 매도) — 기존과 동일
- **자본 모델**: 진짜 K슬롯 (cap=K, 자본 1/K 동적) — 기존과 동일
- **수수료**: 매수 0.015% + 매도 0.015% + 세금 0.18%
- **매트릭스**: N=[3,5,7,10,15] × K=[3,5,7,10,15] = 25 조합
- **결과물**:
  - `research/backtest_n_k_matrix_3000.py` (실행 스크립트)
  - `research/results/backtest_n_k_matrix_3000.md` (결과 리포트)

## 비범위 (Out-of-Scope)

- 임계치를 다른 값(2000억, 5000억 등)으로 변경
- 매도 규칙 변경 (LIST_EXIT 외)
- N/K 외 다른 그리드 (예: N=20, K=20)
- 1500억 결과 재실행 (기존 결과 그대로 사용해 비교)

## 사용자 시나리오

투자자(PO)는 다음 의문을 가진다:
- "임계치 1500억은 너무 느슨한 것 아닌가? 더 큰 거래대금 종목만 잡으면 더 좋지 않을까?"
- "임계치를 올리면 미체결이 줄어 cap 효율이 올라갈까?"
- "긴 룩백(N=15)에서 3000억은 신호가 너무 적어 백테스트 의미가 없는가?"

본 백테스트는 정량 데이터로 답한다.

## 기능 요구사항

### F1. 신호 생성 (3000억 N/N)
- 기존 `compute_5d_filter_signals(threshold_won=...)` 함수 재사용
- `threshold_won = 300_000_000_000` 으로 호출
- N별로 일별 신호 dict 생성

### F2. N×K 매트릭스 백테스트
- 기존 `backtest_top3_indicators.run_backtest` (rule="LIST_EXIT", slots=K, max_concurrent=K) 재사용
- 기존 `backtest_5d_realistic_k.equity_real_k(trades, K=K)` 로 자본 시계열 계산
- 25개 조합 모두 실행

### F3. 결과 리포트
기존 `backtest_n_k_matrix.md` 와 동일한 구조로 다음 섹션 포함:
- Calmar 매트릭스
- CAGR 매트릭스 (%)
- MDD 매트릭스 (%)
- 거래수 매트릭스
- 미체결 매트릭스
- Calmar Top 10 조합
- N≈K 매칭 효과 (대각선)
- **추가**: 1500억 매트릭스(`backtest_n_k_matrix.md`) 대비 비교 표
  - 동일 (N,K)에서 Calmar/CAGR/MDD/거래수 변화 (Δ)

## 비기능 요구사항

- 단일 스크립트 실행으로 매트릭스 완료
- 실행 시간 10분 이내 (1500억 버전과 유사 또는 더 빠름)
- 코드 중복 최소화: 기존 모듈 import 재사용

## 완료 기준 (Definition of Done)

- [ ] `backtest_n_k_matrix_3000.py` 가 정상 실행됨 (25개 조합)
- [ ] `results/backtest_n_k_matrix_3000.md` 가 모든 섹션 채워서 생성됨
- [ ] 1500억 vs 3000억 비교 표가 리포트에 포함됨
- [ ] 신호 빈도 감소가 의도대로 발생함을 로그/리포트로 확인
- [ ] Reviewer 승인 (`APPROVED`)

## 테스트 관점 핵심 검증 항목

- **T1**: `compute_5d_filter_signals(threshold_won=300_000_000_000, lookback=N)` 호출 시 N별 신호 일수가 1500억 대비 감소
- **T2**: 동일 N에 대해 일별 평균 신호 수가 1500억 대비 감소
- **T3**: 25개 조합 결과 dict의 키가 모두 (n,k) ∈ {3,5,7,10,15}² 임
- **T4**: 각 조합 결과에 cagr, mdd, calmar, total, missed 키 존재
- **T5**: 리포트 마크다운 파일이 생성되고 모든 매트릭스 섹션 포함

## 제약사항

- KOSPI200 외 종목 미포함
- 키움 데이터 기반 amount 사용 (DB 의존)
- N=15 + 3000억 조합에서 신호가 0 또는 매우 적을 가능성 — 그 경우 Calmar=0/거래수=0 으로 그대로 표기

## 가정사항

- 1500억 결과 (`results/backtest_n_k_matrix.md`) 가 최신 상태로 존재한다
- DB의 daily 테이블에 충분한 거래일 데이터가 있다
- `backtest_top3_indicators.run_backtest` 의 LIST_EXIT 매도 로직은 변경되지 않는다
