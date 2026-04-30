# TASK - Top 3 지표 비교 백테스트

## 진행 상태

- 현재 단계: 1단계 계획 수립 (Reviewer 리뷰 대기)
- 목표: PRD/TASK 승인 → 시가총액 수집 → 백테스트 → 리포트

## 작업 분해

### 1단계. 계획 수립 (현재)

- [x] 요구사항 분석 (사용자 AskUserQuestion 응답 기준)
- [x] 데이터 가용성 조사 (stock_all.amount 존재, mcap 미존재 → 키움 API 수집 필요)
- [x] PRD.md 작성
- [x] TASK.md 작성
- [ ] **Reviewer 1차 검토 → APPROVED 대기**

### 2단계. 시가총액 데이터 수집

- [ ] `research/fetch_market_cap.py` 작성
  - kospi200_members 200종목 ticker 조회
  - 키움 ka10001로 각 종목의 `mac`(시가총액), `cur_prc`(현재가) 수집
  - `report/top3_indicators/market_cap_snapshot.csv`에 저장
- [ ] 200종목 수집 완료 검증 (NULL/실패 종목 확인)
- [ ] **Reviewer 2차 검토**

### 3단계. 백테스트 엔진 구현

- [ ] `research/backtest_top3_indicators.py` 작성
  - 데이터 로드 (stocks + stock_all + mcap_snapshot)
  - 일별 지표 계산 (amount/mcap/turnover)
  - 일별 Top 3 추출
  - 매도 규칙 6종 시뮬레이션 (LIST_EXIT/MA5/MA20/HOLD_5/HOLD_10/HOLD_20)
  - 통계 집계
- [ ] 단위 테스트:
  - 1일치 미니 데이터로 Top3 추출 검증
  - 각 매도 규칙별 진입/청산 로직 검증
- [ ] **Reviewer 3차 검토**

### 4단계. 리포트 생성

- [ ] 18 조합 종합 비교표
- [ ] 연도별 비교표
- [ ] 최우수 조합 상세 (Top/Bottom 거래)
- [ ] `results/backtest_top3_indicators.md` 저장
- [ ] **Reviewer 4차 최종 검토**

## 검증 항목

- [ ] 거래대금 Top3 백테스트 결과가 기존 `backtest_amount_hold.md`와 일관성 있는가?
- [ ] 시가총액 Top3 결과가 KOSPI200 가중 평균 대비 합리적인가?
- [ ] MA5/MA20 이탈 매도 로직이 정확한가?

## 의존 관계

- 1단계 승인 → 2단계 진행
- 2단계 완료 → 3단계 진행
- 3단계 완료 → 4단계 진행

## Reviewer 결정 대기 항목

**현재 1단계 계획 수립 완료. 다음 사항에 대해 Reviewer 판정 요청:**

1. PRD 범위가 사용자 요구사항을 정확히 반영하는가?
2. 시가총액 추정 방식(상장주식수 = mcap/close)이 KOSPI200 대형주 백테스트에 충분한 정확도인가?
3. 매도 규칙 6종(4 + N=3종)이 적절한 비교 대상 집합인가?
4. forward-looking bias(현재 KOSPI200 200종목 기준 과거 백테스트) 수용 가능한가?
