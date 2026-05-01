# PRD - add_daily_stocks.py 일일 업데이트에 amount 추가

## 프로젝트 목적

`data_center/add_daily_stocks.py` 의 stocks 테이블 일일 업데이트 로직에 **amount(거래대금) 컬럼 함께 업데이트** 기능 추가. 또한 기존 `start_date = last_date + 1일` 로직을 **`last_date` 부터 재수집**으로 변경 (마지막 날 데이터를 다시 받아와 갱신).

## 해결하려는 문제

- 기존 `add_daily_stocks.py` 는 OHLCV 만 수집 → stocks.amount 컬럼 업데이트 안 됨
- 직전 PDCA (`amount_backfill.py`) 로 기존 데이터 (2019~현재) 는 백필되었지만, 일일 신규 데이터에는 amount 가 없음
- 기존 start_date 로직은 `last_date + 1일` 부터 → 장중 미완성 데이터가 들어간 마지막 날이 갱신 안 됨
  - 사용자 요구: "마지막 날 포함하여 오늘까지 다시 받아와 업데이트"
  - 오늘 = last_date 케이스도 동일 (오늘 데이터 다시 받아오기)

## 범위 (In-Scope)

### 변경 1: start_date 로직 통합

기존:

```python
if last_date.date() >= today_date:
    # 오늘 이후 삭제 후 재수집
    start_date_obj = today
    delete_data_from_date(today)
else:
    # 마지막 날 다음날부터
    start_date_obj = last_date + 1day
```

변경:

```python
# 항상 last_date 포함하여 재수집 (마지막 날 데이터 갱신 보장)
start_date_obj = last_date
delete_data_from_date(last_date)
```

### 변경 2: amount 수집 추가

OHLCV (FDR) 저장 후, 같은 기간에 대해 키움 ka10081 호출하여 stocks.amount UPDATE.

- 함수: `update_amount_for_period(token, tickers, start_date, end_date)`
  - 각 ticker 에 대해 ka10081 (base_dt=end_date) 1회 호출
  - 응답 600일치에서 [start_date, end_date] 구간만 추출
  - stocks.amount UPDATE
- 단위: 백만원 (stock_all 일관성, ka10081 trde_prica 그대로)
- 기존 `amount_backfill.py` 의 `parse_amount`, `build_amount_records`, `bulk_update_amount` 재사용

### 적용 흐름

```text
1. last_date 조회
2. start_date = last_date
3. delete_data_from_date(stocks, start_date)
4. fetch_and_save_data (FDR로 OHLCV 수집 + INSERT)  ← 기존
5. update_amount_for_period (Kiwoom ka10081로 amount 수집 + UPDATE)  ← 신규
6. market_indices 업데이트 (기존)
```

## 비범위 (Out-of-Scope)

- 기존 OHLCV 수집 로직 변경 (FDR 그대로 유지)
- 신규 종목 추가 (existing tickers 만)
- stock_all 테이블 변경
- amount_backfill 재실행 (이미 완료)

## 사용자 시나리오

운영자(PO)는 매일 장 마감 후:

- `python add_daily_stocks.py` 한 번 실행 → OHLCV + amount 자동 업데이트
- 장중 실행해도 안전 (last_date 부터 삭제 후 재수집이라 idempotent)
- amount 가 stocks 테이블에 함께 저장 → 백테스트에서 추정값 사용 불필요

## 가설

- **H1**: start_date = last_date 로직 변경 시 정상 동작 (DELETE + INSERT)
- **H2**: ka10081 base_dt=오늘 1회 호출로 최근 며칠치 amount 충분 (보통 1~5일치 갱신)
- **H3**: 4067 종목 × 1 호출 ≈ 20분 (rate limit 0.3s)

## 기능 요구사항

### F1. start_date 로직 변경

`get_last_update_date` 결과를 그대로 start_date 로 사용 (last_date 포함). 분기 제거.

### F2. amount 수집 함수

```python
def update_amount_for_period(token, tickers, start_date, end_date):
    """[start_date, end_date] 구간의 amount 를 키움 ka10081 로 수집해 UPDATE."""
    for ticker in tickers:
        rows = fetch_ka10081(token, ticker, end_date.replace("-", ""))
        records = build_amount_records(rows, ticker)
        # start_date 이상 필터
        records = [r for r in records if r[1] >= pd.Timestamp(start_date)]
        bulk_update_amount(records → DataFrame)
        sleep 0.3s
```

### F3. main() 통합

OHLCV 저장 후 amount 수집 추가:

```python
# 기존
fetch_and_save_data(...)

# 신규
print("Updating amount via Kiwoom ka10081...")
token = get_kiwoom_token()
update_amount_for_period(token, all_ticker_names, start_date_str, end_date_str)
```

### F4. 단위 일관성

- ka10081 `trde_prica` 단위 = 백만원 (Smoke test 확정, amount_backfill PDCA)
- stocks.amount 도 백만원 단위로 저장 (변환 없음)

## 비기능 요구사항

- 단일 명령 (`python add_daily_stocks.py`) 실행
- 기존 동작 (OHLCV 수집) 안 깨짐
- amount 수집 추가 시간: 4067 종목 × 0.3s ≈ 20분 (1회 호출/종목)
- 실패 종목 로깅, 전체 중단 방지

## 완료 기준 (Definition of Done)

- [ ] `add_daily_stocks.py` 수정 완료
- [ ] start_date 로직 변경 (last_date 포함 재수집)
- [ ] amount 수집 함수 추가 + main 통합
- [ ] 단위 테스트 추가 (Red→Green)
- [ ] 실제 실행 후 검증 (마지막 1~5일 amount 채워짐 확인)
- [ ] Reviewer 승인

## 테스트 관점 핵심 검증 항목

- **T1**: 모듈 임포트
- **T2**: `compute_start_date(last_date, today)` — last_date 그대로 반환 (분기 없이)
- **T3**: `filter_records_by_date_range(records, start, end)` — 구간 필터
- **T4**: `update_amount_for_period` 호출 흐름 (mock)

## 제약사항

- 기존 코드 (FDR 부분) 변경 최소화
- 키움 토큰 만료 (장시간 실행 시 재발급)
- ka10081 응답에 어떤 ticker 가 빠지면 그 ticker 만 amount 누락 (재시도 별도)

## 가정사항

- amount_backfill 의 헬퍼 (parse_amount, build_amount_records, bulk_update_amount) 재사용 가능
- start_date 가 last_date 보다 이전 데이터는 amount 가 이미 백필됨 (직전 작업)
- ka10081 의 trde_prica 단위 = 백만원 (확정)
