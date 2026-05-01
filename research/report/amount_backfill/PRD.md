# PRD - stocks 테이블 amount 컬럼 백필 (2019~현재)

## 프로젝트 목적

기존 `stocks` 테이블에 **amount (거래대금) 컬럼이 부재**하여 2019~2023 백테스트가 추정값(close × volume)에 의존하는 한계를 해결한다. 키움 REST API `ka10081` 로 정확한 amount 데이터를 수집하여 stocks 테이블에 추가한다.

## 해결하려는 문제

- 현재 `stocks` 테이블 (4067 종목, 2019-01-02 ~ 2026-04-30): close, open, high, low, volume 만 있음
- `stock_all.amount` 는 2023-09-25 이후만 가용
- 2019~2023 기간 (15,3) 백테스트는 amount = close × volume **추정값** 사용 (직전 PDCA 발견)
- 추정값 한계: 일중 평균가 vs 종가 차이 → 신호 식별 정확도 영향
- 해결: 키움 ka10081 의 `trde_prica` (거래대금) 를 stocks 테이블 amount 컬럼에 저장

## 범위 (In-Scope)

- **DB 스키마 변경**: `ALTER TABLE stocks ADD COLUMN amount BIGINT`
- **수집 대상**: stocks 테이블의 모든 ticker (4067 종목)
- **수집 기간**: 2019-01-02 ~ 2026-04-30 (현재)
- **API**: 키움 REST API `ka10081` (주식일봉차트조회)
- **단위**: **백만원** (stock_all.amount 일관성)
- **호출 전략**:
  - ka10081 = base_dt 기준 과거 600일치 1회 호출
  - 2026-04-30 부터 600일 단위로 과거 방향 호출
  - 7.4년 / 600일 = 약 4~5 호출/종목
  - 4067 종목 × 5 호출 = 약 20,000 호출
- **rate limit**: 0.3초/호출 (기존 코드 동일) → 약 100분 소요
- **검증**: stock_all.amount (2023-09-25~) 와 새 stocks.amount 일치성 95%+ 확인

## 비범위 (Out-of-Scope)

- stock_all 테이블 변경 (별도)
- 투자자별 순매수 (ka10060) — 본 작업은 amount만
- 신규 종목 추가 (현재 stocks 의 4067 종목만)
- 실시간 업데이트 (배치 백필만)
- KOSDAQ150 universe 추가 (별도)

## 사용자 시나리오

투자자(PO)는:
- "2019~2023 (15,3) Calmar 0.29 결과가 amount 추정 한계 때문일까? 정확한 amount 로 재검증하자"
- "향후 모든 백테스트에서 amount 추정 없이 정확한 값 사용 가능"
- "stock_all 과 stocks 의 amount 일치성으로 데이터 정확도 확인"

## 기능 요구사항

### F1. DB 스키마 변경

```sql
ALTER TABLE stocks ADD COLUMN IF NOT EXISTS amount BIGINT;
CREATE INDEX IF NOT EXISTS idx_stocks_ticker_date ON stocks(ticker, date);
```

### F2. 키움 API 호출 (ka10081)

기존 `get_stock_all.fetch_ka10081` 재사용:
- `base_dt` 기준 과거 600일치 OHLCV + amount 수집
- 호출 결과의 `trde_prica` 추출 → 백만원 단위로 변환
  - ka10081 의 trde_prica 단위 확인 필요 (raw 데이터 첫 호출로 검증)
  - stock_all 코드 보면 그대로 저장 후 SQL에서 1_000_000 곱함 → 백만원 단위로 저장
- 응답 `stk_dt_pole_chart_qry` 파싱

### F3. 종목별 반복 + 600일 단위 페이지네이션

```text
for ticker in tickers:
    base_dt = "20260430"
    while base_dt > "20190102":
        rows = fetch_ka10081(token, ticker, base_dt)
        if not rows: break
        save rows
        oldest_date = min(rows.dates)
        if oldest_date <= "20190102": break
        base_dt = oldest_date - 1 day
        sleep 0.3s
```

### F4. DB 저장

```sql
UPDATE stocks SET amount = :amount WHERE ticker = :ticker AND date = :date
```

또는 일괄 UPDATE for performance:

```sql
UPDATE stocks s SET amount = t.amount
FROM (VALUES ...) t(ticker, date, amount)
WHERE s.ticker = t.ticker AND s.date = t.date
```

### F5. 진행 상황 저장 (resume 가능)

- 종목별 완료 여부를 별도 파일 또는 DB 테이블에 기록
- 중단 시 다음 실행에서 미완료 종목만 처리
- `report/amount_backfill/progress.json` 또는 `amount_backfill_progress` 테이블

### F6. 검증

수집 완료 후 stock_all.amount 와 비교:

```sql
SELECT s.ticker, s.date, s.amount AS s_amt, sa.amount AS sa_amt
FROM stocks s
JOIN stock_all sa ON s.ticker = sa.ticker AND s.date::date = sa.date
WHERE s.date >= '2023-09-25'
LIMIT 100;
```

일치율 95%+ 확인 (단위·반올림 차이 약간 허용).

### F7. Smoke Test (전 단계)

전체 실행 전 KOSPI200 중 005930 (삼성전자) 1 종목으로 1회 호출 → 데이터 형식·단위 확인.

## 비기능 요구사항

- 단일 명령으로 실행 (`python amount_backfill.py`)
- 백그라운드 실행 가능 (`run_in_background=true`)
- 중단 후 재시작 가능 (resume)
- 약 100분 소요 (4067 종목)
- API 실패 시 자동 재시도 (3회 까지)

## 완료 기준 (Definition of Done)

- [ ] stocks 테이블에 amount 컬럼 추가
- [ ] 4067 종목 모두 amount 채워짐
- [ ] stock_all.amount 와 95%+ 일치율 확인
- [ ] 진행 상황 저장 + resume 동작 확인
- [ ] Reviewer 승인

## 테스트 관점 핵심 검증 항목

- **T1**: 모듈 임포트
- **T2**: `parse_amount(raw_value)` — ka10081 trde_prica 파싱 (단위 변환)
- **T3**: `build_amount_records(api_rows, ticker)` — API 응답을 (ticker, date, amount) 리스트로
- **T4**: 600일 단위 페이지네이션 종료 조건 (base_dt < 2019-01-02)
- **T5**: 진행 상황 저장 + resume

## 제약사항

- 키움 API rate limit: 0.3초/호출 (보수적)
- 4067 종목 × ~5 호출 = 20,000 호출 → 100분
- API 인증 토큰 만료 시 재발급 필요 (장시간 실행)
- 실패 종목 별도 기록 → 재시도 가능

## 가정사항

- stocks.date 와 ka10081 응답의 dt 가 같은 거래일 기준
- ka10081 trde_prica 단위 = 백만원 (stock_all 일관성, smoke test 로 확인)
- 4067 종목 모두 키움에서 ka10081 응답 가능 (KOSPI + KOSDAQ + ETF 일부)
- 키움 API 키 가용 (data_center/.env 확인 완료)
