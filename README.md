# 주식 데이터 센터 및 차트 대시보드 (Stock Data Center & Chart Dashboard)

KOSPI/KOSDAQ 전 종목의 일봉 데이터와 테마/업종 정보를 수집, 저장하고 이를 시각화하여 분석할 수 있는 로컬 데이터 플랫폼입니다. PostgreSQL에 데이터를 저장하며, Streamlit 차트 대시보드와 터미널 기반 테마 편집 도구를 제공합니다.

## 📁 프로젝트 구조

### 주가 데이터 수집

| 파일 | 설명 |
|------|------|
| `get_stocks.py` | **초기 데이터 구축**. 전 종목 OHLCV 데이터를 수집하여 `stocks` 테이블 초기화 (멀티프로세싱 4코어, 배치 저장) |
| `add_daily_stocks.py` | **일일 업데이트**. DB 마지막 날짜 이후 데이터 추가 수집 + `market_indices` 테이블에 KOSPI/KOSDAQ 지수 업데이트 |
| `get_stock_all.py` | **종합 데이터 수집 (키움 REST API)**. 전 종목 일봉 OHLCV + 거래대금 + 투자자별 순매수를 `stock_all` 테이블에 저장 |
| `verify_stocks.py` | **데이터 검증**. 행 개수, 고유 티커 수, 컬럼 구조 등 DB 정합성 확인 |

### 테마/업종 데이터 수집

| 파일 | 설명 |
|------|------|
| `get_stock_themes.py` | **키움 REST API** 기반 테마/업종 수집. `themes` 테이블에 저장 (142개 테마그룹, 648종목) |
| `scrape_naver_themes.py` | **네이버 증권 테마 스크래핑**. 전체 테마별 구성 종목을 수집하여 CSV 생성 (~2,397 종목) |
| `add_summary_to_themes.py` | `themes` 테이블에 기업 요약(summary) 컬럼 추가 및 CSV 데이터 매핑 |

### 재무데이터 크롤링 (네이버 증권)

| 파일 | 설명 |
|------|------|
| `crawl_financial_summary.py` | **Financial Summary** (cF1002). 매출액, 영업이익, EPS, PER, PBR, ROE 등 → `financial_summary` 테이블 (UPSERT) |
| `crawl_financial.py` | **연간+분기 재무데이터** (cF3002+cF4002). 손익계산서 + 재무비율 → `financial_annual` 테이블 (quarter=0: 연간, 1~4: 분기) |
| `crawl_consensus.py` | **투자의견 컨센서스** (c1010001). 목표주가, 제공처별 의견 → `consensus_summary`, `consensus_provider` 테이블 + `financial_summary` PCR/BPS/DPS 업데이트 |
| `run_financial_crawl.py` | **통합 실행**. 위 3개 스크립트를 순서대로 실행 |

### 시각화 및 편집 도구

| 파일 | 설명 |
|------|------|
| `chart_test.py` | **Streamlit 대시보드**. 캔들/하이킨아시 차트, MA, 볼린저밴드, 스무스드 하이킨아시 지표 |
| `theme_edit.py` | **터미널 UI (Textual)**. 테마/요약 정보를 검색하고 개별/일괄 편집하는 TUI 앱 |

### 유틸리티

| 파일 | 설명 |
|------|------|
| `debug_issue.py` | 데이터 수집 이슈 디버깅 (DB 최신일자, pykrx 데이터 확인 등) |

## 📦 데이터베이스 스키마

| 테이블 | 컬럼 | 설명 |
|--------|-------|------|
| `stocks` | ticker, name, date, open, high, low, close, volume | 전 종목 일봉 데이터 |
| `market_indices` | symbol, name, date, open, high, low, close, volume | KOSPI/KOSDAQ 지수 |
| `themes` | ticker, name, themes, sector, summary | 종목별 테마, 업종, 기업 요약 |
| `trading_details` | ticker, date, close, volume, trading_value, foreign_net, inst_net, foreign_ratio | 일별 거래대금 + 기관/외인 순매수 (ka10015+ka10045) |
| `stock_all` | ticker, name, date, open, high, low, close, amount, volume, 외국인, 개인, 기관계 | 종합 일봉 + 투자자별 순매수 (ka10081+ka10060) |
| `financial_summary` | ticker, year, is_estimate, revenue, operating_income, eps, per, pbr, roe, ev_ebitda, pcr, bps, dps, dividend_yield 등 | Financial Summary 연간 (cF1002) |
| `financial_annual` | ticker, year, quarter, revenue, operating_income, net_income, eps, roe, roa, roic 등 25개 컬럼 | 연간+분기 손익계산서+재무비율 (cF3002+cF4002) |
| `consensus_summary` | ticker, rating, target_price, eps, per, analyst_count | 투자의견 컨센서스 요약 |
| `consensus_provider` | ticker, provider, report_date, target_price, prev_target, change_pct, opinion | 제공처별 투자의견 |

### `stock_all` 테이블 상세

전 종목의 일봉 OHLCV, 거래대금, 투자자별 순매수 데이터를 하나의 테이블에 통합한 종합 데이터입니다.

| 컬럼 | 타입 | 설명 | API 출처 |
|------|------|------|----------|
| `ticker` | VARCHAR(20) | 종목코드 (PK) | stocks 테이블 |
| `name` | VARCHAR(100) | 종목명 | stocks 테이블 |
| `date` | DATE | 날짜 (PK) | ka10081 `dt` |
| `open` | BIGINT | 시가 | ka10081 `open_pric` |
| `high` | BIGINT | 고가 | ka10081 `high_pric` |
| `low` | BIGINT | 저가 | ka10081 `low_pric` |
| `close` | BIGINT | 종가 | ka10081 `cur_prc` |
| `amount` | BIGINT | 거래대금 (백만원) | ka10081 `trde_prica` |
| `volume` | BIGINT | 거래량 (주) | ka10081 `trde_qty` |
| `외국인` | BIGINT | 외국인 순매수 (주) | ka10060 `frgnr_invsr` |
| `개인` | BIGINT | 개인 순매수 (주) | ka10060 `ind_invsr` |
| `기관계` | BIGINT | 기관계 순매수 (주) | ka10060 `orgn` |

**데이터 범위:**
- ka10081 (일봉): 1회 호출 시 기준일 기준 과거 **600 거래일** 반환 (약 2.5년)
- ka10060 (투자자): 1회 호출 시 기준일 기준 과거 **100 거래일** 반환 (약 5개월)
- 투자자 데이터가 없는 날짜의 외국인/개인/기관계 값은 NULL

## 📄 데이터 파일 (CSV)

| 파일 | 설명 |
|------|------|
| `korean_stock_company_summaries.csv` | 기업 요약 데이터 (2,794종목) |
| `naver_themes.csv` | 네이버 증권 테마-종목 매핑 원본 (6,511건) |
| `naver_themes_by_ticker.csv` | 티커별 네이버 테마 집계 (2,397종목) |

## 🛠️ 설치 및 환경 설정

### 필수 요구사항
*   **Python 3.x**
*   **PostgreSQL**

### 라이브러리 설치
```bash
pip install pandas sqlalchemy psycopg2-binary pykrx streamlit plotly tqdm python-dotenv FinanceDataReader requests beautifulsoup4 textual
```

### 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성합니다.

```env
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="postgres"
DB_USER="사용자명"
DB_PASSWORD="비밀번호"
```

키움 REST API 사용 시 추가 설정이 필요합니다:
```env
KIWOOM_APP_KEY="앱키"
KIWOOM_APP_SECRET="앱시크릿"
```

## 🚀 사용 방법

### 1단계: 초기 데이터 구축
```bash
python get_stocks.py          # 전 종목 OHLCV 수집 (기본: 2019-01-01 ~ 오늘)
```

### 2단계: 일일 데이터 업데이트
```bash
python add_daily_stocks.py    # 최신 주가 + 시장 지수 업데이트
```

### 3단계: 종합 데이터 수집 (키움 REST API)

```bash
python get_stock_all.py       # 전 종목 일봉 + 거래대금 + 투자자별 순매수 수집
```

- **첫 실행**: `stock_all` 테이블 생성 → 과거 600거래일 전체 수집 (종목당 API 2회, 약 30분 소요)
- **재실행**: DB 최종일 이후 ~ 오늘 데이터만 증분 수집
- **당일 재실행**: 오늘 데이터 삭제 후 재수집 (장중 갱신용)

### 4단계: 테마/업종 데이터 수집
```bash
python get_stock_themes.py    # 키움 API로 테마/업종 수집
python scrape_naver_themes.py # 네이버 증권 테마 스크래핑
python add_summary_to_themes.py  # 기업 요약 추가
```

### 5단계: 재무데이터 수집 (네이버 증권)
```bash
python run_financial_crawl.py   # 3개 크롤러 통합 실행
```
또는 개별 실행:
```bash
python crawl_financial_summary.py  # Financial Summary (cF1002)
python crawl_financial.py          # 연간+분기 재무데이터 (cF3002+cF4002)
python crawl_consensus.py          # 컨센서스 + 주요지표 (c1010001)
```

### 6단계: 대시보드 실행

```bash
streamlit run chart_test.py   # 웹 차트 대시보드
```

### 테마 편집
```bash
python theme_edit.py          # 터미널 UI로 테마/요약 편집
```

## 📊 대시보드 주요 기능
*   **종목 검색**: 한글 종목명 또는 티커 코드로 검색
*   **차트 모드**: 캔들(Candle) / 하이킨아시(Heikin-Ashi) 전환
*   **기술적 지표**:
    *   이동평균선(MA) - 기간, 색상, 굵기 커스터마이징
    *   볼린저밴드(BB) - 기간, 승수(k), 상/하한선 설정
    *   스무스드 하이킨아시(SHA) - 추세 파악용 오버레이
*   **인터랙티브**: 확대/축소, 패닝, 마우스 오버 상세 정보
