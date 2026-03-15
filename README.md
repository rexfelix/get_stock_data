# 주식 데이터 센터 및 차트 대시보드 (Stock Data Center & Chart Dashboard)

KOSPI/KOSDAQ 전 종목의 일봉 데이터와 테마/업종 정보를 수집, 저장하고 이를 시각화하여 분석할 수 있는 로컬 데이터 플랫폼입니다. PostgreSQL에 데이터를 저장하며, Streamlit 차트 대시보드와 터미널 기반 테마 편집 도구를 제공합니다.

## 📁 프로젝트 구조

### 주가 데이터 수집

| 파일 | 설명 |
|------|------|
| `get_stocks.py` | **초기 데이터 구축**. 전 종목 OHLCV 데이터를 수집하여 `stocks` 테이블 초기화 (멀티프로세싱 4코어, 배치 저장) |
| `add_daily_stocks.py` | **일일 업데이트**. DB 마지막 날짜 이후 데이터 추가 수집 + `market_indices` 테이블에 KOSPI/KOSDAQ 지수 업데이트 |
| `verify_stocks.py` | **데이터 검증**. 행 개수, 고유 티커 수, 컬럼 구조 등 DB 정합성 확인 |

### 테마/업종 데이터 수집

| 파일 | 설명 |
|------|------|
| `get_stock_themes.py` | **키움 REST API** 기반 테마/업종 수집. `themes` 테이블에 저장 (142개 테마그룹, 648종목) |
| `scrape_naver_themes.py` | **네이버 증권 테마 스크래핑**. 전체 테마별 구성 종목을 수집하여 CSV 생성 (~2,397 종목) |
| `add_summary_to_themes.py` | `themes` 테이블에 기업 요약(summary) 컬럼 추가 및 CSV 데이터 매핑 |

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

### 3단계: 테마/업종 데이터 수집
```bash
python get_stock_themes.py    # 키움 API로 테마/업종 수집
python scrape_naver_themes.py # 네이버 증권 테마 스크래핑
python add_summary_to_themes.py  # 기업 요약 추가
```

### 4단계: 대시보드 실행
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
