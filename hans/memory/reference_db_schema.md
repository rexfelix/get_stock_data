---
name: reference_db_schema
description: hans 프로젝트 DB 연결/스키마 주의사항 (PostgreSQL, db.py, 테이블 함정)
metadata:
  type: reference
---

hans 프로젝트는 `hans/db.py`의 `ENGINE`으로 DB에 접속한다(backtest_crash 의존 제거). db.py는 `../.env`(data_center/.env) 또는 `HANS_ENV_PATH` 환경변수에서 접속정보를 읽는다. 점검: `python hans/db.py`.

**DB**: PostgreSQL, 기본 `stock_db` @ localhost:5432 (user rexfelix). MySQL/pymysql 아님 — pymysql import 금지.

**스키마 함정(검증 중 실측)**:
- `kospi200_members`: 컬럼 `ticker / name / updated_date` 3개, 200행. **`is_active` 컬럼 없음**. 현재 구성만 보유(과거 편입·퇴출 미보존 → 생존편향).
- `stocks`: 일봉. 컬럼 date/open/high/low/close/volume/ticker/name/amount. **날짜 범위 2019-01-02 ~ 현재**(그 이전 없음). close 등은 bigint.
- `financial_annual`: 영업이익 컬럼명은 **`operating_income`** (operating_profit 아님). `quarter` 컬럼 존재 — **quarter=0 이 연간 누적치**, quarter=1~4는 분기 단건. 연간 분석 시 반드시 `WHERE quarter=0`. `is_estimate`(추정치) 혼재. double precision.
- 기타 가용 테이블: stock_master(유니버스 단일소스, is_listed 보존), consensus_summary, market_indices, themes 등.

영업이익 분석 시 주의: 흑자전환·저기저·분할로 성장률 노이즈 큼(예: +3018%, 적자전환). 견고한 신호는 표본 중앙값. 가격(정배열)이 영업이익을 약 1년 선행 → 온셋연도=FY0 매핑. 상세 [[project_hans_rule_verify]].
