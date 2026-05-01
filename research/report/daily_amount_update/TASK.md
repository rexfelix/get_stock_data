# TASK - add_daily_stocks.py 일일 amount 업데이트

## 작업 우선순위

1. PRD 승인
2. 테스트 작성 (Red)
3. 구현 (Green)
4. 실제 실행 + 검증

## 작업 단위 목록

### 단계 1. 계획 수립

- [x] PRD / TASK 작성 + 기존 코드 분석
- [x] 옵션 A (FDR + Kiwoom 분리) 채택
- [x] Reviewer 승인 (APPROVED)

### 단계 2. 테스트 작성 (Red)

- [x] T1~T4 (8 테스트) + Red 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 3. 구현 (Green)

- [x] `add_daily_stocks.py` 수정 — 신규 함수 7개 추가 + main() 변경
- [x] start_date 로직 변경 / amount 수집 main() 통합 / 헬퍼 inline
- [x] **옵션 C 적용**: ThreadPoolExecutor(max_workers=3) 병렬화 (20분→7분)
- [x] 8/8 Green + 기존 8 회귀 없음 (16/16 PASS)
- [x] Reviewer 승인 (APPROVED)

### 단계 4. 실제 실행 + 검증 (운영자 책임)

- [-] 운영자가 매일 `python add_daily_stocks.py` 실행 시 자동 검증
- [-] 본 PDCA 에서는 코드 수정만 진행 (사용자 결정)

### 전체 작업 ✅

## 의존 관계

- `data_center/add_daily_stocks.py` (기존)
- `research/amount_backfill.py` (헬퍼 재사용)
- DB: stocks 테이블 (amount 컬럼 이미 추가됨)
- 키움 API 키 (data_center/.env)

## 현재 진행 상태

- 단계 1 작성 완료, Reviewer 승인 대기

## 완료 여부 체크박스 합계

- 단계 1: 2/3 (승인 대기)
- 단계 2: 0/5
- 단계 3: 0/4
- 단계 4: 0/4
