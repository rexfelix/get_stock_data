# TASK - stocks 테이블 amount 컬럼 백필

## 작업 우선순위

1. PRD 승인
2. 테스트 작성 (Red)
3. 구현 (Green)
4. Smoke test (1 종목)
5. stock_all 검증 (소규모)
6. 전체 백필 실행 (백그라운드)
7. 검증 + 결론

## 작업 단위 목록

### 단계 1. 계획 수립

- [x] PRD / TASK 작성 + 사전 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 2. 테스트 작성 (Red)

- [x] T1~T5 (13 테스트) + Red 확인
- [x] Reviewer 승인 (APPROVED)

### 단계 3. 구현 (Green)

- [x] `amount_backfill.py` 작성 + 13/13 Green
- [x] Smoke 단위 오류 발견 → 단위 수정 (÷1_000_000 제거, trde_prica 그대로 백만원)
- [x] Reviewer 승인 (APPROVED)

### 단계 4. Smoke Test (005930)

- [x] 1800행 저장, 100% stock_all 일치 (627일)
- [x] 단위 = 백만원 확정
- [x] Reviewer 승인 (APPROVED)

### 단계 5. 5~10 종목 검증

- [x] 10 종목 추가 (18,000행), 100.00% stock_all 일치 (6,279일)
- [x] Reviewer 승인 (APPROVED)

### 단계 6. 전체 백필 (백그라운드)

- [x] 4067 종목 실행, 77.6분 소요
- [x] 성공 4030 / 실패 36
- [x] amount 채움률 99.0% (4.27M/4.32M 행)
- [x] Reviewer 승인 (APPROVED)

### 단계 7. 검증 + 결론

- [x] stock_all 일치율 99.71% (2023-09-25~ 1.65M 일 중 1.65M 정확 일치)
- [x] 2019~2023 정확값 교체 구간: 2.63M 행 (2496 종목)
- [x] 실패 36 종목 = 상장폐지/거래정지 추정 (별도 분석 후속)
- [x] 후속 작업: (15,3) 2019~2023 정확 amount 재검증 (새 PDCA)
- [ ] **결론 Reviewer 승인 대기**

### 전체 작업 ✅

## 의존 관계

- 키움 API 키 (data_center/.env): KIWOOM_APPKEY, KIWOOM_SECRETKEY
- 기존 코드: `data_center/get_stock_all.py` 의 `fetch_ka10081`, `parse_int`
- DB: stocks 테이블 (4067 종목, 4.3M 행)

## 현재 진행 상태

- 단계 1 작성 완료, Reviewer 승인 대기
- DB 스키마 / API 키 가용 확인 완료

## 완료 여부 체크박스 합계

- 단계 1: 2/3 (승인 대기)
- 단계 2: 0/6
- 단계 3: 0/3
- 단계 4: 0/4
- 단계 5: 0/4
- 단계 6: 0/4
- 단계 7: 0/4
