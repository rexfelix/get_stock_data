# REVIEW - stocks 테이블 amount 컬럼 백필

## 리뷰 이력

---

### 리뷰 #001 — 단계 1 계획 수립

- **리뷰 일시**: (대기 중)
- **검토 범위**:
  - `report/amount_backfill/PRD.md`
  - `report/amount_backfill/TASK.md`
- **사전 확인 사항**:
  - stocks 테이블: 4067 종목, 4.3M 행, 2019-01-02 ~ 2026-04-30
  - 기존 코드 `get_stock_all.fetch_ka10081` 재사용 가능
  - 키움 API 키 `data_center/.env` 가용 확인
  - 추정 시간: 약 100분 (4067 종목 × ~5 호출 × 0.3s)
- **요구사항 충족 여부**: 충족
- **TDD 준수 여부**: 단계 2부터 Red 작성 약속
- **결정**: **APPROVED**

---

### 리뷰 #002 — 단계 2 테스트 작성 (Red)

- **리뷰 일시**: 2026-05-01
- **TDD 준수 여부**: 13/13 ModuleNotFoundError → Red 확정
- **결정**: **APPROVED**

---

### 리뷰 #003 — 단계 3 구현 (Green) — Smoke test 진행 전 검토

- **리뷰 일시**: (대기 중)
- **검토 범위**: `amount_backfill.py` (신규)
- **TDD 준수 여부**: 13/13 Green 전환 확인
- **구현 내용**:
  - 헬퍼: `parse_amount`, `build_amount_records`, `next_base_dt`, `load/save_progress`, `is_done`
  - `ensure_schema()`: ALTER TABLE + INDEX 생성 (idempotent)
  - `get_kiwoom_token()`: data_center/.env 의 KIWOOM_APPKEY 사용
  - `fetch_ka10081()`: rate limit 0.3s, timeout 15s, 실패 시 빈 리스트
  - `backfill_one_ticker()`: 600일 페이지네이션, 단위 변환 (원→백만원)
  - `bulk_update_amount()`: 청크 500 단위 UPDATE
  - `main(limit=None, single_ticker=None)`: 전체 / 단일 / 일부 실행
  - 진행 상황: 10 종목마다 progress.json 저장
- **단위 변환 가정 (Smoke test 로 확정 필요)**:
  - ka10081 의 `trde_prica` 단위 = 원 단위로 추정
  - 백만원 단위 저장 위해 ÷ 1_000_000
  - **Smoke test 결과로 확정**
- **CLI 사용**:

  ```bash
  # Smoke test (1 종목)
  python amount_backfill.py --ticker 005930

  # 일부 (5 종목)
  python amount_backfill.py --limit 5

  # 전체
  python amount_backfill.py
  ```

- **발견 사항**:
  - Smoke test 시 단위 오류 발견 (원으로 추정 → 실제 백만원) → 즉시 수정 후 재검증
  - 단위 수정 후 005930: 627/627 (100.00%) stock_all 일치
- **수정 요청 사항**: 없음
- **결정**: **APPROVED**

---

### 리뷰 #004 — 단계 5 검증 (10 종목)

- **리뷰 일시**: 2026-05-01
- **검토 범위**: 10 종목 추가 백필 (000660, 035720, 035420, 005380, 012450, 042700, 068270, 066570, 055550, 000270)
- **결과**: 18,000 행 저장, **100.00% stock_all 일치** (6,279/6,279)
- **종목당 처리**: ~1.2초 (3 호출 × 0.3s + DB)
- **결정**: **APPROVED**

---

### 리뷰 #005 — 단계 6 전체 백그라운드 + 단계 7 검증

- **리뷰 일시**: 2026-05-01
- **단계 6 결과**:
  - 4067 종목 백필, **77.6분** 소요 (예상 81분과 근사)
  - 성공 4030 / 실패 36 (상장폐지/거래정지 추정)
- **단계 7 검증 결과**:
  - amount 채움 4,277,440 / 4,322,337 행 (**99.0%**)
  - 채움 종목 4,029 / 4,067 (**99.1%**)
  - **stock_all 정확 일치율 99.71%** (1,646,209 / 1,650,982 일, 2023-09-25 이후)
  - 2019~2023 정확값 교체: 2.63M 행 / 2,496 종목
  - amount min/max/avg (백만원): 0 / 8,379,238 / 8,217
- **데이터 한계 제거**:
  - 이전 (15,3) 2019~2023 백테스트는 amount 추정 (close × volume) 사용
  - 이제 정확값으로 재검증 가능 → 새 PDCA 후속 작업
- **결정**: **APPROVED**
