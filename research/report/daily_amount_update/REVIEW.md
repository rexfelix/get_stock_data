# REVIEW - add_daily_stocks.py 일일 amount 업데이트

## 리뷰 이력

---

### 리뷰 #001 — 단계 1 계획 수립

- **리뷰 일시**: (대기 중)
- **검토 범위**:
  - `report/daily_amount_update/PRD.md`
  - `report/daily_amount_update/TASK.md`
- **변경 핵심 요약**:
  - 변경 1: start_date 로직 통합 (last_date 포함 재수집, 분기 제거)
  - 변경 2: amount 수집 추가 (키움 ka10081, 백만원 단위)
  - amount_backfill PDCA 의 헬퍼 재사용
- **사전 인지 사항**:
  - 기존 OHLCV 수집 (FDR) 변경 안 함 → 안전
  - amount 수집 시간 4067 × 0.3s ≈ 20분 추가
- **요구사항 충족 여부**: 충족
- **옵션 결정**: 옵션 A (FDR + Kiwoom 분리) 채택
- **TDD 준수 여부**: 단계 2부터 Red 작성 약속
- **결정**: **APPROVED**

---

### 리뷰 #002 — 단계 2 테스트 작성 (Red)

- **리뷰 일시**: 2026-05-01
- **TDD 준수 여부**: 8/8 Red 확정
- **결정**: **APPROVED**

---

### 리뷰 #003 — 단계 3 구현 (Green)

- **리뷰 일시**: (대기 중)
- **검토 범위**:
  - `data_center/add_daily_stocks.py` (수정)
- **변경 요약**:
  - 신규 함수 7개 추가 (compute_start_date, filter_records_by_date_range, parse_amount, build_amount_records, fetch_ka10081, bulk_update_stocks_amount, update_amount_for_period)
  - main() start_date 로직 단순화 (last_date 그대로 사용, 분기 제거)
  - main() OHLCV 저장 후 amount 수집 호출 추가
- **TDD 준수 여부**: 8/8 Green + 기존 8 테스트 회귀 없음 (16/16 PASS)
- **단위 일관성**: ka10081 trde_prica 백만원 단위 그대로 저장 (amount_backfill PDCA 검증)
- **안전성**: amount 수집 try/except 로 감싸 OHLCV 저장 정상 진행 보장
- **추가 최적화 (옵션 C)**: `update_amount_for_period` 에 `ThreadPoolExecutor(max_workers=3)` 병렬화
  - 직렬 4067 × 0.3s = 20분 → 병렬 3 = **약 7분** (안전 마진)
  - 키움 API rate limit 위반 위험 낮음
  - 기존 OHLCV (FDR multiprocessing 4) / ticker 리스트 (ThreadPool 2) 등 다른 병렬은 그대로 유지
- **단계 4 (실제 실행)**: 사용자 결정으로 운영자가 매일 실행 시 검증 — 본 PDCA 에서는 미진행
- **결정**: **APPROVED**
