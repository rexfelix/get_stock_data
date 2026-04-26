# TASK - Envelope 매매 KODEX 레버리지 단일 종목 최적화

## 작업 우선순위

1. PRD/TASK 작성 → Reviewer 승인 (1단계)
2. 신규 실패 테스트 T7·T8 작성 → Reviewer 승인 (2단계)
3. fdr 로더 + 단일 종목 진입점 구현 → Reviewer 승인 (3단계)
4. 48 조합 그리드 실행 + 리포트 → Reviewer 승인 (4단계)
5. 최종 보고서 + 메모리 갱신 (5단계)

## 의존 관계

- 인터넷 접근 가능해야 fdr이 122630 데이터 수집 가능 (사전 검증 완료).
- `backtest_envelope.py`의 기존 함수(`_precompute_ma_per_ticker`,
  `_simulate_with_precomputed`)를 모듈 상수 의존 제거 후 파라미터로 주입하도록
  리팩토링 필요. **단, 기존 9 테스트가 통과 유지되도록 보호**.
- 신규 진입점은 별도 스크립트 `backtest_envelope_kodex_lev.py` 또는
  `backtest_envelope.py`에 함수 추가.

## 작업 단위별 목표 및 체크박스

### 1단계: 계획 수립

- [x] PRD 작성 (`report/envelope_kodex_lev/PRD.md`)
- [x] TASK 작성 (`report/envelope_kodex_lev/TASK.md`)
- [ ] Reviewer 승인 (REVIEW 1단계 기록)

### 2단계: 실패 테스트 작성 (Red)

- [ ] T7: `test_load_kodex_lev_format` — fdr 결과(mock) → 표준 OHLCV 컬럼 변환
  (date/open/high/low/close/volume/ticker/name) + dtype 확인
- [ ] T8: `test_simulate_etf_no_tax` — 동일 매매 시퀀스에서 tax=0 vs tax=0.0018
  매도 순익 차이가 정확히 `sell_value × 0.0018` 만큼 발생하는지 검증
- [ ] 스텁/인터페이스 정의 후 pytest로 실패 확인
- [ ] Reviewer 승인 (REVIEW 2단계 기록)

### 3단계: 최소 구현 (Green)

- [ ] `_simulate_with_precomputed` 시그니처 변경:
  - 모듈 상수(MAX_POSITIONS, SLOT_CAPITAL, COMMISSION 등) 의존 제거
  - 파라미터로 주입 (또는 별도 헬퍼 함수 추가)
- [ ] `load_kodex_lev(start)` 함수 추가
- [ ] `run_grid_single_ticker(df, ...)` 단일 종목 그리드 러너
- [ ] `write_report_kodex_lev(...)` 리포트 작성기
- [ ] `main_kodex_lev()` 진입점
- [ ] **기존 9 테스트 + 신규 2 테스트 모두 통과 확인**
- [ ] Reviewer 승인 (REVIEW 3단계 기록)

### 4단계: 그리드 실행 + 리포트

- [ ] `python -c "from backtest_envelope import main_kodex_lev; main_kodex_lev()"`
  실행 → 48 조합 결과 생성
- [ ] `results/backtest_envelope_kodex_lev.md` 작성:
  - 헤더(종목·기간·자본·비용·룰·그리드)
  - 조합별 표 (CAGR 내림차순, Calmar 칼럼 포함)
  - 상위 5 상세 (X% 무효성 재발생 시 유효 결과 5선으로 축약)
  - 결론 (이전 사이클 KOSPI/KOSDAQ 결과와의 비교)
- [ ] 테스트 재실행 → 11건 통과 유지
- [ ] Reviewer 승인 (REVIEW 4단계 기록)

### 5단계: 최종 보고서 + 메모리

- [ ] `report/envelope_kodex_lev/FINAL_REPORT.md` 작성
- [ ] 메모리 `project_envelope_kodex_lev.md` + MEMORY.md 갱신

## 테스트 항목

| ID | 검증 대상 | 상태 |
| --- | --- | --- |
| T1~T6, T4-a/b, T6-a/b (기존 9건) | 회귀 보호 | 통과 유지 필요 |
| T7 | fdr → 표준 변환 | 미작성 |
| T8 | ETF 매도세 0% 반영 | 미작성 |

## 현재 진행 상태

- **현재 단계**: 5단계 종결 (모든 단계 APPROVED, 사이클 완료)
- **다음 게이트**: 없음 (사이클 종결)
- **블로커**: 없음
- 최종 보고서: `report/envelope_kodex_lev/FINAL_REPORT.md`
- 핵심 결과: MA120 ±5% CAGR 13.92%, MDD −51.76%, Calmar 0.269
