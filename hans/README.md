# Hans — 한의 법칙 검증 & 주도주 생애주기 스크리닝

한국 주식 주도주의 "공세 유한성"을 설명하는 **한의 법칙(Han's Rule)**을 KOSPI200 데이터로 검증하고, 현재 어느 종목이 생애주기 어느 단계에 있는지 분류하는 독립 프로젝트.

## 구조

```
hans/
├── Hans_rule.md                  # 법칙 원문
├── db.py                         # 독립 DB 연결 (PostgreSQL ENGINE)
├── verify_hans_rule.py           # 법칙 검증 (시간구조 + 실적델타)
├── screen_diffusion_stage.py     # 생애주기 4단계 스크리너
├── results/
│   ├── Hans_rule_verification.md     # 검증 리포트 (자동생성)
│   ├── Hans_rule_diffusion_stage.md  # 단계 분류 리포트 (자동생성)
│   └── hans_rule_list_202605.md      # 2026-05 확산기·형성기 큐레이션
└── memory/                       # 프로젝트 영속 메모리 (세션 간 컨텍스트)
    ├── MEMORY.md                     # 인덱스
    ├── reference_hans_rule_law.md
    ├── reference_db_schema.md
    ├── project_hans_rule_verify.md
    └── project_diffusion_stage_screener.md
```

## 실행

```bash
cd hans
python db.py                      # DB 연결 점검
python verify_hans_rule.py        # 법칙 검증 → results/Hans_rule_verification.md
LAGSCAN=1 python verify_hans_rule.py   # 실적 선행성 시차 스캔(콘솔)
python screen_diffusion_stage.py  # 현재 단계 분류 → results/Hans_rule_diffusion_stage.md
```

의존: `pandas`, `numpy`, `sqlalchemy`, `psycopg2`(postgres), `python-dotenv`.
DB 접속정보는 `../.env`(data_center/.env) 또는 환경변수 `HANS_ENV_PATH`에서 읽는다.

## 검증 결론 (2026-05-31)

- **시간구조 ✅**: 주도주(정배열 후 50%+ 상승) 공세는 중앙 30주, **104주(2년) 이내 99.6% 종료**.
- **실적델타 ✅**: 영업이익이 정배열을 약 1년 후행하여 폭발(FY1 +39%) 후 피크아웃(FY2 +14%).
- 상세는 `results/` 리포트와 `memory/` 참조.

## 메모리 사용 규약

새 세션에서 이 프로젝트를 이어갈 때는 먼저 `memory/MEMORY.md`를 읽어 컨텍스트를 복원하고, 사실이 바뀌면 해당 메모리 파일을 갱신한 뒤 `MEMORY.md` 인덱스 한 줄을 유지한다. 메모리는 작성 시점의 사실이므로, 파일/컬럼/수치를 인용하기 전 현재 코드·DB로 재확인한다.
