---
name: project_hans_rule_verify
description: 한의 법칙 KOSPI200 검증 결과 — 시간구조·실적델타 모두 지지, 가격이 실적 1년 선행
metadata:
  type: project
---

한의 법칙(Han's Rule) KOSPI200 실증 검증 (2026-05-31). 스크립트 `hans/verify_hans_rule.py`, 리포트 `hans/results/Hans_rule_verification.md`. 재현 `cd hans && python verify_hans_rule.py`(약 2초). `LAGSCAN=1 python verify_hans_rule.py`로 실적 선행성만 콘솔 출력.

방법: kospi200_members 200종목(199개 주봉≥60), stocks 일봉(2019~2026)→주봉 W-FRI→MA4/13/26/52. 온셋=정배열(MA4>13>26>52) 거짓→참 전환주. 공세종말점=추세붕괴 직전 최고 주봉종가, 생존=온셋→고점(주). 붕괴정의 2종: sensitive(MA4<MA13|close<MA52), structural(MA13<MA26|close<MA52). 주도주=structural∩고점수익≥50%.

**시간구조 ✅지지**:
- sensitive 1559건(1472종료) 중앙3주 104주내 100.0%
- structural 1559건(1353종료) 중앙5주 p95 42주 104주내 99.9%
- 주도주 356건(233종료) 중앙30주 mean33.4 p95 74주 max128 104주내 99.6% (누적 ≤52주85%·≤78주96%·≤104주100%) → 미국 138건 중앙59주·130주내93%와 구조 일치(한국이 더 짧음)

**실적델타 ✅지지 — 핵심: 가격이 영업이익 ~1년 선행**: LAGSCAN(주도주, 온셋연도 Y 대비 영업이익 성장 중앙): Y-2→Y-1 +17.2%, Y-1→Y +33.1%, **Y→Y+1 +40.6%(정점)**, Y+1→Y+2 +11.0%, Y+2→Y+3 +11.8%. 최대성장 해의 온셋대비 중앙 오프셋 +1.0년. 온셋연도=FY0로 매핑하면 주도주 FY0→FY1 +39.0% 폭발 → FY1→FY2 +14.3% 둔화, 피크아웃 64%. (온셋=FY1로 잘못 매핑하면 반증처럼 보임 — 선행성 보정 필수.)

대표 주도주: HD현대일렉트릭·삼성전기·SK스퀘어·SK하이닉스·효성중공업·한미반도체·한화에어로스페이스·두산 (2022~2025 반도체/조선/방산/전력기기 cohort).

결론: 시간구조·실적델타 모두 KOSPI200에서 성립. 진짜 주도주(≥50%)로 좁혀야 또렷. 한계: 2019~데이터(censoring), 현재구성 생존편향, 50%컷 임의, KM 미적용. 스키마 주의는 [[reference_db_schema]], 법칙 원문은 [[reference_hans_rule_law]], 현재 단계 스크리닝은 [[project_diffusion_stage_screener]].
