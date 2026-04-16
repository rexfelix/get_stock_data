# MTL/MTH V2 + S10+F1 전략 — 자동매매 규칙

> 실전 자동매매 코딩을 위한 전략 설명 및 의사코드

## 1. 전략 구성 요소

### MTL (Medium-Term Low, 중기저점)
- 먼저 STL(단기저점) 계산: 어느 봉의 저가가 좌우 봉보다 낮으면 STL 확정 (다음 봉에서)
- **연속 3개의 STL 중 가운데 STL의 저가가 가장 낮을 때** → 가운데 STL이 MTL로 확정
- 확정 시점: 3번째 STL이 확정되는 날

### MTH (Medium-Term High, 중기고점)
- STH(단기고점) 동일 방식: 좌우 봉보다 높은 고가
- **연속 3개 STH 중 가운데가 최고일 때** MTH 확정

### S10
- 최대 10종목 동시 보유
- 슬롯당 자본 1천만원 (초기자본 = 1억)

### F1 (지수 필터)
- KOSPI 지수 종가가 KOSPI 60일 이동평균보다 위일 때만 **신규 매수** 허용
- F1 미통과 시 신규 매수 전면 중단 (보유분은 매도 규칙 따름)

### V2 (손절선)
- 진입 기준이 된 MTH 캔들의 **저가**를 손절선으로 사용

### 유니버스
- KOSPI200 ∪ 시가총액 3조원 이상 (약 349종목)

---

## 2. 매매 흐름

### 매일 장 마감 직전 실행

1. **모든 종목의 STL/STH/MTL/MTH를 최신까지 확정**
2. **매도 먼저 체크**
   - 보유 종목 중, 당일 종가가 진입 MTH 캔들 저가보다 아래면 → **손절 매도**
   - 손절 미발동 종목 중, 당일 종가가 최신 MTL 저가보다 아래면 → **트레일링 매도**
3. **매수 조건 확인**
   - KOSPI 종가 > KOSPI MA60 인지 확인 (아니면 오늘 신규 매수 전면 중단)
   - 각 종목별: 최신 MTL 확정 이후 형성된 MTH가 있고, 당일 종가가 그 MTH 고가를 상향 돌파하는가?
4. **매수 실행**
   - 조건 통과한 종목들을 **당일 거래대금 순으로 정렬**
   - 비어있는 슬롯 수(최대 10 - 현재 보유수) 만큼 상위 종목부터 종가 매수
   - 슬롯당 1천만원 할당

---

## 3. 자동매매 의사코드

```python
# ──────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────
SLOT_CAPITAL = 10_000_000        # 슬롯당 1천만원
MAX_POSITIONS = 10               # 최대 동시 보유
UNIVERSE = KOSPI200 ∪ 시가총액_3조_이상   # 약 349종목

# ──────────────────────────────────────────────
# 스윙 포인트 탐지 함수
# ──────────────────────────────────────────────
def find_stl_sth(ohlcv):
    stls, sths = [], []
    for i in range(1, len(ohlcv) - 1):
        # STL: 좌우 봉보다 저가가 낮음
        if ohlcv.low[i] < ohlcv.low[i-1] and ohlcv.low[i] < ohlcv.low[i+1]:
            stls.append({pivot_idx: i, confirmed_at: i+1, low: ohlcv.low[i]})
        # STH: 좌우 봉보다 고가가 높음
        if ohlcv.high[i] > ohlcv.high[i-1] and ohlcv.high[i] > ohlcv.high[i+1]:
            sths.append({pivot_idx: i, confirmed_at: i+1, high: ohlcv.high[i]})
    return stls, sths


def find_mtl_mth(stls, sths):
    mtls, mths = [], []
    # MTL: 연속 3개 STL 중 가운데가 최저
    for k in range(2, len(stls)):
        a, b, c = stls[k-2], stls[k-1], stls[k]
        if b.low < a.low and b.low < c.low:
            mtls.append({pivot_idx: b.pivot_idx, confirmed_at: c.confirmed_at,
                         low: b.low, high: b.high})  # b 캔들의 OHLC
    # MTH: 연속 3개 STH 중 가운데가 최고
    for k in range(2, len(sths)):
        a, b, c = sths[k-2], sths[k-1], sths[k]
        if b.high > a.high and b.high > c.high:
            mths.append({pivot_idx: b.pivot_idx, confirmed_at: c.confirmed_at,
                         low: b.low, high: b.high})
    return mtls, mths


# ──────────────────────────────────────────────
# 일일 매매 루프 (매일 장 마감 10~30분 전 실행)
# ──────────────────────────────────────────────
def daily_trading(today):
    # ── STEP 1: 지수 필터 계산 (KOSPI > MA60)
    kospi = get_index_ohlcv("^KS11", lookback=80)
    kospi_ma60 = kospi.close.rolling(60).mean().iloc[-1]
    market_ok = kospi.close.iloc[-1] > kospi_ma60   # 오늘 신규 매수 허용 여부

    # ── STEP 2: 매도 체크 (보유 종목)
    for pos in open_positions():
        ohlcv = get_stock_ohlcv(pos.ticker)
        close_today = ohlcv.close.iloc[-1]
        stls, sths = find_stl_sth(ohlcv)
        mtls, _ = find_mtl_mth(stls, sths)

        # [손절] 진입 MTH 캔들 저가 이탈
        if close_today < pos.entry_mth_low:
            sell(pos.ticker, qty=pos.quantity, reason="MTH저가손절")
            continue

        # [트레일링] 최신 확정 MTL 저가 이탈
        latest_mtl = max(m for m in mtls if m.confirmed_at <= today_idx)
        if latest_mtl and close_today < latest_mtl.low:
            sell(pos.ticker, qty=pos.quantity, reason="MTL이탈")

    # ── STEP 3: 매수 시그널 스캔 (F1 통과 시에만)
    if not market_ok:
        return   # 신규 매수 중단 — 보유분만 매도 규칙 따름

    candidates = []
    for ticker in UNIVERSE:
        if ticker in held_tickers():
            continue

        ohlcv = get_stock_ohlcv(ticker)
        if len(ohlcv) < 20:
            continue

        stls, sths = find_stl_sth(ohlcv)
        mtls, mths = find_mtl_mth(stls, sths)
        today_idx = len(ohlcv) - 1

        # 최신 확정 MTL
        latest_mtl = last(m for m in mtls if m.confirmed_at <= today_idx)
        if not latest_mtl:
            continue

        # MTL 이후 확정된 MTH
        latest_mth = last(m for m in mths
                          if m.confirmed_at <= today_idx
                          and m.pivot_idx > latest_mtl.pivot_idx)
        if not latest_mth:
            continue

        # 돌파 조건: 전일 종가 ≤ MTH 고가 < 당일 종가
        prev_close = ohlcv.close.iloc[-2]
        close_today = ohlcv.close.iloc[-1]
        if prev_close <= latest_mth.high and close_today > latest_mth.high:
            amount_today = ohlcv.volume.iloc[-1] * close_today   # 거래대금
            candidates.append({
                ticker: ticker,
                entry_price: close_today,
                mth_low: latest_mth.low,     # 손절선으로 저장
                amount: amount_today,
            })

    # ── STEP 4: 거래대금 순 매수
    candidates.sort(key=lambda c: -c.amount)
    slots_free = MAX_POSITIONS - len(open_positions())

    for cand in candidates[:slots_free]:
        if cash() < SLOT_CAPITAL:
            break
        qty = floor(SLOT_CAPITAL / cand.entry_price)
        if qty <= 0:
            continue

        buy(cand.ticker, qty=qty, price=cand.entry_price)
        save_position({
            ticker: cand.ticker,
            entry_price: cand.entry_price,
            quantity: qty,
            entry_date: today,
            entry_mth_low: cand.mth_low,     # 손절선
        })
```

---

## 4. 실전 주의사항

| 항목 | 설명 |
|---|---|
| **실행 타이밍** | 일봉 기반 전략 → 장 마감 20~30분 전 OR 다음날 시가 근처. 종가 기준 시그널이므로 마감 임박 실행 권장 |
| **슬리피지** | 돌파 매수 시 상승 중 체결 → 실제 체결가는 종가보다 높을 수 있음 (0.3~0.5% 여유) |
| **거래대금 데이터** | 한국투자증권/키움 API는 실시간 `amount` 제공. 장중 누적치 사용 |
| **MTL/MTH 실시간 확정** | STL/STH는 "다음 봉"에서 확정 → 실시간 판단 불가, **최소 1봉 지연** 필연적 |
| **손절은 종가 기준** | 장중 손절가 터치해도 종가가 회복하면 유지 → 실전에서 **장 마감 30분 전 스냅샷**으로 판단 |
| **자본 배분** | 슬롯당 1천만 고정 가정. 실전 자본 확대 시 슬롯당 자본 비례 증가하되 슬롯 수는 유지(10개) |

---

## 5. 연결 가능한 실전 시스템

- **키움 REST API**: `ka10081`(일봉), `ka10015`(거래대금), 종목별 OHLCV 수집
- **시그널 계산**: Python 스크립트로 매일 장 마감 전 실행, 매수/매도 후보 리스트 생성
- **주문 실행**: 키움 API로 지정가 or 시장가 매수/매도
- **모니터링**: 포지션 테이블에 `entry_mth_low` 반드시 저장 (손절 기준)

---

## 6. 백테스트 검증 성과 (2023-01-01 ~ 2026-04-17)

| 지표 | 값 |
|---|---|
| 초기자본 | 100,000,000원 |
| 최종자본 | 215,390,000원 (+115%) |
| **CAGR** | **26.31%** |
| **MDD** | **-51.23%** |
| Calmar | 0.51 |
| 거래수 | 198건 |
| 승률 | 31.3% |
| 손익비 (PF) | **1.93** |
| 평균 보유일 | 49.2일 |

### 연도별 성과

| 연도 | 수익률 | MDD | 거래수 | 승률 |
|---|:---:|:---:|:---:|:---:|
| 2023 | +39.08% | -15.39% | 62건 | 29.0% |
| 2024 | -1.52% 🛡 | -19.21% | 49건 | 22.4% |
| 2025 | +36.50% | -51.23% | 66건 | 40.9% |
| 2026 (YTD) | +9.83% | -16.81% | 21건 | 28.6% |

### TOP 수익 거래

| 종목 | 수익률 | 보유일 | 사유 |
|---|:---:|:---:|---|
| 이수페타시스 | +175.90% | 117일 | MTL이탈 |
| 실리콘투 | +172.67% | 103일 | MTL이탈 |
| 현대무벡스 | +125.28% | 82일 | MTL이탈 |
| 삼성전기 | +125.21% | 84일 | 미청산 |
| 현대차 | +117.90% | 243일 | MTL이탈 |
| 한화에어로스페이스 | +113.90% | 301일 | MTL이탈 |
| 한미반도체 | +109.11% | 90일 | MTL이탈 |

---

**관련 파일**
- 백테스트 스크립트: `backtest_mtl_mth_portfolio_v2.py`
- 백테스트 결과: `results/backtest_mtl_mth_portfolio_v2.md`
- V1 (손절 없는 버전): `backtest_mtl_mth_portfolio.py`

**생성일시**: 2026-04-17
