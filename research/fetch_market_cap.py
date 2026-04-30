"""
KOSPI200 시가총액 스냅샷 수집 (ka10001).
- kospi200_members 200종목 ticker 조회
- 키움 ka10001로 mac(시가총액), cur_prc(현재가) 수집
- 상장주식수 = mac / cur_prc 추정
- report/top3_indicators/market_cap_snapshot.csv 저장
"""

import os
import time
import requests
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .env: data_center/.env에 KIWOOM 키, research/.env에 DB 정보가 있음
load_dotenv("/Volumes/SSD/project/py/invest/data_center/.env")
load_dotenv("/Volumes/SSD/project/py/invest/data_center/research/.env", override=False)

DB_USER = os.getenv("DB_USER", "rexfelix")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "stock_db")

KIWOOM_APPKEY = os.getenv("KIWOOM_APPKEY", "")
KIWOOM_SECRETKEY = os.getenv("KIWOOM_SECRETKEY", "")
KIWOOM_DOMAIN = "https://api.kiwoom.com"

ENGINE = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

OUTPUT_CSV = "/Volumes/SSD/project/py/invest/data_center/research/report/top3_indicators/market_cap_snapshot.csv"


def get_kiwoom_token():
    url = f"{KIWOOM_DOMAIN}/oauth2/token"
    headers = {"Content-Type": "application/json;charset=UTF-8", "api-id": "au10001"}
    body = {
        "grant_type": "client_credentials",
        "appkey": KIWOOM_APPKEY,
        "secretkey": KIWOOM_SECRETKEY,
    }
    r = requests.post(url, json=body, headers=headers, timeout=10)
    data = r.json()
    if data.get("return_code") != 0:
        raise RuntimeError(f"토큰 발급 실패: {data.get('return_msg')}")
    print(f"토큰 발급 성공 (만료: {data.get('expires_dt')})")
    return data["token"]


def fetch_ka10001(token, ticker):
    """ka10001 주식기본정보요청 → mac(시가총액), cur_prc(현재가) 등."""
    url = f"{KIWOOM_DOMAIN}/api/dostk/stkinfo"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "ka10001",
        "authorization": f"Bearer {token}",
    }
    body = {"stk_cd": ticker}

    time.sleep(0.25)
    r = requests.post(url, json=body, headers=headers, timeout=10)
    data = r.json()
    if data.get("return_code") != 0:
        return None
    return data


def parse_int_safe(val):
    """음수 부호는 등락방향 표시 → 절댓값으로 변환."""
    if val is None or val == "":
        return None
    try:
        s = str(val).replace("+", "").replace(",", "").replace("-", "")
        return int(s)
    except (ValueError, TypeError):
        return None


def get_kospi200_tickers():
    df = pd.read_sql("SELECT ticker, name FROM kospi200_members ORDER BY ticker", ENGINE)
    return list(zip(df["ticker"], df["name"]))


def main():
    tickers = get_kospi200_tickers()
    print(f"KOSPI200 {len(tickers)}종목 시가총액 수집 시작")

    token = get_kiwoom_token()

    rows = []
    fails = []
    for i, (ticker, name) in enumerate(tickers, 1):
        data = fetch_ka10001(token, ticker)
        if not data:
            fails.append((ticker, name, "API 실패"))
            continue
        mac = parse_int_safe(data.get("mac"))
        cur_prc = parse_int_safe(data.get("cur_prc"))
        flo_stk = parse_int_safe(data.get("flo_stk"))  # 상장주식수 (천주 단위)
        stk_nm = data.get("stk_nm", name)

        if not mac or not cur_prc or not flo_stk:
            fails.append((ticker, name, f"mac={mac}, cur_prc={cur_prc}, flo_stk={flo_stk}"))
            continue

        # 단위 정리:
        # - mac: 억원 단위 → 원 단위로 변환 시 ×1억
        # - cur_prc: 원 단위 (절댓값, 음수는 등락방향 표시)
        # - flo_stk: 천주 단위 → 주 단위로 변환 시 ×1000
        mac_won = mac * 100_000_000
        shares = flo_stk * 1000

        rows.append({
            "ticker": ticker,
            "name": stk_nm,
            "mcap_eok": mac,                    # 억원 단위 (원본)
            "mcap_won": mac_won,                # 원 단위
            "cur_prc": cur_prc,                 # 현재가(원, 절댓값)
            "shares_outstanding": shares,       # 상장주식수 (주 단위)
        })

        if i % 20 == 0:
            print(f"  [{i:3}/{len(tickers)}] {ticker} {stk_nm} mac={mac:,}억 close={cur_prc:,} 상장주식수={shares:,}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n저장: {OUTPUT_CSV}")
    print(f"성공: {len(rows)}/{len(tickers)}, 실패: {len(fails)}")
    if fails:
        print("실패 목록:")
        for t, n, msg in fails[:20]:
            print(f"  {t} {n}: {msg}")

    # 시가총액 Top10 미리보기
    print("\n시가총액 상위 10:")
    df_sorted = df.sort_values("mcap_eok", ascending=False).head(10)
    for _, r in df_sorted.iterrows():
        print(f"  {r['ticker']} {r['name']:20s} {r['mcap_eok']:>10,}억")


if __name__ == "__main__":
    main()
