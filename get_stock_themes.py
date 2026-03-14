import time
import requests
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 데이터베이스 연결 정보
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "stock_db")
DB_USER = os.getenv("DB_USER", "rexfelix")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# 키움 API 설정
KIWOOM_APPKEY = os.getenv("KIWOOM_APPKEY", "")
KIWOOM_SECRETKEY = os.getenv("KIWOOM_SECRETKEY", "")
KIWOOM_DOMAIN = "https://api.kiwoom.com"


def get_db_engine():
    """PostgreSQL 연결 엔진 생성"""
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)


def get_kiwoom_token():
    """키움 REST API OAuth2 토큰 발급 (au10001)"""
    url = f"{KIWOOM_DOMAIN}/oauth2/token"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "au10001",
    }
    body = {
        "grant_type": "client_credentials",
        "appkey": KIWOOM_APPKEY,
        "secretkey": KIWOOM_SECRETKEY,
    }

    response = requests.post(url, json=body, headers=headers)
    data = response.json()

    if data.get("return_code") != 0:
        raise Exception(f"토큰 발급 실패: {data.get('return_msg')}")

    print(f"토큰 발급 성공 (만료: {data.get('expires_dt')})")
    return data["token"]


def get_stock_list(token, mrkt_tp):
    """종목정보 리스트 조회 (ka10099) - 업종명(upName) 포함"""
    url = f"{KIWOOM_DOMAIN}/api/dostk/stkinfo"
    all_stocks = []

    req_headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "ka10099",
        "authorization": f"Bearer {token}",
    }
    body = {"mrkt_tp": mrkt_tp}

    while True:
        time.sleep(0.5)
        response = requests.post(url, json=body, headers=req_headers)
        data = response.json()

        if data.get("return_code") != 0:
            print(f"종목 리스트 조회 오류: {data.get('return_msg')}")
            break

        stocks = data.get("list", [])
        if not stocks:
            break

        all_stocks.extend(stocks)

        cont_yn = response.headers.get("cont-yn", "N")
        if cont_yn != "Y":
            break

        req_headers["cont-yn"] = "Y"
        req_headers["next-key"] = response.headers.get("next-key", "")

    return all_stocks


def get_all_stock_sectors(token):
    """KOSPI + KOSDAQ 전체 종목의 업종 정보 조회"""
    # ticker -> upName 매핑
    sector_map = {}

    for mrkt_tp, market_name in [("0", "코스피"), ("10", "코스닥")]:
        stocks = get_stock_list(token, mrkt_tp)
        for stk in stocks:
            code = stk.get("code", "")
            if code:
                sector_map[code] = stk.get("upName", "")
        print(f"{market_name} 종목 {len(stocks)}개 업종 정보 조회 완료")

    return sector_map


def get_theme_groups(token):
    """전체 테마 그룹 목록 조회 (ka90001)"""
    url = f"{KIWOOM_DOMAIN}/api/dostk/thme"
    all_themes = []

    req_headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "ka90001",
        "authorization": f"Bearer {token}",
    }
    body = {
        "qry_tp": "0",
        "stk_cd": "",
        "date_tp": "1",
        "thema_nm": "",
        "flu_pl_amt_tp": "1",
        "stex_tp": "1",
    }

    while True:
        time.sleep(0.5)
        response = requests.post(url, json=body, headers=req_headers)
        data = response.json()

        if data.get("return_code") != 0:
            print(f"테마 그룹 조회 오류: {data.get('return_msg')}")
            break

        themes = data.get("thema_grp", [])
        if not themes:
            break

        all_themes.extend(themes)

        cont_yn = response.headers.get("cont-yn", "N")
        if cont_yn != "Y":
            break

        req_headers["cont-yn"] = "Y"
        req_headers["next-key"] = response.headers.get("next-key", "")

    print(f"테마 그룹 {len(all_themes)}개 조회 완료")
    return all_themes


def get_theme_stocks(token, thema_grp_cd):
    """특정 테마의 구성 종목 조회 (ka90002)"""
    url = f"{KIWOOM_DOMAIN}/api/dostk/thme"
    all_stocks = []

    req_headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "api-id": "ka90002",
        "authorization": f"Bearer {token}",
    }
    body = {
        "date_tp": "1",
        "thema_grp_cd": thema_grp_cd,
        "stex_tp": "1",
    }

    while True:
        time.sleep(0.3)
        response = requests.post(url, json=body, headers=req_headers)
        data = response.json()

        if data.get("return_code") != 0:
            break

        stocks = data.get("thema_comp_stk", [])
        if not stocks:
            break

        all_stocks.extend(stocks)

        cont_yn = response.headers.get("cont-yn", "N")
        if cont_yn != "Y":
            break

        req_headers["cont-yn"] = "Y"
        req_headers["next-key"] = response.headers.get("next-key", "")

    return all_stocks


def build_theme_mapping(token, theme_groups):
    """전체 테마를 순회하며 종목별 테마 매핑 구축"""
    # ticker -> set of theme names
    theme_map = {}

    for theme in tqdm(theme_groups, desc="테마별 종목 조회", unit="theme"):
        grp_cd = theme["thema_grp_cd"]
        thema_nm = theme["thema_nm"]

        stocks = get_theme_stocks(token, grp_cd)

        for stk in stocks:
            ticker = stk.get("stk_cd", "")
            if ticker:
                if ticker not in theme_map:
                    theme_map[ticker] = set()
                theme_map[ticker].add(thema_nm)

    return theme_map


def save_to_db(all_tickers, theme_map, sector_map, engine):
    """전체 종목의 테마+업종 정보를 themes 테이블에 저장"""
    rows = []
    for ticker, name in all_tickers:
        themes = ",".join(sorted(theme_map.get(ticker, set()))) if ticker in theme_map else ""
        sector = sector_map.get(ticker, "")
        rows.append({
            "ticker": ticker,
            "name": name,
            "themes": themes,
            "sector": sector,
        })

    if not rows:
        print("저장할 데이터가 없습니다.")
        return

    df = pd.DataFrame(rows)
    df.to_sql("themes", engine, if_exists="replace", index=False)

    has_theme = df[df["themes"] != ""].shape[0]
    has_sector = df[df["sector"] != ""].shape[0]
    print(f"themes 테이블에 {len(df)}개 종목 저장 완료 (테마 보유: {has_theme}, 업종 보유: {has_sector})")


def main():
    print("=== 종목별 테마/업종 검색 시작 ===")

    # 1. 토큰 발급
    token = get_kiwoom_token()

    # 2. DB에서 전체 종목 목록 가져오기
    engine = get_db_engine()
    df_stocks = pd.read_sql("SELECT DISTINCT ticker, name FROM stocks", engine)
    all_tickers = list(zip(df_stocks["ticker"], df_stocks["name"]))
    print(f"stocks 테이블 고유 종목: {len(all_tickers)}개")

    # 3. 전체 종목 업종 정보 조회 (ka10099)
    sector_map = get_all_stock_sectors(token)
    print(f"업종 정보 수집 완료: {len(sector_map)}개 종목")

    # 4. 전체 테마 그룹 조회 (ka90001)
    theme_groups = get_theme_groups(token)

    # 5. 각 테마별 구성 종목 조회 → 종목별 테마 매핑 (ka90002)
    theme_map = build_theme_mapping(token, theme_groups)
    print(f"테마 보유 종목: {len(theme_map)}개")

    # 6. DB 저장
    save_to_db(all_tickers, theme_map, sector_map, engine)

    print("=== 완료 ===")


if __name__ == "__main__":
    main()
