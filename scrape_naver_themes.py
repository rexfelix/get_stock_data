"""네이버 증권 테마 페이지에서 모든 테마와 구성 종목을 스크래핑하여 CSV로 저장"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://finance.naver.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_theme_list():
    """테마 목록 페이지에서 모든 테마명과 no 파라미터를 추출"""
    themes = []
    page = 1

    while True:
        url = f"{BASE_URL}/sise/theme.naver?&page={page}"
        resp = requests.get(url, headers=HEADERS)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        # 테마 링크 추출
        found = False
        for a_tag in soup.select("td.col_type1 a"):
            href = a_tag.get("href", "")
            if "type=theme&no=" in href:
                no = href.split("no=")[-1]
                name = a_tag.get_text(strip=True)
                if name and no:
                    themes.append({"theme_no": no, "theme_name": name})
                    found = True

        if not found:
            break

        # 다음 페이지 확인
        paging = soup.select("table.Nnavi td.pgRR a")
        if not paging:
            # 현재 페이지가 마지막인지 확인
            current_pages = soup.select("table.Nnavi td a, table.Nnavi td strong")
            max_page = 0
            for el in current_pages:
                try:
                    p = int(el.get_text(strip=True))
                    if p > max_page:
                        max_page = p
                except ValueError:
                    continue
            if page >= max_page:
                break

        page += 1
        time.sleep(0.3)

    # 중복 제거
    seen = set()
    unique_themes = []
    for t in themes:
        if t["theme_no"] not in seen:
            seen.add(t["theme_no"])
            unique_themes.append(t)

    return unique_themes


def get_theme_stocks(theme_no, theme_name):
    """특정 테마의 구성 종목 목록 추출"""
    url = f"{BASE_URL}/sise/sise_group_detail.naver?type=theme&no={theme_no}"
    resp = requests.get(url, headers=HEADERS)
    resp.encoding = "euc-kr"
    soup = BeautifulSoup(resp.text, "html.parser")

    stocks = []
    # type_5 테이블에서 종목 추출 (첫번째 td에 종목명 링크)
    for tr in soup.select("table.type_5 tr"):
        tds = tr.select("td")
        if not tds:
            continue
        # 첫번째 td에서 /item/main.naver?code= 링크 찾기
        a_tag = tds[0].select_one("a[href*='/item/main.naver?code=']")
        if not a_tag:
            continue
        href = a_tag.get("href", "")
        ticker = href.split("code=")[-1].strip()
        name = a_tag.get_text(strip=True).rstrip("*")
        if ticker and name:
            stocks.append({
                "ticker": ticker,
                "name": name,
                "theme_no": theme_no,
                "theme_name": theme_name,
            })

    return stocks


def main():
    print("=== 네이버 증권 테마 스크래핑 시작 ===")

    # 1. 테마 목록 수집
    print("테마 목록 수집 중...")
    themes = get_theme_list()
    print(f"테마 {len(themes)}개 발견")

    # 2. 각 테마별 구성 종목 수집
    all_rows = []
    for theme in tqdm(themes, desc="테마별 종목 수집", unit="theme"):
        stocks = get_theme_stocks(theme["theme_no"], theme["theme_name"])
        all_rows.extend(stocks)
        time.sleep(0.3)

    print(f"총 {len(all_rows)}개 종목-테마 레코드 수집")

    if not all_rows:
        print("종목 데이터가 없습니다. 종료합니다.")
        return

    # 3. CSV 저장
    df = pd.DataFrame(all_rows)
    csv_path = "naver_themes.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 저장 완료: {csv_path}")

    # 4. 종목별 테마 요약
    theme_by_ticker = df.groupby(["ticker", "name"])["theme_name"].apply(
        lambda x: ",".join(sorted(x.unique()))
    ).reset_index()
    theme_by_ticker.columns = ["ticker", "name", "naver_themes"]
    summary_path = "naver_themes_by_ticker.csv"
    theme_by_ticker.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"종목별 테마 요약 CSV 저장: {summary_path}")
    print(f"  - 고유 종목 수: {theme_by_ticker.shape[0]}")
    print(f"  - 고유 테마 수: {df['theme_name'].nunique()}")

    print("=== 완료 ===")


if __name__ == "__main__":
    main()
