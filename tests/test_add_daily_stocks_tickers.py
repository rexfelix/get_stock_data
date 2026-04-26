"""add_daily_stocks.py 의 ticker 수집 경로(키움 ka10099 + 병렬 + DB Fallback) 검증.

PRD §B-8 핵심 검증 항목 4가지를 모두 커버한다.
"""
import os
import sys
import time
from unittest.mock import patch, MagicMock

import pytest

# tests/ 디렉토리에서 부모 디렉토리(스크립트 위치)를 import path에 추가
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import add_daily_stocks as ads  # noqa: E402


# ---------------------------------------------------------------------------
# get_stock_list_kiwoom: ka10099 응답을 (code, name) 튜플 리스트로 변환
# ---------------------------------------------------------------------------

def _make_ka10099_response(items, cont_yn="N", next_key=""):
    """requests.post의 응답 mock 생성."""
    resp = MagicMock()
    resp.json.return_value = {
        "return_code": 0,
        "return_msg": "정상적으로 처리되었습니다",
        "list": items,
    }
    resp.headers = {"cont-yn": cont_yn, "next-key": next_key}
    return resp


def test_get_stock_list_kiwoom_returns_code_name_tuples():
    """ka10099 정상 응답이 (code, name) 튜플 리스트로 변환되어야 한다."""
    items = [
        {"code": "005930", "name": "삼성전자", "marketCode": "0", "upName": "전기전자"},
        {"code": "000660", "name": "SK하이닉스", "marketCode": "0", "upName": "전기전자"},
    ]
    with patch.object(ads, "requests") as mock_requests:
        mock_requests.post.return_value = _make_ka10099_response(items)
        result = ads.get_stock_list_kiwoom("dummy_token", "0")

    assert result == [("005930", "삼성전자"), ("000660", "SK하이닉스")]


def test_get_stock_list_kiwoom_handles_pagination():
    """cont-yn=Y인 경우 next-key로 다음 페이지를 호출해야 한다."""
    page1 = _make_ka10099_response(
        [{"code": "000020", "name": "동화약품"}],
        cont_yn="Y",
        next_key="NEXT123",
    )
    page2 = _make_ka10099_response(
        [{"code": "000040", "name": "KR모터스"}],
        cont_yn="N",
    )
    with patch.object(ads, "requests") as mock_requests:
        mock_requests.post.side_effect = [page1, page2]
        result = ads.get_stock_list_kiwoom("dummy_token", "0")

    assert result == [("000020", "동화약품"), ("000040", "KR모터스")]
    # 두 번째 호출의 헤더에 cont-yn/next-key가 실려야 한다
    second_call_headers = mock_requests.post.call_args_list[1].kwargs["headers"]
    assert second_call_headers.get("cont-yn") == "Y"
    assert second_call_headers.get("next-key") == "NEXT123"


# ---------------------------------------------------------------------------
# fetch_tickers_kiwoom_parallel: KOSPI("0") + KOSDAQ("10") 병렬 호출 후 합산
# ---------------------------------------------------------------------------

def test_fetch_tickers_kiwoom_parallel_calls_both_markets():
    """KOSPI와 KOSDAQ 두 시장을 모두 호출하고 결과를 합쳐서 반환해야 한다."""
    def fake_get_stock_list(token, mrkt_tp):
        if mrkt_tp == "0":
            return [("005930", "삼성전자")]
        if mrkt_tp == "10":
            return [("035720", "카카오")]
        raise AssertionError(f"unexpected mrkt_tp: {mrkt_tp}")

    with patch.object(ads, "get_stock_list_kiwoom", side_effect=fake_get_stock_list) as m:
        result = ads.fetch_tickers_kiwoom_parallel("dummy_token")

    called_markets = sorted(call.args[1] for call in m.call_args_list)
    assert called_markets == ["0", "10"]
    assert sorted(result) == sorted([("005930", "삼성전자"), ("035720", "카카오")])


def test_fetch_tickers_kiwoom_parallel_runs_concurrently():
    """두 시장 호출이 직렬이 아닌 병렬로 실행되어야 한다.

    각 호출이 0.3초 sleep을 소모할 때, 직렬이면 0.6초 이상이 걸리지만 병렬이면 0.5초 미만이어야 한다.
    """
    def slow_get_stock_list(token, mrkt_tp):
        time.sleep(0.3)
        return [(f"code_{mrkt_tp}", f"name_{mrkt_tp}")]

    with patch.object(ads, "get_stock_list_kiwoom", side_effect=slow_get_stock_list):
        t0 = time.time()
        ads.fetch_tickers_kiwoom_parallel("dummy_token")
        elapsed = time.time() - t0

    assert elapsed < 0.5, f"병렬 실행이어야 하는데 {elapsed:.3f}s 소요됨 (직렬 가능성)"


# ---------------------------------------------------------------------------
# resolve_target_tickers: 키움 시도 → 실패시 DB Fallback
# ---------------------------------------------------------------------------

def test_resolve_target_tickers_uses_kiwoom_when_available():
    """키움 호출이 성공하면 그 결과를 반환하고 DB Fallback은 호출되지 않는다."""
    fake_engine = MagicMock()
    with patch.object(ads, "get_kiwoom_token", return_value="tok"), \
         patch.object(ads, "fetch_tickers_kiwoom_parallel", return_value=[("005930", "삼성전자")]) as m_kw, \
         patch.object(ads, "get_tickers_from_db") as m_db:
        result = ads.resolve_target_tickers(fake_engine)

    assert result == [("005930", "삼성전자")]
    m_kw.assert_called_once()
    m_db.assert_not_called()


def test_resolve_target_tickers_falls_back_to_db_on_kiwoom_error():
    """키움 호출이 예외를 던지면 DB Fallback을 사용한다."""
    fake_engine = MagicMock()
    with patch.object(ads, "get_kiwoom_token", side_effect=RuntimeError("token fail")), \
         patch.object(ads, "fetch_tickers_kiwoom_parallel") as m_kw, \
         patch.object(ads, "get_tickers_from_db", return_value=[("000020", "동화약품")]) as m_db:
        result = ads.resolve_target_tickers(fake_engine)

    assert result == [("000020", "동화약품")]
    m_kw.assert_not_called()  # 토큰 자체가 실패했으므로 호출되지 않아야 함
    m_db.assert_called_once_with(fake_engine)


def test_resolve_target_tickers_falls_back_when_kiwoom_returns_empty():
    """키움이 0건을 반환하면 DB Fallback으로 우회한다."""
    fake_engine = MagicMock()
    with patch.object(ads, "get_kiwoom_token", return_value="tok"), \
         patch.object(ads, "fetch_tickers_kiwoom_parallel", return_value=[]), \
         patch.object(ads, "get_tickers_from_db", return_value=[("000020", "동화약품")]) as m_db:
        result = ads.resolve_target_tickers(fake_engine)

    assert result == [("000020", "동화약품")]
    m_db.assert_called_once_with(fake_engine)


def test_resolve_target_tickers_returns_empty_when_both_sources_fail():
    """키움과 DB 둘 다 0건이면 빈 리스트를 반환한다."""
    fake_engine = MagicMock()
    with patch.object(ads, "get_kiwoom_token", side_effect=RuntimeError("token fail")), \
         patch.object(ads, "get_tickers_from_db", return_value=[]):
        result = ads.resolve_target_tickers(fake_engine)

    assert result == []
