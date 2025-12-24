import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# 데이터베이스 연결 정보
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


def get_db_engine():
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)


def get_stock_data(search_term, start_date, end_date):
    """종목명 또는 티커로 데이터 조회"""
    engine = get_db_engine()

    # 먼저 종목 확인
    search_term_upper = search_term.upper()
    query_ticker = f"""
    SELECT DISTINCT ticker, name 
    FROM stocks 
    WHERE UPPER(ticker) = '{search_term_upper}' OR UPPER(name) = '{search_term_upper}'
    """
    try:
        ticker_info = pd.read_sql(query_ticker, engine)
        if ticker_info.empty:
            return None, None

        ticker = ticker_info.iloc[0]["ticker"]
        name = ticker_info.iloc[0]["name"]

        # 데이터 조회
        query_data = f"""
        SELECT * 
        FROM stocks 
        WHERE ticker = '{ticker}' 
          AND date BETWEEN '{start_date}' AND '{end_date}' 
        ORDER BY date
        """
        df = pd.read_sql(query_data, engine)
        return df, name
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None


def calculate_heikin_ashi(df):
    """하이킨아시 캔들 계산"""
    ha_df = df.copy()

    # HA Close
    ha_df["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    # HA Open
    ha_open = [df["open"].iloc[0]]  # 첫 값은 일반 Open
    for i in range(1, len(df)):
        new_open = (ha_open[-1] + ha_df["close"].iloc[i - 1]) / 2
        ha_open.append(new_open)
    ha_df["open"] = ha_open

    # HA High, HA Low
    ha_df["high"] = ha_df[["high", "open", "close"]].max(axis=1)
    ha_df["low"] = ha_df[["low", "open", "close"]].min(axis=1)

    return ha_df


def calculate_smoothed_heikin_ashi(df, smooth_period=4):
    """스무스드 하이킨아시 계산 (Pandas ewm 사용)"""
    # 1. 일반 하이킨아시 계산
    df_ha = calculate_heikin_ashi(df)

    # 2. EMA 적용 (Smoothing)
    df_sha = pd.DataFrame()
    # 인덱스 유지를 위해
    df_sha.index = df_ha.index

    for col in ["open", "high", "low", "close"]:
        # span=smooth_period corresponds to standard EMA window
        df_sha[col] = df_ha[col].ewm(span=smooth_period, adjust=False).mean()

    return df_sha


def update_param(list_key, index, item_key, widget_key):
    """Callback to update session state list items"""
    st.session_state[list_key][index][item_key] = st.session_state[widget_key]


def main():
    st.set_page_config(page_title="Stock Chart Dashboard", layout="wide")
    st.title("📈 Stock Chart Dashboard")

    # 이동평균선 설정을 위한 세션 상태 초기화
    if "ma_lines" not in st.session_state:
        st.session_state["ma_lines"] = [
            {"period": 20, "color": "#FF9900", "width": 1}  # 기본 20일선
        ]

    # 볼린저밴드 설정을 위한 세션 상태 초기화 (기본값)
    if "bb_settings" not in st.session_state:
        st.session_state["bb_settings"] = {
            "enabled": False,
            "period": 20,
            "std_dev": 2.0,
            "color": "#808080",
            "width": 1,
            "show_mid": True,
            "show_upper": True,
            "show_lower": True,
        }

    # 스무스드 하이킨아시 상태 초기화
    if "sha_list" not in st.session_state:
        st.session_state["sha_list"] = []

    # 사이드바 설정
    st.sidebar.header("Configuration")

    # 1. 종목 입력
    search_term = st.sidebar.text_input("종목명 또는 코드 입력", value="삼성전자")

    # 2. 날짜 선택
    today = datetime.now()
    one_year_ago = today - timedelta(days=365)

    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("시작일", one_year_ago)
    end_date = col2.date_input("종료일", today)

    # 3. 차트 종류 선택
    chart_type = st.sidebar.radio("차트 종류", ["Candle", "Heikin-Ashi"])

    # 4. 이동평균선 설정
    st.sidebar.divider()
    st.sidebar.subheader("이동평균선 (MA)")

    # MA 추가/삭제 버튼 (크기 동일하게 배치)
    col_add, col_del = st.sidebar.columns([1, 1])
    with col_add:
        if st.button("MA 추가 ➕", use_container_width=True):
            st.session_state["ma_lines"].append(
                {"period": 5, "color": "#000000", "width": 1}
            )
    with col_del:
        if st.button("MA 삭제 ➖", use_container_width=True):
            if st.session_state["ma_lines"]:
                st.session_state["ma_lines"].pop()

    # MA 설정 입력창 동적 생성
    max_ma_period = 0
    for i, ma in enumerate(st.session_state["ma_lines"]):
        st.sidebar.markdown(f"**MA {i+1}**")
        sc1, sc2, sc3 = st.sidebar.columns([1, 1, 1])
        sc1.number_input(
            f"기간",
            value=ma["period"],
            min_value=1,
            key=f"ma_p_{i}",
            on_change=update_param,
            args=("ma_lines", i, "period", f"ma_p_{i}"),
        )
        sc2.color_picker(
            f"색상",
            value=ma["color"],
            key=f"ma_c_{i}",
            on_change=update_param,
            args=("ma_lines", i, "color", f"ma_c_{i}"),
        )
        sc3.number_input(
            f"굵기",
            value=ma["width"],
            min_value=1,
            max_value=10,
            key=f"ma_w_{i}",
            on_change=update_param,
            args=("ma_lines", i, "width", f"ma_w_{i}"),
        )

        # Update max_ma_period using the current value in session state (which is up to date thanks to callback)
        if ma["period"] > max_ma_period:
            max_ma_period = ma["period"]

    # 5. 볼린저밴드 설정
    st.sidebar.divider()
    st.sidebar.subheader("볼린저밴드 (BB)")

    bb = st.session_state["bb_settings"]
    # BB settings are static dict, so regular update is fine or use callback.
    # User complained about "Period" +/- buttons specifically, which implies dynamic number inputs.
    # BB period is static single input, usually less laggy, but worth checking?
    # Keeping BB as is for now unless requested, as it's not in a dynamic loop.
    bb["enabled"] = st.sidebar.checkbox("볼린저밴드 켜기", value=bb["enabled"])

    if bb["enabled"]:
        c1, c2 = st.sidebar.columns(2)
        bb["period"] = c1.number_input("기간 (N)", value=bb["period"], min_value=1)
        bb["std_dev"] = c2.number_input(
            "승수 (k)", value=bb["std_dev"], min_value=0.1, step=0.1
        )

        c3, c4 = st.sidebar.columns(2)
        bb["color"] = c3.color_picker("색상", value=bb["color"])
        bb["width"] = c4.number_input(
            "굵기", value=bb["width"], min_value=1, max_value=5
        )

        st.sidebar.markdown("**표시 여부**")
        rc1, rc2, rc3 = st.sidebar.columns(3)
        bb["show_upper"] = rc1.checkbox("상한선", value=bb["show_upper"])
        bb["show_mid"] = rc2.checkbox("기준선", value=bb["show_mid"])
        bb["show_lower"] = rc3.checkbox("하한선", value=bb["show_lower"])

        # 데이터 버퍼 계산을 위해 max period 업데이트
        if bb["period"] > max_ma_period:
            max_ma_period = bb["period"]

    # 6. 스무스드 하이킨아시 (SHA) 설정
    st.sidebar.divider()
    st.sidebar.subheader("스무스드 하이킨아시 (SHA)")

    col_sha_add, col_sha_del = st.sidebar.columns([1, 1])
    with col_sha_add:
        if st.button("SHA 추가 ➕", use_container_width=True):
            st.session_state["sha_list"].append({"period": 10})
    with col_sha_del:
        if st.button("SHA 삭제 ➖", use_container_width=True):
            if st.session_state["sha_list"]:
                st.session_state["sha_list"].pop()

    for i, sha in enumerate(st.session_state["sha_list"]):
        st.sidebar.markdown(f"**SHA {i+1}**")
        st.sidebar.number_input(
            f"기간",
            value=sha["period"],
            min_value=1,
            key=f"sha_p_{i}",
            on_change=update_param,
            args=("sha_list", i, "period", f"sha_p_{i}"),
        )
        if sha["period"] > max_ma_period:
            max_ma_period = sha["period"]

    st.sidebar.divider()

    if st.sidebar.button("조회", type="primary", use_container_width=True):
        with st.spinner("데이터 조회 중..."):
            # 이동평균선/볼린저밴드 계산을 위해 앞쪽 데이터를 넉넉히 가져옴
            buffer_days = int(max_ma_period * 2) + 20
            fetch_start_date = start_date - timedelta(days=buffer_days)

            df, stock_name = get_stock_data(search_term, fetch_start_date, end_date)

            if df is not None and not df.empty:
                st.subheader(f"{stock_name} ({df['ticker'].iloc[0]})")

                # 차트 데이터 준비
                if chart_type == "Heikin-Ashi":
                    chart_df = calculate_heikin_ashi(df)
                    title_prefix = "Heikin-Ashi"
                else:
                    chart_df = df
                    title_prefix = "Candle"

                # 이동평균선 추가 (전체 데이터에 대해 계산)
                for i, ma_conf in enumerate(st.session_state["ma_lines"]):
                    period = ma_conf["period"]
                    ma_series = chart_df["close"].rolling(window=period).mean()
                    col_name = f"ma_{period}_{i}"
                    chart_df[col_name] = ma_series
                    ma_conf["col_name"] = col_name

                # 볼린저밴드 계산
                if bb["enabled"]:
                    bb_col_mid = "bb_mid"
                    bb_col_upper = "bb_upper"
                    bb_col_lower = "bb_lower"

                    sma = chart_df["close"].rolling(window=bb["period"]).mean()
                    std = chart_df["close"].rolling(window=bb["period"]).std()

                    chart_df[bb_col_mid] = sma
                    chart_df[bb_col_upper] = sma + (std * bb["std_dev"])
                    chart_df[bb_col_lower] = sma - (std * bb["std_dev"])

                # -----------------------------------------------------------
                # 중요: 사용자가 요청한 날짜로 데이터 슬라이싱
                chart_df["date"] = pd.to_datetime(chart_df["date"])
                mask = (chart_df["date"] >= pd.Timestamp(start_date)) & (
                    chart_df["date"] <= pd.Timestamp(end_date)
                )
                view_df = chart_df.loc[mask].copy()
                # -----------------------------------------------------------

                if view_df.empty:
                    st.warning("선택한 기간에 데이터가 충분하지 않습니다.")
                    st.stop()

                # 날짜 문자열 변환 (X축 카테고리화)
                view_df["date_str"] = view_df["date"].dt.strftime("%Y-%m-%d")

                # 1. 캔들/하이킨아시 차트 추가
                # -----------------------------------------------------------
                # Plotly 렌더링 (TradingView 스타일 최적화)
                # -----------------------------------------------------------

                # 1. 캔들/하이킨아시 차트 추가
                traces = [
                    go.Candlestick(
                        x=view_df["date_str"],
                        open=view_df["open"],
                        high=view_df["high"],
                        low=view_df["low"],
                        close=view_df["close"],
                        name=title_prefix,
                        increasing_line_color="red",
                        decreasing_line_color="green",
                    )
                ]

                # 2. 이동평균선 Trace 생성
                for ma_conf in st.session_state["ma_lines"]:
                    period = ma_conf["period"]
                    color = ma_conf["color"]
                    width = ma_conf["width"]
                    col_name = ma_conf.get("col_name")

                    if col_name and col_name in view_df.columns:
                        traces.append(
                            go.Scatter(
                                x=view_df["date_str"],
                                y=view_df[col_name],
                                mode="lines",
                                line=dict(color=color, width=width),
                                name=f"MA {period}",
                            )
                        )

                # 3. 볼린저밴드 Trace 생성
                if bb["enabled"]:
                    if bb["show_upper"] and bb_col_upper in view_df.columns:
                        traces.append(
                            go.Scatter(
                                x=view_df["date_str"],
                                y=view_df[bb_col_upper],
                                mode="lines",
                                line=dict(
                                    color=bb["color"], width=bb["width"], dash="dot"
                                ),
                                name="BB Upper",
                            )
                        )
                    if bb["show_mid"] and bb_col_mid in view_df.columns:
                        traces.append(
                            go.Scatter(
                                x=view_df["date_str"],
                                y=view_df[bb_col_mid],
                                mode="lines",
                                line=dict(color=bb["color"], width=bb["width"]),
                                name="BB Mid",
                            )
                        )
                    if bb["show_lower"] and bb_col_lower in view_df.columns:
                        traces.append(
                            go.Scatter(
                                x=view_df["date_str"],
                                y=view_df[bb_col_lower],
                                mode="lines",
                                line=dict(
                                    color=bb["color"], width=bb["width"], dash="dot"
                                ),
                                name="BB Lower",
                            )
                        )

                # 4. 스무스드 하이킨아시 (SHA) Trace 생성 (Overlay)
                for sha_conf in st.session_state["sha_list"]:
                    period = sha_conf["period"]
                    sha_res = calculate_smoothed_heikin_ashi(df, smooth_period=period)

                    sha_res["date"] = pd.to_datetime(df["date"])
                    mask_sha = (sha_res["date"] >= pd.Timestamp(start_date)) & (
                        sha_res["date"] <= pd.Timestamp(end_date)
                    )
                    view_sha = sha_res.loc[mask_sha].copy()
                    view_sha["date_str"] = view_sha["date"].dt.strftime("%Y-%m-%d")

                    if not view_sha.empty:
                        traces.append(
                            go.Candlestick(
                                x=view_sha["date_str"],
                                open=view_sha["open"],
                                high=view_sha["high"],
                                low=view_sha["low"],
                                close=view_sha["close"],
                                name=f"SHA ({period})",
                                increasing_line_color="#FF00FF",  # Magenta
                                decreasing_line_color="#FFFFFF",  # White
                                opacity=0.7,
                            )
                        )

                fig = go.Figure(data=traces)

                # TradingView 스타일 UX 설정
                fig.update_layout(
                    title=f"{title_prefix} Chart - {stock_name}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,  # 하단 슬라이더 제거 (공간 확보)
                    xaxis_type="category",
                    height=600,
                    dragmode="pan",  # 기본 드래그 모드를 이동(Pan)으로 설정
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                    hovermode="x unified",  # 십자선 마우스 오버 효과 강화
                )

                # X축 틱 레이블 과밀 방지
                if len(view_df) > 30:
                    fig.update_xaxes(dtick=max(1, len(view_df) // 10))

                # 스크롤 줌 활성화
                config = {
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                }

                st.plotly_chart(fig, use_container_width=True, config=config)

                with st.expander("Raw Data 보기"):
                    st.dataframe(df)
            else:
                st.warning("데이터를 찾을 수 없습니다.")


if __name__ == "__main__":
    main()
