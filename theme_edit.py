"""themes 테이블의 themes/summary 컬럼 편집 TUI 앱"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.widgets import (
    Header,
    Footer,
    Input,
    Button,
    Static,
    TextArea,
    Label,
)
from textual.widget import Widget
from textual.binding import Binding
from textual.message import Message

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "stock_db")
DB_USER = os.getenv("DB_USER", "rexfelix")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

MAX_LENGTH = 1000


def get_db_engine():
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)


def search_stocks(query: str) -> pd.DataFrame:
    """ticker 또는 name으로 종목 검색 (쉼표 구분 복수 검색 지원)"""
    engine = get_db_engine()
    queries = [q.strip() for q in query.split(",") if q.strip()]
    if not queries:
        return pd.DataFrame()

    conditions = []
    params = {}
    for i, q in enumerate(queries):
        conditions.append(f"(ticker ILIKE :q{i} OR name ILIKE :q{i})")
        params[f"q{i}"] = f"%{q}%"

    where = " OR ".join(conditions)
    sql = f"SELECT ticker, name, themes, sector, summary FROM themes WHERE {where} ORDER BY name"
    return pd.read_sql(text(sql), engine, params=params)


def update_stock(ticker: str, themes_val: str, summary_val: str) -> bool:
    """단일 종목의 themes, summary 업데이트"""
    engine = get_db_engine()
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE themes SET themes = :themes, summary = :summary WHERE ticker = :ticker"),
            {"themes": themes_val, "summary": summary_val, "ticker": ticker},
        )
        conn.commit()
    return True


def bulk_update(tickers: list[str], themes_val: str | None, summary_val: str | None) -> int:
    """복수 종목에 동일 값 일괄 적용 (기존 값에 ',' 구분자로 이어붙임)"""
    engine = get_db_engine()
    sets = []
    params: dict = {}
    if themes_val is not None:
        # 기존 값이 있으면 ','로 이어붙이고, 없으면 새 값만
        sets.append(
            "themes = CASE WHEN themes IS NOT NULL AND themes != '' "
            "THEN themes || ',' || :themes ELSE :themes END"
        )
        params["themes"] = themes_val
    if summary_val is not None:
        sets.append(
            "summary = CASE WHEN summary IS NOT NULL AND summary != '' "
            "THEN summary || ',' || :summary ELSE :summary END"
        )
        params["summary"] = summary_val
    if not sets:
        return 0

    placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
    for i, t in enumerate(tickers):
        params[f"t{i}"] = t

    sql = f"UPDATE themes SET {', '.join(sets)} WHERE ticker IN ({placeholders})"
    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        conn.commit()
        return result.rowcount


class StockCard(Widget):
    """개별 종목 편집 카드"""

    DEFAULT_CSS = """
    StockCard {
        border: solid $primary;
        margin: 1 0;
        padding: 1;
        height: auto;
    }
    StockCard .stock-header {
        text-style: bold;
        margin-bottom: 1;
    }
    StockCard .field-label {
        margin-top: 1;
        color: $text-muted;
    }
    StockCard .char-count {
        color: $text-muted;
        text-align: right;
    }
    StockCard .char-count.over-limit {
        color: $error;
        text-style: bold;
    }
    StockCard .card-buttons {
        margin-top: 1;
        height: 3;
        align: right middle;
    }
    """

    class Saved(Message):
        def __init__(self, ticker: str, name: str, success: bool, msg: str) -> None:
            super().__init__()
            self.ticker = ticker
            self.name = name
            self.success = success
            self.msg = msg

    def __init__(self, ticker: str, name: str, sector: str, themes: str, summary: str) -> None:
        super().__init__(id=f"card-{ticker}")
        self.ticker = ticker
        self.stock_name = name
        self.sector = sector
        self.themes_text = themes
        self.summary_text = summary

    def _calc_height(self, text: str, min_lines: int = 1) -> int:
        """텍스트 줄 수 기반 높이 계산 (최소 min_lines)"""
        lines = max(text.count("\n") + 1, min_lines)
        return lines + 1  # border/padding 보정

    def compose(self) -> ComposeResult:
        total = len(self.themes_text) + len(self.summary_text)
        yield Static(
            f"[b]{self.stock_name}[/b]  ({self.ticker})  |  {self.sector}",
            classes="stock-header",
        )
        yield Label("Themes:", classes="field-label")
        ta_themes = TextArea(self.themes_text, id=f"themes-{self.ticker}", soft_wrap=True)
        ta_themes.styles.height = self._calc_height(self.themes_text)
        yield ta_themes
        yield Label("Summary:", classes="field-label")
        ta_summary = TextArea(self.summary_text, id=f"summary-{self.ticker}", soft_wrap=True)
        ta_summary.styles.height = self._calc_height(self.summary_text)
        yield ta_summary
        yield Static(f"{total} / {MAX_LENGTH}", id=f"count-{self.ticker}", classes="char-count")
        with Horizontal(classes="card-buttons"):
            yield Button("저장", id=f"save-{self.ticker}", variant="success")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._update_count()
        self._update_heights()

    def _update_heights(self) -> None:
        """TextArea 높이를 내용에 맞게 조정"""
        try:
            ta_themes = self.query_one(f"#themes-{self.ticker}", TextArea)
            ta_themes.styles.height = self._calc_height(ta_themes.text)
            ta_summary = self.query_one(f"#summary-{self.ticker}", TextArea)
            ta_summary.styles.height = self._calc_height(ta_summary.text)
        except Exception:
            pass

    def _update_count(self) -> None:
        try:
            t_len = len(self.query_one(f"#themes-{self.ticker}", TextArea).text)
            s_len = len(self.query_one(f"#summary-{self.ticker}", TextArea).text)
            total = t_len + s_len
            label = self.query_one(f"#count-{self.ticker}", Static)
            label.update(f"{total} / {MAX_LENGTH}")
            if total > MAX_LENGTH:
                label.add_class("over-limit")
            else:
                label.remove_class("over-limit")
        except Exception:
            pass

    def get_values(self) -> tuple[str, str]:
        themes_val = self.query_one(f"#themes-{self.ticker}", TextArea).text.strip()
        summary_val = self.query_one(f"#summary-{self.ticker}", TextArea).text.strip()
        return themes_val, summary_val

    def set_values(self, themes: str | None = None, summary: str | None = None) -> None:
        if themes is not None:
            self.query_one(f"#themes-{self.ticker}", TextArea).text = themes
        if summary is not None:
            self.query_one(f"#summary-{self.ticker}", TextArea).text = summary
        self._update_count()

    def save(self) -> None:
        themes_val, summary_val = self.get_values()
        total = len(themes_val) + len(summary_val)
        if total > MAX_LENGTH:
            self.post_message(self.Saved(self.ticker, self.stock_name, False,
                                         f"글자 수 초과! ({total}/{MAX_LENGTH})"))
            return
        try:
            update_stock(self.ticker, themes_val, summary_val)
            self.post_message(self.Saved(self.ticker, self.stock_name, True,
                                         f"저장 완료: {self.stock_name} ({self.ticker})"))
        except Exception as e:
            self.post_message(self.Saved(self.ticker, self.stock_name, False,
                                         f"저장 실패: {e}"))


class BulkEditor(Widget):
    """일괄 편집 위젯"""

    DEFAULT_CSS = """
    BulkEditor {
        border: solid $warning;
        margin: 1 0;
        padding: 1;
        height: auto;
    }
    BulkEditor .bulk-header {
        color: $warning;
        text-style: bold;
        margin-bottom: 1;
    }
    BulkEditor .field-label {
        margin-top: 1;
        color: $text-muted;
    }
    BulkEditor .bulk-note {
        color: $text-muted;
        text-align: right;
    }
    BulkEditor .card-buttons {
        margin-top: 1;
        height: 3;
        align: right middle;
    }
    """

    class Applied(Message):
        def __init__(self, success: bool, msg: str) -> None:
            super().__init__()
            self.success = success
            self.msg = msg

    def __init__(self, tickers_info: list[tuple[str, str]]) -> None:
        super().__init__(id="bulk-section")
        self.tickers_info = tickers_info  # [(ticker, name), ...]

    def compose(self) -> ComposeResult:
        names = ", ".join(f"{name}({ticker})" for ticker, name in self.tickers_info)
        yield Static(f"일괄 편집 대상: {names}", classes="bulk-header")
        yield Label("Themes (일괄):", classes="field-label")
        yield TextArea(id="bulk-themes", soft_wrap=True)
        yield Label("Summary (일괄):", classes="field-label")
        yield TextArea(id="bulk-summary", soft_wrap=True)
        yield Static("빈 칸은 변경하지 않음", classes="bulk-note")
        with Horizontal(classes="card-buttons"):
            yield Button("일괄 적용", id="bulk-apply", variant="warning")

    def apply(self) -> tuple[str | None, str | None]:
        themes_val = self.query_one("#bulk-themes", TextArea).text.strip() or None
        summary_val = self.query_one("#bulk-summary", TextArea).text.strip() or None

        if themes_val is None and summary_val is None:
            self.post_message(self.Applied(False, "일괄 적용할 내용을 입력하세요."))
            return None, None

        total = len(themes_val or "") + len(summary_val or "")
        if total > MAX_LENGTH:
            self.post_message(self.Applied(False, f"글자 수 초과! ({total}/{MAX_LENGTH})"))
            return None, None

        tickers = [t for t, _ in self.tickers_info]
        try:
            count = bulk_update(tickers, themes_val, summary_val)
            self.post_message(self.Applied(True, f"일괄 적용 완료: {count}개 종목 업데이트"))
            return themes_val, summary_val
        except Exception as e:
            self.post_message(self.Applied(False, f"일괄 적용 실패: {e}"))
            return None, None


CSS = """
Screen {
    background: $surface;
}

#search-box {
    height: 3;
    dock: top;
    padding: 0 1;
}

#search-input {
    width: 1fr;
}

#search-btn {
    width: 12;
    min-width: 12;
}

#result-area {
    height: 1fr;
    padding: 0 1;
}

#status-bar {
    dock: bottom;
    height: 1;
    padding: 0 1;
    background: $primary-background;
    color: $text;
}
"""


class ThemeEditApp(App):
    TITLE = "Themes Editor"
    SUB_TITLE = "themes / summary 편집"
    CSS = CSS
    BINDINGS = [
        Binding("ctrl+q", "quit", "종료"),
        Binding("ctrl+s", "save_all", "전체 저장"),
        Binding("escape", "clear_results", "결과 초기화"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="search-box"):
            yield Input(
                placeholder="ticker 또는 종목명 검색 (쉼표로 복수 검색: 삼성전자,SK하이닉스)",
                id="search-input",
            )
            yield Button("검색", id="search-btn", variant="primary")
        yield VerticalScroll(id="result-area")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.set_status("ticker 또는 종목명을 입력하고 검색하세요.")
        self.query_one("#search-input", Input).focus()

    def set_status(self, msg: str) -> None:
        self.query_one("#status-bar", Static).update(msg)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "search-btn":
            await self.do_search()
        elif btn_id and btn_id.startswith("save-"):
            ticker = btn_id.removeprefix("save-")
            card = self.query_one(f"#card-{ticker}", StockCard)
            card.save()
        elif btn_id == "bulk-apply":
            self.do_bulk_apply()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input":
            await self.do_search()

    def on_stock_card_saved(self, event: StockCard.Saved) -> None:
        self.set_status(event.msg)

    def on_bulk_editor_applied(self, event: BulkEditor.Applied) -> None:
        self.set_status(event.msg)

    async def do_search(self) -> None:
        search_input = self.query_one("#search-input", Input)
        query = search_input.value.strip()
        if not query:
            self.set_status("검색어를 입력하세요.")
            return

        try:
            df = search_stocks(query)
        except Exception as e:
            self.set_status(f"DB 오류: {e}")
            return

        if df.empty:
            self.set_status(f"'{query}' 검색 결과 없음")
            return

        result_area = self.query_one("#result-area", VerticalScroll)
        await result_area.remove_children()

        tickers_info = []
        for _, row in df.iterrows():
            ticker = str(row["ticker"])
            name = str(row.get("name", ""))
            sector = str(row.get("sector", "") or "")
            themes = str(row.get("themes", "") or "")
            summary = str(row.get("summary", "") or "")
            tickers_info.append((ticker, name))
            await result_area.mount(StockCard(ticker, name, sector, themes, summary))

        if len(df) > 1:
            await result_area.mount(BulkEditor(tickers_info))

        self.set_status(f"{len(df)}개 종목 검색 완료")

    def do_bulk_apply(self) -> None:
        try:
            bulk = self.query_one("#bulk-section", BulkEditor)
        except Exception:
            self.set_status("일괄 편집 위젯을 찾을 수 없음")
            return

        themes_val, summary_val = bulk.apply()
        if themes_val is None and summary_val is None:
            return

        # 카드의 TextArea도 기존 값에 이어붙여 업데이트
        for card in self.query(StockCard):
            current_themes, current_summary = card.get_values()
            new_themes = None
            new_summary = None
            if themes_val is not None:
                new_themes = f"{current_themes},{themes_val}" if current_themes else themes_val
            if summary_val is not None:
                new_summary = f"{current_summary},{summary_val}" if current_summary else summary_val
            card.set_values(themes=new_themes, summary=new_summary)

    def action_save_all(self) -> None:
        """Ctrl+S: 모든 종목 저장"""
        cards = list(self.query(StockCard))
        for card in cards:
            themes_val, summary_val = card.get_values()
            total = len(themes_val) + len(summary_val)
            if total > MAX_LENGTH:
                self.set_status(f"글자 수 초과: {card.stock_name} ({total}/{MAX_LENGTH})")
                return
        saved = 0
        for card in cards:
            try:
                card.save()
                saved += 1
            except Exception:
                pass
        self.set_status(f"전체 저장 완료: {saved}개 종목")

    async def action_clear_results(self) -> None:
        """ESC: 결과 초기화"""
        await self.query_one("#result-area", VerticalScroll).remove_children()
        self.set_status("초기화 완료. 새 검색어를 입력하세요.")
        self.query_one("#search-input", Input).focus()


if __name__ == "__main__":
    app = ThemeEditApp()
    app.run()
