"""Column-header detection for L3 text-row tables."""
import pytest

from src.models.odl_structured import ODLStructuredDocument
from src.services.document_chunker import DocumentChunker
from src.services.textrow_extractor import (
    detect_column_header, rows_to_markdown, text_to_table_rows, text_to_table_with_header,
)

ROWS = "Total assets $ 27,787 $ 26,075\nTotal liabilities 10,100 9,900"


@pytest.mark.parametrize("head", [
    "2023 2022",
    "December 31, 2023  2022",
    "For the Years Ended December 31, 2023 2022",
    "Fiscal 2023 Fiscal 2022",
    "December 31, 2023 December 31, 2022",
])
def test_header_forms_above_rows(head):
    assert detect_column_header(f"Balance Sheet\n{head}\n", 2) == ["", "2023", "2022"]
    rows, has = text_to_table_with_header(f"Balance Sheet\n{head}\n{ROWS}")
    assert has and rows[0] == ["", "2023", "2022"]
    assert rows[1] == ["Total assets", "27787", "26075"] and len(rows) == 3


def test_three_year_header():
    h = "For the Years Ended December 31, 2023 2022 2021"
    assert detect_column_header(h, 3) == ["", "2023", "2022", "2021"]


def test_split_two_line_header():
    t = "(in millions)\nDecember 31,\n2023 2022\n" + ROWS
    rows, has = text_to_table_with_header(t)
    assert has and rows[0] == ["", "2023", "2022"]
    assert detect_column_header("As of\nDecember 31,\n2023 2022", 2) == ["", "2023", "2022"]


def test_no_header_keeps_current_behaviour():
    rows, has = text_to_table_with_header("Some prose about 2023 and 2022 results.\n" + ROWS)
    assert not has and rows == [["Total assets", "27787", "26075"], ["Total liabilities", "10100", "9900"]]
    assert detect_column_header("nothing here", 2) is None
    assert rows_to_markdown(rows).splitlines()[0] == "|  |  |  |"


def test_mismatched_count_not_applied():
    # header lists 3 periods, rows carry 2 values: never guess
    t = "For the Years Ended December 31, 2023 2022 2021\nRevenue note\n" + ROWS
    rows, has = text_to_table_with_header(t)
    assert detect_column_header(t.split("\n" + ROWS)[0], 2) is None
    assert not has and rows == text_to_table_rows(t)
    # header-line-as-row with a different count than the data rows is left as data
    rows2, has2 = text_to_table_with_header("Years Ended December 31, 2023 2022 2021\n" + ROWS)
    assert not has2


def test_rows_with_other_count_kept_unchanged():
    t = "2023 2022\nTotal assets 1,000 900\nShares 5 6 7"
    rows, has = text_to_table_with_header(t)
    assert has and rows[0] == ["", "2023", "2022"] and rows[2] == ["Shares", "5", "6", "7"]
    md = rows_to_markdown(rows, has).splitlines()
    assert md[0] == "|  | 2023 | 2022 |  |" and md[2] == "| Total assets | 1000 | 900 |  |"


def test_markdown_uses_header():
    rows, has = text_to_table_with_header("2023 2022\n" + ROWS)
    md = rows_to_markdown(rows, has).splitlines()
    assert md[0] == "|  | 2023 | 2022 |" and md[2].startswith("| Total assets")


def test_chunker_hook_emits_header(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_L3_TEXTROW", "1")
    doc = {"number of pages": 2, "kids": [
        {"type": "paragraph", "page number": 1, "content": "Hello prose only."},
        {"type": "paragraph", "page number": 2, "content": "December 31, 2023 2022\n" + ROWS}]}
    t = DocumentChunker().extract_tables_from_structured(ODLStructuredDocument.from_json(doc))
    assert len(t) == 1 and t[0].method == "textrow" and t[0].page == 2
    assert t[0].rows[0] == ["", "2023", "2022"] and t[0].rows[1][0] == "Total assets"
    assert t[0].markdown.splitlines()[0] == "|  | 2023 | 2022 |"
