"""Tests for the UI helpers in app.py that do not need a browser.

The sidebar timestamp path matters more than it looks: a database created by init.sql
stores created_at as timestamptz (psycopg2 hands back an aware datetime), while one
upgraded by migrations/001 keeps the original timestamp column (naive datetime). Both
shapes, plus the ISO strings they become once they round-trip through a dcc.Store, have
to render.
"""

from datetime import datetime, timedelta, timezone

import app


def test_aware_datetime_today_renders_a_clock_time():
    now = datetime.now(timezone.utc)
    assert app.format_relative_time(now) == now.strftime("%I:%M %p").lstrip("0")


def test_naive_datetime_today_renders_a_clock_time():
    # The migrated-schema shape: timestamp without time zone.
    now = datetime.now()
    assert app.format_relative_time(now) == now.strftime("%I:%M %p").lstrip("0")


def test_yesterday_is_labelled():
    assert app.format_relative_time(datetime.now() - timedelta(days=1)) == "Yesterday"
    assert app.format_relative_time(datetime.now(timezone.utc) - timedelta(days=1)) == "Yesterday"


def test_older_dates_render_as_a_short_date_with_no_zero_padding():
    assert app.format_relative_time(datetime(2026, 3, 5, 14, 30)) == "Mar 5"
    assert app.format_relative_time(datetime(2026, 12, 25, 1, 0, tzinfo=timezone.utc)) == "Dec 25"


def test_iso_strings_from_a_store_round_trip():
    # What a dcc.Store hands back, with and without an offset.
    assert app.format_relative_time("2026-03-05T14:30:00+00:00") == "Mar 5"
    assert app.format_relative_time("2026-03-05T14:30:00") == "Mar 5"
    assert app.format_relative_time("2026-03-05T14:30:00Z") == "Mar 5"


def test_unparseable_values_render_as_empty_rather_than_raising():
    assert app.format_relative_time(None) == ""
    assert app.format_relative_time("") == ""
    assert app.format_relative_time("not a date") == ""


def test_sidebar_shows_a_placeholder_when_there_are_no_conversations():
    rendered = str(app.render_sidebar_items([], None))
    assert "Nothing yet" in rendered


def test_sidebar_marks_the_active_conversation():
    conversations = [
        {"id": 1, "title": "first", "created_at": datetime.now()},
        {"id": 2, "title": "second", "created_at": datetime.now()},
    ]
    rendered = str(app.render_sidebar_items(conversations, 2))
    assert "vi-sidebar-item active" in rendered
    assert rendered.count("vi-sidebar-item active") == 1


def test_every_code_block_gets_its_own_copy_button():
    import backend

    parsed = backend.parse_response(
        "Create it:\n```sql\nCREATE TABLE t (id int);\n```\nThen read it:\n```sql\nSELECT * FROM t;\n```"
    )
    rendered = str(app.render_answer_body(0, {**parsed, "answer": None}))
    # One copy button and one code block per parsed block, with distinct composite ids.
    assert rendered.count("'copy-btn'") == 2
    assert "'0-0'" in rendered and "'0-1'" in rendered


def test_an_error_bubble_renders_without_feedback_buttons():
    history = [{"question": "q", "answer": "The model is unreachable.", "error": True}]
    rendered = str(app.render_bubbles(history))
    assert "The model is unreachable." in rendered
    assert "upvote-btn" not in rendered


def test_a_pending_bubble_shows_only_the_question():
    history = [{"question": "my question", "answer": None, "pending": True}]
    rendered = str(app.render_bubbles(history))
    assert "my question" in rendered
    assert "upvote-btn" not in rendered


def test_an_attached_file_is_badged_on_the_question():
    history = [{"question": "explain", "answer": "text", "file_name": "etl.sql", "blocks": []}]
    rendered = str(app.render_bubbles(history))
    assert "etl.sql" in rendered and "vi-file-badge" in rendered
