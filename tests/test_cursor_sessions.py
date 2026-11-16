"""Tests for `tp.read.cursor` session transcript adapter."""

from __future__ import annotations

import json

import pytest

import tracepress as tp
from tracepress.adapters.atif import IMAGE_PLACEHOLDER


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


@pytest.fixture
def cursor_dir(tmp_path):
    projects = tmp_path / "projects"
    sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    main_file = projects / "Users-me-myapp" / "agent-transcripts" / sid / f"{sid}.jsonl"
    _write_jsonl(
        main_file,
        [
            {
                "role": "user",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<user_info>\nWorkspace Path: /Users/me/myapp\n</user_info>\n"
                                "<timestamp>Sunday, Jul 26, 2026, 1:06 PM (UTC)</timestamp>\n"
                                "<user_query>\nFix the flaky login test\n</user_query>"
                            ),
                        }
                    ]
                },
            },
            {
                "role": "assistant",
                "message": {
                    "model": "claude-sonnet-5-5",
                    "content": [
                        {"type": "thinking", "thinking": "Inspect auth_test.py first."},
                        {"type": "text", "text": "Reading the login test."},
                        {"type": "tool_use", "id": "c1", "name": "Read", "input": {"path": "auth_test.py"}},
                        {"type": "tool_use", "id": "c2", "name": "Read", "input": {"path": "screen.png"}},
                    ],
                },
            },
            {
                "role": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "c1", "content": "File not found", "is_error": True},
                        {
                            "type": "tool_result",
                            "tool_use_id": "c2",
                            "content": [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}],
                        },
                    ]
                },
            },
            {"type": "turn_ended"},
            {
                "role": "assistant",
                "timestamp": "2026-07-26T13:07:00Z",
                "message": {"content": [{"type": "text", "text": "Updated the path and fixed the test."}]},
            },
        ],
    )
    sub_file = projects / "Users-me-myapp" / "agent-transcripts" / sid / "subagents" / "agent-sub1.jsonl"
    _write_jsonl(
        sub_file,
        [
            {"role": "user", "timestamp": "2026-07-26T13:06:30Z", "message": {"content": "Find login tests"}},
            {
                "role": "assistant",
                "timestamp": "2026-07-26T13:06:40Z",
                "message": {"content": "Located tests/test_login.py."},
            },
        ],
    )
    return projects


def test_cursor_sessions_and_steps(cursor_dir):
    src = tp.read.cursor(cursor_dir)
    sessions = src.sessions.collect()
    assert len(sessions) == 1
    s = sessions[0]
    assert s.key == (("project", "/Users/me/myapp"), ("session", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"))
    assert s.title == "Fix the flaky login test"
    assert s.prompt == "Fix the flaky login test"
    assert (s.n_steps, s.n_subagents, s.n_tool_errors) == (5, 1, 1)
    assert s.duration_s == 60.0

    steps = src.steps.collect()
    assert [(st.source, st.agent_id, st.is_sidechain) for st in steps] == [
        ("user", None, False),
        ("agent", None, False),
        ("agent", None, False),
        ("user", "sub1", True),
        ("agent", "sub1", True),
    ]
    assert steps[1].reasoning == "Inspect auth_test.py first."
    assert [(c["name"], c["result"], c["is_error"]) for c in steps[1].tool_calls] == [
        ("Read", "File not found", True),
        ("Read", IMAGE_PLACEHOLDER, False),
    ]
    assert len(src.tool_calls.collect()) == 2


def test_cursor_replay_matches_batch(cursor_dir):
    src = tp.read.cursor(cursor_dir, subagents=False)
    count = lambda t: [(r.key, r.values) for r in t.group_by("session").reduce(lambda rs: {"n": len(rs)}).collect()]
    batch = count(src.steps)
    for seed in range(3):
        live = tp.replay(src, seed=seed, shuffle_within=4, duplicates=0.3)
        assert count(live.steps) == batch
