"""Cursor Agent transcripts, from `~/.cursor/projects/<slug>/agent-transcripts/**/*.jsonl`.

Levels: project > session > step (> call). The project is the workspace directory decoded
from the project slug (or `<workspace_path>` / `Workspace Path:` when embedded in the
opening user prompt).

Cursor writes JSONL transcripts with `{role: "user"|"assistant", message: {content: [...]}}`
records (plus `type: "turn_ended"` markers). Content blocks cover `text`, `thinking`,
`tool_use`, and `tool_result`. Subagent transcripts under `subagents/*.jsonl` become steps
of the parent session with `is_sidechain=True` and their `agent_id`, matching Claude Code.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atif import clean_content
from .sessions import (
    Parsed,
    SessionAdapter,
    SessionFile,
    find_files,
    first_prompt,
    join,
    new_step,
    read_lines,
    seconds,
    select,
)

SESSION_COLUMNS = {
    "title": str,
    "prompt": str,
    "cwd": str,
    "started_at": str,
    "finished_at": str,
    "duration_s": float,
    "n_steps": int,
    "n_subagents": int,
    "n_tool_errors": int,
    "path": str,
}

_TS_BLOCK = re.compile(r"<timestamp>\s*(.*?)\s*</timestamp>", re.I | re.S)
_USER_QUERY = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.I | re.S)
_WORKSPACE_PATH = re.compile(r"Workspace Path:\s*([^\n<]+)", re.I)


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except OSError:
        return None


def _parse_timestamp(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", " ".join(raw.strip().split()))
    for fmt in (
        "%A, %b %d, %Y, %I:%M %p",
        "%A, %B %d, %Y, %I:%M %p",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def _extract_embedded(text: str) -> tuple[str | None, str | None, str | None]:
    """Return `(timestamp_iso, workspace_path, user_query)` from Cursor's wrapper tags."""
    ts_match = _TS_BLOCK.search(text)
    ts = _parse_timestamp(ts_match.group(1)) if ts_match else None
    ws_match = _WORKSPACE_PATH.search(text)
    ws = ws_match.group(1).strip() if ws_match else None
    q_match = _USER_QUERY.search(text)
    query = q_match.group(1).strip() if q_match else None
    return ts, ws, query


def _blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _parse_thread(path: Path, agent_id: str | None, calls: dict[str, dict]) -> tuple[list[dict], str | None, str | None, list[str]]:
    steps: list[dict] = []
    cwd_found: str | None = None
    title: str | None = None
    stamps: list[str] = []
    current_ts: str | None = None

    for line in read_lines(path):
        if line.get("type") == "turn_ended":
            continue
        role = line.get("role") or line.get("type")
        if role not in ("user", "assistant"):
            continue
        if isinstance(line.get("timestamp"), str):
            current_ts = line["timestamp"]
            stamps.append(current_ts)

        blocks = _blocks(line.get("message"))
        if role == "assistant":
            step = new_step("agent", current_ts, agent_id, model_name=(line.get("message") or {}).get("model"),
                            _text=[], _thinking=[])
            for block in blocks:
                btype = block.get("type")
                if btype == "text" and block.get("text"):
                    step["_text"].append(str(block["text"]))
                elif btype == "thinking" and block.get("thinking"):
                    step["_thinking"].append(str(block["thinking"]))
                elif btype in ("tool_use", "server_tool_use"):
                    call_id = block.get("id")
                    call = {
                        "id": call_id,
                        "name": block.get("name") or "tool",
                        "arguments": block.get("input"),
                        "result": None,
                        "is_error": None,
                    }
                    step["tool_calls"].append(call)
                    if call_id:
                        calls[str(call_id)] = call
            step["message"] = join(step.pop("_text"))
            step["reasoning"] = join(step.pop("_thinking"))
            if step["message"] or step["reasoning"] or step["tool_calls"]:
                steps.append(step)
            continue

        # User turn: may contain tool_result blocks and/or user text (with optional <user_query> wrapper).
        said: list[str] = []
        unmatched: list[str] = []
        for block in blocks:
            btype = block.get("type")
            if btype == "tool_result":
                result = clean_content(block.get("content"))
                call = calls.get(str(block.get("tool_use_id") or ""))
                if call is not None:
                    call["result"] = result
                    call["is_error"] = bool(block.get("is_error"))
                elif result:
                    unmatched.append(str(result))
            elif btype == "text" and block.get("text"):
                raw = str(block["text"])
                emb_ts, emb_ws, emb_query = _extract_embedded(raw)
                if emb_ts:
                    current_ts = emb_ts
                    stamps.append(emb_ts)
                if emb_ws and not cwd_found:
                    cwd_found = emb_ws
                display = emb_query if emb_query is not None else raw
                if title is None and agent_id is None and display.strip():
                    title = " ".join(display.strip().split())[:80]
                said.append(display)
            else:
                cleaned = clean_content([block])
                if cleaned:
                    said.append(str(cleaned))

        if unmatched and steps:
            steps[-1]["observation"] = join([steps[-1]["observation"], *unmatched])
        text = join(said)
        if text:
            step = new_step("user", current_ts, agent_id)
            step["message"] = text
            steps.append(step)

    return steps, cwd_found, title, stamps


def _slug_to_project(path: Path) -> str:
    parts = path.parts
    if "agent-transcripts" in parts:
        idx = parts.index("agent-transcripts")
        if idx >= 1:
            slug = parts[idx - 1].lstrip("-")
            if slug:
                return "/" + slug.replace("-", "/")
    return "unknown"


def _peek_cwd(path: Path) -> str | None:
    for i, line in enumerate(read_lines(path)):
        if i >= 20:
            break
        if isinstance(line.get("cwd"), str) and line["cwd"]:
            return line["cwd"]
        for block in _blocks(line.get("message")):
            if block.get("type") == "text" and block.get("text"):
                _, ws, _ = _extract_embedded(str(block["text"]))
                if ws:
                    return ws
    return None


class CursorAdapter(SessionAdapter):
    """Cursor Agent transcripts. Levels: project > session > step (> call)."""

    session_columns = SESSION_COLUMNS

    def parse(self, file: SessionFile) -> Parsed:
        calls: dict[str, dict] = {}
        steps, cwd, title, stamps = _parse_thread(file.path, None, calls)
        for sub in file.extra:
            sub_id = sub.stem.removeprefix("agent-")
            sub_steps, _, _, sub_stamps = _parse_thread(sub, sub_id, calls)
            steps.extend(sub_steps)
            stamps.extend(sub_stamps)

        mtime = _mtime_iso(file.path)
        started = min(stamps) if stamps else mtime
        finished = max(stamps) if stamps else mtime
        return Parsed({
            "title": title,
            "prompt": first_prompt(steps),
            "cwd": cwd or file.project,
            "started_at": started,
            "finished_at": finished,
            "duration_s": seconds(started, finished),
            "n_steps": len(steps),
            "n_subagents": len({s["agent_id"] for s in steps if s["agent_id"]}),
            "n_tool_errors": sum(1 for c in calls.values() if c["is_error"]),
            "path": str(file.path),
        }, steps)


def default_root() -> Path:
    return Path(os.environ.get("CURSOR_CONFIG_DIR") or Path.home() / ".cursor") / "projects"


def _is_subagent(path: Path) -> bool:
    return "subagents" in path.parts or path.name.startswith("agent-")


def read_cursor(
    path: str | Path | None = None,
    project: str | None = None,
    limit: int | None = None,
    subagents: bool = True,
) -> CursorAdapter:
    """Cursor Agent sessions from `~/.cursor/projects/*/agent-transcripts`.

    `path` defaults to `~/.cursor/projects` (or `$CURSOR_CONFIG_DIR/projects`); it can also
    be one project directory, an `agent-transcripts` folder, one `.jsonl` transcript, or a
    glob. `project` filters by workspace path substring; `limit` keeps the N most recently
    updated sessions.
    """
    root = Path(path).expanduser() if path is not None else default_root()
    candidates = [
        f for f in find_files(root, "*.jsonl", skip=_is_subagent)
        if root.is_file() or "agent-transcripts" in f.parts or f.parent == root or f.parent.parent == root
    ]
    found: list[SessionFile] = []
    for f in candidates:
        proj = _peek_cwd(f) or _slug_to_project(f)
        sid = f.stem
        sub_dir = f.parent / "subagents" if f.parent.name == sid else f.with_suffix("") / "subagents"
        extra = sorted(sub_dir.glob("*.jsonl")) if subagents and sub_dir.is_dir() else []
        found.append(SessionFile(f, proj, sid, extra))
    found = select(found, project, limit)
    if not found:
        raise FileNotFoundError(f"no Cursor sessions found at {root}")
    return CursorAdapter(found)
