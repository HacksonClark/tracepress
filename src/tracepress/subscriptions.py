"""Use existing AI coding subscriptions (Claude Code, Codex, Cursor) as TracePress LMs.

Supports two ways to run against a subscription you already pay for:

1. Direct subscription OAuth HTTP (built into DSPy 3.4 for Claude Code and Codex when
   local OAuth token files exist in `~/.claude/.credentials.json` or `~/.codex/auth.json`).
2. Headless CLI subprocess engines (`claude -p`, `codex exec`, `agent -p`) wrapped as
   native DSPy 3.4 `lm15` engines, so DSPy's disk cache, structured adapters, retries
   and `tp.last_run()` token accounting all work identically to HTTP gateways — including
   on macOS where Claude Code and Cursor keep OAuth credentials in the system Keychain.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy
from dspy._vendor import lm15

CLAUDE_MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5",
}
DEFAULT_CLAUDE_CLI_MODEL = "sonnet"
DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_CURSOR_MODEL = "auto"

# Environment variables set by an active Claude Code parent session that would block
# nested `claude -p` calls, plus API keys that would override subscription billing.
_CLAUDE_STRIP_ENV = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_PID",
    "ANTHROPIC_API_KEY",
)
_CODEX_STRIP_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY")

_AUTH_MARKERS = (
    "not logged in",
    "please run `claude`",
    "login required",
    "run `agent login`",
    "agent login",
    "codex login",
    "unauthorized",
    "authentication",
    "invalid api key",
    "missing api key",
    "expired token",
    "401",
)
_BILLING_MARKERS = (
    "usage limit",
    "quota exceeded",
    "insufficient_quota",
    "billing",
    "credit balance",
    "out of credits",
    "plan limit",
    "spend limit",
)


def _search_path() -> str:
    """PATH augmented with common user CLI install locations (`~/.local/bin`, etc.)."""
    current = os.environ.get("PATH", "")
    extras = [str(Path.home() / ".local" / "bin"), "/usr/local/bin", "/opt/homebrew/bin"]
    parts = [p for p in current.split(os.pathsep) if p]
    for extra in extras:
        if extra not in parts:
            parts.append(extra)
    return os.pathsep.join(parts)


def find_cli(*names: str) -> str | None:
    """Return the executable path for the first CLI binary found, or None."""
    path = _search_path()
    for name in names:
        found = shutil.which(name, path=path)
        if found:
            return found
    return None


def _claude_credentials_path() -> Path:
    base = Path(os.environ.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude"))
    return base / ".credentials.json"


def has_claude_oauth_file(path: Path | None = None) -> bool:
    p = path or _claude_credentials_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
    return isinstance(oauth, dict) and bool(oauth.get("accessToken") or oauth.get("refreshToken"))


def _codex_auth_path() -> Path:
    base = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    return base / "auth.json"


def has_codex_oauth_file(path: Path | None = None) -> bool:
    p = path or _codex_auth_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    tokens = data.get("tokens") if isinstance(data, dict) else None
    return isinstance(tokens, dict) and bool(tokens.get("access_token") or tokens.get("refresh_token"))


@dataclass(frozen=True)
class SubscriptionStatus:
    name: str
    available: bool
    mode: str | None  # "oauth", "cli", or None
    spec: str  # canonical TracePress LM spec when available
    detail: str
    login_hint: str


def doctor() -> dict[str, Any]:
    """Inspect local Claude Code, Codex and Cursor subscriptions without making LM calls."""
    claude_oauth = has_claude_oauth_file()
    claude_bin = find_cli("claude")
    claude_token_env = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))
    claude_ok = claude_oauth or bool(claude_bin) or claude_token_env
    claude_mode = "oauth" if claude_oauth else ("cli" if claude_bin else ("env" if claude_token_env else None))

    codex_oauth = has_codex_oauth_file()
    codex_bin = find_cli("codex")
    codex_ok = codex_oauth or bool(codex_bin)
    codex_mode = "oauth" if codex_oauth else ("cli" if codex_bin else None)

    cursor_bin = find_cli("agent", "cursor-agent")
    cursor_key = bool(os.environ.get("CURSOR_API_KEY"))
    cursor_ok = bool(cursor_bin)
    cursor_mode = "cli" if cursor_bin else None

    statuses = [
        SubscriptionStatus(
            name="claude-code",
            available=claude_ok,
            mode=claude_mode,
            spec=f"claude-code:{DEFAULT_CLAUDE_CLI_MODEL}",
            detail=(
                f"OAuth credentials at {_claude_credentials_path()}" if claude_oauth
                else f"CLI at {claude_bin}" + (" + CLAUDE_CODE_OAUTH_TOKEN" if claude_token_env else "") if claude_bin
                else "CLAUDE_CODE_OAUTH_TOKEN set (install `claude` CLI to invoke)" if claude_token_env
                else "not found"
            ),
            login_hint="Install Claude Code and run `claude` -> `/login` (or `claude setup-token`).",
        ),
        SubscriptionStatus(
            name="codex",
            available=codex_ok,
            mode=codex_mode,
            spec=f"codex:{DEFAULT_CODEX_MODEL}",
            detail=(
                f"OAuth credentials at {_codex_auth_path()}" if codex_oauth
                else f"CLI at {codex_bin}" if codex_bin
                else "not found"
            ),
            login_hint="Install OpenAI Codex CLI and run `codex login`.",
        ),
        SubscriptionStatus(
            name="cursor",
            available=cursor_ok,
            mode=cursor_mode,
            spec=f"cursor:{DEFAULT_CURSOR_MODEL}",
            detail=(
                f"CLI at {cursor_bin}" + (" + CURSOR_API_KEY" if cursor_key else "")
                if cursor_bin else "not found"
            ),
            login_hint="Install Cursor CLI (`curl https://cursor.com/install -fsS | bash`) and run `agent login`.",
        ),
    ]

    recommended = None
    try:
        recommended = detect_subscription()
    except (RuntimeError, dspy.LMError):
        recommended = None

    return {
        "recommended": recommended,
        "providers": {
            s.name: {
                "available": s.available,
                "mode": s.mode,
                "spec": s.spec,
                "detail": s.detail,
                "login_hint": s.login_hint,
            }
            for s in statuses
        },
    }


def detect_subscription() -> str:
    """Return the best available subscription LM spec (e.g. `'claude-code:sonnet'`).

    Prefers the coding agent hosting the current session when its CLI or OAuth token is
    present, then falls back to direct OAuth files, then installed CLIs.
    """
    claude_oauth = has_claude_oauth_file()
    claude_bin = find_cli("claude")
    codex_oauth = has_codex_oauth_file()
    codex_bin = find_cli("codex")
    cursor_bin = find_cli("agent", "cursor-agent")

    # 1. Honour active host-agent hints when that backend is actually usable here.
    if (os.environ.get("CLAUDECODE") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")) and (claude_oauth or claude_bin):
        return f"claude-code:{DEFAULT_CLAUDE_CLI_MODEL}"
    if (os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_CI")) and (codex_oauth or codex_bin):
        return f"codex:{DEFAULT_CODEX_MODEL}"
    if (os.environ.get("CURSOR_AGENT") or os.environ.get("CURSOR_API_KEY")) and cursor_bin:
        return f"cursor:{DEFAULT_CURSOR_MODEL}"

    # 2. Direct OAuth token files (fastest: async HTTP with no subprocess overhead).
    if claude_oauth:
        return f"claude-code:{DEFAULT_CLAUDE_CLI_MODEL}"
    if codex_oauth:
        return f"codex:{DEFAULT_CODEX_MODEL}"

    # 3. Installed CLI binaries (uses OS Keychain / CLI login).
    if claude_bin:
        return f"claude-cli:{DEFAULT_CLAUDE_CLI_MODEL}"
    if codex_bin:
        return f"codex-cli:{DEFAULT_CODEX_MODEL}"
    if cursor_bin:
        return f"cursor:{DEFAULT_CURSOR_MODEL}"

    raise dspy.LMAuthError(
        "No AI coding subscription found. Log in to one of:\n"
        "  - Claude Code: run `claude` and `/login` (or `claude setup-token`)\n"
        "  - OpenAI Codex: run `codex login`\n"
        "  - Cursor CLI: install from https://cursor.com/install and run `agent login`\n"
        "Or configure an API model with `tp.configure(lm='openrouter:openai/gpt-6-luna')`.",
        provider="subscription",
    )


# --- Formatting lm15.Request into CLI prompts ------------------------------------------


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    text = getattr(part, "text", None)
    if text is not None:
        return str(text)
    return str(part)


def extract_prompt(request: lm15.Request) -> tuple[str | None, str]:
    """Return `(system_prompt, user_prompt)` for a DSPy `lm15.Request`."""
    system: str | None = None
    if request.system:
        if isinstance(request.system, str):
            system = request.system
        else:
            system = "\n".join(_part_text(p) for p in request.system if _part_text(p))

    msgs = list(request.messages)
    if not msgs:
        return system, ""
    if len(msgs) == 1 and msgs[0].role == "user":
        return system, msgs[0].text or ""

    blocks: list[str] = []
    for msg in msgs:
        role = (msg.role or "user").capitalize()
        body = msg.text or "\n".join(_part_text(p) for p in getattr(msg, "parts", ()) if _part_text(p))
        blocks.append(f"{role}:\n{body}" if len(msgs) > 1 else body)
    return system, "\n\n".join(blocks)


def _classify_cli_error(provider: str, model: str, detail: str, login_hint: str) -> Exception:
    lowered = detail.lower()
    if any(m in lowered for m in _BILLING_MARKERS):
        return dspy.LMBillingError(f"{provider} subscription limit hit: {detail}", provider=provider, model=model)
    if any(m in lowered for m in _AUTH_MARKERS):
        return dspy.LMAuthError(f"{provider} authentication failed: {detail}. {login_hint}", provider=provider, model=model)
    return RuntimeError(f"{provider} CLI failed ({model}): {detail}")


# --- CLI subprocess concurrency & execution --------------------------------------------

_loop_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]] = (
    weakref.WeakKeyDictionary()
)


def _cli_semaphore() -> asyncio.Semaphore:
    from .lm import config

    limit = max(1, int(os.environ.get("TRACEPRESS_CLI_CONCURRENCY") or config().cli_concurrency))
    loop = asyncio.get_running_loop()
    cached = _loop_semaphores.get(loop)
    if cached is None or cached[0] != limit:
        sem = asyncio.Semaphore(limit)
        _loop_semaphores[loop] = (limit, sem)
        return sem
    return cached[1]


def _clean_env(strip: tuple[str, ...]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in strip}
    env["PATH"] = _search_path()
    return env


def _run_subprocess_sync(
    argv: list[str],
    stdin_text: str,
    *,
    env: dict[str, str],
    cwd: str | Path | None = None,
) -> tuple[int, str, str]:
    proc = subprocess.run(
        argv,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(cwd) if cwd else None,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


async def _run_subprocess_async(
    argv: list[str],
    stdin_text: str,
    *,
    env: dict[str, str],
    cwd: str | Path | None = None,
) -> tuple[int, str, str]:
    async with _cli_semaphore():
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        try:
            out_b, err_b = await proc.communicate(stdin_text.encode("utf-8"))
        except BaseException:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
            raise
        return (
            proc.returncode or 0,
            out_b.decode("utf-8", errors="replace"),
            err_b.decode("utf-8", errors="replace"),
        )


# --- Provider-specific CLI command & output parsers ------------------------------------


def _build_claude_invocation(binary: str, model: str, system: str | None) -> tuple[list[str], dict[str, str]]:
    argv = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--tools",
        "",
        "--disallowedTools",
        "mcp__*",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    if model and model != "default":
        argv.extend(["--model", model])
    if system:
        argv.extend(["--system-prompt", system])
    env = _clean_env(_CLAUDE_STRIP_ENV)
    env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
    return argv, env


def _parse_claude_output(model: str, code: int, stdout: str, stderr: str) -> lm15.Response:
    hint = "Run `claude` and `/login` (or `claude setup-token`)."
    payload = None
    for candidate in (stdout.strip(), *(line for line in reversed(stdout.splitlines()) if line.strip().startswith("{"))):
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if not isinstance(payload, dict):
        detail = (stderr or stdout or f"exit code {code}").strip()
        raise _classify_cli_error("claude-code", model, detail, hint)

    if payload.get("is_error") or payload.get("subtype", "success") != "success" or code != 0:
        detail = str(payload.get("result") or payload.get("error") or stderr or f"exit code {code}").strip()
        raise _classify_cli_error("claude-code", model, detail, hint)

    text = str(payload.get("result") or "")
    u = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    in_tok = int(u.get("input_tokens") or 0)
    cache_read = int(u.get("cache_read_input_tokens") or 0)
    cache_write = int(u.get("cache_creation_input_tokens") or 0)
    out_tok = int(u.get("output_tokens") or 0)
    prompt_tok = in_tok + cache_read + cache_write
    usage = lm15.Usage(
        input_tokens=prompt_tok,
        output_tokens=out_tok,
        total_tokens=prompt_tok + out_tok,
        cache_read_tokens=cache_read or None,
        cache_write_tokens=cache_write or None,
    )
    return lm15.Response(
        id=payload.get("session_id"),
        model=f"claude-cli/{model}",
        message=lm15.Message.assistant(text),
        finish_reason="stop",
        usage=usage,
    )


def _build_codex_invocation(
    binary: str, model: str, system: str | None, workdir: Path
) -> tuple[list[str], dict[str, str], Path]:
    out_file = workdir / "last_message.txt"
    argv = [
        binary,
        "exec",
        "-",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        "-o",
        str(out_file),
    ]
    if model and model != "default":
        argv.extend(["-m", model])
    instructions = (
        f"{system}\n\nAnswer directly from the prompt without running shell commands or tools."
        if system
        else "Answer directly from the prompt without running shell commands or tools."
    )
    inst_file = workdir / "instructions.md"
    inst_file.write_text(instructions, encoding="utf-8")
    argv.extend(["-c", f'model_instructions_file="{inst_file.as_posix()}"'])
    return argv, _clean_env(_CODEX_STRIP_ENV), out_file


def _parse_codex_output(model: str, code: int, stdout: str, stderr: str, out_file: Path) -> lm15.Response:
    hint = "Run `codex login`."
    thread_id: str | None = None
    last_msg: str | None = None
    usage_dict: dict[str, Any] = {}
    errors: list[str] = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "thread.started":
            thread_id = event.get("thread_id")
        elif etype == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text") is not None:
                last_msg = str(item["text"])
        elif etype == "turn.completed" and isinstance(event.get("usage"), dict):
            usage_dict = event["usage"]
        elif etype == "turn.failed":
            err = event.get("error") or {}
            errors.append(str(err.get("message") or err))
        elif etype == "error":
            errors.append(str(event.get("message") or event))

    file_text = out_file.read_text(encoding="utf-8").strip() if out_file.exists() else None
    text = file_text if file_text is not None else last_msg

    if code != 0 or (errors and text is None):
        detail = "; ".join(errors) or (stderr or stdout or f"exit code {code}").strip()
        raise _classify_cli_error("codex", model, detail, hint)
    if text is None:
        detail = (stderr or stdout or "empty response from codex exec").strip()
        raise _classify_cli_error("codex", model, detail, hint)

    # In Codex `turn.completed`, input_tokens already includes cached_input_tokens and
    # output_tokens already includes reasoning_output_tokens.
    in_tok = int(usage_dict.get("input_tokens") or 0)
    out_tok = int(usage_dict.get("output_tokens") or 0)
    cache_read = int(usage_dict.get("cached_input_tokens") or 0)
    cache_write = int(usage_dict.get("cache_write_input_tokens") or 0)
    reasoning = int(usage_dict.get("reasoning_output_tokens") or 0)
    usage = lm15.Usage(
        input_tokens=in_tok if usage_dict else None,
        output_tokens=out_tok if usage_dict else None,
        total_tokens=(in_tok + out_tok) if usage_dict else None,
        cache_read_tokens=cache_read or None,
        cache_write_tokens=cache_write or None,
        reasoning_tokens=reasoning or None,
    )
    return lm15.Response(
        id=thread_id,
        model=f"codex-cli/{model}",
        message=lm15.Message.assistant(text),
        finish_reason="stop",
        usage=usage,
    )


def _prepare_cursor_workspace(workdir: Path) -> None:
    """Configure an isolated temp workspace with all tools denied so Cursor runs pure Q&A."""
    cfg_dir = workdir / ".cursor"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "cli.json").write_text(
        json.dumps({
            "permissions": {
                "allow": [],
                "deny": ["Read(**)", "Write(**)", "Shell(*)", "WebFetch(*)", "Mcp(*:*)"],
            }
        }),
        encoding="utf-8",
    )


def _build_cursor_invocation(
    binary: str, model: str, system: str | None, prompt: str, workdir: Path
) -> tuple[list[str], str, dict[str, str]]:
    _prepare_cursor_workspace(workdir)
    argv = [
        binary,
        "-p",
        "--mode=ask",
        "--trust",
        "--output-format",
        "json",
        "--workspace",
        str(workdir),
    ]
    if model and model != "default":
        argv.extend(["--model", model])
    combined = f"System instructions:\n{system}\n\n{prompt}" if system else prompt
    return argv, combined, _clean_env(())


def _parse_cursor_output(model: str, code: int, stdout: str, stderr: str) -> lm15.Response:
    hint = "Run `agent login` or set `CURSOR_API_KEY`."
    payload = None
    for candidate in (stdout.strip(), *(line for line in reversed(stdout.splitlines()) if line.strip().startswith("{"))):
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if code != 0 or not isinstance(payload, dict):
        detail = (
            str(payload.get("result") or payload.get("error"))
            if isinstance(payload, dict) and (payload.get("is_error") or payload.get("error"))
            else (stderr or stdout or f"exit code {code}").strip()
        )
        raise _classify_cli_error("cursor", model, detail, hint)

    if payload.get("is_error") or payload.get("subtype", "success") != "success":
        detail = str(payload.get("result") or payload.get("error") or stderr or "Cursor CLI returned an error").strip()
        raise _classify_cli_error("cursor", model, detail, hint)

    text = str(payload.get("result") or "")
    u = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    in_tok = u.get("input_tokens") or u.get("prompt_tokens")
    out_tok = u.get("output_tokens") or u.get("completion_tokens")
    usage = lm15.Usage(
        input_tokens=int(in_tok) if in_tok is not None else None,
        output_tokens=int(out_tok) if out_tok is not None else None,
        total_tokens=(int(in_tok or 0) + int(out_tok or 0)) if (in_tok is not None or out_tok is not None) else None,
    )
    return lm15.Response(
        id=payload.get("session_id") or payload.get("request_id"),
        model=f"cursor-cli/{model}",
        message=lm15.Message.assistant(text),
        finish_reason="stop",
        usage=usage,
    )


# --- DSPy 3.4 lm15 Engines -------------------------------------------------------------


class CLISyncEngine:
    def __init__(self, backend: str, model: str, binary: str):
        self.backend = backend
        self.model = model
        self.binary = binary

    def dump_state(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model}

    def complete(self, request: lm15.Request) -> lm15.Response:
        system, prompt = extract_prompt(request)
        if self.backend == "claude-cli":
            argv, env = _build_claude_invocation(self.binary, self.model, system)
            code, out, err = _run_subprocess_sync(argv, prompt, env=env)
            return _parse_claude_output(self.model, code, out, err)
        with tempfile.TemporaryDirectory(prefix=f"tp-{self.backend}-") as tmp:
            workdir = Path(tmp)
            if self.backend == "codex-cli":
                argv, env, out_file = _build_codex_invocation(self.binary, self.model, system, workdir)
                code, out, err = _run_subprocess_sync(argv, prompt, env=env, cwd=workdir)
                return _parse_codex_output(self.model, code, out, err, out_file)
            if self.backend == "cursor-cli":
                argv, combined, env = _build_cursor_invocation(self.binary, self.model, system, prompt, workdir)
                code, out, err = _run_subprocess_sync(argv, combined, env=env, cwd=workdir)
                return _parse_cursor_output(self.model, code, out, err)
        raise ValueError(f"unknown CLI backend {self.backend!r}")


class CLIAsyncEngine:
    def __init__(self, backend: str, model: str, binary: str):
        self.backend = backend
        self.model = model
        self.binary = binary

    def dump_state(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model}

    async def complete(self, request: lm15.Request) -> lm15.Response:
        system, prompt = extract_prompt(request)
        if self.backend == "claude-cli":
            argv, env = _build_claude_invocation(self.binary, self.model, system)
            code, out, err = await _run_subprocess_async(argv, prompt, env=env)
            return _parse_claude_output(self.model, code, out, err)
        with tempfile.TemporaryDirectory(prefix=f"tp-{self.backend}-") as tmp:
            workdir = Path(tmp)
            if self.backend == "codex-cli":
                argv, env, out_file = _build_codex_invocation(self.binary, self.model, system, workdir)
                code, out, err = await _run_subprocess_async(argv, prompt, env=env, cwd=workdir)
                return _parse_codex_output(self.model, code, out, err, out_file)
            if self.backend == "cursor-cli":
                argv, combined, env = _build_cursor_invocation(self.binary, self.model, system, prompt, workdir)
                code, out, err = await _run_subprocess_async(argv, combined, env=env, cwd=workdir)
                return _parse_cursor_output(self.model, code, out, err)
        raise ValueError(f"unknown CLI backend {self.backend!r}")


def _make_cli_lm(backend: str, gateway: str, model: str, cli_names: tuple[str, ...], login_hint: str) -> dspy.LM:
    binary = find_cli(*cli_names)
    if not binary:
        names = " / ".join(f"`{n}`" for n in cli_names)
        raise dspy.LMAuthError(f"{gateway} CLI ({names}) not found on PATH. {login_hint}", provider=gateway, model=model)
    lm = dspy.LM(
        f"{backend}/{model}",
        engine=CLISyncEngine(backend, model, binary),
        async_engine=CLIAsyncEngine(backend, model, binary),
    )
    lm._tp_label = (gateway, model)
    lm._tp_cli = True
    return lm


# --- Provider builders registered in `tracepress.lm.PROVIDERS` -------------------------


def build_claude_code(model: str, spec: str, *, mode: str = "auto") -> dspy.LM:
    chosen = model or DEFAULT_CLAUDE_CLI_MODEL
    if mode == "oauth" or (mode == "auto" and has_claude_oauth_file()):
        full = CLAUDE_MODEL_ALIASES.get(chosen, chosen)
        lm = dspy.LM(f"claude-code/{full}")
        lm._tp_label = ("claude-code", full)
        return lm
    return _make_cli_lm(
        "claude-cli",
        "claude-code",
        chosen,
        ("claude",),
        "Install Claude Code and run `claude` -> `/login` (or `claude setup-token`).",
    )


def build_codex(model: str, spec: str, *, mode: str = "auto") -> dspy.LM:
    chosen = model or DEFAULT_CODEX_MODEL
    if mode == "oauth" or (mode == "auto" and has_codex_oauth_file()):
        lm = dspy.LM(f"openai-codex/{chosen}")
        lm._tp_label = ("codex", chosen)
        return lm
    return _make_cli_lm(
        "codex-cli",
        "codex",
        chosen,
        ("codex",),
        "Install OpenAI Codex CLI and run `codex login`.",
    )


def build_cursor(model: str, spec: str) -> dspy.LM:
    chosen = model or DEFAULT_CURSOR_MODEL
    return _make_cli_lm(
        "cursor-cli",
        "cursor",
        chosen,
        ("agent", "cursor-agent"),
        "Install Cursor CLI (`curl https://cursor.com/install -fsS | bash`) and run `agent login`.",
    )


def build_subscription(model: str, spec: str) -> dspy.LM:
    from .lm import make_lm

    if model and model not in ("auto", "default"):
        raise ValueError(
            f"{spec!r}: `subscription` only accepts `:auto` (or bare `subscription`). "
            "Pick a specific tool with `claude-code:<model>`, `codex:<model>` or `cursor:<model>`."
        )
    return make_lm(detect_subscription())
