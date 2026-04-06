# cli_executor.py — запуск Claude CLI с поддержкой нескольких параллельных сессий
#
# Каждая вкладка браузера (client_id = ws_id) получает:
#   - Свою Claude CLI сессию (--resume <session_id>)
#   - Свой Lock (параллельные вкладки не блокируют друг друга)
#
# Claude CLI сам управляет историей, контекстом и compacting.

import asyncio
import json
import logging
import os
import sys
import shutil
import config as _config
from config import CLAUDE_CLI_PATH, CLAUDE_CLI_TIMEOUT, AGENT_INACTIVITY_TIMEOUT, AGENT_MAX_TIMEOUT, AGENT_BUFFER_LIMIT

logger = logging.getLogger(__name__)


# ============================================================
#  Multi-session state: per client_id
# ============================================================

class _SessionState:
    """Tracks Claude CLI session for one client (browser tab)."""
    __slots__ = ('cli_session_id', 'agent_session_id', 'lock')

    def __init__(self):
        self.cli_session_id: str | None = None      # --print mode session
        self.agent_session_id: str | None = None     # agent mode session
        self.lock = asyncio.Lock()                    # per-client lock


_sessions: dict[str, _SessionState] = {}


def _get_state(client_id: str) -> _SessionState:
    """Get or create session state for a client."""
    if client_id not in _sessions:
        _sessions[client_id] = _SessionState()
    return _sessions[client_id]


# ============================================================
#  Simple --print mode (legacy, used when AGENT_MODE_ENABLED=False)
# ============================================================

async def run_claude_streaming(
    text: str,
    on_chunk: callable = None,
    on_status: callable = None,
    new_session: bool = False,
    timeout: int = None,
    client_id: str = "",
) -> str:
    """
    Отправляет текст в Claude CLI и возвращает полный ответ.
    Каждый client_id получает отдельную сессию.
    """
    state = _get_state(client_id)

    async with state.lock:
        cmd = _build_cmd(state, new_session=new_session)

        if new_session:
            state.cli_session_id = None

        is_new = state.cli_session_id is None
        logger.info(
            f"Claude CLI [{client_id[:8]}]: {'новая сессия' if is_new else f'продолжение ({state.cli_session_id[:12]})'}, "
            f"команда: {' '.join(cmd)}, длина запроса: {len(text)} символов"
        )

        if on_status:
            session_info = "новая сессия" if is_new else "продолжение"
            await on_status(f"Claude CLI ({session_info})...")

        try:
            # CREATE_NEW_PROCESS_GROUP isolates the child from Ctrl+C/window-close signals
            _flags = {"creationflags": 0x00000200} if sys.platform == "win32" else {}
            # Force UTF-8 in all child Python processes (fixes charmap errors on Windows)
            _env = os.environ.copy()
            _env["PYTHONUTF8"] = "1"
            _env["PYTHONIOENCODING"] = "utf-8"
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_env,
                **_flags,
            )

            process.stdin.write(text.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

            full_response = []
            effective_timeout = timeout or CLAUDE_CLI_TIMEOUT
            deadline = asyncio.get_event_loop().time() + effective_timeout

            while True:
                if asyncio.get_event_loop().time() > deadline:
                    process.kill()
                    raise asyncio.TimeoutError()

                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    break

                if not line:
                    break

                decoded = line.decode("utf-8", errors="replace")
                full_response.append(decoded)

                if on_chunk:
                    await on_chunk(decoded)

            await process.wait()

            if process.returncode != 0:
                stderr_data = await process.stderr.read()
                error = stderr_data.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Claude CLI ошибка (код {process.returncode}): {error}")

            response = "".join(full_response).strip()
            logger.info(f"Claude CLI [{client_id[:8]}] ответил: {len(response)} символов")

            # Mark session as started (simple mode uses --continue, no session_id tracking)
            if state.cli_session_id is None:
                state.cli_session_id = "started"

            if on_status:
                await on_status("Ответ от Claude получен")

            return response

        except asyncio.TimeoutError:
            raise RuntimeError(f"Claude CLI таймаут ({effective_timeout} сек)")
        except FileNotFoundError:
            raise RuntimeError(
                f"Claude CLI не найден. Проверь CLAUDE_CLI_PATH в config.py (сейчас: '{CLAUDE_CLI_PATH}')"
            )


async def new_session(client_id: str = ""):
    """Сбросить сессию для клиента — следующий вызов создаст новый диалог."""
    state = _get_state(client_id)
    state.cli_session_id = None
    logger.info(f"Сессия Claude CLI [{client_id[:8]}] сброшена")


def _build_cmd(state: _SessionState, new_session: bool = False) -> list[str]:
    """Строит команду для Claude CLI --print mode."""
    claude_path = _find_claude()
    cmd = [claude_path, "--print"]

    if state.cli_session_id and not new_session:
        cmd.append("--continue")

    return cmd


# ============================================================
#  Agent mode (primary, with full tool access)
# ============================================================

async def run_claude_agent_streaming(
    text: str,
    on_tool_use: callable = None,
    on_tool_result: callable = None,
    on_text: callable = None,
    on_result: callable = None,
    on_status: callable = None,
    on_session_id: callable = None,
    new_session: bool = False,
    timeout: int = None,
    system_prompt: str = None,
    client_id: str = "",
) -> str:
    """
    Запускает Claude CLI в agent-режиме с полным доступом к инструментам.
    Каждый client_id получает отдельную сессию и отдельный Lock.
    Параллельные вкладки работают одновременно.
    """
    state = _get_state(client_id)

    async with state.lock:
        if new_session:
            state.agent_session_id = None

        cmd = _build_agent_cmd(state, new_session=new_session, system_prompt=system_prompt)

        is_new = state.agent_session_id is None
        logger.info(
            f"Claude Agent [{client_id[:8]}]: "
            f"{'новая сессия' if is_new else f'resume ({state.agent_session_id[:12]})'}, "
            f"команда: {' '.join(cmd)}, длина запроса: {len(text)} символов"
        )

        if on_status:
            session_info = "новая сессия" if is_new else "продолжение"
            await on_status(f"Claude Agent ({session_info})...")

        try:
            # CREATE_NEW_PROCESS_GROUP isolates the child from Ctrl+C/window-close signals
            _flags = {"creationflags": 0x00000200} if sys.platform == "win32" else {}
            # Force UTF-8 in all child Python processes (fixes charmap errors on Windows)
            _env = os.environ.copy()
            _env["PYTHONUTF8"] = "1"
            _env["PYTHONIOENCODING"] = "utf-8"
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=AGENT_BUFFER_LIMIT,
                env=_env,
                **_flags,
            )

            process.stdin.write(text.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

            import time as _time

            inactivity_limit = timeout or AGENT_INACTIVITY_TIMEOUT
            max_deadline = _time.monotonic() + AGENT_MAX_TIMEOUT

            final_result_text = ""
            last_text_content = ""
            last_tool_name = ""
            tool_use_count = 0
            event_count = 0
            start_time = _time.monotonic()
            last_activity = start_time
            tool_start_time = None

            while True:
                now = _time.monotonic()
                elapsed = now - start_time

                # Hard ceiling — emergency stop (2 hours default)
                if now > max_deadline:
                    logger.warning(f"Agent [{client_id[:8]}] MAX TIMEOUT ({AGENT_MAX_TIMEOUT}s) after {elapsed:.1f}s, {tool_use_count} tools")
                    process.kill()
                    raise asyncio.TimeoutError()

                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=inactivity_limit,
                    )
                except asyncio.TimeoutError:
                    silent_sec = _time.monotonic() - last_activity
                    elapsed = _time.monotonic() - start_time

                    # Check if the process is still alive
                    if process.returncode is not None:
                        logger.warning(
                            f"Agent [{client_id[:8]}] process dead (rc={process.returncode}) after "
                            f"silence {silent_sec:.0f}s, elapsed={elapsed:.1f}s"
                        )
                        break

                    # Process alive — log heartbeat and keep waiting
                    elapsed_min = int(elapsed // 60)
                    logger.info(
                        f"Agent [{client_id[:8]}] HEARTBEAT: жив, молчит {silent_sec:.0f}s, "
                        f"elapsed={elapsed_min}m{int(elapsed%60)}s, "
                        f"last_tool={last_tool_name}, tools={tool_use_count}"
                    )
                    if on_status:
                        await on_status(f"Agent думает... ({elapsed_min} мин)")

                    continue

                if not line:
                    break

                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded:
                    continue

                # Activity! Reset inactivity timer
                last_activity = _time.monotonic()
                event_count += 1

                # Parse NDJSON event
                try:
                    event = json.loads(decoded)
                except json.JSONDecodeError:
                    logger.debug(f"Agent [{client_id[:8]}]: non-JSON line: {decoded[:200]}")
                    continue

                event_type = event.get("type", "")
                event_subtype = event.get("subtype", "")
                elapsed = _time.monotonic() - start_time

                # Log every event type for visibility
                if event_type == "system":
                    model = event.get("model", "?")
                    tools = event.get("tools", [])
                    session_id = event.get("session_id", "?")

                    # Capture session_id for --resume on next call
                    if session_id and session_id != "?":
                        state.agent_session_id = session_id
                        logger.info(f"Agent [{client_id[:8]}] captured session_id={session_id[:12]}")
                        if on_session_id:
                            await on_session_id(session_id)

                    logger.info(f"Agent [{client_id[:8]}] [{elapsed:.1f}s] system/{event_subtype}: model={model}, tools={len(tools)}, session={session_id[:12]}")
                    if on_status:
                        await on_status(f"Agent подключён (модель: {model})")

                elif event_type == "assistant":
                    message = event.get("message", {})
                    content_blocks = message.get("content", [])
                    for block in content_blocks:
                        block_type = block.get("type", "")

                        if block_type == "tool_use":
                            tool_name = block.get("name", "unknown")
                            tool_input = block.get("input", {})
                            last_tool_name = tool_name
                            tool_use_count += 1
                            tool_start_time = _time.monotonic()

                            input_summary = _summarize_tool_input(tool_name, tool_input)
                            logger.info(f"Agent [{client_id[:8]}] [{elapsed:.1f}s] tool #{tool_use_count}: {tool_name} → {input_summary}")

                            if on_tool_use:
                                await on_tool_use(tool_name, tool_input)
                            if on_status:
                                await on_status(f"Tool #{tool_use_count}: {tool_name} → {input_summary}")

                        elif block_type == "text":
                            text_content = block.get("text", "")
                            if text_content:
                                last_text_content = text_content
                                logger.info(f"Agent [{client_id[:8]}] [{elapsed:.1f}s] text: {len(text_content)} chars — {text_content[:150]}{'...' if len(text_content) > 150 else ''}")
                                if on_text:
                                    await on_text(text_content)

                elif event_type == "user":
                    tool_elapsed = ""
                    if tool_start_time:
                        tool_ms = (_time.monotonic() - tool_start_time) * 1000
                        tool_elapsed = f" ({tool_ms:.0f}ms)"
                        tool_start_time = None

                    content_blocks = event.get("message", {}).get("content", [])
                    summary = _summarize_tool_result(content_blocks, last_tool_name)
                    logger.info(f"Agent [{client_id[:8]}] [{elapsed:.1f}s] tool_result: {last_tool_name}{tool_elapsed} → {summary}")

                    if on_tool_result and last_tool_name:
                        await on_tool_result(last_tool_name, summary)
                    if on_status:
                        await on_status(f"✓ {last_tool_name}{tool_elapsed}")

                elif event_type == "result":
                    final_result_text = event.get("result", "")
                    result_subtype = event.get("subtype", "")
                    result_errors = event.get("errors", [])

                    # Capture stdout errors (Claude CLI reports errors here, not stderr)
                    if result_subtype == "error_during_execution" and result_errors:
                        stdout_error = "; ".join(result_errors)
                        # Stale --resume session → clear it and retry as new session
                        if "No conversation found" in stdout_error:
                            logger.warning(
                                f"Agent [{client_id[:8]}] stale session_id={state.agent_session_id}, "
                                f"clearing and will retry as new session"
                            )
                            state.agent_session_id = None
                        raise RuntimeError(f"Claude Agent ошибка: {stdout_error}")

                    result_info = {
                        "result": final_result_text,
                        "subtype": result_subtype,
                        "cost_usd": event.get("total_cost_usd", 0),
                        "duration_ms": event.get("duration_ms", 0),
                        "turns": tool_use_count,
                        "num_events": event_count,
                    }
                    logger.info(
                        f"Agent [{client_id[:8]}] [{elapsed:.1f}s] DONE: {len(final_result_text)} chars, "
                        f"cost=${result_info['cost_usd']:.4f}, "
                        f"duration={result_info['duration_ms']}ms, "
                        f"tools={tool_use_count}, events={event_count}"
                    )
                    if on_result:
                        await on_result(result_info)

                else:
                    logger.debug(f"Agent [{client_id[:8]}] [{elapsed:.1f}s] event: {event_type}/{event_subtype}")

            await process.wait()

            if process.returncode != 0:
                stderr_data = await process.stderr.read()
                error = stderr_data.decode("utf-8", errors="replace").strip()
                # Claude CLI writes errors to stdout as JSON, not stderr — stderr may be empty
                if not error:
                    error = "(ошибка в stdout JSON выше)"
                raise RuntimeError(f"Claude Agent ошибка (код {process.returncode}): {error}")

            response = final_result_text or last_text_content
            logger.info(f"Claude Agent [{client_id[:8]}] ответил: {len(response)} символов")

            if on_status:
                await on_status("Ответ от Claude Agent получен")

            return response

        except asyncio.TimeoutError:
            elapsed = _time.monotonic() - start_time
            raise RuntimeError(f"Claude Agent таймаут после {elapsed:.0f} сек ({tool_use_count} tools)")
        except FileNotFoundError:
            raise RuntimeError(
                f"Claude CLI не найден. Проверь CLAUDE_CLI_PATH в config.py (сейчас: '{CLAUDE_CLI_PATH}')"
            )


async def new_agent_session(client_id: str = ""):
    """Сбросить agent-сессию для клиента."""
    state = _get_state(client_id)
    state.agent_session_id = None
    logger.info(f"Agent-сессия [{client_id[:8]}] сброшена")


def restore_agent_session(client_id: str, agent_session_id: str):
    """Восстановить agent-сессию по известному session_id (например, после перезагрузки страницы)."""
    state = _get_state(client_id)
    state.agent_session_id = agent_session_id
    logger.info(f"Agent-сессия [{client_id[:8]}] восстановлена: {agent_session_id[:12]}")


def _build_agent_cmd(state: _SessionState, new_session: bool = False, system_prompt: str = None) -> list[str]:
    """
    Строит команду для Claude CLI в agent-режиме.
    Первый вызов: claude -p ... (новая сессия)
    Повторный:    claude -p ... --resume <session_id>
    """
    claude_path = _find_claude()

    # Determine the effective model:
    # - In Anthropic mode: AGENT_MODEL (e.g. "claude-sonnet-4-6") passed via --model
    # - In OpenRouter mode: ANTHROPIC_MODEL env var in settings.json is used (e.g. "anthropic/claude-sonnet-4-5")
    #   so we must NOT pass --model here (it would override ANTHROPIC_MODEL with a name OpenRouter doesn't know)
    use_openrouter = getattr(_config, "CLAUDE_PROVIDER", "anthropic") == "openrouter"

    cmd = [
        claude_path,
        "-p",
        "--verbose",
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
    ]

    if not use_openrouter:
        # Anthropic direct: pass model explicitly
        cmd.extend(["--model", _config.AGENT_MODEL])

    if state.agent_session_id and not new_session:
        cmd.extend(["--resume", state.agent_session_id])

    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    return cmd


# ============================================================
#  Cleanup: remove session state when client disconnects
# ============================================================

def remove_client(client_id: str):
    """Remove session state for a disconnected client."""
    removed = _sessions.pop(client_id, None)
    if removed:
        logger.info(f"Session state removed for client [{client_id[:8]}]")


# ============================================================
#  Helpers
# ============================================================

def _summarize_tool_input(tool_name: str, tool_input: dict) -> str:
    """Краткое описание входных данных инструмента для логов."""
    try:
        if tool_name == "Read":
            return tool_input.get("file_path", "?")
        elif tool_name == "Write":
            fp = tool_input.get("file_path", "?")
            content = tool_input.get("content", "")
            return f"{fp} ({len(content)} chars)"
        elif tool_name == "Edit":
            fp = tool_input.get("file_path", "?")
            old = tool_input.get("old_string", "")
            return f"{fp} (replace {len(old)} chars)"
        elif tool_name == "Bash":
            cmd = tool_input.get("command", "?")
            return cmd[:120] + ("..." if len(cmd) > 120 else "")
        elif tool_name == "Glob":
            return tool_input.get("pattern", "?")
        elif tool_name == "Grep":
            pattern = tool_input.get("pattern", "?")
            path = tool_input.get("path", ".")
            return f"'{pattern}' in {path}"
        elif tool_name == "WebSearch":
            return tool_input.get("query", "?")
        elif tool_name == "WebFetch":
            return tool_input.get("url", "?")[:100]
        elif tool_name == "TodoWrite":
            todos = tool_input.get("todos", [])
            return f"{len(todos)} items"
        else:
            keys = list(tool_input.keys())[:2]
            parts = [f"{k}={str(tool_input[k])[:60]}" for k in keys]
            return ", ".join(parts) if parts else "(no input)"
    except Exception:
        return str(tool_input)[:100]


def _summarize_tool_result(content_blocks: list, tool_name: str) -> str:
    """Создаёт краткое описание результата инструмента для UI."""
    if not content_blocks:
        return f"{tool_name} completed"

    for block in content_blocks:
        block_type = block.get("type", "")

        if block_type == "tool_result":
            content = block.get("content", "")
            if isinstance(content, str):
                lines = content.strip().split("\n")
                line_count = len(lines)
                if line_count > 3:
                    return f"{tool_name}: {line_count} lines of output"
                return f"{tool_name}: {content[:120]}"
            elif isinstance(content, list):
                return f"{tool_name}: {len(content)} result blocks"

        elif block_type == "text":
            text = block.get("text", "")
            if text:
                lines = text.strip().split("\n")
                if len(lines) > 3:
                    return f"{tool_name}: {len(lines)} lines"
                return f"{tool_name}: {text[:120]}"

    return f"{tool_name} completed"


def _find_claude() -> str:
    """Находит исполняемый файл Claude CLI."""
    if os.path.exists(CLAUDE_CLI_PATH):
        return CLAUDE_CLI_PATH

    found = shutil.which("claude") or shutil.which("claude.cmd")
    if found:
        return found

    if sys.platform == "win32":
        npm_paths = [
            os.path.expanduser(r"~\AppData\Roaming\npm\claude.cmd"),
            r"C:\Program Files\nodejs\claude.cmd",
        ]
        for p in npm_paths:
            if os.path.exists(p):
                return p

    raise FileNotFoundError(
        f"Claude CLI не найден. Убедись что он установлен и доступен.\n"
        f"Проверь CLAUDE_CLI_PATH в config.py (сейчас: '{CLAUDE_CLI_PATH}')"
    )
