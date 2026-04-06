"""
agent_memory.py — Universal Agent Memory System
================================================
Центральный модуль для ВСЕХ проектов.
Принимает project_path как параметр — работает с любым проектом.

Использование:
    from agent_memory import get_context_for_agent, get_research_brief, add_attempt
    ctx = get_context_for_agent("E:/project/translator")
    brief = get_research_brief("ошибка X", "E:/project/translator")
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone


# ─────────────────────────────────────────────
# РЕЕСТР ПРОЕКТОВ (единый источник правды)
# ─────────────────────────────────────────────

KNOWN_PROJECTS = [
    ("CLI Helper",            "E:/project/ceo 2.0/cli helper"),
    ("Market Analyzer",       "E:/market_analyzer"),
    ("Construction Research", "E:/project/ceo 2.0"),
    ("Beauty Hair Instagram", "E:/project/ceo 2.0/hair beauty"),
    ("Marketing School",      "E:/project/marketing_lessons 2.0"),
    ("OmniAvatar",            "E:/project/OmniAvatar"),
    ("MultiTalk",             "E:/project/MultiTalk"),
    ("Translator",            "E:/crewai/translator"),
    ("Reader",                "E:/project/reader"),
    ("Smart System",          "E:/project/smart_system"),
    ("Sales Agent",           "E:/project/ceo 2.0/cli helper/sales_agent"),
]

# Сколько часов считается "acknowledged" (не показывать снова)
ACKNOWLEDGMENT_WINDOW_HOURS = 4


# ─────────────────────────────────────────────
# ПУТИ
# ─────────────────────────────────────────────

def _paths(project_path: str) -> dict:
    """Возвращает все пути для проекта."""
    mem = os.path.join(project_path, ".agent-memory")
    return {
        "mem":     mem,
        "debug":   os.path.join(mem, "debug-map.md"),
        "goal":    os.path.join(mem, "goal-map.md"),
        "archive": os.path.join(mem, "archive"),
    }


def _ensure_dirs(project_path: str):
    p = _paths(project_path)
    os.makedirs(p["mem"], exist_ok=True)
    os.makedirs(p["archive"], exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────────────────────────
# ACKNOWLEDGMENT TRACKING
# ─────────────────────────────────────────────

def _state_path(project_path: str) -> str:
    return os.path.join(project_path, ".agent-memory", "_state.json")


def _mark_acknowledged(project_path: str) -> None:
    """Записывает факт что контекст был прочитан. Вызывается внутри get_context_for_agent()."""
    try:
        path = _state_path(project_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"last_context_read": _now()}, f)
    except Exception:
        pass


def is_acknowledged(project_path: str, window_hours: int = ACKNOWLEDGMENT_WINDOW_HOURS) -> bool:
    """
    Возвращает True если get_context_for_agent() был вызван в течение window_hours.
    Используется хуками enforce_memory.py и inject_agent_memory.py.
    """
    try:
        path = _state_path(project_path)
        if not os.path.exists(path):
            return False
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        ts = state.get("last_context_read", "")
        if not ts:
            return False
        last_read = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - last_read
        return delta.total_seconds() < window_hours * 3600
    except Exception:
        return False


# ─────────────────────────────────────────────
# ЧТЕНИЕ
# ─────────────────────────────────────────────

def read_debug_map(project_path: str) -> str:
    p = _paths(project_path)
    _ensure_dirs(project_path)
    if not os.path.exists(p["debug"]):
        return ""
    return open(p["debug"], encoding="utf-8").read()


def read_goal_map(project_path: str) -> str:
    p = _paths(project_path)
    _ensure_dirs(project_path)
    if not os.path.exists(p["goal"]):
        return ""
    return open(p["goal"], encoding="utf-8").read()


def get_context_for_agent(project_path: str) -> str:
    """
    Вызывать при АКТИВАЦИИ любого проекта.
    Возвращает полный контекст: цель + история + текущие проблемы.
    Автоматически помечает проект как acknowledged — снимает блокировку enforce_memory.py.
    """
    debug = read_debug_map(project_path)
    goal = read_goal_map(project_path)
    parts = []
    if debug:
        parts.append("=== AGENT MEMORY: DEBUG MAP ===\n" + debug)
    if goal:
        parts.append("=== AGENT MEMORY: GOAL MAP ===\n" + goal)
    if not parts:
        result = f"=== AGENT MEMORY: EMPTY for {project_path} (no active maps) ==="
    else:
        result = "\n\n".join(parts)
    # Фиксируем факт прочтения — снимает блокировку инструментов
    _mark_acknowledged(project_path)
    return result


def get_research_brief(current_problem: str, project_path: str) -> str:
    """
    ОБЯЗАТЕЛЬНО вызывать ПЕРЕД любым веб-ресёрчем.
    Даёт полную картину: цель, история, что не работало, как формулировать запрос.
    """
    goal_content = read_goal_map(project_path)
    debug_content = read_debug_map(project_path)

    lines = ["=" * 60,
             "RESEARCH BRIEF — ПРОЧИТАЙ ПЕРЕД ФОРМУЛИРОВКОЙ ЗАПРОСА",
             f"Проект: {project_path}",
             "=" * 60]

    # [1] Цель
    lines.append("\n[1] ЦЕЛЬ ПРОЕКТА (держи в голове при ресёрче):")
    if goal_content:
        m = re.search(r"\*\*Goal:\*\* (.+)", goal_content)
        if m:
            lines.append(f"    {m.group(1)}")
        arch = re.search(r"## System Architecture\n([\s\S]*?)(?=\n##|\Z)", goal_content)
        if arch:
            # Показываем только ASCII схему
            arch_text = arch.group(1).strip()
            for line in arch_text.split("\n")[:12]:
                lines.append(f"    {line}")
        cur = re.search(r"\*\*Current Actual Behavior:\*\* (.+)", goal_content)
        if cur:
            lines.append(f"    Сейчас: {cur.group(1)}")
        changes = re.findall(
            r"\| \d+ \| .+? \| (.+?) \| .+? \| .+? \| (✅|❌|➡)", goal_content
        )
        if changes:
            lines.append("    История изменений (последние 5):")
            for desc, direction in changes[-5:]:
                lines.append(f"      {direction} {desc.strip()}")
    else:
        lines.append("    (goal-map отсутствует — создай через init_goal_map)")

    # [2] Текущая проблема
    lines.append(f"\n[2] ТЕКУЩАЯ ПРОБЛЕМА:")
    lines.append(f"    {current_problem}")

    # [3] Что уже пробовали
    lines.append("\n[3] УЖЕ ПРОБОВАЛИ (НЕ ПОВТОРЯТЬ):")
    if debug_content:
        do_not = re.search(
            r"## DO NOT TRY\n([\s\S]*?)(?=\n##|\Z)", debug_content
        )
        block = do_not.group(1).strip() if do_not else ""
        if block and "Этот раздел читается" not in block:
            lines.append(block)
        else:
            attempts = re.findall(
                r"### Attempt #(\d+)[\s\S]*?- \*\*Action Taken:\*\* (.+?)[\s\S]*?- \*\*Result:\*\* (SUCCESS|FAILED|PARTIAL)",
                debug_content
            )
            if attempts:
                for num, action, result in attempts:
                    icon = "✅" if result == "SUCCESS" else ("❌" if result == "FAILED" else "⚠️")
                    lines.append(f"    {icon} Attempt #{num}: {action.strip()}")
            else:
                lines.append("    (нет истории попыток)")
    else:
        lines.append("    (debug-map отсутствует)")

    # [4] Research Filter
    lines.append("\n[4] КАК ФОРМУЛИРОВАТЬ ЗАПРОС:")
    search_only, do_not_research = "", ""
    for content in [debug_content, goal_content]:
        if not content:
            continue
        m1 = re.search(r"\*\*Search ONLY:\*\* (.+)", content)
        m2 = re.search(r"\*\*Do NOT research:\*\* (.+)", content)
        if m1 and "заполняется" not in m1.group(1) and not search_only:
            search_only = m1.group(1)
        if m2 and "заполняется" not in m2.group(1) and not do_not_research:
            do_not_research = m2.group(1)

    lines.append(f"    ИСКАТЬ: {search_only or 'конкретная ошибка + tech stack + ограничения из истории выше'}")
    if do_not_research:
        lines.append(f"    НЕ ИСКАТЬ: {do_not_research}")
    lines.append("    ПРАВИЛО: запрос = ошибка + версии + OS + что уже НЕ сработало")
    lines.append("    Плохо: 'triton import error'")
    lines.append("    Хорошо: 'torch 2.7.1 triton 3.6.0 triton_key ImportError Windows no downgrade'")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# ИНИЦИАЛИЗАЦИЯ
# ─────────────────────────────────────────────

def init_debug_map(
    project_path: str,
    error_type: str,
    error_message: str,
    error_location: str,
    environment: str,
    reproduction_steps: list[str]
) -> str:
    _ensure_dirs(project_path)
    p = _paths(project_path)
    if os.path.exists(p["debug"]):
        existing = read_debug_map(project_path)
        if "Status:** OPEN" in existing:
            raise RuntimeError(
                "debug-map.md уже OPEN. Закрой текущую проблему перед созданием новой."
            )
    problem_id = str(uuid.uuid4())
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reproduction_steps))
    content = f"""# DEBUG MAP

## Meta
- **Problem ID:** {problem_id}
- **Created:** {_now()}
- **Last Updated:** {_now()}
- **Status:** OPEN
- **Resolved In Attempt:** null

## Problem Statement
**Error Type:** {error_type}
**Error Message:** {error_message}
**Error Location:** {error_location}
**Environment:** {environment}
**Reproduction Steps:**
{steps_text}

## Attempts

<!-- Attempts добавляются автоматически -->

## DO NOT TRY
<!-- Читается ПЕРВЫМ перед любой новой попыткой -->

## Research Filter
**Search ONLY:** (заполняется после первой попытки)
**Do NOT research:** (заполняется после первой попытки)

## Rollback Checkpoints
| Attempt # | Git Hash / File Snapshot | Description |
|-----------|--------------------------|-------------|
| 0 (base)  | no-git                   | Начальное состояние |
"""
    with open(p["debug"], "w", encoding="utf-8") as f:
        f.write(content)
    return problem_id


def init_goal_map(
    project_path: str,
    goal: str,
    success_criteria: list[str],
    failure_criteria: list[str],
    current_behavior: str,
    gap: str
) -> str:
    _ensure_dirs(project_path)
    p = _paths(project_path)
    if os.path.exists(p["goal"]):
        existing = read_goal_map(project_path)
        if "Status:** IN_PROGRESS" in existing:
            raise RuntimeError(
                "goal-map.md уже IN_PROGRESS. Закрой или завершь текущую цель."
            )
    goal_id = str(uuid.uuid4())
    criteria_text = "\n".join(f"- [ ] {c}" for c in success_criteria)
    failure_text = "\n".join(f"- {c}" for c in failure_criteria)
    content = f"""# GOAL MAP

## Meta
- **Goal ID:** {goal_id}
- **Created:** {_now()}
- **Last Updated:** {_now()}
- **Status:** IN_PROGRESS

## Target Definition
**Goal:** {goal}
**Success Criteria:**
{criteria_text}
**Failure Criteria (когда останавливаемся):**
{failure_text}
**Current Actual Behavior:** {current_behavior}
**Gap:** {gap}

## System Architecture
(добавить схему компонентов проекта)

## Change Log

| # | Timestamp | Изменение | Ожидаемый эффект | Реальный результат | Направление |
|---|-----------|-----------|------------------|--------------------|-------------|

## Directional Analysis
**Pattern Toward Goal:** (заполняется после 3+ изменений)
**Pattern Away from Goal:** (заполняется после 3+ изменений)
**Current Best Direction:** (заполняется после 3+ изменений)

## Research Filter
**Search ONLY:** (заполняется на основе паттернов)
**Do NOT research:** (заполняется на основе паттернов)

## Hypothesis Queue
1. [HIGH] (первая гипотеза)
"""
    with open(p["goal"], "w", encoding="utf-8") as f:
        f.write(content)
    return goal_id


# ─────────────────────────────────────────────
# ЗАПИСЬ: DEBUG MAP
# ─────────────────────────────────────────────

def add_attempt(
    project_path: str,
    attempt_number: int,
    hypothesis: str,
    hypothesis_confidence: str,
    action_taken: str,
    files_modified: list[str],
    result: str,
    failure_reason: str = "",
    new_information: str = "",
    ruled_out: str = "",
    git_hash: str = ""
) -> None:
    assert hypothesis_confidence in ("HIGH", "MEDIUM", "LOW")
    assert result in ("SUCCESS", "FAILED", "PARTIAL")

    p = _paths(project_path)
    content = read_debug_map(project_path)
    if not content:
        raise RuntimeError("debug-map.md не найден. Сначала вызови init_debug_map().")

    files_text = ", ".join(files_modified) if files_modified else "none"
    attempt_block = f"""
### Attempt #{attempt_number}
- **Timestamp:** {_now()}
- **Hypothesis:** {hypothesis}
- **Hypothesis Confidence:** {hypothesis_confidence}
- **Action Taken:** {action_taken}
- **Files Modified:** {files_text}
- **Result:** {result}
- **Failure Reason:** {failure_reason or 'N/A'}
- **New Information:** {new_information or 'N/A'}
- **Ruled Out:** {ruled_out or 'N/A'}
"""
    content = content.replace("## DO NOT TRY", attempt_block + "\n## DO NOT TRY")

    if result == "FAILED" and ruled_out:
        content = content.replace(
            "## Research Filter",
            f"- [ ] Attempt #{attempt_number}: {action_taken} — Причина: {ruled_out}\n## Research Filter"
        )

    if git_hash:
        # Append row to end of Rollback Checkpoints section (correct order, no table corruption)
        checkpoint_row = f"| {attempt_number} | {git_hash} | После Attempt #{attempt_number} |"
        content = re.sub(
            r"(## Rollback Checkpoints[\s\S]*?)(\n##|\Z)",
            lambda m: m.group(1).rstrip() + "\n" + checkpoint_row + m.group(2),
            content,
            count=1,
        )

    if result == "SUCCESS":
        content = content.replace("- **Status:** OPEN", "- **Status:** RESOLVED")
        content = content.replace(
            "- **Resolved In Attempt:** null",
            f"- **Resolved In Attempt:** {attempt_number}"
        )

    content = re.sub(r"- \*\*Last Updated:\*\* .*", f"- **Last Updated:** {_now()}", content)
    with open(p["debug"], "w", encoding="utf-8") as f:
        f.write(content)

    if result == "SUCCESS":
        print(f"\n✅ RESOLVED. ОБЯЗАТЕЛЬНЫЙ СЛЕДУЮЩИЙ ШАГ: вызови archive_and_reset('{project_path}', 'short-slug')")


# ─────────────────────────────────────────────
# ЗАПИСЬ: GOAL MAP
# ─────────────────────────────────────────────

def add_change(
    project_path: str,
    change_number: int,
    change_description: str,
    expected_effect: str,
    actual_result: str,
    direction: str
) -> None:
    assert direction in ("toward", "away", "neutral")
    icons = {"toward": "✅ toward", "away": "❌ away", "neutral": "➡ neutral"}
    p = _paths(project_path)
    content = read_goal_map(project_path)
    if not content:
        raise RuntimeError("goal-map.md не найден.")

    new_row = (
        f"| {change_number} | {_now()} | {change_description} | "
        f"{expected_effect} | {actual_result} | {icons[direction]} |"
    )
    content = content.replace(
        "| # | Timestamp | Изменение | Ожидаемый эффект | Реальный результат | Направление |\n"
        "|---|-----------|-----------|------------------|--------------------|-------------|",
        "| # | Timestamp | Изменение | Ожидаемый эффект | Реальный результат | Направление |\n"
        "|---|-----------|-----------|------------------|--------------------|-------------|\n"
        + new_row
    )
    content = re.sub(r"- \*\*Last Updated:\*\* .*", f"- **Last Updated:** {_now()}", content)
    with open(p["goal"], "w", encoding="utf-8") as f:
        f.write(content)


def update_directional_analysis(
    project_path: str,
    pattern_toward: str,
    pattern_away: str,
    current_best_direction: str,
    search_only: str,
    do_not_research: str,
    hypothesis_queue: list[tuple[str, str]],  # [(confidence, hypothesis), ...]
) -> None:
    """
    Обновляет Directional Analysis, Research Filter и Hypothesis Queue в goal-map.md.
    Вызывать после каждых 3 изменений (spec: Rule / Таблица функций).
    hypothesis_queue: список кортежей ("HIGH"/"MEDIUM"/"LOW", "текст гипотезы")
    """
    p = _paths(project_path)
    content = read_goal_map(project_path)
    if not content:
        raise RuntimeError("goal-map.md не найден. Сначала вызови init_goal_map().")

    queue_text = "\n".join(
        f"{i + 1}. [{conf}] {hyp}"
        for i, (conf, hyp) in enumerate(hypothesis_queue)
    )
    content = re.sub(r"\*\*Pattern Toward Goal:\*\* .*",
                     f"**Pattern Toward Goal:** {pattern_toward}", content)
    content = re.sub(r"\*\*Pattern Away from Goal:\*\* .*",
                     f"**Pattern Away from Goal:** {pattern_away}", content)
    content = re.sub(r"\*\*Current Best Direction:\*\* .*",
                     f"**Current Best Direction:** {current_best_direction}", content)
    content = re.sub(r"\*\*Search ONLY:\*\* .*",
                     f"**Search ONLY:** {search_only}", content)
    content = re.sub(r"\*\*Do NOT research:\*\* .*",
                     f"**Do NOT research:** {do_not_research}", content)
    content = re.sub(
        r"## Hypothesis Queue\n[\s\S]*$",
        f"## Hypothesis Queue\n{queue_text}\n",
        content,
    )
    content = re.sub(r"- \*\*Last Updated:\*\* .*", f"- **Last Updated:** {_now()}", content)
    with open(p["goal"], "w", encoding="utf-8") as f:
        f.write(content)


def mark_goal_achieved(project_path: str) -> None:
    p = _paths(project_path)
    content = read_goal_map(project_path)
    content = re.sub(r"- \*\*Status:\*\* .*", "- **Status:** ACHIEVED", content)
    content = re.sub(r"- \*\*Last Updated:\*\* .*", f"- **Last Updated:** {_now()}", content)
    with open(p["goal"], "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ GOAL ACHIEVED. ОБЯЗАТЕЛЬНЫЙ СЛЕДУЮЩИЙ ШАГ: вызови archive_and_reset('{project_path}', 'short-slug')")


# ─────────────────────────────────────────────
# АРХИВАЦИЯ
# ─────────────────────────────────────────────

def archive_and_reset(project_path: str, slug: str) -> str:
    _ensure_dirs(project_path)
    p = _paths(project_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    archive_path = os.path.join(p["archive"], f"{timestamp}_{slug}.md")

    combined = ""
    if os.path.exists(p["debug"]):
        combined += read_debug_map(project_path) + "\n\n---\n\n"
        os.remove(p["debug"])
    if os.path.exists(p["goal"]):
        combined += read_goal_map(project_path)
        os.remove(p["goal"])

    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(combined)
    return archive_path


# ─────────────────────────────────────────────
# БЫСТРЫЙ СТАТУС
# ─────────────────────────────────────────────

def project_status(project_path: str) -> str:
    """Краткий статус проекта для Project Registry."""
    debug = read_debug_map(project_path)
    goal = read_goal_map(project_path)

    status = []
    if goal:
        m = re.search(r"\*\*Goal:\*\* (.+)", goal)
        s = re.search(r"\*\*Status:\*\* (.+)", goal)
        if m:
            status.append(f"GOAL: {m.group(1)[:60]}")
        if s:
            status.append(f"Status: {s.group(1)}")
    if debug:
        s = re.search(r"\*\*Status:\*\* (.+)", debug)
        err = re.search(r"\*\*Error Message:\*\* (.+)", debug)
        if s:
            status.append(f"Debug: {s.group(1)}")
        if err:
            status.append(f"Error: {err.group(1)[:60]}")
    if not status:
        return "No active maps"
    return " | ".join(status)
