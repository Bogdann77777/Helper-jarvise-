"""
enforce_memory.py
PreToolUse hook (matcher: Edit|Write|NotebookEdit):
БЛОКИРУЕТ изменение файлов если есть активный проект с OPEN/IN_PROGRESS памятью,
которая не была прочитана за последние ACKNOWLEDGMENT_WINDOW_HOURS часов.

Bash/Read/Glob/Grep — НЕ блокируются.
Это позволяет Claude вызвать get_context_for_agent() через Bash, после чего
блокировка снимается автоматически.

Exit code 2 = Claude Code блокирует вызов инструмента.
"""

import os
import sys

sys.path.insert(0, "E:/project/ceo 2.0/cli helper")
from agent_memory import KNOWN_PROJECTS, read_goal_map, read_debug_map, is_acknowledged

unacknowledged = []

for name, path in KNOWN_PROJECTS:
    mem_dir = os.path.join(path, ".agent-memory")
    if not os.path.exists(mem_dir):
        continue

    goal  = read_goal_map(path)
    debug = read_debug_map(path)

    has_active_goal = goal  and "Status:** IN_PROGRESS" in goal
    has_open_debug  = debug and "Status:** OPEN"        in debug

    if (has_active_goal or has_open_debug) and not is_acknowledged(path):
        unacknowledged.append((name, path))

if not unacknowledged:
    sys.exit(0)

# БЛОКИРОВКА
lines = [
    "=" * 70,
    "🚫  EDIT/WRITE BLOCKED — PROJECT MEMORY NOT LOADED",
    "=" * 70,
    "",
    "Активные проекты с непрочитанной памятью:",
    "",
]

for name, path in unacknowledged:
    lines.append(f"  ❗ {name}  [{path}]")

lines += [
    "",
    "Выполни через Bash tool (Bash не заблокирован):",
    "",
    "  import sys; sys.path.insert(0, 'E:/project/ceo 2.0/cli helper')",
    "  from agent_memory import get_context_for_agent",
]

for _, path in unacknowledged:
    lines.append(f"  print(get_context_for_agent(r'{path}'))")

lines += [
    "",
    "После этого блокировка снимается автоматически на 4 часа.",
    "=" * 70,
]

print("\n".join(lines))
sys.exit(2)
