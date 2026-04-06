"""
remind_add_change.py
PostToolUse hook: после Edit/Write напоминает залогировать изменение.

Оптимизация: читает путь редактированного файла из stdin (PostToolUse hook API),
и проверяет ТОЛЬКО проекты, к которым относится этот файл.
Fallback: если stdin недоступен — проверяет все проекты.
"""

import json
import os
import sys

sys.path.insert(0, "E:/project/ceo 2.0/cli helper")
from agent_memory import KNOWN_PROJECTS, read_goal_map, read_debug_map

# ── Читаем путь файла из stdin (PostToolUse hook API) ──────────────────────
edited_file = ""
try:
    raw = sys.stdin.read()
    if raw.strip():
        data = json.loads(raw)
        edited_file = data.get("tool_input", {}).get("file_path", "")
except Exception:
    pass

# ── Определяем какие проекты проверять ─────────────────────────────────────
if edited_file:
    # Оптимизация: только проекты, к которым относится редактированный файл
    projects_to_check = [
        (name, path) for name, path in KNOWN_PROJECTS
        if edited_file.replace("\\", "/").startswith(path.replace("\\", "/"))
    ]
    if not projects_to_check:
        sys.exit(0)  # Файл не в известном проекте — выходим
else:
    # Fallback: проверяем все проекты
    projects_to_check = KNOWN_PROJECTS

# ── Проверяем наличие активной памяти ──────────────────────────────────────
active = []
for name, path in projects_to_check:
    goal  = read_goal_map(path)
    debug = read_debug_map(path)
    if (goal and "IN_PROGRESS" in goal) or (debug and "Status:** OPEN" in debug):
        active.append(name)

if active:
    projects_str = ", ".join(active)
    print(
        f"📝 ОБЯЗАТЕЛЬНО: вызови add_change() или add_attempt() для [{projects_str}] — "
        f"каждое изменение должно быть залогировано в goal-map/debug-map. "
        f"Не переходи к следующему шагу без этого."
    )
