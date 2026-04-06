"""
inject_agent_memory.py
UserPromptSubmit hook: показывает КРАТКОЕ РЕЗЮМЕ активной памяти проекта,
НО только если get_context_for_agent() не был вызван в течение последних
ACKNOWLEDGMENT_WINDOW_HOURS часов.

Логика:
- Есть активный проект + НЕ acknowledged → показываем резюме (remind Claude to load full context)
- Есть активный проект + acknowledged      → тихо выходим (не спамим каждое сообщение)
- Нет активных проектов                    → тихо выходим

Полный контекст — через get_context_for_agent(path). Это резюме только для навигации.
"""

import os
import re
import sys

sys.path.insert(0, "E:/project/ceo 2.0/cli helper")
from agent_memory import KNOWN_PROJECTS, read_goal_map, read_debug_map, is_acknowledged

active = []

for name, path in KNOWN_PROJECTS:
    mem_dir = os.path.join(path, ".agent-memory")
    if not os.path.exists(mem_dir):
        continue

    goal  = read_goal_map(path)
    debug = read_debug_map(path)

    has_active_goal = goal  and "Status:** IN_PROGRESS" in goal
    has_open_debug  = debug and "Status:** OPEN"        in debug

    if has_active_goal or has_open_debug:
        # Показываем только если НЕ acknowledged — убираем шум в активной сессии
        if not is_acknowledged(path):
            active.append((name, path, goal, debug))

if not active:
    sys.exit(0)  # Всё acknowledged или нет активных — тихо выходим

print("=" * 70)
print("🧠 AGENT MEMORY — ACTIVE PROJECTS (MANDATORY READ BEFORE RESPONDING)")
print("=" * 70)

for name, path, goal, debug in active:
    print(f"\n📁 {name}  [{path}]")
    print("-" * 60)

    if goal and "Status:** IN_PROGRESS" in goal:
        goal_line = re.search(r"\*\*Goal:\*\* (.+)", goal)
        if goal_line:
            print(f"GOAL: {goal_line.group(1)}")

        current = re.search(r"\*\*Current Actual Behavior:\*\* (.+)", goal)
        if current:
            print(f"NOW:  {current.group(1)[:120]}")

        gap = re.search(r"\*\*Gap:\*\* (.+)", goal)
        if gap:
            print(f"GAP:  {gap.group(1)[:120]}")

        changes = re.findall(
            r"\| (\d+) \| .+? \| (.+?) \| .+? \| (.+?) \| (✅|❌|➡)",
            goal
        )
        if changes:
            print("CHANGES (last 3):")
            for num, desc, result, direction in changes[-3:]:
                print(f"  {direction} #{num} {desc.strip()[:80]}")

        hyp = re.search(r"## Hypothesis Queue\n([\s\S]*?)(?=\n##|\Z)", goal)
        if hyp:
            lines = [l for l in hyp.group(1).strip().split("\n")
                     if l.strip() and not l.strip().startswith("1. [HIGH] (первая")]
            if lines:
                print("NEXT:")
                for l in lines[:2]:
                    print(f"  {l}")

    if debug and "Status:** OPEN" in debug:
        err = re.search(r"\*\*Error Message:\*\* (.+)", debug)
        do_not = re.search(r"## DO NOT TRY\n([\s\S]*?)(?=\n##|\Z)", debug)
        if err:
            print(f"🔴 OPEN BUG: {err.group(1)[:100]}")
        if do_not:
            block = do_not.group(1).strip()
            if block and "Читается ПЕРВЫМ" not in block:
                print("DO NOT TRY:")
                for line in block.split("\n")[:3]:
                    print(f"  {line}")

print("\n" + "=" * 70)
print("⚠️  Вызови get_context_for_agent(path) для полного контекста.")
print("    Это также снимет блокировку enforce_memory.py.")
print("=" * 70 + "\n")
