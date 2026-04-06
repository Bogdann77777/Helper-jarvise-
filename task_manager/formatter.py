"""
formatter.py — Telegram HTML report builder.
"""
from datetime import date
from typing import Optional
from .date_parser import format_date_ru


MONTHS_RU = [
    "", "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]

DAYS_RU = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]


def _today_ru() -> str:
    d = date.today()
    return f"{DAYS_RU[d.weekday()]}, {d.day} {MONTHS_RU[d.month]} {d.year}"


def morning_report(tasks: list[dict]) -> str:
    """Full morning report for daily 9am Telegram."""
    today = date.today().isoformat()
    in_3 = (date.today().replace(day=date.today().day)).isoformat()

    overdue, today_tasks, soon_tasks, undated = [], [], [], []

    for t in tasks:
        due = t.get("due_date")
        if not due:
            undated.append(t)
        elif due < today:
            overdue.append(t)
        elif due == today:
            today_tasks.append(t)
        else:
            soon_tasks.append(t)

    lines = [f"☀️ <b>Доброе утро!</b>  {_today_ru()}\n"]

    def _task_line(t: dict, icon: str) -> str:
        prio = " 🔴 <b>СРОЧНО</b>" if t.get("priority") == "high" else ""
        due_label = f" <i>({format_date_ru(t['due_date'])})</i>" if t.get("due_date") else ""
        return f"{icon} <b>[{t['project']}]</b>{prio} {t['text']}{due_label} <code>#{t['id']}</code>"

    if overdue:
        lines.append("🚨 <b>ПРОСРОЧЕНО</b>")
        lines.append("─────────────────────")
        for t in overdue:
            lines.append(_task_line(t, "❗"))
        lines.append("")

    if today_tasks:
        lines.append("📅 <b>СЕГОДНЯ</b>")
        lines.append("─────────────────────")
        for t in today_tasks:
            lines.append(_task_line(t, "🔴" if t.get("priority") == "high" else "📌"))
        lines.append("")

    if soon_tasks:
        lines.append("📆 <b>БЛИЖАЙШИЕ ДНИ</b>")
        lines.append("─────────────────────")
        for t in soon_tasks:
            lines.append(_task_line(t, "📋"))
        lines.append("")

    if undated:
        lines.append("🔔 <b>НАПОМИНАНИЯ (без даты)</b>")
        lines.append("─────────────────────")
        for t in undated:
            lines.append(_task_line(t, "•"))
        lines.append("")

    if not any([overdue, today_tasks, soon_tasks, undated]):
        lines.append("✅ <i>Нет открытых задач. Отличный день!</i>")
    else:
        total = len(overdue) + len(today_tasks) + len(soon_tasks) + len(undated)
        projects = len(set(t["project"] for t in tasks))
        lines.append(f"<i>Всего: {total} задач по {projects} проектам</i>")

    return "\n".join(lines)


def project_report(project: str, tasks: list[dict]) -> str:
    """Report for a single project."""
    if not tasks:
        return f"📂 <b>[{project}]</b>\n\n<i>Нет открытых задач.</i>"

    lines = [f"📂 <b>Проект [{project}]</b> — {len(tasks)} задач\n"]
    today = date.today().isoformat()

    for t in tasks:
        due = t.get("due_date")
        if due and due < today:
            icon = "❗"
        elif t.get("priority") == "high":
            icon = "🔴"
        elif due:
            icon = "📋"
        else:
            icon = "•"

        due_label = f" <i>({format_date_ru(due)})</i>" if due else ""
        lines.append(f"{icon} {t['text']}{due_label} <code>#{t['id']}</code>")

    return "\n".join(lines)


def full_list_report(tasks: list[dict]) -> str:
    """All projects in one message."""
    if not tasks:
        return "✅ <b>Нет открытых задач.</b>"

    by_project: dict[str, list] = {}
    for t in tasks:
        by_project.setdefault(t["project"], []).append(t)

    lines = [f"📋 <b>Все задачи</b> ({len(tasks)} шт.)\n"]
    today = date.today().isoformat()

    for proj in sorted(by_project):
        lines.append(f"\n<b>▸ [{proj}]</b>")
        for t in by_project[proj]:
            due = t.get("due_date")
            overdue = due and due < today
            icon = "❗" if overdue else ("🔴" if t.get("priority") == "high" else "•")
            due_label = f" ({format_date_ru(due)})" if due else ""
            lines.append(f"  {icon} {t['text']}{due_label} <code>#{t['id']}</code>")

    return "\n".join(lines)


def urgent_alert(task: dict, sent_count: int) -> str:
    """Message for hourly urgent task reminder."""
    due_label = f"\nСрок: {format_date_ru(task['due_date'])}" if task.get("due_date") else ""
    return (
        f"🚨 <b>СРОЧНАЯ ЗАДАЧА</b> (напоминание #{sent_count})\n\n"
        f"<b>[{task['project']}]</b> {task['text']}{due_label}\n\n"
        f"Чтобы закрыть: отправь <code>сделано {task['id']}</code>"
    )


def task_added_confirm(tasks_added: list[dict]) -> str:
    """Confirmation after adding tasks."""
    if len(tasks_added) == 1:
        t = tasks_added[0]
        due = f" | срок: {format_date_ru(t['due_date'])}" if t.get("due_date") else ""
        prio = " | 🔴 СРОЧНО" if t.get("priority") == "high" else ""
        return f"✅ Записал для <b>[{t['project']}]</b>:\n{t['text']}{due}{prio} <code>#{t['id']}</code>"
    lines = [f"✅ Записал {len(tasks_added)} задач:"]
    for t in tasks_added:
        due = f" ({format_date_ru(t['due_date'])})" if t.get("due_date") else ""
        prio = " 🔴" if t.get("priority") == "high" else ""
        lines.append(f"• <b>[{t['project']}]</b>{prio} {t['text']}{due} <code>#{t['id']}</code>")
    return "\n".join(lines)
