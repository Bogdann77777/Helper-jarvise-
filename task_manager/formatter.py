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


# Category icons for the 4 main categories
CATEGORY_ICONS = {
    "объекты": "🏗",
    "офис": "🏢",
    "программирование": "💻",
    "личное": "👤",
}

def _cat_icon(project: str) -> str:
    key = project.lower().split("/")[0].strip()
    for k, icon in CATEGORY_ICONS.items():
        if k in key:
            return icon
    return "📁"


def morning_report(tasks: list[dict]) -> str:
    """Full morning report for daily 9am Telegram."""
    today = date.today().isoformat()

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
        cat = _cat_icon(t["project"])
        return f"{icon} {cat} <b>[{t['project']}]</b>{prio} {t['text']}{due_label} <code>#{t['id']}</code>"

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
        lines.append("🔔 <b>ВСЕ ЗАДАЧИ (без даты)</b>")
        lines.append("─────────────────────")
        # group by category
        by_cat: dict[str, list] = {}
        for t in undated:
            by_cat.setdefault(t["project"], []).append(t)
        for proj in sorted(by_cat):
            lines.append(f"  {_cat_icon(proj)} <b>{proj}</b>")
            for t in by_cat[proj]:
                prio = " 🔴" if t.get("priority") == "high" else ""
                lines.append(f"    •{prio} {t['text']} <code>#{t['id']}</code>")
        lines.append("")

    if not any([overdue, today_tasks, soon_tasks, undated]):
        lines.append("✅ <i>Нет открытых задач. Отличный день!</i>")
    else:
        total = len(overdue) + len(today_tasks) + len(soon_tasks) + len(undated)
        categories = len(set(t["project"] for t in tasks))
        lines.append(f"<i>Всего: {total} задач по {categories} категориям</i>")
        lines.append("\n💡 Напиши <code>план сегодня</code> чтобы выбрать задачи на день")

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


def day_plan_report(tasks: list[dict]) -> str:
    """Today's selected plan grouped by category."""
    if not tasks:
        return (
            "📋 <b>План на сегодня пуст.</b>\n\n"
            "Напиши <code>список</code> чтобы увидеть все задачи,\n"
            "потом скажи мне (Claude) какие берём на сегодня."
        )

    lines = [f"📋 <b>ПЛАН НА {_today_ru().upper()}</b>\n"]

    by_cat: dict[str, list] = {}
    for t in tasks:
        by_cat.setdefault(t["project"], []).append(t)

    num = 1
    for proj in sorted(by_cat):
        lines.append(f"{_cat_icon(proj)} <b>{proj}</b>")
        for t in by_cat[proj]:
            prio = " 🔴" if t.get("priority") == "high" else ""
            lines.append(f"  {num}.{prio} {t['text']} <code>#{t['id']}</code>")
            num += 1
        lines.append("")

    lines.append(f"<i>Итого на сегодня: {len(tasks)} задач</i>")
    return "\n".join(lines)


def day_overview(tasks: list[dict]) -> str:
    """Full day overview: urgent + regular plan. Sent after planning session."""
    if not tasks:
        return "📋 <b>План на сегодня пуст.</b>"

    urgent = [t for t in tasks if t.get("priority") == "high"]
    regular = [t for t in tasks if t.get("priority") != "high"]

    parts = []

    # Header
    parts.append(
        f"🗓 <b>РАБОЧИЙ ДЕНЬ</b>\n"
        f"<i>{_today_ru()}</i>"
    )

    # Urgent block
    if urgent:
        block = ["", "🔴 <b>СРОЧНЫЕ ЗАДАЧИ</b>", "<i>напоминание каждый час</i>", ""]
        by_cat: dict[str, list] = {}
        for t in urgent:
            by_cat.setdefault(t["project"], []).append(t)
        for proj in sorted(by_cat):
            block.append(f"{_cat_icon(proj)}  <b>{proj.upper()}</b>")
            for i, t in enumerate(by_cat[proj], 1):
                block.append(f"    {i}. 🔴 {t['text']}")
                block.append(f"        <code>#{t['id']}</code>")
                block.append("")
        parts.append("\n".join(block))

    # Plan block
    if regular:
        block = ["", "📋 <b>ПЛАН ДНЯ</b>", ""]
        by_cat2: dict[str, list] = {}
        for t in regular:
            by_cat2.setdefault(t["project"], []).append(t)
        for proj in sorted(by_cat2):
            block.append(f"{_cat_icon(proj)}  <b>{proj.upper()}</b>")
            for i, t in enumerate(by_cat2[proj], 1):
                block.append(f"    {i}. {t['text']}")
                block.append(f"        <code>#{t['id']}</code>")
                block.append("")
        parts.append("\n".join(block))

    # Footer
    parts.append(
        f"─────────────────────\n"
        f"<i>Всего: {len(tasks)}  •  Срочных: {len(urgent)}</i>\n\n"
        f"✅ Закрыть задачу: <code>сделано 42</code>\n"
        f"🌆 В 18:00 — подведём итоги дня"
    )

    return "\n\n".join(parts)


def evening_report(tasks: list[dict]) -> str:
    """Evening check-in at 18:00: show today's remaining tasks."""
    d = date.today()
    date_str = f"{d.day} {MONTHS_RU[d.month]}"

    if not tasks:
        return (
            f"🌆 <b>ИТОГИ ДНЯ — {date_str}</b>\n\n"
            "✅ <i>Все задачи выполнены! Отличный день!</i>"
        )

    urgent = [t for t in tasks if t.get("priority") == "high"]
    regular = [t for t in tasks if t.get("priority") != "high"]

    parts = []

    parts.append(
        f"🌆 <b>ИТОГИ ДНЯ — {date_str}</b>\n"
        f"<i>Что из этого сделал сегодня?</i>"
    )

    if urgent:
        block = ["🔴 <b>СРОЧНЫЕ (не закрыты)</b>", ""]
        for i, t in enumerate(urgent, 1):
            block.append(f"    {i}. ❗ {t['text']}")
            block.append(f"        <i>{t['project']}</i>  <code>#{t['id']}</code>")
            block.append("")
        parts.append("\n".join(block))

    if regular:
        block = ["📋 <b>ПЛАНОВЫЕ</b>", ""]
        for i, t in enumerate(regular, 1):
            block.append(f"    {i}. {t['text']}")
            block.append(f"        <i>{t['project']}</i>  <code>#{t['id']}</code>")
            block.append("")
        parts.append("\n".join(block))

    parts.append(
        f"─────────────────────\n"
        f"<i>Незакрытых: {len(tasks)}</i>\n\n"
        f"💬 Открой Claude — надиктуй что вычеркнуть"
    )

    return "\n\n".join(parts)


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
