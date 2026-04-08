"""
morning_report.py — Daily 9am Telegram report.
Run by Windows Task Scheduler every weekday at 09:00.

Skips: weekends, vacation days.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager.db import init_db, get_todays_report_tasks, is_vacation_today
from task_manager.formatter import morning_report


def main():
    init_db()

    today = date.today()

    # Skip weekends (5=Saturday, 6=Sunday)
    if today.weekday() >= 5:
        print(f"[morning_report] Выходной ({today}), пропускаем")
        return

    # Skip vacation days
    if is_vacation_today():
        print(f"[morning_report] Отпуск ({today}), пропускаем")
        return

    tasks = get_todays_report_tasks()
    msg   = morning_report(tasks)

    from task_manager.tg import tg_task_msg
    ok = tg_task_msg(msg)
    print(f"[morning_report] Отправлен отчёт ({len(tasks)} задач), ok={ok}")


if __name__ == "__main__":
    main()
