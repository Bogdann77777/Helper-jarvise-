"""
urgent_checker.py — Hourly alarm for urgent unfinished tasks.
Run by Windows Task Scheduler every hour (08:00–22:00 weekdays).

Behavior:
  - Finds all urgent (priority=high) open tasks
  - If not confirmed today and sent < MAX_SENDS times → sends Telegram
  - Stops sending if user replied "сделано X" (bot sets confirmed=True)
  - Stops after MAX_SENDS per day (no spam at night)
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MAX_SENDS = 8   # max reminders per task per day

from task_manager.db import (
    init_db, get_urgent_pending, is_vacation_today,
    get_urgent_alert_count, increment_urgent_alert,
    is_urgent_confirmed_today,
)
from task_manager.formatter import urgent_alert


def main():
    init_db()

    today = date.today()
    if today.weekday() >= 5 or is_vacation_today():
        return

    tasks = get_urgent_pending()
    if not tasks:
        return

    from task_manager.tg import tg_task_msg
    sent_any = False

    for task in tasks:
        tid = task["id"]

        # Skip if confirmed today
        if is_urgent_confirmed_today(tid):
            continue

        count = get_urgent_alert_count(tid)
        if count >= MAX_SENDS:
            continue

        msg = urgent_alert(task, sent_count=count + 1)
        ok  = tg_task_msg(msg)
        if ok:
            increment_urgent_alert(tid)
            sent_any = True
            print(f"[urgent] Отправлено #{tid} ({task['text'][:40]}), attempt={count+1}")

    if not sent_any:
        print("[urgent] Нет срочных задач для напоминания")


if __name__ == "__main__":
    main()
