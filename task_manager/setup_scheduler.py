"""
setup_scheduler.py — Register Windows Task Scheduler jobs.
Run ONCE as Administrator to set up daily reports + hourly urgent checker.

Creates:
  CEO_MorningReport   — daily 09:00 (Mon-Fri)
  CEO_UrgentChecker   — every hour 08:00-22:00 (Mon-Fri)
  CEO_TaskBot_Start   — on login (auto-start bot)
"""
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).parent.parent
PYTHON = sys.executable


def bat(name: str, script: str) -> str:
    """Write a .bat launcher and return its path."""
    bat_path = str(ROOT / f"{name}.bat")
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(f'@echo off\ncd /d "{ROOT}"\n"{PYTHON}" "{script}"\n')
    return bat_path


def schtask(name: str, bat_path: str, schedule: str, extra: list = None):
    cmd = [
        "schtasks", "/Create", "/F",
        "/TN", name,
        "/TR", bat_path,
    ] + schedule.split() + (extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode == 0:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}: {result.stderr.strip()}")


def main():
    print("Регистрация задач в Windows Task Scheduler...\n")

    # 1. Morning report — every weekday at 09:00
    morning_bat = bat("run_morning_report",
                      str(ROOT / "task_manager" / "morning_report.py"))
    schtask(
        "CEO_MorningReport", morning_bat,
        "/SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:00",
    )

    # 2. Urgent checker — every hour, weekdays 08:00-22:00
    urgent_bat = bat("run_urgent_checker",
                     str(ROOT / "task_manager" / "urgent_checker.py"))
    schtask(
        "CEO_UrgentChecker", urgent_bat,
        "/SC HOURLY /MO 1 /ST 08:00 /ET 22:00 /K",
    )

    # 3. Bot auto-start on user login
    bot_bat = bat("run_task_bot",
                  str(ROOT / "task_manager" / "bot_launcher.py"))
    schtask(
        "CEO_TaskBot_Start", bot_bat,
        "/SC ONLOGON",
    )

    print("\nГотово! Задачи зарегистрированы.")
    print("Проверить: откройте 'Планировщик задач' → папка CEO_*")
    print("\nЗапустить бот сейчас:")
    print(f'  python "{ROOT / "task_manager" / "bot_launcher.py"}"')


if __name__ == "__main__":
    main()
