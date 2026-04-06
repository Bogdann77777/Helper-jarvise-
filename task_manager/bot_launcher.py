"""
bot_launcher.py — Launch Telegram bot as detached background process.
Run this once to start the bot. It survives closing the terminal.

Usage:
  python task_manager/bot_launcher.py
"""
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
PYTHON  = sys.executable
BOT     = str(ROOT / "task_manager" / "bot.py")
LOG     = str(ROOT / "logs" / "task_bot_stdout.txt")


def main():
    Path(ROOT / "logs").mkdir(exist_ok=True)
    log_f = open(LOG, "a", encoding="utf-8")

    proc = subprocess.Popen(
        [PYTHON, BOT],
        cwd=str(ROOT),
        stdout=log_f,
        stderr=log_f,
        stdin=subprocess.DEVNULL,
        creationflags=0x00000008 | 0x00000200,  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    )
    print(f"✅ Task Bot запущен | PID {proc.pid}")
    print(f"📄 Лог: {LOG}")


if __name__ == "__main__":
    main()
