"""
vast_monitor.py — мониторинг vast.ai инстанса через API.
Каждые 10 минут шлёт простой статус в Telegram.

Запуск: python vast_monitor.py [--instance-id 33917364]
Остановка: Ctrl+C или создать файл STOP_VAST_MONITOR
"""

import sys
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from tg_send import tg_msg

try:
    from config import VASTAI_API_KEY, VASTAI_INSTANCE_ID
except ImportError:
    import os
    VASTAI_API_KEY     = os.environ.get("VASTAI_API_KEY", "")
    VASTAI_INSTANCE_ID = int(os.environ.get("VASTAI_INSTANCE_ID", "0"))

CHECK_INTERVAL = 10 * 60  # 10 минут
STOP_FILE      = _HERE / "STOP_VAST_MONITOR"
API_BASE       = "https://console.vast.ai/api/v0"


def get_instance_status(instance_id: int) -> dict:
    try:
        r = requests.get(
            f"{API_BASE}/instances/",
            params={"owner": "me"},
            headers={"Authorization": f"Bearer {VASTAI_API_KEY}"},
            timeout=10,
        )
        if not r.ok:
            return {}
        for inst in r.json().get("instances", []):
            if inst.get("id") == instance_id:
                return inst
        return {}
    except Exception as e:
        return {"error": str(e)}


def format_status(inst: dict, iteration: int) -> str:
    if not inst:
        return f"⚠️ Инстанс не найден (#{iteration})"

    err = inst.get("error")
    if err:
        return f"❌ Ошибка API: {err}"

    status     = inst.get("actual_status") or inst.get("status") or "unknown"
    gpu        = inst.get("gpu_name", "?")
    disk_used  = inst.get("disk_util", 0) or 0
    disk_total = inst.get("disk_space", 0) or 0
    now        = datetime.now().strftime("%H:%M")

    status_emoji = {
        "running": "✅",
        "loading": "⏳",
        "exited":  "⛔",
        "offline": "⛔",
    }.get(status, "❓")

    lines = [
        f"{status_emoji} Инстанс {inst.get('id')} | {now} | #{iteration}",
        f"Статус: {status} | GPU: {gpu}",
    ]
    if disk_total:
        lines.append(f"Диск: {disk_used:.0f} / {disk_total:.0f} GB")

    return "\n".join(lines)


def run(instance_id: int):
    print(f"[vast_monitor] Запуск для инстанса {instance_id}...")
    tg_msg(f"🚀 <b>vast_monitor запущен</b>\nИнстанс: {instance_id}\nОтчёт каждые 10 мин")

    iteration = 0
    while True:
        if STOP_FILE.exists():
            STOP_FILE.unlink(missing_ok=True)
            tg_msg("⏹️ vast_monitor остановлен")
            print("[vast_monitor] Остановлен по STOP файлу")
            break

        time.sleep(CHECK_INTERVAL)
        iteration += 1

        inst = get_instance_status(instance_id)
        msg  = format_status(inst, iteration)
        tg_msg(msg)
        print(f"[{iteration}] {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", type=int, default=VASTAI_INSTANCE_ID)
    args = parser.parse_args()

    try:
        run(args.instance_id)
    except KeyboardInterrupt:
        tg_msg("⏹️ vast_monitor остановлен (Ctrl+C)")
        print("\n[vast_monitor] Остановлен")
