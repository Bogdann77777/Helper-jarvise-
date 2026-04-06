"""
screen_watcher.py — делает скриншот каждые N секунд, хранит последние MAX_KEEP файлов.

Запуск: python screen_watcher.py [--interval 2] [--keep 10] [--browser-only]
Остановка: Ctrl+C или создать файл STOP_WATCHER в папке скриптов

Скриншоты: static/screen/screen_001.jpg ... screen_010.jpg (ротация)
Последний: static/screen/latest.jpg (всегда актуальный)
"""

import os
import sys
import time
import glob
import argparse
import logging
from pathlib import Path

_HERE   = Path(__file__).parent
SCREEN_DIR = _HERE / "static" / "screen"
SCREEN_DIR.mkdir(parents=True, exist_ok=True)

STOP_FILE = _HERE / "STOP_WATCHER"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [screen_watcher] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def take_screenshot(browser_only: bool = False) -> bytes | None:
    """Снимает скриншот. browser_only=True — пытается захватить только браузер."""
    try:
        import mss
        import mss.tools

        with mss.mss() as sct:
            if browser_only:
                # Попытка найти окно браузера (Chrome/Edge/Firefox)
                mon = _find_browser_monitor(sct)
            else:
                mon = sct.monitors[0]  # весь рабочий стол

            img = sct.grab(mon)
            return mss.tools.to_png(img.rgb, img.size)
    except Exception as e:
        log.warning(f"Screenshot failed: {e}")
        return None


def _find_browser_monitor(sct):
    """Пытается найти прямоугольник окна браузера через win32gui."""
    try:
        import win32gui

        browser_titles = ["Chrome", "Edge", "Firefox", "Opera", "Brave", "Vivaldi", "vast.ai"]

        def _cb(hwnd, found):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if any(b.lower() in title.lower() for b in browser_titles):
                rect = win32gui.GetWindowRect(hwnd)
                found.append(rect)

        found = []
        win32gui.EnumWindows(_cb, found)

        if found:
            x1, y1, x2, y2 = found[0]
            w, h = x2 - x1, y2 - y1
            if w > 100 and h > 100:
                return {"left": x1, "top": y1, "width": w, "height": h}
    except ImportError:
        pass  # win32gui не установлен — вернём полный экран

    import mss
    with mss.mss() as sct:
        return sct.monitors[0]


_shot_counter = 0

def save_screenshot(png_bytes: bytes, keep: int = 20):
    """Сохраняет скриншот, сжимает в JPEG, ротирует буфер. Хранит последние keep файлов."""
    global _shot_counter
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(png_bytes))
    if img.width > 1920:
        ratio = 1920 / img.width
        img = img.resize((1920, int(img.height * ratio)), Image.LANCZOS)

    _shot_counter += 1
    out_path = SCREEN_DIR / f"screen_{_shot_counter:05d}.jpg"
    img.save(out_path, "JPEG", quality=75, optimize=True)

    # latest.jpg — всегда последний
    latest = SCREEN_DIR / "latest.jpg"
    img.save(latest, "JPEG", quality=75, optimize=True)

    # Удаляем лишние — сортируем по имени, оставляем последние keep
    existing = sorted(SCREEN_DIR.glob("screen_?????.jpg"))
    for old in existing[:-keep]:
        try:
            old.unlink(missing_ok=True)
        except PermissionError:
            pass

    return out_path


def run(interval: float = 2.0, keep: int = 10, browser_only: bool = False):
    log.info(f"Started — interval={interval}s keep={keep} browser_only={browser_only}")
    log.info(f"Screenshots → {SCREEN_DIR}")
    log.info("Press Ctrl+C to stop, or create STOP_WATCHER file")

    count = 0
    try:
        while True:
            if STOP_FILE.exists():
                log.info("STOP_WATCHER file found — stopping")
                STOP_FILE.unlink(missing_ok=True)
                break

            png = take_screenshot(browser_only)
            if png:
                path = save_screenshot(png, keep=keep)
                count += 1
                if count % 30 == 1:  # лог каждую минуту
                    log.info(f"Shot #{count} → {path.name} ({len(png)//1024}KB)")

            time.sleep(interval)

    except KeyboardInterrupt:
        log.info(f"Stopped. Total shots: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Desktop/browser screenshot watcher")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between shots (default 2)")
    parser.add_argument("--keep",     type=int,   default=10,  help="How many screenshots to keep (default 10)")
    parser.add_argument("--browser",  action="store_true",     help="Try to capture only browser window")
    args = parser.parse_args()

    run(interval=args.interval, keep=args.keep, browser_only=args.browser)
