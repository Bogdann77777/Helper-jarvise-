"""
wan22_t2v_tool.py — Wan 2.2 Text-to-Video generator для Claude CLI Helper.

Использует WanGP (deepbeepmeep/Wan2GP) с layer-offloading.
Модели: Wan 2.2 T2V 14B (int8 quantized).
Установлено: F:/project/Wan2GP/  venv: F:/project/wan2gp_env/

PRODUCTION DEFAULTS:
  - resolution: 1280x720 (GPU = RTX 5060 Ti 15GB × 2, fits at 720p)
    ⚠️  161+ кадров → "Sliding Windows" error для T2V — НЕ использовать
  - frames: 81 (max native T2V, ~5 sec at 16fps)
  - steps: 20 (confirmed working; 30 = higher quality but slower)
  - guidance_scale: 5.0
  - GPU: CUDA_VISIBLE_DEVICES=0,1 (оба RTX 5060 Ti 15GB)
  - TeaCache: skip_steps_cache_type=tea, multiplier=2.0 (~2× ускорение)

TELEGRAM MONITORING (автоматически при generate_video):
  - Старт — уведомление в Telegram
  - Каждые 10 мин — прогресс: шаги, скорость (сек/шаг), время до конца
  - Финал — отправляет видео в Telegram

Использование:
    from wan22_t2v_tool import generate_video

    task = generate_video(
        prompt="A young woman drinking coffee at a sunny table",
        output_path="outputs/my_video.mp4",
    )
    print(f"Started: {task['pid']}, log: {task['log']}")
    task["monitor_thread"].join()  # wait for completion (blocks)
"""

import subprocess
import sys
import os
import json
import time
import threading
import re
from pathlib import Path
from datetime import datetime, timedelta

WANGP_DIR = Path("F:/project/Wan2GP")
WANGP_PYTHON = Path("F:/project/wan2gp_env/Scripts/python.exe")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "wan22_t2v"
LOG_DIR = WANGP_DIR / "logs"
CLI_HELPER_DIR = Path(__file__).resolve().parent
TG_REPORT_INTERVAL = 600  # seconds (10 min)


def _parse_progress(log_path: Path):
    """
    Parse latest tqdm progress line from WanGP log.
    Returns dict or None if not yet available.
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None

    # tqdm format: "25/50 [10:30<10:30, 25.2s/it]" or "25/50 [10:30<10:30, 0.04it/s]"
    pattern = r'(\d+)/(\d+)\s*\[(\d+:\d+)<([^,\]]+),\s*([\d.]+)(s/it|it/s)\]'
    matches = re.findall(pattern, content)
    if not matches:
        return None

    cur_s, total_s, elapsed_s, remaining_s, speed_s, unit = matches[-1]
    cur, total = int(cur_s), int(total_s)
    if total == 0:
        return None

    # Parse remaining time "MM:SS" or "H:MM:SS"
    parts = remaining_s.strip().split(":")
    try:
        if len(parts) == 2:
            rem_sec = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            rem_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            rem_sec = 0
    except ValueError:
        rem_sec = 0

    speed_val = float(speed_s)
    speed_spit = speed_val if unit == "s/it" else (1.0 / speed_val if speed_val > 0 else 0)

    return {
        "current": cur,
        "total": total,
        "percent": round(cur / total * 100, 1),
        "speed_spit": speed_spit,      # seconds per step
        "remaining_sec": rem_sec,
        "remaining_str": str(timedelta(seconds=rem_sec)),
        "elapsed_str": elapsed_s,
    }


def _monitor_thread(proc, log_path: Path, output_dir: str, started_at: str,
                    total_steps: int, prompt: str):
    """
    Background thread: Telegram progress reports every 10 min.
    Sends final video when done.
    """
    try:
        if str(CLI_HELPER_DIR) not in sys.path:
            sys.path.insert(0, str(CLI_HELPER_DIR))
        from tg_send import tg_msg, tg_video as tg_vid
    except Exception as e:
        print(f"[wangp-monitor] Failed to import tg_send: {e}")
        return

    start_time = time.time()
    last_report = start_time - TG_REPORT_INTERVAL  # force immediate report in 10 min

    tg_msg(
        f"🎬 <b>WanGP T2V — генерация началась</b>\n"
        f"📝 {prompt[:120]}\n"
        f"⚙️ Шаги: {total_steps} | 832×480 | 33 кадра (~2s) | TeaCache×2 | profile5\n"
        f"📊 Отчёты каждые 10 мин"
    )

    while True:
        time.sleep(30)
        retcode = proc.poll()
        now = time.time()

        if now - last_report >= TG_REPORT_INTERVAL:
            progress = _parse_progress(log_path)
            elapsed_total = int(now - start_time)
            elapsed_str = str(timedelta(seconds=elapsed_total))

            if progress and progress["current"] > 0:
                pct = progress["percent"]
                bar_filled = int(pct / 10)
                bar = "█" * bar_filled + "░" * (10 - bar_filled)
                speed = progress["speed_spit"]
                rem = progress["remaining_str"]
                cur = progress["current"]
                total = progress["total"]

                tg_msg(
                    f"📊 <b>WanGP T2V — прогресс</b>\n"
                    f"[{bar}] {pct}%\n"
                    f"🔢 Шаги: {cur}/{total}\n"
                    f"⚡ Скорость: {speed:.1f} сек/шаг\n"
                    f"⏳ Осталось: {rem}\n"
                    f"🕐 Прошло: {elapsed_str}"
                )
            else:
                tg_msg(
                    f"📊 <b>WanGP T2V</b> — генерация идёт\n"
                    f"🕐 Прошло: {elapsed_str} | прогресс ещё не виден в логе"
                )
            last_report = now

        if retcode is not None:
            break

    # Generation finished
    videos = list(Path(output_dir).glob("*.mp4"))
    video = str(max(videos, key=lambda f: f.stat().st_mtime)) if videos else None
    total_elapsed = int(time.time() - start_time)
    elapsed_str = str(timedelta(seconds=total_elapsed))

    if retcode == 0 and video:
        tg_msg(
            f"✅ <b>WanGP T2V — готово!</b>\n"
            f"⏱ Общее время: {elapsed_str}\n"
            f"Отправляю видео..."
        )
        ok = tg_vid(video, caption=f"🎬 {prompt[:80]}")
        if not ok:
            tg_msg(f"⚠️ Видео слишком большое для Telegram.\nПуть: <code>{video}</code>")
    else:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read()[-800:]
        except Exception:
            tail = "(не удалось прочитать лог)"
        tg_msg(
            f"❌ <b>WanGP T2V — ошибка</b> (код {retcode})\n"
            f"<pre>{tail[-400:]}</pre>"
        )


def generate_video(
    prompt: str,
    output_path: str = None,
    resolution: str = "832x480",
    frames: int = 33,
    steps: int = 20,
    seed: int = 42,
    attention: str = "sage2",
    profile: int = 5,
    preload: int = 0,
    negative_prompt: str = "blurry, low quality, static, distorted, noise, pixelated",
    guidance_scale: float = 5.0,
) -> dict:
    """
    Запускает Wan 2.2 T2V генерацию в фоне.
    Автоматически стартует Telegram мониторинг (отчёт каждые 10 мин).

    Returns:
        dict: pid, log, settings_file, output_dir, started_at, proc, monitor_thread
    """
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path is None:
        output_dir = str(DEFAULT_OUTPUT_DIR)
    else:
        output_dir = str(Path(output_path).parent)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    settings = {
        "model_type": "t2v_2_2",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": resolution,
        "video_length": frames,
        "num_inference_steps": steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "guidance_phases": 1,
        "skip_steps_cache_type": "tea",
        "skip_steps_multiplier": 2.0,
    }

    settings_file = WANGP_DIR / f"task_{timestamp}.json"
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    log_file = LOG_DIR / f"t2v_{timestamp}.log"

    cmd = [
        str(WANGP_PYTHON),
        "-u", "wgp.py",
        "--process", str(settings_file),
        "--output-dir", output_dir,
        "--attention", attention,
        "--profile", str(profile),
        "--verbose", "1",
    ]

    # CUDA_VISIBLE_DEVICES=0,1 → оба RTX 5060 Ti 15GB
    # --gpu cuda:N does NOT work, use CUDA_VISIBLE_DEVICES instead
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"

    log_f = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(WANGP_DIR),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
    )

    monitor = threading.Thread(
        target=_monitor_thread,
        args=(proc, log_file, output_dir, timestamp, steps, prompt),
        daemon=True,
        name="wangp-monitor",
    )
    monitor.start()

    return {
        "pid": proc.pid,
        "log": str(log_file),
        "settings_file": str(settings_file),
        "output_dir": output_dir,
        "started_at": timestamp,
        "proc": proc,
        "monitor_thread": monitor,
    }


def check_status(pid: int) -> str:
    """Check if WanGP process is still running."""
    import psutil
    try:
        p = psutil.Process(pid)
        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
            return "running"
        return "done"
    except psutil.NoSuchProcess:
        return "not_found"
    except ImportError:
        try:
            os.kill(pid, 0)
            return "running"
        except OSError:
            return "not_found"


def get_latest_video(output_dir: str):
    """Find the most recently created .mp4 in output_dir."""
    files = list(Path(output_dir).glob("*.mp4"))
    if not files:
        return None
    return str(max(files, key=lambda f: f.stat().st_mtime))


def wait_for_completion(task: dict, timeout_sec: int = 7200, poll_sec: int = 30) -> dict:
    """
    Wait for generation to complete (blocking).
    Note: Telegram monitoring runs automatically in background thread.
    Returns: {"success": bool, "video": path or None, "log_tail": str}
    """
    proc = task.get("proc")
    if proc is None:
        return {"success": False, "video": None, "log_tail": "No process object"}

    start = time.time()
    while True:
        retcode = proc.poll()
        if retcode is not None:
            break
        if time.time() - start > timeout_sec:
            proc.kill()
            return {"success": False, "video": None, "log_tail": "Timeout"}
        time.sleep(poll_sec)

    video = get_latest_video(task["output_dir"])
    try:
        with open(task["log"], "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        log_tail = content[-1000:]
    except Exception:
        log_tail = ""

    return {
        "success": retcode == 0,
        "video": video,
        "log_tail": log_tail,
        "returncode": retcode,
    }


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "A young woman sitting at a wooden table by a bright morning window, "
        "holding a cup of coffee with both hands, smiling warmly, talking softly "
        "about her plans for the day, cozy home interior, golden morning light, "
        "authentic UGC style, cinematic, sharp, 4K, photorealistic"
    )
    print(f"Launching: {prompt[:80]}...")
    task = generate_video(prompt=prompt)
    print(f"PID: {task['pid']}")
    print(f"Log: {task['log']}")
    print(f"Output: {task['output_dir']}")
    print(f"Settings: 832x480 | 33 frames (~2s) | 20 steps | gs=5.0 | TeaCache×2 | profile=5 | GPU1")
    print("Monitoring active. Telegram reports every 10 min.")
    print("Waiting for completion...")
    task["monitor_thread"].join()  # blocks until generation + final TG report done
    print("Done.")
