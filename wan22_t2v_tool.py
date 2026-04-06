"""
wan22_t2v_tool.py — Wan 2.2 Text-to-Video generator для Claude CLI Helper.

Использует WanGP (deepbeepmeep/Wan2GP) с layer-offloading для 16GB VRAM.
Модели: Wan 2.2 T2V 14B (int8 quantized, ~14GB high + ~14GB low noise).
Установлено: F:/project/Wan2GP/  venv: F:/project/wan2gp_env/

Использование:
    from wan22_t2v_tool import generate_video, check_status

    # Запуск генерации (возвращает task_id)
    task = generate_video(
        prompt="A beautiful sunset over mountains",
        output_path="outputs/my_video.mp4",
        resolution="832x480",     # 480p (default)
        frames=33,                # ~2 sec at 16fps
        steps=30,                 # denoising steps
        seed=42,
    )
    print(f"Started: {task['pid']}, log: {task['log']}")

    # Проверить статус
    status = check_status(task['pid'])
    print(status)  # "running" / "done" / "failed" / "not found"
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

WANGP_DIR = Path("F:/project/Wan2GP")
WANGP_PYTHON = Path("F:/project/wan2gp_env/Scripts/python.exe")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "wan22_t2v"
LOG_DIR = WANGP_DIR / "logs"


def generate_video(
    prompt: str,
    output_path: str = None,
    resolution: str = "832x480",
    frames: int = 33,
    steps: int = 30,
    seed: int = -1,
    attention: str = "sage2",
    profile: int = 4,
    negative_prompt: str = "blurry, low quality, static, distorted",
    guidance_scale: float = 4.0,
) -> dict:
    """
    Запускает Wan 2.2 T2V генерацию в фоне.

    Returns:
        dict with keys: pid, log, settings_file, output_dir, started_at
    """
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path is None:
        output_dir = str(DEFAULT_OUTPUT_DIR)
    else:
        output_dir = str(Path(output_path).parent)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Создать settings JSON
    settings = {
        "model_type": "t2v_2_2",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": resolution,
        "video_length": frames,
        "num_inference_steps": steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "guidance2_scale": 3.0,
        "guidance_phases": 2,
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

    log_f = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(WANGP_DIR),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )

    return {
        "pid": proc.pid,
        "log": str(log_file),
        "settings_file": str(settings_file),
        "output_dir": output_dir,
        "started_at": timestamp,
        "proc": proc,
    }


def check_status(pid: int) -> str:
    """Check if process is still running."""
    import psutil
    try:
        p = psutil.Process(pid)
        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
            return "running"
        return "done"
    except psutil.NoSuchProcess:
        return "not_found"
    except ImportError:
        # fallback without psutil
        try:
            os.kill(pid, 0)
            return "running"
        except OSError:
            return "not_found"


def get_latest_video(output_dir: str) -> str | None:
    """Find the most recently created .mp4 in output_dir."""
    files = list(Path(output_dir).glob("*.mp4"))
    if not files:
        return None
    return str(max(files, key=lambda f: f.stat().st_mtime))


def wait_for_completion(task: dict, timeout_sec: int = 7200, poll_sec: int = 30) -> dict:
    """
    Wait for generation to complete (blocking).
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
    except:
        log_tail = ""

    return {
        "success": retcode == 0,
        "video": video,
        "log_tail": log_tail,
        "returncode": retcode,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing Wan 2.2 T2V tool...")
    task = generate_video(
        prompt="A majestic eagle soaring over mountains, cinematic",
        frames=17,
        steps=20,
        seed=42,
    )
    print(f"Started PID {task['pid']}")
    print(f"Log: {task['log']}")
    print(f"Output: {task['output_dir']}")
