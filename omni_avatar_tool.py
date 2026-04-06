"""
omni_avatar_tool.py — OmniAvatar как инструмент по вызову.

Использование:
    result = generate_avatar_video(
        image_path="path/to/face.jpg",
        audio_path="path/to/voice.wav",
        prompt="A woman speaking directly to camera...",
        output_name="ad_video"
    )
    print(result["video_path"])

Работает на GPU 1 (GPU 0 занят CLI helper).
Логи пишет в omni_avatar.log.
"""

import os
import subprocess
import sys
import logging
import logging.handlers
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# --- Пути ---
_HERE     = Path(__file__).resolve().parent
_PROJECT  = _HERE.parent.parent
OMNI_DIR  = _PROJECT / "OmniAvatar"
if sys.platform == "win32":
    VENV_PYTHON = _PROJECT / "omniavatar_env" / "Scripts" / "python.exe"
else:
    VENV_PYTHON = _PROJECT / "omniavatar_env_linux" / "bin" / "python"
CONFIG_1_3B   = OMNI_DIR / "configs/inference_1.3B.yaml"
OUTPUT_DIR    = OMNI_DIR / "demo_out"
CLI_VIDEO_DIR = _HERE / "outputs" / "omniavatar"
LOG_FILE      = _HERE / "omni_avatar.log"

# GPU 1 — свободная карта (GPU 0 занят CLI helper)
TARGET_GPU = "1"

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [OmniAvatar] %(levelname)s: %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("omni_avatar")


def check_models_ready() -> bool:
    """Проверяет что все модели скачаны."""
    required = [
        OMNI_DIR / "pretrained_models/Wan2.1-T2V-1.3B",
        OMNI_DIR / "pretrained_models/OmniAvatar-1.3B",
        OMNI_DIR / "pretrained_models/wav2vec2-base-960h",
    ]
    for path in required:
        if not path.exists():
            logger.error(f"Модель не найдена: {path}")
            return False
    logger.info("Все модели на месте.")
    return True


def generate_avatar_video(
    image_path: str,
    audio_path: str,
    prompt: str = None,
    output_name: str = None,
    num_steps: int = 20,
    guidance_scale: float = 4.5,
    audio_scale: float = 5.0,
    max_tokens: int = 15000,
    overlap_frame: int = 1,
) -> dict:
    """
    Генерирует говорящего аватара из фото + аудио.

    Args:
        image_path: путь к фото (jpg/png)
        audio_path: путь к аудио (wav/mp3)
        prompt: описание поведения (если None — базовый промпт)
        output_name: имя выходного файла (без расширения)
        num_steps: шаги генерации (20-50, меньше = быстрее)
        guidance_scale: сила промпта (4-6)
        audio_scale: сила lip-sync (3-6, выше = точнее)

    Returns:
        dict с ключами: video_path, success, error, log_path
    """
    logger.info("=" * 60)
    logger.info(f"СТАРТ генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Фото:  {image_path}")
    logger.info(f"  Аудио: {audio_path}")
    logger.info(f"  GPU:   {TARGET_GPU}")

    # Проверяем модели
    if not check_models_ready():
        return {
            "success": False,
            "error": "Модели не скачаны. Запусти download_models().",
            "video_path": None,
            "log_path": str(LOG_FILE),
        }

    # Проверяем входные файлы
    for f, name in [(image_path, "image"), (audio_path, "audio")]:
        if not Path(f).exists():
            err = f"Файл не найден: {f}"
            logger.error(err)
            return {"success": False, "error": err, "video_path": None, "log_path": str(LOG_FILE)}

    # Дефолтный промпт
    if prompt is None:
        prompt = (
            "A realistic video of a woman speaking directly to the camera, "
            "with natural facial expressions and subtle head movements. "
            "The camera remains steady, capturing sharp, clear movements."
        )

    # Имя выходного файла
    if output_name is None:
        output_name = f"avatar_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Создаём temp input файл
    input_file = OMNI_DIR / f"_temp_input_{output_name}.txt"

    try:
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(f"{prompt}@@{Path(image_path).absolute()}@@{Path(audio_path).absolute()}\n")
        logger.info(f"Input файл создан: {input_file}")

        # Команда запуска — напрямую (без torchrun: Windows несовместим)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = TARGET_GPU
        env["PYTHONUTF8"] = "1"
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["LOCAL_RANK"] = "0"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "29503"

        cmd = [
            str(VENV_PYTHON),
            str(OMNI_DIR / "scripts/inference.py"),
            "--config", str(CONFIG_1_3B),
            "--input_file", str(input_file),
            "-hp", f"num_steps={num_steps},guidance_scale={guidance_scale},audio_scale={audio_scale},max_tokens={max_tokens},overlap_frame={overlap_frame}",
        ]

        logger.info(f"Запуск: {' '.join(cmd)}")
        logger.info(f"GPU: {TARGET_GPU} | steps: {num_steps}")

        # Запуск с live-логами
        process = subprocess.Popen(
            cmd,
            cwd=str(OMNI_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"  [omni] {line}")

        process.wait()

        if process.returncode != 0:
            err = f"Процесс завершился с кодом {process.returncode}"
            logger.error(err)
            return {"success": False, "error": err, "video_path": None, "log_path": str(LOG_FILE)}

        # Ищем выходное видео в demo_out/**/*.mp4
        video_files = list(OUTPUT_DIR.rglob("*.mp4"))
        if not video_files:
            err = "Видео не найдено в demo_out/"
            logger.error(err)
            return {"success": False, "error": err, "video_path": None, "log_path": str(LOG_FILE)}

        # Берём самый свежий файл
        latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"ГОТОВО (оригинал): {latest_video}")

        # Копируем в CLI Helper / outputs / omniavatar
        CLI_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        dest_name = f"{output_name}.mp4"
        dest_path = CLI_VIDEO_DIR / dest_name
        # Если файл уже есть — добавляем метку времени
        if dest_path.exists():
            ts = datetime.now().strftime("%H%M%S")
            dest_path = CLI_VIDEO_DIR / f"{output_name}_{ts}.mp4"
        shutil.copy2(str(latest_video), str(dest_path))
        logger.info(f"Скопировано в: {dest_path}")
        logger.info("=" * 60)

        return {
            "success": True,
            "video_path": str(dest_path),
            "original_path": str(latest_video),
            "error": None,
            "log_path": str(LOG_FILE),
        }

    finally:
        # Убираем temp файл
        if input_file.exists():
            input_file.unlink()


def download_models():
    """Скачивает все нужные модели для 1.3B."""
    logger.info("Скачиваю модели для OmniAvatar-1.3B...")
    models = [
        ("Wan-AI/Wan2.1-T2V-1.3B", "pretrained_models/Wan2.1-T2V-1.3B"),
        ("OmniAvatar/OmniAvatar-1.3B", "pretrained_models/OmniAvatar-1.3B"),
        ("facebook/wav2vec2-base-960h", "pretrained_models/wav2vec2-base-960h"),
    ]

    hf_cli = VENV_PYTHON.parent / ("huggingface-cli.exe" if sys.platform == "win32" else "huggingface-cli")

    for repo, local_dir in models:
        dest = OMNI_DIR / local_dir
        if dest.exists() and any(dest.iterdir()):
            logger.info(f"Уже скачано: {local_dir}")
            continue

        logger.info(f"Скачиваю: {repo} → {local_dir}")
        result = subprocess.run(
            [str(hf_cli), "download", repo, "--local-dir", str(OMNI_DIR / local_dir)],
            capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"OK: {repo}")
        else:
            logger.error(f"ОШИБКА: {repo}")


def gpu_status():
    """Показывает текущую нагрузку GPU."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True
    )
    print(result.stdout)


def check_status() -> dict:
    """
    Парсит omni_avatar.log и возвращает текущий статус генерации.

    Returns:
        dict с ключами:
            running: bool — идёт ли генерация прямо сейчас
            done: bool — завершена ли успешно
            segment_current: int
            segment_total: int
            step_current: int
            step_total: int
            pct_segments: float — % по сегментам
            pct_overall: float — общий % (сегменты × шаги)
            eta_seconds: float | None — оценка оставшегося времени
            avg_step_sec: float | None — среднее время одного шага
            image: str | None
            audio: str | None
            prompt: str | None
            last_line: str
            summary: str — готовая строка для вывода
    """
    import re
    from datetime import datetime

    result = {
        "running": False,
        "done": False,
        "segment_current": 0,
        "segment_total": 0,
        "step_current": 0,
        "step_total": 20,
        "pct_segments": 0.0,
        "pct_overall": 0.0,
        "eta_seconds": None,
        "avg_step_sec": None,
        "image": None,
        "audio": None,
        "prompt": None,
        "last_line": "",
        "summary": "Лог не найден.",
    }

    if not LOG_FILE.exists():
        return result

    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return result

    result["last_line"] = lines[-1]

    # --- Парсим входные данные (берём из последнего запуска) ---
    # Ищем последний блок START (=====)
    start_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "=====" in lines[i] and "СТАРТ" in (lines[i + 1] if i + 1 < len(lines) else ""):
            start_idx = i
            break
        if "СТАРТ генерации" in lines[i]:
            start_idx = i
            break

    if start_idx is not None:
        block = lines[start_idx:]
        for line in block:
            if "\u0424\u043e\u0442\u043e:" in line or "Фото:" in line:
                result["image"] = line.split(":")[-1].strip()
            if "\u0410\u0443\u0434\u0438\u043e:" in line or "Аудио:" in line:
                result["audio"] = line.split("Аудио:")[-1].strip() if "Аудио:" in line else line.split(":")[-1].strip()
            if "\u0413\u041e\u0422\u041e\u0412\u041e:" in line or "ГОТОВО:" in line:
                result["done"] = True
                result["pct_overall"] = 100.0
                result["pct_segments"] = 100.0
    else:
        block = lines
        # fallback: scan entire log for ГОТОВО
        for line in lines:
            if "ГОТОВО:" in line or "\u0413\u041e\u0422\u041e\u0412\u041e:" in line:
                result["done"] = True
                result["pct_overall"] = 100.0
                result["pct_segments"] = 100.0

    # --- Парсим прогресс сегментов [N/M] ---
    seg_pattern = re.compile(r'\[(\d+)/(\d+)\]')
    seg_current, seg_total = 0, 0
    for line in reversed(block):
        m = seg_pattern.search(line)
        if m:
            seg_current = int(m.group(1))
            seg_total = int(m.group(2))
            break

    result["segment_current"] = seg_current
    result["segment_total"] = seg_total

    # --- Парсим текущий шаг диффузии из tqdm ---
    # Формат: " 45%|████      | 9/20 [01:52<02:18,  ...]"
    step_pattern = re.compile(r'(\d+)/(\d+)\s*\[(\d+):(\d+)<(\d+):(\d+)')
    step_current, step_total = 0, 20
    eta_from_tqdm = None
    for line in reversed(block):
        m = step_pattern.search(line)
        if m:
            step_current = int(m.group(1))
            step_total = int(m.group(2))
            eta_min = int(m.group(5))
            eta_sec = int(m.group(6))
            eta_from_tqdm = eta_min * 60 + eta_sec
            break

    result["step_current"] = step_current
    result["step_total"] = step_total

    # --- Считаем avg_step_sec по timestamp'ам в логе ---
    ts_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    timestamps = []
    for line in block:
        m = ts_pattern.match(line)
        if m and step_pattern.search(line):
            try:
                timestamps.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            except Exception:
                pass

    avg_step_sec = None
    if len(timestamps) >= 2:
        deltas = [(timestamps[i + 1] - timestamps[i]).total_seconds()
                  for i in range(len(timestamps) - 1)
                  if 0 < (timestamps[i + 1] - timestamps[i]).total_seconds() < 300]
        if deltas:
            avg_step_sec = sum(deltas) / len(deltas)
            result["avg_step_sec"] = avg_step_sec

    # --- ETA ---
    if not result["done"]:
        result["running"] = True
        if seg_total > 0 and step_total > 0:
            completed_steps = (seg_current - 1) * step_total + step_current
            total_steps = seg_total * step_total
            overall_pct = completed_steps / total_steps * 100 if total_steps > 0 else 0
            result["pct_overall"] = round(overall_pct, 1)
            result["pct_segments"] = round(seg_current / seg_total * 100, 1) if seg_total > 0 else 0

            remaining_steps = total_steps - completed_steps
            if avg_step_sec:
                result["eta_seconds"] = remaining_steps * avg_step_sec
            elif eta_from_tqdm is not None:
                # tqdm показывает ETA только для текущего сегмента — добавляем оставшиеся сегменты
                remaining_segs = seg_total - seg_current
                result["eta_seconds"] = eta_from_tqdm + remaining_segs * step_total * 12.5  # 12.5 sec/step estimate

    # --- Собираем summary ---
    if result["done"]:
        summary = "✅ ГОТОВО! Генерация завершена."
    elif result["running"] and seg_total > 0:
        eta = result.get("eta_seconds")
        if eta:
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            eta_str = f" | ETA: ~{eta_min}м {eta_sec}с"
        else:
            eta_str = ""
        summary = (
            f"⏳ Генерация: сегмент {seg_current}/{seg_total} "
            f"| шаг {step_current}/{step_total} "
            f"| {result['pct_overall']:.1f}%{eta_str}"
        )
        if result["image"]:
            summary += f"\n   Фото:  {result['image']}"
        if result["audio"]:
            summary += f"\n   Аудио: {result['audio']}"
    else:
        summary = "ℹ️  Генерация не запущена или лог пустой."

    result["summary"] = summary
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_models()
    elif len(sys.argv) > 1 and sys.argv[1] == "gpu":
        gpu_status()
    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        s = check_status()
        print(s["summary"])
        if s.get("avg_step_sec"):
            print(f"   Avg step: {s['avg_step_sec']:.1f}s")
    else:
        print("Использование:")
        print("  python omni_avatar_tool.py download  — скачать модели")
        print("  python omni_avatar_tool.py gpu       — статус GPU")
        print("  python omni_avatar_tool.py status    — прогресс генерации")
