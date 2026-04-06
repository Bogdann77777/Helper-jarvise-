"""
video_analyzer_tool.py — Инструмент анализа видео для Claude.

Позволяет Claude "видеть" видео:
  1. extract_frames()  — нарезает кадры из видео через FFmpeg
  2. video_info()      — метаданные видео (длительность, fps, разрешение)
  3. analyze_video()   — полный анализ: кадры + информация (всё в одном)

Кадры сохраняются в outputs/video_frames/<video_name>/ и могут быть
прочитаны напрямую через инструмент Read (Claude поддерживает изображения).

Зависимости: только ffmpeg (уже установлен в PATH).

Использование:
    from video_analyzer_tool import analyze_video, extract_frames, video_info

    # Полный анализ — кадры + мета
    result = analyze_video("path/to/video.mp4", fps=0.5)
    print(result["summary"])
    # result["frames"] — список путей к .jpg файлам, читай через Read

    # Только кадры
    frames = extract_frames("video.mp4", fps=1, max_frames=20)

    # Только метаданные
    info = video_info("video.mp4")
"""

import subprocess
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Выходная папка для кадров
FRAMES_BASE_DIR = Path(__file__).resolve().parent / "outputs" / "video_frames"


def video_info(video_path: str) -> dict:
    """
    Получает метаданные видео через ffprobe.

    Returns:
        dict: duration_sec, fps, width, height, codec, size_mb, streams
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {"error": f"Файл не найден: {video_path}"}

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"error": f"ffprobe error: {result.stderr}"}

        data = json.loads(result.stdout)
        info = {
            "path": str(video_path),
            "filename": video_path.name,
            "size_mb": round(video_path.stat().st_size / 1024 / 1024, 2),
            "duration_sec": None,
            "fps": None,
            "width": None,
            "height": None,
            "codec": None,
            "audio_codec": None,
            "streams": [],
        }

        # Формат (общая длительность)
        fmt = data.get("format", {})
        if "duration" in fmt:
            info["duration_sec"] = round(float(fmt["duration"]), 2)

        # Потоки
        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type", "")
            info["streams"].append(f"{codec_type}: {stream.get('codec_name', '?')}")

            if codec_type == "video":
                info["codec"] = stream.get("codec_name")
                info["width"] = stream.get("width")
                info["height"] = stream.get("height")
                # FPS
                fps_str = stream.get("r_frame_rate", "0/1")
                try:
                    num, den = fps_str.split("/")
                    info["fps"] = round(int(num) / int(den), 2) if int(den) > 0 else None
                except Exception:
                    pass

            elif codec_type == "audio":
                info["audio_codec"] = stream.get("codec_name")

        # Summary
        dur = f"{info['duration_sec']}s" if info["duration_sec"] else "?"
        res = f"{info['width']}x{info['height']}" if info["width"] else "?"
        info["summary"] = (
            f"📹 {video_path.name} | {dur} | {res} | {info['fps']} fps | "
            f"{info['codec']} | аудио: {info['audio_codec']} | {info['size_mb']} MB"
        )

        return info

    except Exception as e:
        return {"error": str(e)}


def extract_frames(
    video_path: str,
    fps: float = 1.0,
    max_frames: int = 30,
    output_dir: str = None,
    quality: int = 2,
) -> dict:
    """
    Извлекает кадры из видео через FFmpeg.

    Args:
        video_path: путь к видео
        fps: кадров в секунду (0.5 = каждые 2с, 1 = каждую секунду, 2 = каждые 0.5с)
        max_frames: максимум кадров (защита от огромных видео)
        output_dir: куда сохранять (по умолчанию outputs/video_frames/<name>/)
        quality: качество JPEG 1-31 (1=лучшее, 2=хорошее, 5=норм)

    Returns:
        dict: frames (list of paths), count, output_dir, error
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {"error": f"Файл не найден: {video_path}", "frames": [], "count": 0}

    # Создаём папку для кадров
    if output_dir is None:
        safe_name = video_path.stem.replace(" ", "_")[:40]
        ts = datetime.now().strftime("%H%M%S")
        output_dir = FRAMES_BASE_DIR / f"{safe_name}_{ts}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Узнаём длительность видео
    info = video_info(str(video_path))
    duration = info.get("duration_sec", 0) or 0

    # Подбираем fps чтобы не превышать max_frames
    if duration > 0 and fps * duration > max_frames:
        fps = round(max_frames / duration, 3)

    # FFmpeg: извлекаем кадры
    output_pattern = str(output_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", str(quality),
        output_pattern
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            return {
                "error": f"FFmpeg error: {result.stderr[-500:]}",
                "frames": [],
                "count": 0,
                "output_dir": str(output_dir),
            }

        # Собираем список файлов
        frames = sorted(output_dir.glob("frame_*.jpg"))
        frame_paths = [str(f) for f in frames]

        summary_lines = [
            f"🎞️  Извлечено {len(frames)} кадров из {video_path.name}",
            f"   fps={fps} | max_frames={max_frames} | duration={duration}s",
            f"   Папка: {output_dir}",
        ]
        if frames:
            summary_lines.append(f"   Первый: {frames[0].name} | Последний: {frames[-1].name}")

        return {
            "frames": frame_paths,
            "count": len(frames),
            "output_dir": str(output_dir),
            "fps_used": fps,
            "duration_sec": duration,
            "error": None,
            "summary": "\n".join(summary_lines),
        }

    except Exception as e:
        return {"error": str(e), "frames": [], "count": 0, "output_dir": str(output_dir)}


def analyze_video(
    video_path: str,
    fps: float = 0.5,
    max_frames: int = 20,
) -> dict:
    """
    Полный анализ видео: метаданные + кадры.
    Кадры читай через инструмент Read — Claude видит изображения.

    Args:
        video_path: путь к видео
        fps: кадров в секунду для извлечения (0.5 = каждые 2с)
        max_frames: максимум кадров

    Returns:
        dict: info, frames, summary, error
    """
    info = video_info(video_path)
    if "error" in info and not info.get("duration_sec"):
        return {"error": info["error"], "frames": [], "info": info}

    frames_result = extract_frames(video_path, fps=fps, max_frames=max_frames)

    summary = info.get("summary", "?") + "\n" + frames_result.get("summary", "")

    return {
        "info": info,
        "frames": frames_result.get("frames", []),
        "count": frames_result.get("count", 0),
        "output_dir": frames_result.get("output_dir"),
        "error": frames_result.get("error"),
        "summary": summary,
    }


def cleanup_frames(output_dir: str):
    """Удаляет папку с кадрами после анализа."""
    p = Path(output_dir)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
        print(f"Удалено: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python video_analyzer_tool.py <video.mp4>              — анализ (кадры каждые 2с)")
        print("  python video_analyzer_tool.py <video.mp4> info         — только метаданные")
        print("  python video_analyzer_tool.py <video.mp4> frames 1 30  — кадры: fps=1, max=30")
        sys.exit(0)

    vpath = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "analyze"

    if mode == "info":
        info = video_info(vpath)
        print(info.get("summary", info))

    elif mode == "frames":
        fps_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        max_arg = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        result = extract_frames(vpath, fps=fps_arg, max_frames=max_arg)
        print(result.get("summary", result))
        if result["frames"]:
            print("\nКадры:")
            for f in result["frames"][:5]:
                print(f"  {f}")
            if result["count"] > 5:
                print(f"  ... ещё {result['count'] - 5} кадров")

    else:
        result = analyze_video(vpath)
        print(result.get("summary", result))
        if result["frames"]:
            print("\nПервые 5 кадров (читай через Read):")
            for f in result["frames"][:5]:
                print(f"  {f}")
