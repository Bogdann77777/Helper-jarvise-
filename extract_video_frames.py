"""
extract_video_frames.py — раскадровка видео + метаданные через OpenCV.

Использование как инструмент:
    from extract_video_frames import extract_frames, get_metadata

    # Получить метаданные
    meta = get_metadata("path/to/video.mp4")
    print(meta)

    # Извлечь кадры (1 кадр в секунду)
    frames = extract_frames("path/to/video.mp4", every_n_sec=1.0)
    print(f"Извлечено: {len(frames)} кадров → {frames[0]}")

CLI:
    python extract_video_frames.py video.mp4
    python extract_video_frames.py video.mp4 --every 2.0
    python extract_video_frames.py video.mp4 --out ./frames --every 0.5
    python extract_video_frames.py video.mp4 --meta-only
"""

import sys
import os
from pathlib import Path


def get_metadata(video_path: str) -> dict:
    """
    Возвращает метаданные видеофайла.

    Returns:
        {
            "path": str,
            "fps": float,
            "width": int,
            "height": int,
            "total_frames": int,
            "duration_sec": float,
            "codec": str,
        }
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не могу открыть видео: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int   = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec        = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
    duration     = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "path":         str(video_path),
        "fps":          round(fps, 2),
        "width":        width,
        "height":       height,
        "total_frames": total_frames,
        "duration_sec": round(duration, 2),
        "codec":        codec.strip(),
    }


def extract_frames(
    video_path: str,
    output_dir: str = None,
    every_n_sec: float = 1.0,
    max_frames: int = 500,
    quality: int = 90,
) -> list:
    """
    Извлекает кадры из видео с заданным интервалом.

    Args:
        video_path:  путь к видео (.mp4, .avi, .mov, …)
        output_dir:  куда сохранять кадры (default: рядом с видео, папка _frames)
        every_n_sec: интервал между кадрами в секундах (default: 1.0)
        max_frames:  максимум кадров (защита от огромных файлов)
        quality:     JPEG quality 1-100 (default: 90)

    Returns:
        list[str] — пути к сохранённым PNG/JPEG файлам (хронологический порядок)
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Файл не найден: {video_path}")

    # Папка вывода
    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_frames"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не могу открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise ValueError(f"Не удалось определить FPS: {video_path}")

    # Шаг в кадрах
    frame_step = max(1, int(fps * every_n_sec))

    saved = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp = frame_idx / fps
            out_name = f"frame_{saved_count:05d}_{timestamp:.2f}s.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved.append(str(out_path))
            saved_count += 1

            if saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def extract_keyframes(
    video_path: str,
    output_dir: str = None,
    threshold: float = 30.0,
    max_frames: int = 200,
) -> list:
    """
    Извлекает только ключевые кадры (где картинка сильно меняется).
    Полезно для длинных видео — не дублирует похожие кадры.

    Args:
        threshold: порог изменения (выше = меньше кадров, grubее отбор)

    Returns:
        list[str] — пути к сохранённым файлам
    """
    import cv2
    import numpy as np

    video_path = Path(video_path)
    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_keyframes"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не могу открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_gray = None
    saved = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            # Всегда сохраняем первый кадр
            diff = threshold + 1
        else:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))

        if diff >= threshold:
            timestamp = frame_idx / fps if fps > 0 else frame_idx
            out_name = f"key_{saved_count:04d}_{timestamp:.2f}s.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved.append(str(out_path))
            saved_count += 1
            prev_gray = gray

            if saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Раскадровка видео")
    parser.add_argument("video", help="Путь к видеофайлу")
    parser.add_argument("--out", default=None, help="Папка для кадров")
    parser.add_argument("--every", type=float, default=1.0,
                        help="Интервал между кадрами в секундах (default: 1.0)")
    parser.add_argument("--keyframes", action="store_true",
                        help="Извлечь только ключевые кадры (по изменению)")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Порог ключевых кадров (default: 30.0)")
    parser.add_argument("--max", type=int, default=500,
                        help="Максимум кадров (default: 500)")
    parser.add_argument("--meta-only", action="store_true",
                        help="Только метаданные, без извлечения")
    args = parser.parse_args()

    # Метаданные
    meta = get_metadata(args.video)
    print("=== МЕТАДАННЫЕ ===")
    print(f"  Разрешение: {meta['width']}×{meta['height']}")
    print(f"  FPS:        {meta['fps']}")
    print(f"  Кадров:     {meta['total_frames']}")
    print(f"  Длина:      {meta['duration_sec']} сек")
    print(f"  Кодек:      {meta['codec']}")

    if args.meta_only:
        sys.exit(0)

    print()

    # Извлечение кадров
    if args.keyframes:
        print(f"Извлекаю ключевые кадры (порог={args.threshold})...")
        frames = extract_keyframes(args.video, output_dir=args.out,
                                   threshold=args.threshold, max_frames=args.max)
    else:
        print(f"Извлекаю кадры (каждые {args.every}с)...")
        frames = extract_frames(args.video, output_dir=args.out,
                                every_n_sec=args.every, max_frames=args.max)

    print(f"Готово! Извлечено {len(frames)} кадров → {Path(frames[0]).parent if frames else 'н/д'}")
