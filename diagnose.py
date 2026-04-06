"""Startup diagnostics — writes to both console and startup.log."""
import sys
import os
import datetime

LOGFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startup.log")

_log_file = None

def _open_log():
    global _log_file
    _log_file = open(LOGFILE, "w", encoding="utf-8")

def p(msg=""):
    """Print to both console and log file."""
    print(msg)
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()

def check_import(name, version_attr="__version__", label=None):
    """Try to import a module and print its version. Returns (success, version_or_error)."""
    label = label or name
    try:
        mod = __import__(name)
        ver = getattr(mod, version_attr, "OK")
        return True, ver
    except Exception as e:
        return False, str(e)

def run():
    _open_log()
    errors = []

    p("=" * 60)
    p(f"  CEO 2.0 — Startup Diagnostics")
    p(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 60)
    p()

    # --- Python ---
    p("--- Python ---")
    p(f"  Version:    {sys.version}")
    p(f"  Executable: {sys.executable}")
    p(f"  Arch:       {'x64' if sys.maxsize > 2**32 else 'x86 (32-bit!!)'}")
    p(f"  Platform:   {sys.platform}")
    p()

    # --- torch ---
    p("--- [1/9] torch ---")
    ok, ver = check_import("torch")
    if ok:
        import torch
        p(f"  version:        {torch.__version__}")
        p(f"  CUDA available: {torch.cuda.is_available()}")
        p(f"  CUDA version:   {torch.version.cuda or 'N/A'}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p(f"  GPU {i}:          {torch.cuda.get_device_name(i)}")
                try:
                    props = torch.cuda.get_device_properties(i)
                    p(f"  GPU {i} VRAM:     {props.total_memory / 1024**3:.1f} GB")
                except Exception as e:
                    p(f"  GPU {i} VRAM:     ? ({e})")
        else:
            p("  WARNING: CUDA not available, XTTS will run on CPU (very slow)")
    else:
        p(f"  FAILED: {ver}")
        errors.append("torch")
    p()

    # --- torchaudio ---
    p("--- [2/9] torchaudio ---")
    ok, ver = check_import("torchaudio")
    if ok:
        p(f"  version: {ver}")
    else:
        p(f"  FAILED: {ver}")
        errors.append("torchaudio")
    p()

    # --- numpy ---
    p("--- [3/9] numpy ---")
    ok, ver = check_import("numpy")
    if ok:
        p(f"  version: {ver}")
        major = int(ver.split(".")[0])
        if major >= 2:
            p("  WARNING: numpy 2.x may cause issues with TTS/XTTS")
            p("  Fix: pip install 'numpy<2'")
    else:
        p(f"  FAILED: {ver}")
        errors.append("numpy")
    p()

    # --- TTS (top-level) ---
    p("--- [4/9] TTS package (XTTS, optional — Qwen3-TTS is primary) ---")
    ok, ver = check_import("TTS")
    if ok:
        p(f"  version: {ver}")
    else:
        p(f"  WARNING: {ver}")
        p("  (XTTS unavailable — server uses Qwen3-TTS as primary engine)")
    p()

    # --- TTS.api (what xtts_manager.py actually imports) ---
    p("--- [5/9] TTS.api (XTTS engine import, optional) ---")
    try:
        from TTS.api import TTS as _TTS
        p("  from TTS.api import TTS: OK")
    except Exception as e:
        p(f"  WARNING: {e}")
        p("  (XTTS unavailable — Qwen3-TTS will be used instead)")
    p()

    # --- XTTS sub-dependencies ---
    p("--- [6/9] XTTS sub-dependencies ---")
    for dep in ["librosa", "scipy", "soundfile"]:
        ok, ver = check_import(dep)
        if ok:
            p(f"  {dep}: {ver}")
        else:
            p(f"  {dep}: FAILED — {ver}")
            errors.append(dep)
    p()

    # --- faster_whisper ---
    p("--- [7/9] faster_whisper (STT) ---")
    ok, ver = check_import("faster_whisper")
    if ok:
        p(f"  faster_whisper: OK")
    else:
        p(f"  FAILED: {ver}")
        errors.append("faster_whisper")
    p()

    # --- Web framework ---
    p("--- [8/9] Web framework ---")
    for dep in ["fastapi", "httpx", "pydantic", "websockets", "uvicorn"]:
        ok, ver = check_import(dep)
        p(f"  {dep}: {ver}" if ok else f"  {dep}: FAILED — {ver}")
        if not ok:
            errors.append(dep)
    p()

    # --- Voice files ---
    p("--- [9/9] Voice files ---")
    base = os.path.dirname(os.path.abspath(__file__))
    voice_files = [
        ("voices/olena.wav", "General voice (default)"),
        ("voices/board/cso.wav", "CSO board voice"),
        ("voices/board/cfo.wav", "CFO board voice"),
        ("voices/board/cto.wav", "CTO board voice"),
        ("voices/board/ceo.wav", "CEO board voice"),
    ]
    for rel_path, label in voice_files:
        full = os.path.join(base, rel_path.replace("/", os.sep))
        if os.path.exists(full):
            size_kb = os.path.getsize(full) / 1024
            p(f"  [OK]   {rel_path} ({size_kb:.0f} KB)")
        else:
            p(f"  [MISS] {rel_path} — {label}")
    p()

    # --- pip list of key packages ---
    p("--- Installed versions (pip) ---")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=columns"],
            capture_output=True, text=True, timeout=30
        )
        keys = [
            "torch", "torchaudio", "tts", "numpy", "scipy", "librosa",
            "soundfile", "faster-whisper", "fastapi", "httpx", "edge-tts",
            "pydantic", "coqui", "trainer", "coqpit", "uvicorn",
        ]
        for line in result.stdout.splitlines():
            if any(line.lower().startswith(k.lower()) for k in keys):
                p(f"  {line.strip()}")
    except Exception as e:
        p(f"  Could not get pip list: {e}")
    p()

    # --- OpenRouter key ---
    p("--- Environment ---")
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if or_key:
        p(f"  OPENROUTER_API_KEY: set ({len(or_key)} chars)")
    else:
        p("  OPENROUTER_API_KEY: NOT SET (board will use config.py fallback)")
    p()

    # --- Summary ---
    p("=" * 60)
    if errors:
        p(f"  ERRORS: {len(errors)} failed — {', '.join(errors)}")
        p("  Server will NOT start.")
    else:
        p("  All checks passed.")
    p("=" * 60)
    p()

    if _log_file:
        _log_file.close()

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(run())
