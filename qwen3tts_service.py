# qwen3tts_service.py — subprocess TTS server (runs in qwen3tts_env Python)
#
# Запуск: E:/project/qwen3tts_env/Scripts/python.exe qwen3tts_service.py
# Протокол stdin/stdout JSON lines:
#   → {"text": "...", "speaker": "Ryan", "language": "Russian", "instruct": "..."}
#   ← {"ok": true, "url": "/static/audio/xxx.wav", "gen_sec": 2.1, "rtf": 0.42}
#   ← {"ok": false, "error": "..."}
# Первая строка после старта: {"status": "ready"}
# Выход: {"cmd": "quit"} или EOF stdin

import os
import sys
import json
import logging

# ─── ENV ────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault("HF_HOME",              os.path.join(_PROJECT, "Qwen3TTS"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTHONIOENCODING",     "utf-8")

# ─── PATH ────────────────────────────────────────────────────────────────────
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ─── IMPORT MODEL MODULE ─────────────────────────────────────────────────────
# Import first — only registers the logger, no model loading yet.
from qwen3tts_manager import get_qwen3tts_manager  # noqa: E402

# Remove ALL StreamHandlers that write to stdout from every logger.
# qwen3tts_manager sets up logging.StreamHandler(sys.stdout) at module level;
# we don't want that polluting our JSON IPC pipe.
def _remove_stdout_handlers():
    all_loggers = [logging.getLogger()] + [
        logging.getLogger(name)
        for name in logging.Logger.manager.loggerDict
        if isinstance(logging.Logger.manager.loggerDict.get(name), logging.Logger)
    ]
    for lg in all_loggers:
        for h in list(lg.handlers):
            # Remove StreamHandlers that are NOT file-based
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                lg.removeHandler(h)

_remove_stdout_handlers()

# Redirect sys.stdout → stderr during model load to swallow any tqdm/HF progress
# bars that write directly to sys.stdout (not via the logging system).
# We keep a reference to the real stdout so we can restore it before the JSON loop.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DEFAULT_INSTRUCT = (
    "Speak with genuine emotion and natural intonation. "
    "Be warm, confident, and expressive. "
    "Emphasize key points, vary tone naturally, add feeling to the words."
)


def main():
    mgr = get_qwen3tts_manager()
    # Preload model — logs go to qwen3tts.log; tqdm/HF bars → stderr pipe
    mgr._ensure_loaded("custom")

    # Trigger CUDA graph warmup NOW while stdout is still redirected to stderr.
    # faster_qwen3_tts prints "Warming up predictor...", "Capturing CUDA graph..."
    # directly to sys.stdout on the first inference — those must NOT reach the
    # JSON IPC pipe.  After this call the graphs are captured and won't print again.
    try:
        mgr.synthesize_custom(text="warmup", speaker="Ryan",
                               language="Russian", instruct="")
    except Exception:
        pass  # warmup failure is non-fatal; real requests will still work

    # Restore real stdout BEFORE any print() so JSON lines reach the parent.
    sys.stdout = _real_stdout

    # Signal parent: ready for JSON requests
    print(json.dumps({"status": "ready"}), flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"JSON decode: {e}"}), flush=True)
            continue

        if req.get("cmd") == "quit":
            break

        text = req.get("text", "").strip()
        if not text:
            print(json.dumps({"ok": False, "error": "empty text"}), flush=True)
            continue

        try:
            result = mgr.synthesize_custom(
                text     = text,
                speaker  = req.get("speaker",  "Ryan"),
                language = req.get("language", "Russian"),
                instruct = req.get("instruct", DEFAULT_INSTRUCT),
            )
            print(json.dumps({"ok": True, **result}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
