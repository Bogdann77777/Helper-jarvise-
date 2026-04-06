"""
setup_board_voices.py — Generate reference WAV files for Board of Directors voices.

Uses Edge TTS to create 4 distinct voice samples in voices/board/:
  - cso.wav  (en-US-GuyNeural, male)
  - cfo.wav  (en-US-JennyNeural, female)
  - cto.wav  (en-US-DavisNeural, male)
  - ceo.wav  (en-US-AriaNeural, female)

Usage: python setup_board_voices.py
"""

import asyncio
import os

VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices", "board")

VOICES = {
    "cso": {
        "voice": "en-US-GuyNeural",
        "text": (
            "As Chief Strategy Officer, I analyze market positioning, competitive "
            "dynamics, and long-term strategic opportunities. My role is to ensure "
            "the company makes informed decisions about where to compete and how to win. "
            "I evaluate market trends, assess competitive advantages, and recommend "
            "strategic pivots when necessary."
        ),
    },
    "cfo": {
        "voice": "en-US-JennyNeural",
        "text": (
            "As Chief Financial Officer, I oversee the company's financial health, "
            "cash flow management, and investment decisions. My analysis covers "
            "revenue projections, cost optimization, runway calculations, and "
            "financial risk assessment. Every recommendation I make is grounded "
            "in solid financial data and conservative estimates."
        ),
    },
    "cto": {
        "voice": "en-US-DavisNeural",
        "text": (
            "As Chief Technology Officer, I evaluate technical architecture, "
            "engineering capacity, and technology stack decisions. I assess the "
            "feasibility of proposed solutions, identify technical debt risks, "
            "and recommend the most efficient engineering approaches. My focus "
            "is on building scalable, maintainable systems."
        ),
    },
    "ceo": {
        "voice": "en-US-AriaNeural",
        "text": (
            "As the Chief Executive Officer, I synthesize insights from all board "
            "members to make final strategic decisions. I balance competing priorities, "
            "resolve conflicts between departments, and chart the course forward. "
            "My decisions consider both short-term execution and long-term vision "
            "for the company."
        ),
    },
}


async def generate_voice(role: str, config: dict):
    """Generate a single voice WAV file using Edge TTS."""
    import edge_tts

    output_path = os.path.join(VOICES_DIR, f"{role}.wav")

    print(f"  Generating {role}.wav with {config['voice']}...")

    communicate = edge_tts.Communicate(config["text"], config["voice"])

    # Edge TTS outputs MP3 by default, save to temp then convert
    mp3_path = output_path.replace(".wav", "_temp.mp3")
    await communicate.save(mp3_path)

    # Convert MP3 → WAV (16kHz mono) using torchaudio
    import torchaudio

    waveform, sr = torchaudio.load(mp3_path)
    # Resample to 22050Hz (good for XTTS reference)
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resampler(waveform)
    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    torchaudio.save(output_path, waveform, 22050)
    os.remove(mp3_path)

    print(f"  Done: {output_path}")


async def main():
    os.makedirs(VOICES_DIR, exist_ok=True)

    print(f"Generating board voices in: {VOICES_DIR}")
    print()

    for role, config in VOICES.items():
        await generate_voice(role, config)

    print()
    print("All board voices generated successfully!")
    print(f"Files: {os.listdir(VOICES_DIR)}")


if __name__ == "__main__":
    asyncio.run(main())
