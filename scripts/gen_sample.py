"""Generate a small Russian TTS WAV fixture for tests.

Uses `pyttsx3` (Windows SAPI5, offline). A Russian voice is expected to be
available on Windows 10+. The resulting file is written as
`tests/fixtures/sample.wav`. The downstream `audio_prep` step will normalize
the file to 16 kHz mono PCM16 before feeding it to Whisper, so we don't have
to be strict about SAPI's native sample format.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyttsx3

TEXT = (
    "Меня зовут Иван Петров, "
    "мой телефон плюс семь девятьсот сто двадцать три сорок пять шестьдесят семь. "
    "Номер ИНН один два три четыре пять шесть семь восемь девять ноль. "
    "Электронная почта test@example.com."
)


def pick_russian_voice(engine: "pyttsx3.Engine") -> str | None:
    """Return the id of a Russian SAPI5 voice if one is installed."""
    voices = engine.getProperty("voices") or []
    for v in voices:
        name = (getattr(v, "name", "") or "").lower()
        langs = getattr(v, "languages", []) or []
        lang_blob = " ".join(
            (lang.decode("utf-8", errors="ignore") if isinstance(lang, bytes) else str(lang)).lower()
            for lang in langs
        )
        if "russian" in name or "русск" in name or "ru" in lang_blob or "1049" in lang_blob:
            return v.id
    # Fallback: look for typical Windows Russian voice ids.
    for v in voices:
        vid = (getattr(v, "id", "") or "").lower()
        if "irina" in vid or "pavel" in vid or "ru-ru" in vid or "ru_ru" in vid:
            return v.id
    return None


def main() -> int:
    out_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine = pyttsx3.init()  # defaults to SAPI5 on Windows

    voice_id = pick_russian_voice(engine)
    if voice_id is not None:
        print(f"[gen_sample] using Russian voice: {voice_id}")
        engine.setProperty("voice", voice_id)
    else:
        voices = engine.getProperty("voices") or []
        print(
            "[gen_sample] WARNING: no Russian voice found, available voices:",
            [getattr(v, "name", "?") for v in voices],
        )

    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    engine.save_to_file(TEXT, str(out_path))
    engine.runAndWait()
    engine.stop()

    if not out_path.exists() or out_path.stat().st_size == 0:
        print(f"[gen_sample] FAIL: {out_path} not created or empty", file=sys.stderr)
        return 1

    print(f"[gen_sample] wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
