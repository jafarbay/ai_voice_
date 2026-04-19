"""Скачивает все модели для проекта.

Запускать ОДИН РАЗ после ``pip install -r requirements.txt``.

Что делает:
  1. Скачивает GGUF-файл Qwen 2.5 3B Instruct (Q4_K_M) в ``settings.models_dir``,
     если его ещё нет. Используется llm_detector'ом через llama-cpp-python.
  2. Пред-кеширует Whisper-модель (по умолчанию ``large-v3-turbo``) —
     faster-whisper сам кладёт веса в ``~/.cache/huggingface``. Здесь
     создаём объект на CPU с маленьким compute_type, чтобы скачать веса,
     не занимая VRAM. Реальная транскрипция потом пойдёт на GPU.

Запуск:
    python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as ``python scripts/download_models.py`` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from faster_whisper import WhisperModel  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from app.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()

    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Qwen GGUF --------------------------------------------------------
    gguf_filename = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    gguf_path = models_dir / gguf_filename
    print(f"[1/2] Qwen 2.5 3B Instruct Q4_K_M -> {gguf_path}")
    if gguf_path.exists():
        size_mb = gguf_path.stat().st_size / (1024 * 1024)
        print(f"      already present ({size_mb:.1f} MB), skipping.")
    else:
        print("      downloading from Hugging Face (~1.9 GB)...")
        hf_hub_download(
            repo_id="bartowski/Qwen2.5-3B-Instruct-GGUF",
            filename=gguf_filename,
            local_dir=str(models_dir),
            token=settings.huggingface_token or None,
        )
        print(f"      done: {gguf_path}")

    # 2. Whisper pre-cache ------------------------------------------------
    print(f"[2/2] Pre-caching Whisper model '{settings.whisper_model}'...")
    # device=cpu + int8 — дёшево по памяти; цель не инференс, а скачать веса.
    model = WhisperModel(
        settings.whisper_model, device="cpu", compute_type="int8"
    )
    del model
    print("      done.")

    print("\nAll models ready. You can now start the server:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
