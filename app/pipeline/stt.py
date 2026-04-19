"""Speech-to-text wrapper around `faster-whisper`.

Exposes a singleton transcriber that loads the Whisper model lazily, prefers
GPU + float16, and falls back to CPU + int8 if CUDA initialization fails
(e.g. missing cuDNN). Word-level timestamps and VAD filtering are always on.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from threading import Lock

from app.config import get_settings
from app.schemas import TranscriptResult, Word

log = logging.getLogger(__name__)

_DEFAULT_LANGUAGE = "ru"

_CUDA_DLL_DIRS_REGISTERED = False


def _register_cuda_dll_dirs() -> None:
    """On Windows, make sure cuBLAS / cuDNN DLLs from pip-installed NVIDIA
    wheels are reachable by ctranslate2's dynamic loader.

    Without this, ``faster-whisper`` may load fine but fail at encode time with
    ``Library cublas64_12.dll is not found or cannot be loaded`` because
    ctranslate2 only ships cuDNN (not cuBLAS), and the system CUDA Toolkit is
    not installed.
    """
    global _CUDA_DLL_DIRS_REGISTERED
    if _CUDA_DLL_DIRS_REGISTERED or sys.platform != "win32":
        return

    # nvidia-*-cu12 wheels install DLLs under <site-packages>/nvidia/<lib>/bin/.
    # `nvidia` is a namespace package, so it doesn't have __file__ — use __path__.
    try:
        import nvidia  # type: ignore[import-not-found]
    except ImportError:
        _CUDA_DLL_DIRS_REGISTERED = True
        return

    nvidia_paths = [Path(p).resolve() for p in getattr(nvidia, "__path__", [])]

    candidates: list[Path] = []
    for nvidia_root in nvidia_paths:
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            candidates.append(nvidia_root / sub / "bin")

    for p in candidates:
        if p.is_dir():
            try:
                os.add_dll_directory(str(p))
            except (OSError, AttributeError):
                # Older Python or weird filesystem — fall back to PATH.
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
            # Also extend PATH so child processes / late-loaded DLLs see it.
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

    _CUDA_DLL_DIRS_REGISTERED = True


class WhisperTranscriber:
    """Lazy wrapper around `faster_whisper.WhisperModel`."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        preferred_device: str = "cuda",
        preferred_compute_type: str = "float16",
        fallback_device: str = "cpu",
        fallback_compute_type: str = "int8",
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.whisper_model
        self.preferred_device = preferred_device
        self.preferred_compute_type = preferred_compute_type
        self.fallback_device = fallback_device
        self.fallback_compute_type = fallback_compute_type

        self._model = None
        self._device: str | None = None
        self._compute_type: str | None = None
        self._lock = Lock()

    @property
    def device(self) -> str | None:
        """The device the loaded model is using (`cuda` or `cpu`)."""
        return self._device

    @property
    def compute_type(self) -> str | None:
        """The compute type the loaded model is using (`float16`, `int8`, ...)."""
        return self._compute_type

    def _load(self):
        """Load the model with GPU fallback on failure. Idempotent."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            _register_cuda_dll_dirs()
            from faster_whisper import WhisperModel  # local import, heavy

            try:
                log.info(
                    "loading whisper model %s on %s/%s",
                    self.model_name, self.preferred_device, self.preferred_compute_type,
                )
                self._model = WhisperModel(
                    self.model_name,
                    device=self.preferred_device,
                    compute_type=self.preferred_compute_type,
                )
                self._device = self.preferred_device
                self._compute_type = self.preferred_compute_type
            except Exception as exc:  # noqa: BLE001 - we want any CUDA/cuDNN/OOM failure
                log.warning(
                    "whisper GPU load failed (%s), falling back to %s/%s",
                    exc, self.fallback_device, self.fallback_compute_type,
                )
                self._model = WhisperModel(
                    self.model_name,
                    device=self.fallback_device,
                    compute_type=self.fallback_compute_type,
                )
                self._device = self.fallback_device
                self._compute_type = self.fallback_compute_type

            return self._model

    def transcribe(
        self,
        wav_path: Path,
        *,
        language: str = _DEFAULT_LANGUAGE,
        vad_filter: bool = True,
    ) -> TranscriptResult:
        """Transcribe a WAV file into a `TranscriptResult` with word timestamps."""
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"audio not found: {wav_path}")

        model = self._load()

        segments, info = model.transcribe(
            str(wav_path),
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True,
        )

        words: list[Word] = []
        text_parts: list[str] = []
        for seg in segments:
            # Collect the segment text as a safety net for the full transcript,
            # even when the model returns empty word lists (shouldn't happen
            # with word_timestamps=True, but be defensive).
            if seg.text:
                text_parts.append(seg.text)

            seg_words = getattr(seg, "words", None) or []
            for w in seg_words:
                if w.start is None or w.end is None:
                    continue
                words.append(
                    Word(
                        word=w.word,
                        start=float(w.start),
                        end=float(w.end),
                        probability=float(getattr(w, "probability", 0.0) or 0.0),
                    )
                )

        # Prefer reconstructing text from words (preserves exact spacing that
        # downstream char-offset code will rely on); fall back to segment text.
        if words:
            text = "".join(w.word for w in words).strip()
        else:
            text = "".join(text_parts).strip()

        return TranscriptResult(
            language=str(getattr(info, "language", language) or language),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            text=text,
            words=words,
        )


_transcriber: WhisperTranscriber | None = None


def get_transcriber() -> WhisperTranscriber:
    """Return a process-wide singleton transcriber (model loaded on first use)."""
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber


def transcribe(wav_path: Path, **kwargs) -> TranscriptResult:
    """Convenience wrapper that uses the shared singleton transcriber."""
    return get_transcriber().transcribe(wav_path, **kwargs)
