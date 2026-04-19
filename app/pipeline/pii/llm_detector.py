"""LLM-based PII detector via llama-cpp-python + Qwen2.5-3B Instruct Q4_K_M GGUF.

Strategy:
- Lazy init of the Llama instance so import is cheap.
- If model load fails we log and return [] (service keeps running on regex+natasha).
- Prompt instructs the model to return a JSON array of {"type", "value"} objects
  whose ``value`` is a substring of the input. We map those back to char spans.
"""

from __future__ import annotations

import json
import logging
import os
import re
from threading import Lock
from typing import Any

from app.config import get_settings
from app.pipeline.pii.types import PIISpan, PIIType

log = logging.getLogger(__name__)

SOURCE = "llm"
CONFIDENCE = 0.8

_SYSTEM_PROMPT = (
    "Ты извлекаешь персональные данные (PII) из русскоязычного транскрипта "
    "телефонного разговора. Верни СТРОГО JSON-объект вида "
    '{"items": [{"type": "<TYPE>", "value": "<точная подстрока из текста>"}]}. '
    "Допустимые type: PASSPORT, INN, SNILS, PHONE, EMAIL, ADDRESS, PERSON. "
    "PERSON — это имена и фамилии людей (например: Иван Петров, Анна Сидорова). "
    "ADDRESS — города, улицы, адреса. PHONE — номера телефонов в любой форме, "
    "включая словесную (например: плюс семь девятьсот двадцать семь...). "
    "EMAIL — адреса e-mail. INN — 10 или 12 цифр ИНН. SNILS — номер СНИЛС. "
    "PASSPORT — серия и номер паспорта. "
    "КРИТИЧНО: value ДОЛЖНА быть точной посимвольной подстрокой исходного "
    "текста. НЕ переводи числовые слова в цифры. Если в тексте написано "
    "«плюс семь девятьсот», то value = «плюс семь девятьсот», а НЕ «+7 900». "
    "Если PII нет — {\"items\": []}. "
    "Примеры:\n"
    'Вход: "Меня зовут Иван Петров, телефон +7 900 123 45 67."\n'
    'Выход: {"items": [{"type": "PERSON", "value": "Иван Петров"}, '
    '{"type": "PHONE", "value": "+7 900 123 45 67"}]}\n'
    'Вход: "Мой номер телефона плюс семь девятьсот двадцать семь четыреста '
    'пять девяносто один двадцать два."\n'
    'Выход: {"items": [{"type": "PHONE", "value": "плюс семь девятьсот '
    'двадцать семь четыреста пять девяносто один двадцать два"}]}\n'
    'Вход: "Здравствуйте, я хотел уточнить расписание."\n'
    'Выход: {"items": []}'
)

_VALID_TYPES = {t.value for t in PIIType}


class LLMDetector:
    """Holds a single Llama instance (thread-safe lazy init)."""

    def __init__(self) -> None:
        self._llm: Any | None = None
        self._load_failed = False
        self._lock = Lock()

    def _ensure_loaded(self) -> Any | None:
        if self._llm is not None or self._load_failed:
            return self._llm
        with self._lock:
            if self._llm is not None or self._load_failed:
                return self._llm

            # The prebuilt llama-cpp-python cu124 wheel for Windows links
            # against cudart64_12.dll / cublas64_12.dll, but those live in
            # pip-installed nvidia subpackages whose bin dirs are not on
            # PATH. Register them with the DLL loader and preload in the
            # right order so llama.dll can resolve its CUDA backend.
            _preload_cuda_dlls()

            try:
                from llama_cpp import Llama  # type: ignore
            except Exception as exc:  # pragma: no cover - depends on env
                log.warning("llama-cpp-python import failed: %s", exc)
                self._load_failed = True
                return None

            settings = get_settings()
            model_path = settings.llm_gguf_path
            if not os.path.isabs(model_path):
                # Resolve relative to project root (cwd of uvicorn / pytest).
                model_path = os.path.abspath(model_path)
            if not os.path.exists(model_path):
                log.warning("LLM GGUF not found at %s; detector disabled", model_path)
                self._load_failed = True
                return None

            try:
                self._llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,  # offload everything possible
                    n_ctx=4096,
                    verbose=False,
                )
                log.info("LLM loaded from %s", model_path)
            except Exception as exc:
                log.warning("LLM load failed: %s", exc)
                self._llm = None
                self._load_failed = True
            return self._llm

    def detect(self, text: str) -> list[PIISpan]:
        llm = self._ensure_loaded()
        if llm is None or not text.strip():
            return []

        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as exc:
            log.warning("LLM inference failed: %s", exc)
            return []

        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            log.warning("LLM response has unexpected shape: %s", exc)
            return []

        items = _parse_items(content)
        return _items_to_spans(items, text)


_cuda_preloaded = False


def _preload_cuda_dlls() -> None:
    """On Windows, make sure cudart / cublas DLLs shipped via pip nvidia
    packages are loadable before we import llama_cpp."""
    global _cuda_preloaded
    if _cuda_preloaded or os.name != "nt":
        _cuda_preloaded = True
        return

    import ctypes
    import sysconfig

    site_packages = sysconfig.get_paths().get("purelib")
    if not site_packages:
        _cuda_preloaded = True
        return
    nvidia = os.path.join(site_packages, "nvidia")
    if not os.path.isdir(nvidia):
        _cuda_preloaded = True
        return

    # Register bin dirs so dependent DLLs can be located.
    for sub in ("cuda_runtime", "cublas", "cuda_nvrtc", "cudnn"):
        p = os.path.join(nvidia, sub, "bin")
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except OSError as exc:
                log.debug("add_dll_directory(%s) failed: %s", p, exc)

    # Preload in dependency order; ignore any that don't exist.
    for rel in (
        r"cuda_runtime\bin\cudart64_12.dll",
        r"cublas\bin\cublasLt64_12.dll",
        r"cublas\bin\cublas64_12.dll",
    ):
        full = os.path.join(nvidia, rel)
        if os.path.exists(full):
            try:
                ctypes.WinDLL(full)
            except OSError as exc:
                log.debug("Preload %s failed: %s", full, exc)

    _cuda_preloaded = True


# Module-level singleton so callers don't reload the model.
_detector: LLMDetector | None = None


def get_llm_detector() -> LLMDetector:
    global _detector
    if _detector is None:
        _detector = LLMDetector()
    return _detector


def detect_llm(text: str) -> list[PIISpan]:
    """Convenience wrapper around the singleton detector."""
    return get_llm_detector().detect(text)


# --- Internal helpers -------------------------------------------------------


def _parse_items(content: str) -> list[dict[str, str]]:
    """Accept either {"items": [...]} or a bare [...] array."""
    if not content:
        return []
    content = content.strip()
    # Some models wrap output in ```json ... ```; strip fences.
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        log.warning("LLM returned invalid JSON: %s", exc)
        return []

    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        for key in ("items", "pii", "data", "result"):
            if isinstance(parsed.get(key), list):
                return [x for x in parsed[key] if isinstance(x, dict)]
        # Some models return a single object — wrap it.
        if parsed.get("type") and parsed.get("value"):
            return [parsed]
    return []


def _items_to_spans(items: list[dict[str, str]], text: str) -> list[PIISpan]:
    spans: list[PIISpan] = []
    # To avoid duplicate matches when the same substring repeats, track the
    # offset we've already consumed *per exact value*.
    offsets: dict[str, int] = {}

    for item in items:
        raw_type = str(item.get("type", "")).strip().upper()
        value = str(item.get("value", "")).strip()
        if not value or raw_type not in _VALID_TYPES:
            continue

        search_from = offsets.get(value, 0)
        idx = text.find(value, search_from)
        if idx < 0:
            # Value is not a literal substring of the input — skip.
            continue

        start = idx
        end = idx + len(value)
        offsets[value] = end

        spans.append(
            PIISpan(
                type=PIIType(raw_type),
                text=value,
                start_char=start,
                end_char=end,
                source=SOURCE,
                confidence=CONFIDENCE,
            )
        )
    return spans
