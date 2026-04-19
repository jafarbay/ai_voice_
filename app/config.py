"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the voice anonymization backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # HuggingFace token (needed by pyannote.audio).
    huggingface_token: str = Field(default="", alias="HUGGINGFACE_TOKEN")

    # Enable real pyannote diarization. Default True: requires HF token + accepted terms.
    enable_diarization: bool = Field(default=True, alias="ENABLE_DIARIZATION")

    # faster-whisper model id (e.g. large-v3-turbo, medium, small).
    whisper_model: str = Field(default="large-v3-turbo", alias="WHISPER_MODEL")

    # Path to GGUF file for llama-cpp-python.
    llm_gguf_path: str = Field(
        default="models/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        alias="LLM_GGUF_PATH",
    )

    # FS layout.
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    models_dir: str = Field(default="./models", alias="MODELS_DIR")
    database_path: str = Field(default="./data/app.db", alias="DATABASE_PATH")

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir).resolve()

    @property
    def database_file(self) -> Path:
        return Path(self.database_path).resolve()


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
