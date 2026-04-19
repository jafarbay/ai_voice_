# Voice Anonymization Service

Сервис анонимизации русскоязычных голосовых записей:
- распознаёт речь (faster-whisper, word-level таймстампы),
- размечает спикеров (`pyannote.audio` 3.1),
- находит ПДн (паспорт, ИНН, СНИЛС, телефон, email, адрес, имя — regex + Natasha + локальная LLM),
- заглушает их бипом 1 кГц в аудио и `[PII:TYPE]` в транскрипте,
- отдаёт оригинал и «чистую» версию + лог событий.

Стек: FastAPI · faster-whisper · pyannote.audio · llama-cpp-python (Qwen 2.5 3B Q4) · Vue 3 + Vite. Работает локально в Docker с GPU (NVIDIA).

---

## Быстрый старт через Docker (рекомендуется)

Требуется:
- Docker Engine + NVIDIA drivers + `nvidia-container-toolkit` (Linux) или Docker Desktop с WSL2 GPU (Windows).
- HuggingFace token в `.env` (приняты условия для `pyannote/speaker-diarization-3.1`).
- Файл модели `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf`.

```bash
# 1. Клонируем
git clone git@github.com:jafarbay/ai_voice_.git
cd ai_voice_

# 2. Готовим .env
cp .env.example .env
# отредактировать и вписать HUGGINGFACE_TOKEN=hf_...

# 3. Качаем LLM (~1.9 GB) — только первый раз
mkdir -p models
python scripts/download_models.py

# 4. Поднимаем
docker compose up -d --build
```

Фронт: <http://localhost:5180> · API: <http://localhost:8000/docs>

### Публичная ссылка (опционально)

Чтобы поделиться демо с внешним миром без открытия портов — поднимаем Cloudflare Quick Tunnel:

```bash
docker compose --profile share up -d
docker compose logs tunnel | grep -E "trycloudflare\.com"
```

Вывод содержит одноразовый URL `https://*.trycloudflare.com` — по нему откроется тот же фронт.

### Остановить

```bash
docker compose down
```

---

## Что получается на выходе

Для каждой загруженной записи сервис делает:
- `input.wav` — исходное аудио в 16 kHz mono (хранится только на сервере, через API/UI можно прослушать),
- `redacted.wav` — то же аудио с биппированными PII,
- `transcript_full.json` — полная транскрипция со спикерами (`Спикер 1`, `Спикер 2`, …) и PII-пометками (доступна через API/UI),
- `transcript_redacted.json` — транскрипция с `[PII:TYPE]` вместо ПДн,
- `events.jsonl` — журнал найденных ПДн (тип, таймстампы, источник, уверенность).

На странице задачи в UI есть кнопка **⬇ Скачать ZIP** — собирает только анонимизированные артефакты (`redacted.wav`, `transcript_redacted.json`, `events.jsonl`) в архив `<job_id>.zip`. Исходное аудио и полный транскрипт в ZIP **не включаются**, чтобы ПДн не покидали сервер.

---

## Что детектируется

- **Паспорт** — серия + номер (4 + 6 цифр)
- **ИНН** — 10 или 12 цифр с валидацией контрольной суммы
- **СНИЛС** — 11 цифр
- **Телефоны** — российские форматы (+7, 8, с разделителями и без), включая номера, произнесённые словами
- **Email**
- **Адреса** — через Natasha (тег `LOC`)
- **Имена** — Natasha `PER` + LLM Qwen

---

## API

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/jobs` | загрузить файл (`multipart/form-data`, поле `file`). Возвращает `job_id`, пайплайн в фоне. Лимит 200 MB; `.wav .mp3 .m4a .ogg .flac .opus .webm` |
| `GET` | `/jobs` | последние 50 задач |
| `GET` | `/jobs/{id}` | статус и метаданные |
| `GET` | `/jobs/{id}/transcript?version=full\|redacted` | JSON-транскрипт |
| `GET` | `/jobs/{id}/audio?version=original\|redacted` | WAV |
| `GET` | `/jobs/{id}/events` | лог PII-событий |
| `GET` | `/jobs/{id}/export` | ZIP со всеми артефактами |
| `GET` | `/health` | `{"status": "ok"}` |

### Пример через curl

```bash
curl -s -X POST -F "file=@path/to/audio.wav" http://localhost:8000/jobs
# -> {"job_id":"abc123...","status":"queued","filename":"audio.wav"}

JOB=abc123...
while :; do
  S=$(curl -s http://localhost:8000/jobs/$JOB | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo $S
  [[ "$S" == "done" || "$S" == "failed" ]] && break
  sleep 1
done

curl -o artifacts.zip http://localhost:8000/jobs/$JOB/export
```

---

## Локальная разработка (без Docker)

Требуется Python 3.12+, `ffmpeg` в `PATH`, Node 20+.

### Backend

```bash
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt

# GPU (Windows + CUDA 12.x): поставить колёса
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12
pip install llama-cpp-python --prefer-binary \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

cp .env.example .env    # вписать HUGGINGFACE_TOKEN
python scripts/download_models.py   # скачать Qwen GGUF

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# -> http://localhost:5180
```

---

## Тесты

```bash
# Быстрые юнит-тесты
python -m pytest tests/ -v -m "not integration"

# Полный прогон (включая LLM и реальный Whisper)
python -m pytest tests/ -v
```

Маркеры из `pytest.ini`:
- `slow` — загружают тяжёлые модели (LLM, Whisper);
- `integration` — выполняют полный end-to-end пайплайн.

---

## Переменные окружения (`.env.example`)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `HUGGINGFACE_TOKEN` | — | токен HF (нужен для pyannote) |
| `WHISPER_MODEL` | `large-v3-turbo` | id модели faster-whisper |
| `LLM_GGUF_PATH` | `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` | путь к GGUF |
| `DATA_DIR` | `./data` | каталог задач и БД |
| `MODELS_DIR` | `./models` | каталог GGUF и кешей |
| `DATABASE_PATH` | `./data/app.db` | SQLite-файл |
| `ENABLE_DIARIZATION` | `true` | включить pyannote |

---

## Структура

```
.
├── app/                    FastAPI backend
│   ├── main.py             entry point, CORS, lifespan
│   ├── config.py           pydantic-settings
│   ├── db.py               SQLite схема и CRUD
│   ├── schemas.py          pydantic-модели API
│   ├── api/jobs.py         REST-роуты
│   ├── storage/files.py    JobPaths: data/<job_id>/*
│   └── pipeline/
│       ├── orchestrator.py end-to-end пайплайн
│       ├── audio_prep.py   ffmpeg → 16 kHz mono PCM
│       ├── stt.py          faster-whisper + word timestamps
│       ├── diarization.py  pyannote 3.1 (singleton, с фоллбеком)
│       ├── align.py        привязка слов к спикерам
│       ├── redaction.py    1 kHz beep с fade
│       ├── events.py       EventLogger → events.jsonl + SQLite
│       ├── export.py       ZIP со всеми артефактами
│       └── pii/            regex + Natasha + LLM + merger
├── frontend/               Vue 3 + Vite + TypeScript SPA
│   ├── src/
│   ├── Dockerfile          multi-stage build → nginx:alpine
│   └── nginx.conf          SPA + прокси на backend
├── tests/                  pytest, 60+ тестов
├── scripts/                download_models, gen_sample, quick_test
├── Dockerfile              CUDA runtime + Python 3.11 + deps
├── docker-compose.yml      backend + frontend + (optional) tunnel
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Лицензия

MIT (see `LICENSE` или используйте по своему усмотрению).
