FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    DATABASE_PATH=/app/data/app.db

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg libsndfile1 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

COPY requirements.txt ./
# PyPI has no prebuilt Linux wheel for llama-cpp-python on our Python +
# abi combination, so `pip install -r requirements.txt` would fall back
# to compiling from source (fails without the full CUDA toolkit in the
# image). Install the prebuilt CUDA 12.4 wheel FIRST — with --no-deps so
# we don't look for numpy/etc in that single-package index. Then the
# main requirements.txt install sees llama-cpp-python already satisfies
# the constraint and skips it. The result: LLM runs on GPU via
# n_gpu_layers=-1 and no compiler is needed.
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir --no-deps \
        --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
        "llama-cpp-python>=0.2.70,<0.4" && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /app/data /app/models /app/examples

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
