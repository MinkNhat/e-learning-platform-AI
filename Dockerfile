# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Compiler tools are needed only while Python dependencies are being built.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install --prefer-binary -r requirements.txt


FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser

# FlashRank/ONNX Runtime requires the OpenMP runtime, but not the compiler toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home \
        --home-dir /home/appuser --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app/ ./app/

# FlashRank creates this cache lazily on the first reranking request.
RUN mkdir -p /app/.cache/flashrank \
    && chown -R appuser:appuser /app/.cache

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/', timeout=3).close()"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
