# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src/ /app/src/
# Include your *checked-in* tiny sample
COPY data/ /app/data/

RUN python -m pip install --upgrade pip && \
    pip install -e .

# Optional: drop privileges
RUN useradd -m -u 1000 appuser
USER appuser

ENTRYPOINT ["python", "-m", "myproj.train"]

