# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder

ENV PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    VIRTUAL_ENV=/app/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH"

# System deps required to build llama-cpp-python from sdist when wheels are unavailable
# (gcc/g++, cmake, ninja; OpenBLAS is optional but improves performance if enabled)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and prepare venv
RUN python -m venv $VIRTUAL_ENV && python -m pip install --upgrade pip

# Prefer binary wheels where possible; still allow source build if needed
# (we don't preinstall torch here; nl_agent doesn't require it)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# ---------- Runtime image ----------
FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH"

# Runtime needs OpenBLAS shared libs if llama-cpp was built with BLAS
RUN apt-get update && apt-get install -y --no-install-recommends     libopenblas0     libgomp1 
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/venv /app/venv
COPY --from=builder /app /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
