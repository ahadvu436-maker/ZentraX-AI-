# backend/Dockerfile
#
# Lightweight production image for the ZentraX AI FastAPI backend.
#
# Build:
#   docker build -t zentrax-backend ./backend
# Run:
#   docker run -p 8000:8000 --env-file .env zentrax-backend
#
# Note: if any package in requirements.txt needs to compile a C extension
# (no prebuilt wheel for slim/manylinux), pip install below will fail with
# a missing-compiler error. If that happens, switch to the multi-stage
# builder pattern (compile in a build stage with build-essential, copy only
# the resulting venv into this slim runtime) rather than installing
# build-essential here permanently — it roughly doubles image size.

FROM python:3.10-slim

WORKDIR /app

# Keeps Python from writing .pyc files and from buffering stdout/stderr,
# so logs show up immediately under `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl is needed only for the HEALTHCHECK below; kept minimal and cleaned
# up in the same layer so it doesn't linger in the image's apt cache.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system app && adduser --system --ingroup app app

# Copy only the requirements file first so this layer is cached across
# builds until dependencies actually change — source edits alone won't
# force a full pip reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, owned by the non-root user that will run it.
COPY --chown=app:app . .

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1

# Single Uvicorn process — sufficient for local/dev and low-to-moderate
# traffic. For higher-concurrency production workloads, run behind
# Gunicorn with the Uvicorn worker class so you get multiple worker
# processes and proper process management:
#   CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", \
#        "--bind", "0.0.0.0:8000", "--workers", "4"]
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
