"""
main.py

Application entry point for the ZentraX AI backend.

Responsibilities:
    - Initialize the FastAPI app with metadata (title, version, docs config).
    - Configure CORS so the Next.js frontend can call the API.
    - Wire up domain routers (chat, user, toolkit) under a versioned prefix.
    - Expose a lightweight health check endpoint for load balancers / Docker.
    - Manage startup/shutdown of shared resources (DB engine, etc.) via lifespan.

Run locally:
    uvicorn app.main:app --reload --port 8000

Run in production (see infrastructure/docker-compose.yml):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import os
import sys

# রেন্ডার সার্ভারের জন্য সঠিক পাথ অ্যাডজাস্টমেন্ট
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
backend_dir = os.path.dirname(app_dir)

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.api.routes import chat, toolkit, user

logger = logging.getLogger("zentrax.main")

settings = get_settings()


# -----------------------------------------------------------------------------
# Lifespan: startup / shutdown of shared resources
# -----------------------------------------------------------------------------
# Anything expensive to create (DB connection pool, cache clients, provider
# SDKs) should be initialized once here and attached to `app.state`, rather
# than re-created per-request.

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup env=%s", settings.environment)

    # Example: initialize a shared async DB engine / session factory here.
    # from app.core.database import engine
    # app.state.db_engine = engine

    yield  # --- application runs while suspended here ---

    logger.info("shutdown")
    # Example: dispose of the engine / close pooled connections cleanly.
    # await app.state.db_engine.dispose()


# -----------------------------------------------------------------------------
# App initialization
# -----------------------------------------------------------------------------

app = FastAPI(
    title="ZentraX AI API",
    description=(
        "Backend API for ZentraX AI — a privacy-first AI platform. "
        "Handles conversation orchestration, authentication, and modular AI tooling."
    ),
    version="1.0.0",
    contact={"name": "ZentraX AI", "url": "https://zentrax.ai"},
    license_info={"name": "Proprietary"},
    # Hide interactive docs in production to reduce surface area / avoid
    # leaking internal route structure; keep them on for dev/staging.
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    openapi_url="/openapi.json" if settings.environment != "production" else None,
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# CORS
# -----------------------------------------------------------------------------
# `allow_origins` comes from settings (env-driven), never hardcoded or "*" in
# production — an open CORS policy combined with cookie-based auth is a
# classic cross-site data exposure risk. Configure the real frontend
# origin(s) via the ALLOWED_ORIGINS environment variable.

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)

# -----------------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------------
# Each domain owns its own router (see app/api/routes/*.py); main.py only
# wires them together under a shared, versioned prefix so breaking changes
# can be introduced behind /api/v2 later without disrupting /api/v1 clients.

API_PREFIX = "/api/v1"

app.include_router(chat.router, prefix=f"{API_PREFIX}/chat", tags=["Chat"])
app.include_router(user.router, prefix=f"{API_PREFIX}/users", tags=["Users"])
app.include_router(toolkit.router, prefix=f"{API_PREFIX}/toolkit", tags=["Toolkit"])


# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
# Kept intentionally minimal and public (no auth, no DB round-trip) so it's
# fast and reliable for Docker healthchecks / load balancer probes. Add a
# separate `/health/deep` endpoint later if you need dependency checks
# (DB, Redis, provider reachability) for more thorough monitoring.

@app.get("/", tags=["Health"], summary="Root health check")
async def root() -> dict:
    return {
        "status": "ok",
        "service": "ZentraX AI API",
        "version": app.version,
    }


@app.get("/health", tags=["Health"], summary="Liveness/readiness probe")
async def health() -> dict:
    return {"status": "healthy"}


# -----------------------------------------------------------------------------
# Global exception handling
# -----------------------------------------------------------------------------
# Ensures unhandled exceptions return a consistent, non-leaky JSON shape
# instead of a raw traceback — internal error details go to logs only.

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception path=%s method=%s error_type=%s",
        request.url.path, request.method, type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )
