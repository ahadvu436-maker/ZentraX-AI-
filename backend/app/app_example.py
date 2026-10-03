"""Example FastAPI wiring. Run with: uvicorn app_example:app --reload

pip install fastapi uvicorn httpx
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from local_store import LocalStore, LocalStoreError
from network_monitor import MonitorConfig, NetworkMonitor, Probe
from sync_manager import HttpCloudClient, PermanentSyncError, SyncManager

logging.basicConfig(level=logging.INFO)

CLOUD_URL = os.getenv("ZENTRAX_CLOUD_URL", "https://api.example.com")
CLOUD_HOST = CLOUD_URL.split("://", 1)[-1].split("/", 1)[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    monitor = NetworkMonitor(
        MonitorConfig(probes=(Probe(CLOUD_HOST, 443), Probe("1.1.1.1", 443)))
    )
    store = LocalStore("data/zentrax_local.db")
    client = HttpCloudClient(CLOUD_URL, api_key=os.getenv("ZENTRAX_API_KEY"))
    sync = SyncManager(monitor, store, client)

    await store.open()
    await monitor.start()
    await sync.start()
    app.state.sync = sync
    try:
        yield
    finally:
        await sync.stop()
        await monitor.stop()
        await client.aclose()
        await store.close()


app = FastAPI(title="ZentraX AI", lifespan=lifespan)


class EventIn(BaseModel):
    operation: str = "events"
    payload: dict[str, Any]
    idempotency_key: str | None = None


@app.post("/submit")
async def submit(body: EventIn):
    sync: SyncManager = app.state.sync
    try:
        result = await sync.submit(body.operation, body.payload, body.idempotency_key)
    except PermanentSyncError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except LocalStoreError as exc:
        raise HTTPException(status_code=503, detail=f"Local storage failure: {exc}")
    return {"status": result.status, "idempotency_key": result.idempotency_key}


@app.get("/sync/status")
async def sync_status():
    return await app.state.sync.status()


@app.post("/sync/now")
async def sync_now():
    await app.state.sync.sync_now()
    return await app.state.sync.status()
