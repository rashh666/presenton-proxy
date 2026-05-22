#!/usr/bin/env python3
"""
presenton_proxy.py — v7.0.0 "GPU Manager"
Pure llama-server lifecycle manager for AMD ROCm (dual RX 9060 XT).

Role: manages llama-server on the host and exposes a transparent
OpenAI-compatible passthrough at /v1/chat/completions.

Presenton integrates by setting:
  LLM=custom
  CUSTOM_LLM_URL=http://host.docker.internal:8000/v1
  CUSTOM_MODEL=gemma3
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Optional faster JSON
try:
    import orjson as _fj

    def _jloads(s: bytes | str) -> Any:
        return _fj.loads(s)
except ImportError:
    import json as _sj

    def _jloads(s: bytes | str) -> Any:
        return _sj.loads(s)


# ---------------------------------------------------------------------------
# Settings
class Settings(BaseSettings):
    # llama-server endpoints
    reasoner_url: str = "http://localhost:8081/v1/chat/completions"
    reasoner_health_url: str = "http://localhost:8081/health"
    reasoner_base_url: str = "http://localhost:8081/v1"

    # Native process management
    llama_server_bin: str = "/home/rashid/llama.cpp/build/bin/llama-server"
    models_host_path: str = os.environ.get("MIA_MODELS_PATH", "/models")
    reasoner_model_key: str = "gemma3"
    process_startup_timeout: int = 120
    idle_timeout: int = 600
    llm_timeout: int = 600
    unified_single_model: bool = True

    # Auth (optional — set API_KEY to require Bearer token from callers)
    api_key: Optional[str] = None

    # CORS — comma-separated extra origins beyond localhost:3000 and localhost:5000
    cors_origins: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ---------------------------------------------------------------------------
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
_log = logging.getLogger("presenton-proxy")

# ---------------------------------------------------------------------------
# Metrics
_REQUESTS_TOTAL = Counter(
    "presenton_requests_total",
    "Total proxy requests",
    ["endpoint", "status"],
)
_REQUEST_LATENCY = Histogram(
    "presenton_request_duration_seconds",
    "End-to-end request latency",
    buckets=[0.1, 0.5, 1.0, 5.0, 30.0, 60.0, 120.0, 300.0, 600.0],
)

# ---------------------------------------------------------------------------
# Auth
_http_bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_http_bearer),
) -> None:
    if settings.api_key is None:
        return
    token = credentials.credentials if credentials else None
    if not token or token != settings.api_key:
        raise HTTPException(status_code=401, detail={"message": "Invalid or missing API key."})


# ---------------------------------------------------------------------------
# Circuit Breaker
class CircuitBreakerOpen(Exception):
    pass


class CircuitBreaker:
    """Three-state circuit breaker (CLOSED → OPEN → HALF_OPEN → CLOSED).

    Trips to OPEN after `failure_threshold` consecutive failures; re-tests
    after `reset_timeout` seconds via a single HALF_OPEN probe.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0) -> None:
        self._state = self.CLOSED
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    async def call(self, func, *args, **kwargs):
        async with self._lock:
            if self._state == self.OPEN:
                elapsed = (
                    time.time() - self._last_failure_time
                    if self._last_failure_time is not None
                    else 0.0
                )
                if elapsed >= self._reset_timeout:
                    self._state = self.HALF_OPEN
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit open – downstream stalled. "
                        f"Retry in {self._reset_timeout - elapsed:.0f}s."
                    )
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                self._failure_count = 0
                self._state = self.CLOSED
            return result
        except CircuitBreakerOpen:
            raise
        except Exception:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                if self._failure_count >= self._failure_threshold:
                    if self._state != self.OPEN:
                        _log.warning(
                            "Circuit OPEN after %d consecutive failures",
                            self._failure_count,
                        )
                    self._state = self.OPEN
            raise


_REASONER_CIRCUIT = CircuitBreaker(failure_threshold=5, reset_timeout=30.0)

# ---------------------------------------------------------------------------
# Native Model Manager
class NativeModelManager:
    """Manages llama-server host processes on AMD ROCm GPUs.

    Unified mode (default): one process on port 8081 spanning both GPUs
    via HIP_VISIBLE_DEVICES=0,1.

    Dual mode: reasoner on GPU 0 / port 8081, coder on GPU 1 / port 8082.

    Idle watchdog terminates any process idle longer than settings.idle_timeout
    to flush VRAM; the next request triggers on-demand restart.
    """

    _ROLE_CONFIG: Dict[str, Dict[str, Any]] = {
        "reasoner": {"port": 8081, "gpu_idx": 0, "health_url": "http://localhost:8081/health"},
        "coder":    {"port": 8082, "gpu_idx": 1, "health_url": "http://localhost:8082/health"},
    }

    def __init__(self) -> None:
        self._procs: Dict[str, Optional[asyncio.subprocess.Process]] = {
            "reasoner": None, "coder": None,
        }
        self._locks: Dict[str, asyncio.Lock] = {
            "reasoner": asyncio.Lock(), "coder": asyncio.Lock(),
        }
        self._last_activity: Dict[str, float] = {
            "reasoner": time.time(), "coder": time.time(),
        }
        self._watchdog_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public interface

    async def start_all(self) -> bool:
        if settings.unified_single_model:
            _log.info(
                "Unified mode — single llama-server on port 8081 (HIP_VISIBLE_DEVICES=0,1)."
            )
            ok = await self._start_role("reasoner")
            if ok:
                _log.info("Unified model process healthy.")
            else:
                _log.warning("Unified model process failed to become healthy.")
        else:
            _log.info("Dual mode — spawning reasoner (GPU 0) and coder (GPU 1)...")
            reasoner_ok = await self._start_role("reasoner")
            coder_ok = await self._start_role("coder")
            ok = reasoner_ok and coder_ok
            if not ok:
                _log.warning("One or both model processes failed to become healthy.")
        self._watchdog_task = asyncio.create_task(self._watchdog())
        return ok

    async def ensure_role_ready(self, role: str) -> bool:
        """Touch activity timer; restart the process if exited or unresponsive.
        In unified mode both roles route through the single 'reasoner' process."""
        effective_role = "reasoner" if settings.unified_single_model else role
        self._last_activity[effective_role] = time.time()
        config = self._ROLE_CONFIG[effective_role]
        proc = self._procs.get(effective_role)

        if proc is not None and proc.returncode is None:
            if await self._wait_for_health(config["health_url"], timeout=5):
                return True

        _log.warning("%s is unresponsive or exited — restarting on demand", effective_role)
        return await self._start_role(effective_role)

    def process_status(self) -> Dict[str, str]:
        if settings.unified_single_model:
            proc = self._procs.get("reasoner")
            status = (
                "idle (terminated)" if proc is None
                else "running" if proc.returncode is None
                else f"exited({proc.returncode})"
            )
            return {"unified (GPU 0+1 / port 8081)": status}
        result = {}
        for role, proc in self._procs.items():
            result[role] = (
                "idle (terminated)" if proc is None
                else "running" if proc.returncode is None
                else f"exited({proc.returncode})"
            )
        return result

    def touch(self, role: str) -> None:
        """Heartbeat: reset the idle timer so the watchdog won't evict the process.
        In unified mode always updates 'reasoner' regardless of role passed."""
        effective = "reasoner" if settings.unified_single_model else role
        self._last_activity[effective] = time.time()

    async def shutdown(self) -> None:
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        for role in ("reasoner", "coder"):
            await self._terminate_role(role)
        _log.info("All llama-server processes terminated.")

    # ------------------------------------------------------------------
    # Internal helpers

    def _build_cmd(self, role: str) -> List[str]:
        config = self._ROLE_CONFIG[role]
        model_key = settings.reasoner_model_key
        base = settings.models_host_path
        model_map: Dict[str, Tuple[str, str]] = {
            "gemma3":     (f"{base}/reasoner/gemma3.gguf",  "8192"),
            "dagger":     (f"{base}/reasoner/dagger.gguf",  "8192"),
            "codegeex4":  (f"{base}/coder/codegeex4.gguf",  "16384"),
            "gemma2":     (f"{base}/coder/verifier.gguf",   "8192"),
            "qwen36_35b": (f"{base}/reasoner/Qwen3.6-35B-A3B-UD-Q5_K_S.gguf", "8192"),
        }
        model_path, ctx_size = model_map.get(model_key, (f"{base}/{model_key}.gguf", "8192"))
        cmd = [
            settings.llama_server_bin,
            "--host", "127.0.0.1",
            "--port", str(config["port"]),
            "-m", model_path,
            "-c", ctx_size,
            "-fa", "on",
            "-ngl", "99",
            "--mmap",
        ]
        if model_key == "qwen36_35b":
            cmd.extend([
                "-np", "1",       # single-slot mode — protects VRAM headroom
                "--kv-unified",   # unified KV cache across both GPUs
                "-sm", "row",     # row-split tensor parallelism across GPU 0 & 1
            ])
        return cmd

    def _env_for_role(self, role: str) -> Dict[str, str]:
        visible = "0,1" if settings.unified_single_model else str(self._ROLE_CONFIG[role]["gpu_idx"])
        return {
            **os.environ,
            "HIP_VISIBLE_DEVICES":      visible,
            "ROCR_VISIBLE_DEVICES":     visible,
            "HSA_OVERRIDE_GFX_VERSION": "12.0.0",
            "HSA_ENABLE_SDMA":          "0",
            "ROCM_OVERCOMMIT":          "1",
        }

    async def _start_role(self, role: str) -> bool:
        config = self._ROLE_CONFIG[role]
        port = config["port"]
        gpus = "0,1" if settings.unified_single_model else str(config["gpu_idx"])

        async with self._locks[role]:
            await self._terminate_role(role)
            env = self._env_for_role(role)
            cmd = self._build_cmd(role)
            _log.info("Spawning %s (GPU %s / port %d): %s", role, gpus, port, " ".join(cmd))

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    env=env,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._procs[role] = proc
                self._last_activity[role] = time.time()
                asyncio.create_task(self._pipe_stderr(role, proc))
            except Exception as exc:
                _log.error("Failed to spawn %s: %s", role, exc)
                return False

        if not await self._wait_for_health(config["health_url"], timeout=settings.process_startup_timeout):
            _log.error("%s did not become healthy within %ds", role, settings.process_startup_timeout)
            return False

        _log.info("%s ready on port %d (GPU %s)", role, port, gpus)
        return True

    async def _terminate_role(self, role: str) -> None:
        proc = self._procs.get(role)
        if proc is None or proc.returncode is not None:
            self._procs[role] = None
            return
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
        self._procs[role] = None
        _log.info("Terminated %s process", role)

    async def _watchdog(self) -> None:
        """Every 15s: terminate any role idle longer than settings.idle_timeout."""
        roles_to_watch = ("reasoner",) if settings.unified_single_model else ("reasoner", "coder")
        while True:
            await asyncio.sleep(15)
            now = time.time()
            for role in roles_to_watch:
                idle = now - self._last_activity[role]
                if idle > settings.idle_timeout:
                    proc = self._procs.get(role)
                    if proc is not None and proc.returncode is None:
                        _log.info(
                            "Watchdog: %s idle %.0fs > %ds — terminating to flush VRAM",
                            role, idle, settings.idle_timeout,
                        )
                        async with self._locks[role]:
                            await self._terminate_role(role)

    async def _pipe_stderr(self, role: str, proc: asyncio.subprocess.Process) -> None:
        stderr_log = logging.getLogger(f"llama-server.{role}")
        assert proc.stderr is not None
        try:
            async for raw in proc.stderr:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    stderr_log.info("%s", line)
        except Exception:
            pass

    async def _wait_for_health(self, health_url: str, timeout: int) -> bool:
        start = time.time()
        async with httpx.AsyncClient(timeout=2) as client:
            while time.time() - start < timeout:
                try:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(1.0)
        return False


NATIVE_MODEL_MANAGER = NativeModelManager()


# ---------------------------------------------------------------------------
# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("Presenton proxy v7.0.0 starting (GPU Manager mode)...")

    ready = await NATIVE_MODEL_MANAGER.start_all()
    if not ready:
        _log.warning("Model process failed to start — will retry on first request.")
    else:
        _log.info("llama-server ready. Presenton should set CUSTOM_LLM_URL=http://localhost:8000/v1")

    yield

    _log.info("Shutting down...")
    await NATIVE_MODEL_MANAGER.shutdown()
    _log.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Application
app = FastAPI(
    title="Presenton GPU Manager",
    description="AMD ROCm llama-server lifecycle manager + OpenAI-compatible passthrough",
    version="7.0.0",
    lifespan=lifespan,
)

_cors_origins = list({
    "http://localhost:3000",
    "http://localhost:5000",
    *[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
})
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers

def _forward_headers(request: Request) -> Dict[str, str]:
    """Strip hop-by-hop headers; inject a local auth token for llama-server."""
    skip = {"host", "content-length", "transfer-encoding", "connection", "authorization"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
    headers["authorization"] = "Bearer local"
    return headers


# ---------------------------------------------------------------------------
# Core passthrough — POST /v1/chat/completions
@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    _auth: None = Depends(require_api_key),
) -> Response:
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    t_start = time.perf_counter()
    error_occurred = False

    # Boot model if necessary (on-demand start)
    ok = await NATIVE_MODEL_MANAGER.ensure_role_ready("reasoner")
    if not ok:
        _REQUESTS_TOTAL.labels(endpoint="chat_completions", status="error").inc()
        raise HTTPException(503, detail={"message": "Model failed to start — check proxy logs."})

    body = await request.body()
    is_stream = False
    try:
        is_stream = bool(_jloads(body).get("stream", False))
    except Exception:
        pass

    headers = _forward_headers(request)
    target = settings.reasoner_url

    try:
        if is_stream:
            async def _stream_gen():
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(settings.llm_timeout)
                ) as client:
                    async with client.stream(
                        "POST", target, content=body, headers=headers
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            # Each chunk keeps the watchdog timer alive
                            NATIVE_MODEL_MANAGER.touch("reasoner")
                            yield chunk

            return StreamingResponse(
                _stream_gen(),
                media_type="text/event-stream",
                headers={"X-Request-ID": request_id},
            )

        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.llm_timeout)) as client:

            async def _do_post():
                resp = await client.post(target, content=body, headers=headers)
                resp.raise_for_status()
                return resp

            resp = await _REASONER_CIRCUIT.call(_do_post)
            NATIVE_MODEL_MANAGER.touch("reasoner")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
                headers={"X-Request-ID": request_id},
            )

    except CircuitBreakerOpen as exc:
        error_occurred = True
        raise HTTPException(503, detail={"message": str(exc)})
    except httpx.TimeoutException:
        error_occurred = True
        raise HTTPException(504, detail={"message": f"LLM timed out after {settings.llm_timeout}s."})
    except httpx.HTTPStatusError as exc:
        error_occurred = True
        raise HTTPException(exc.response.status_code, detail={"message": "Upstream error from llama-server."})
    except Exception:
        error_occurred = True
        _log.exception("Passthrough error for request_id=%s", request_id)
        raise HTTPException(502, detail={"message": "Upstream error."})
    finally:
        latency = time.perf_counter() - t_start
        _REQUEST_LATENCY.observe(latency)
        _REQUESTS_TOTAL.labels(
            endpoint="chat_completions",
            status="error" if error_occurred else "ok",
        ).inc()
        _log.info(
            "request_id=%s stream=%s latency=%.3fs error=%s",
            request_id, is_stream, latency, error_occurred,
        )


# ---------------------------------------------------------------------------
# Models list — Presenton queries this to validate the custom provider
@app.get("/v1/models")
async def list_models(_auth: None = Depends(require_api_key)) -> JSONResponse:
    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": settings.reasoner_model_key,
                "object": "model",
                "created": 0,
                "owned_by": "local-rocm",
            }
        ],
    })


# ---------------------------------------------------------------------------
# Explicit wake endpoint — call this to pre-load the model before Presenton starts
@app.post("/v1/models/wake")
async def wake_model(_auth: None = Depends(require_api_key)) -> JSONResponse:
    ok = await NATIVE_MODEL_MANAGER.ensure_role_ready("reasoner")
    proc_status = NATIVE_MODEL_MANAGER.process_status()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "ready": ok,
            "processes": proc_status,
            "message": "Model is ready." if ok else "Model failed to start — check proxy logs.",
        },
    )


# ---------------------------------------------------------------------------
# Health
@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={
        "status": "ok",
        "version": "7.0.0",
        "mode": "unified" if settings.unified_single_model else "dual",
    })


@app.get("/health/detailed")
async def health_detailed() -> JSONResponse:
    async def _check(url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(url)
                return r.status_code == 200
        except Exception:
            return False

    reasoner_ok = await _check(settings.reasoner_health_url)
    proc_status = NATIVE_MODEL_MANAGER.process_status()

    return JSONResponse(content={
        "status": "ok" if reasoner_ok else "degraded",
        "proxy": "ok",
        "llama_server": "ok" if reasoner_ok else "unreachable",
        "processes": proc_status,
        "circuit_breaker": _REASONER_CIRCUIT.state,
        "idle_timeout_seconds": settings.idle_timeout,
        "llm_timeout_seconds": settings.llm_timeout,
    })


# ---------------------------------------------------------------------------
# Metrics
@app.get("/metrics")
async def get_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("presenton_proxy:app", host="0.0.0.0", port=8000, log_level="info")
