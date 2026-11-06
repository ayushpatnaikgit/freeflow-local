"""Local backend for FreeFlow: Orukeet ASR on MLX, plus an on-demand Gemma cleanup LLM.

One OpenAI-compatible endpoint (FreeFlow's API Base URL):
  /v1/audio/transcriptions  Orukeet (Parakeet TDT v3 fine-tune) via parakeet-mlx + fast_tdt
  /v1/chat/completions      forwarded to an mlx_lm.server child process, model pinned

Models load on demand and unload after IDLE_UNLOAD_SECONDS without requests. FreeFlow sends
its context request the moment recording starts, so that request wakes everything up while
the user is still speaking. After the LLM starts, the last-seen cleanup system prompt (kept
in memory only) is pre-filled so the prompt cache is warm before the audio arrives.

All MLX work in this process runs on one dedicated thread (MLX streams are per-thread).
"""

import asyncio
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

import fast_tdt

HERE = Path(__file__).resolve().parent
ASR_MODEL = os.environ.get("ASR_MODEL", "itsreidar/orukeet-mlx")
LLM_MODEL = os.environ.get("LLM_MODEL", "mlx-community/gemma-4-e4b-it-4bit")
LLM_PORT = int(os.environ.get("LLM_PORT", "8080"))
LLM_URL = f"http://127.0.0.1:{LLM_PORT}"
IDLE_UNLOAD_SECONDS = float(os.environ.get("IDLE_UNLOAD_SECONDS", "600"))  # 0 = always warm
CHUNK_SECONDS = float(os.environ.get("ASR_CHUNK_SECONDS", "120"))


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------- ASR (in-process)

_mlx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_asr = None


def _load_asr() -> None:
    global _asr
    if _asr is not None:
        return
    from parakeet_mlx import from_pretrained

    t = time.perf_counter()
    _asr = from_pretrained(ASR_MODEL)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:  # compile Metal kernels once
        with wave.open(f.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.random.default_rng(0).normal(0, 300, 32000).astype(np.int16).tobytes())
        fast_tdt.transcribe(_asr, f.name)
    log(f"asr loaded in {time.perf_counter() - t:.1f}s")


def _unload_asr() -> None:
    global _asr
    if _asr is None:
        return
    _asr = None
    gc.collect()
    mx.clear_cache()
    log("asr unloaded")


def _transcribe(path: str) -> str:
    _load_asr()
    text = fast_tdt.transcribe(_asr, path, chunk_seconds=CHUNK_SECONDS)
    mx.clear_cache()
    return text


# ---------------------------------------------------------------- LLM (child process)

_llm: subprocess.Popen | None = None
_llm_lock = asyncio.Lock()
_cleanup_system_prompt: str | None = None  # memory only; contains the user's vocabulary


def _llm_alive() -> bool:
    return _llm is not None and _llm.poll() is None


def _http(method: str, path: str, body: dict | None = None, timeout: float = 300) -> tuple[int, bytes, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{LLM_URL}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "application/json")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        reason = getattr(e, "reason", e)
        return 502, json.dumps({"error": {"message": f"LLM server unreachable: {reason}"}}).encode(), "application/json"


async def _ensure_llm() -> None:
    """Start mlx_lm.server if needed and wait until it answers HTTP."""
    global _llm
    async with _llm_lock:
        if _llm_alive():
            return
        t = time.perf_counter()
        logf = open(HERE / "logs" / "llm.log", "ab")
        _llm = subprocess.Popen(
            [str(Path(sys.executable).parent / "mlx_lm.server"), "--model", LLM_MODEL,
             "--host", "127.0.0.1", "--port", str(LLM_PORT), "--max-tokens", "4096",
             "--chat-template-args", '{"enable_thinking":false}'],
            stdout=logf, stderr=subprocess.STDOUT, cwd=HERE)
        while time.perf_counter() - t < 60:
            if not _llm_alive():
                raise RuntimeError("mlx_lm.server exited during startup; see logs/llm.log")
            status, _, _ = await asyncio.to_thread(_http, "GET", "/v1/models", None, 2)
            if status == 200:
                break
            await asyncio.sleep(0.2)
        log(f"llm process up in {time.perf_counter() - t:.1f}s (pid {_llm.pid})")
        asyncio.create_task(_prefill())  # load weights + warm the cleanup prompt cache


async def _prefill() -> None:
    t = time.perf_counter()
    system = _cleanup_system_prompt or "You clean up dictated text."
    await asyncio.to_thread(_http, "POST", "/v1/chat/completions", {
        "model": LLM_MODEL, "temperature": 0, "max_tokens": 1,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": "Instructions: Clean up RAW_TRANSCRIPTION."}]})
    log(f"llm weights + prompt cache warm in {time.perf_counter() - t:.1f}s"
        f"{'' if _cleanup_system_prompt else ' (no cleanup prompt seen yet)'}")


def _stop_llm() -> None:
    global _llm
    if _llm is None:
        return
    if _llm.poll() is None:
        _llm.terminate()
        try:
            _llm.wait(10)
        except subprocess.TimeoutExpired:
            _llm.kill()
    _llm = None
    log("llm stopped")


# ---------------------------------------------------------------- activity + idle unloading

_last_activity = time.monotonic()
_inflight = 0


class _Activity:
    async def __aenter__(self):
        global _inflight, _last_activity
        _inflight += 1
        _last_activity = time.monotonic()

    async def __aexit__(self, *exc):
        global _inflight, _last_activity
        _inflight -= 1
        _last_activity = time.monotonic()


def _wake() -> None:
    """Start loading both models in the background without waiting."""
    loop = asyncio.get_running_loop()
    if _asr is None:
        loop.run_in_executor(_mlx, _load_asr)
    if not _llm_alive():
        asyncio.create_task(_ensure_llm())


async def _idle_watch() -> None:
    while True:
        await asyncio.sleep(15)
        if IDLE_UNLOAD_SECONDS <= 0 or _inflight:
            continue
        if time.monotonic() - _last_activity < IDLE_UNLOAD_SECONDS:
            continue
        if _llm is not None or _asr is not None:
            log(f"idle for {IDLE_UNLOAD_SECONDS:.0f}s: unloading")
            async with _llm_lock:
                _stop_llm()
            await asyncio.get_running_loop().run_in_executor(_mlx, _unload_asr)


@asynccontextmanager
async def lifespan(_app):
    (HERE / "logs").mkdir(exist_ok=True)
    watcher = asyncio.create_task(_idle_watch())
    if IDLE_UNLOAD_SECONDS <= 0:  # always-warm mode: load everything now
        _wake()
    log(f"ready (idle unload after {IDLE_UNLOAD_SECONDS:.0f}s; 0 = never)")
    yield
    watcher.cancel()
    _stop_llm()


app = FastAPI(lifespan=lifespan)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [
        {"id": ASR_MODEL, "object": "model", "owned_by": "local"},
        {"id": LLM_MODEL, "object": "model", "owned_by": "local"},
    ]}


@app.get("/health")
async def health():
    return {"ok": True, "asr_loaded": _asr is not None, "llm_running": _llm_alive(),
            "idle_seconds": round(time.monotonic() - _last_activity),
            "idle_unload_after": IDLE_UNLOAD_SECONDS, "asr_model": ASR_MODEL, "llm_model": LLM_MODEL}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Forward to mlx_lm.server, pinning the model so FreeFlow's model fields don't matter."""
    global _cleanup_system_prompt
    async with _Activity():
        _wake()  # recording just started (context request) or cleanup is due
        payload = await request.json()
        payload["model"] = LLM_MODEL
        payload["stream"] = False
        for hosted_only in ("reasoning_effort", "include_reasoning"):  # Groq/OpenAI options
            payload.pop(hosted_only, None)
        messages = payload.get("messages", [])
        for message in messages:
            # mlx_lm can't read images for this model and reports that as a 404, which FreeFlow
            # treats as fatal (no text-only retry). Keep only text so context uses metadata.
            if isinstance(message.get("content"), list):
                message["content"] = [p for p in message["content"] if p.get("type") == "text"]
        if (len(messages) >= 2 and messages[0].get("role") == "system"
                and "RAW_TRANSCRIPTION" in str(messages[-1].get("content", ""))):
            _cleanup_system_prompt = messages[0]["content"]
        await _ensure_llm()
        status, data, ctype = await asyncio.to_thread(_http, "POST", "/v1/chat/completions", payload)
        return Response(content=data, status_code=status, media_type=ctype)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(None),  # ignored: one model is served
    response_format: str = Form("json"),
    language: str = Form(None),  # ignored: Orukeet / Parakeet v3 auto-detect language
):
    async with _Activity():
        _wake()  # if context capture is off, this is the first signal: start the LLM during ASR
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            path = tmp.name
        try:
            t = time.perf_counter()
            text = await asyncio.get_running_loop().run_in_executor(_mlx, _transcribe, path)
            log(f"asr {(time.perf_counter() - t) * 1000:.0f}ms chars={len(text)}")
        finally:
            os.unlink(path)

    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text})
