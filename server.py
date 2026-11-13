"""Local backend for FreeFlow: Orukeet ASR (MLX or Neural Engine), plus an on-demand Gemma cleanup LLM.

One OpenAI-compatible endpoint (FreeFlow's API Base URL):
  /v1/audio/transcriptions  Orukeet (Parakeet TDT v3 fine-tune) via parakeet-mlx + fast_tdt on the GPU,
                            or via the orukeet-ane Core ML helper on the Neural Engine (ASR_BACKEND=ane)
  /v1/chat/completions      forwarded to an mlx_lm.server child process, model pinned

Screen words: when FreeFlow's context request carries a screenshot (FreeFlow has Screen Recording permission), the
screen-ocr helper reads it on device while the user speaks, and the unusual words on screen (LoRA, MLX, names) are
added to that dictation's cleanup request, so a term heard as "Laura" can be written "LoRA". Screenshot, text and
words stay in memory, are never logged, and are dropped after one dictation. SCREEN_WORDS=0 turns this off.

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
import re
import socket
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

import compact_prompt
import fast_tdt
import screen_words

HERE = Path(__file__).resolve().parent


def _load_settings(path: Path) -> None:
    """KEY=VALUE lines from settings.env (written by `freeflowctl`) override the environment."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ[key.strip()] = value.strip()


_load_settings(HERE / "settings.env")
ASR_BACKEND = os.environ.get("ASR_BACKEND", "mlx")  # "mlx" (GPU) or "ane" (Neural Engine via Core ML)
ANE_BUNDLE = HERE / "coreml" / "orukeet-r3-coreml-greedy"
ANE_CACHE = HERE / "coreml" / "compiled"
ANE_HELPER = HERE / "bin" / "orukeet-ane"
ASR_MODEL = os.environ.get("ASR_MODEL", "itsreidar/orukeet-mlx")
LLM_MODEL = os.environ.get("LLM_MODEL", "mlx-community/gemma-4-e4b-it-4bit")


def read_pins(text: str) -> dict[str, str]:
    """pinned-models.txt: "<repo> <commit>" per line; # comments."""
    pins = {}
    for line in text.splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) == 2:
            pins[parts[0]] = parts[1]
    return pins


def model_source(repo: str) -> str:
    """Local folder of the pinned version of a model if it is pinned and downloaded, else the repo id itself."""
    pins_file = HERE / "pinned-models.txt"
    revision = read_pins(pins_file.read_text()).get(repo) if pins_file.exists() else None
    if not revision:
        return repo
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - not downloaded at that version: fall back to the cached latest, and say so
        print(f"note: pinned version of {repo} is not downloaded; run install.sh to fetch it", flush=True)
        return repo


def adapter_path(setting: str) -> str | None:
    """LLM_ADAPTER: a LoRA adapter folder (absolute, or relative to the install folder); None if unset or missing."""
    if not setting:
        return None
    path = Path(setting).expanduser()
    path = path if path.is_absolute() else HERE / path
    if not (path / "adapters.safetensors").exists():
        print(f"note: LLM_ADAPTER {setting} has no adapters.safetensors; serving the base model", flush=True)
        return None
    return str(path)


ASR_SOURCE = model_source(ASR_MODEL)
LLM_SOURCE = model_source(LLM_MODEL)
LLM_ADAPTER = adapter_path(os.environ.get("LLM_ADAPTER", ""))
# mlx_lm.server attaches --adapter-path only to "default_model"; naming the model loads a second copy WITHOUT it.
LLM_REQUEST_MODEL = "default_model" if LLM_ADAPTER else LLM_SOURCE
LLM_PROMPT = os.environ.get("LLM_PROMPT", "freeflow")  # "compact": swap FreeFlow's default prompt for the trained one
LLM_PORT = int(os.environ.get("LLM_PORT", "0"))  # 0 = pick a free port each time the LLM starts
IDLE_UNLOAD_SECONDS = float(os.environ.get("IDLE_UNLOAD_SECONDS", "600"))  # 0 = always warm
# Keep the speech model loaded when idle (only the cleanup LLM unloads). It is small, and a cold Neural Engine
# load is sometimes 15 s instead of 0.2 s when Core ML has to recompile it.
KEEP_ASR_LOADED = os.environ.get("KEEP_ASR_LOADED", "0") == "1"
CHUNK_SECONDS = float(os.environ.get("ASR_CHUNK_SECONDS", "120"))
STRIP_FILLERS = os.environ.get("STRIP_FILLERS", "1") != "0"  # remove uh/um/hmm after transcription
SCREEN_WORDS = os.environ.get("SCREEN_WORDS", "1") != "0"  # use words read from FreeFlow's context screenshot
OCR_HELPER = HERE / "bin" / "screen-ocr"
SCREEN_WORDS_MAX_AGE = 300.0  # seconds from recording start (context request) to cleanup; older words are ignored
# Vocabulary boosting in Orukeet (Neural Engine only): the user's FreeFlow vocabulary and this dictation's screen
# words tip close calls in the recogniser toward those spellings. Only whole terms are accepted. Off by default.
ASR_BOOST = os.environ.get("ASR_BOOST", "0") == "1"
BOOST_MAX_TERMS = 100
SCREEN_WORDS_WAIT = 0.5  # at cleanup, wait at most this long for OCR still running (it normally finished during speech)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------- ASR (in-process)

_mlx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")  # all ASR work runs on this thread
_asr = None


class _AneHelper:
    """The orukeet-ane Swift process: one WAV path per line in, one JSON line out."""

    def __init__(self) -> None:
        t = time.perf_counter()
        self.proc = subprocess.Popen([str(ANE_HELPER), str(ANE_BUNDLE), str(ANE_CACHE)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=HERE)
        ready = json.loads(self.proc.stdout.readline() or '{"error": "helper exited"}')
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(ready.get("error", "orukeet-ane did not start"))
        self.can_boost = bool(ready.get("boost"))  # older helpers read every line as a file path
        log(f"asr (Neural Engine) loaded in {time.perf_counter() - t:.2f}s")

    def transcribe(self, path: str, boost: list[str] | None = None) -> str:
        with wave.open(path, "rb") as w:
            ok = w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000
        if not ok:
            raise ValueError("Neural Engine path needs 16 kHz mono 16-bit WAV")
        self.proc.stdin.write(helper_request(path, boost if self.can_boost else None) + "\n")
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline() or '{"error": "helper exited"}')
        if "error" in reply:
            raise RuntimeError(reply["error"])
        if boost and self.can_boost:
            log(f"asr boost: {len(boost)} terms, {reply.get('boosted', 0)} swapped in")  # counts only
        return reply["text"]

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def helper_request(path: str, boost: list[str] | None) -> str:
    """One stdin line for orukeet-ane: the bare path (today's protocol), or JSON with the terms to boost."""
    return json.dumps({"path": path, "boost": boost}) if boost else path


_VOCAB_MARKER = "Use these spellings exactly in the output when relevant:\n"


def vocabulary_terms(system_prompt: str | None) -> list[str]:
    """The user's FreeFlow vocabulary: FreeFlow appends it as the last line of the cleanup system prompt."""
    if not system_prompt or _VOCAB_MARKER not in system_prompt:
        return []
    line = system_prompt.rsplit(_VOCAB_MARKER, 1)[1].split("\n", 1)[0]
    return [t.strip() for t in line.split(",") if t.strip()]


def boost_terms(vocabulary: list[str], screen: list[str]) -> list[str]:
    """Vocabulary first, then screen words; one spelling per term, at least 3 letters or digits, capped."""
    out, seen = [], set()
    for term in vocabulary + screen:
        key = term.lower()
        if len(re.sub(r"[^a-z0-9]", "", key)) < 3 or key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out[:BOOST_MAX_TERMS]


_ane: _AneHelper | None = None
_ane_failed = False  # after a failure, stay on MLX until restart


def _ane_wanted() -> bool:
    return ASR_BACKEND == "ane" and not _ane_failed and ANE_HELPER.exists() and ANE_BUNDLE.exists()


def _load_asr() -> None:
    """Load the configured speech model: the Neural Engine helper, or MLX if that is off or fails."""
    global _ane, _ane_failed
    if _ane_wanted():
        if _ane is None:
            try:
                _ane = _AneHelper()
            except Exception as e:  # noqa: BLE001 - any helper failure falls back to MLX
                _ane_failed = True
                log(f"Neural Engine ASR unavailable ({e}); using MLX")
        if _ane is not None:
            return
    _load_mlx_asr()


def _load_mlx_asr() -> None:
    global _asr
    if _asr is not None:
        return
    from parakeet_mlx import from_pretrained

    t = time.perf_counter()
    _asr = from_pretrained(ASR_SOURCE)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:  # compile Metal kernels once
        with wave.open(f.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.random.default_rng(0).normal(0, 300, 32000).astype(np.int16).tobytes())
        fast_tdt.transcribe(_asr, f.name)
    log(f"asr loaded in {time.perf_counter() - t:.1f}s")


def _unload_asr() -> None:
    global _asr, _ane
    if _ane is not None:
        _ane.close()
        _ane = None
        log("asr (Neural Engine) unloaded")
    if _asr is not None:
        _asr = None
        gc.collect()
        mx.clear_cache()
        log("asr unloaded")


def _transcribe(path: str, boost: list[str] | None = None) -> str:
    global _ane, _ane_failed
    _load_asr()
    text = None
    if _ane is not None:
        try:
            text = _ane.transcribe(path, boost)
        except ValueError:
            pass  # not 16 kHz mono WAV: MLX reads any format via ffmpeg
        except Exception as e:  # noqa: BLE001
            # error type only: helper messages can echo the request (file path, boost terms read from the screen)
            log(f"Neural Engine ASR failed ({type(e).__name__}); using MLX from now on")
            _ane.close()
            _ane, _ane_failed = None, True
    if text is None:
        _load_mlx_asr()
        text = fast_tdt.transcribe(_asr, path, chunk_seconds=CHUNK_SECONDS)
        mx.clear_cache()
    return strip_fillers(text) if STRIP_FILLERS else text


# Hesitation sounds only: "er" (German "he"), "ah" and "mm" (millimetres) are real words, so they stay.
_FILLER = re.compile(
    r"(?:\s*,)?\s*(?<![\w'])(?:u+h+m*|u+m+|h+m{2,})(?![\w'])([,.!?…]*)\s*", re.IGNORECASE)


def strip_fillers(text: str) -> str:
    """Remove uh/um/uhm/hmm with the commas around them, keeping any sentence-ending mark."""
    def drop(m: re.Match) -> str:
        end = "".join(c for c in m.group(1) if c in ".!?…")
        return f"{end} " if end else " "

    out = _FILLER.sub(drop, text)
    out = re.sub(r"\s+([,.!?…])", r"\1", out)                       # "noon ." -> "noon."
    out = re.sub(r"(^|[.!?…]\s+)[,.!?…\s]+", r"\1", out)            # punctuation left at a sentence start
    out = re.sub(r"(^|[.!?…]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out if re.search(r"\w", out) else ""


# ---------------------------------------------------------------- screen words (OCR helper process)

_ocr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")


class _OcrHelper:
    """The screen-ocr Swift process: one base64 image per stdin line in, one JSON line of text lines out."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen([str(OCR_HELPER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, cwd=HERE)
        ready = json.loads(self.proc.stdout.readline() or '{"error": "helper exited"}')
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(ready.get("error", "screen-ocr did not start"))

    def read(self, data_url: str) -> list[str]:
        self.proc.stdin.write(data_url.replace("\n", "") + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("screen-ocr exited")
        reply = json.loads(line)
        if "error" in reply:
            raise ValueError(reply["error"])  # this image only, e.g. undecodable
        return reply["lines"]

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


_ocr: _OcrHelper | None = None
_ocr_failed = False  # after a failure, screen words stay off until restart
_screen_task: asyncio.Future | None = None  # words for the dictation whose recording just started
_screen_task_at = 0.0


def _read_screen(data_url: str) -> list[str]:
    """Runs on the OCR thread. Returns the unusual words; logs only counts and timing."""
    global _ocr, _ocr_failed
    t = time.perf_counter()
    try:
        if _ocr is None:
            _ocr = _OcrHelper()
        words = screen_words.screen_words(_ocr.read(data_url))
    except ValueError:
        log("screen words: could not read this screenshot")
        return []
    except Exception as e:  # noqa: BLE001 - any OCR failure only loses the screen words
        log(f"screen words unavailable ({type(e).__name__}); off until restart")
        _ocr_failed = True
        _close_ocr()
        return []
    log(f"screen words: {len(words)} in {(time.perf_counter() - t) * 1000:.0f}ms")
    return words


def _close_ocr() -> None:
    global _ocr
    if _ocr is not None:
        _ocr.close()
        _ocr = None


def _screenshot_url(messages: list) -> str | None:
    """The data: URL of the screenshot in FreeFlow's context request, if any."""
    content = messages[-1].get("content") if messages else None
    if not isinstance(content, list):
        return None
    for part in content:
        image = part.get("image_url") if isinstance(part, dict) and part.get("type") == "image_url" else None
        url = image.get("url") if isinstance(image, dict) else image
        if isinstance(url, str) and url.startswith("data:image/"):
            return url
    return None


def _start_screen_words(messages: list) -> None:
    global _screen_task, _screen_task_at
    if not SCREEN_WORDS or _ocr_failed or not OCR_HELPER.exists():
        return
    url = _screenshot_url(messages)
    if url is None:
        return
    _screen_task = asyncio.get_running_loop().run_in_executor(_ocr_pool, _read_screen, url)
    _screen_task_at = time.monotonic()


def _peek_screen_words() -> list[str]:
    """This dictation's screen words if OCR has already finished; never waits, and leaves them for cleanup."""
    task = _screen_task
    if task is None or not task.done() or task.cancelled() or time.monotonic() - _screen_task_at > SCREEN_WORDS_MAX_AGE:
        return []
    return task.result() if task.exception() is None else []


def _take_screen_task() -> asyncio.Future | None:
    """This dictation's screen-words task, if recent; each one is used for one cleanup only."""
    global _screen_task
    task, _screen_task = _screen_task, None
    if task is None or time.monotonic() - _screen_task_at > SCREEN_WORDS_MAX_AGE:
        return None
    return task


async def _with_screen_words(messages: list, task: asyncio.Future | None, transcript: str) -> None:
    """Add the screen words that sound like part of the transcript to the cleanup request's user message (not the
    system prompt, which stays cached). Most dictations match none, and then the request is unchanged."""
    if task is None:
        return
    try:
        words = await asyncio.wait_for(asyncio.shield(task), SCREEN_WORDS_WAIT)
    except asyncio.TimeoutError:
        log("screen words not ready in time; cleaned up without them")
        return
    found = screen_words.relevant(words, transcript)
    log(f"screen words: {len(found)} of {len(words)} match the transcript")  # counts only, never the words
    section = screen_words.prompt_section(found)
    content = messages[-1].get("content")
    marker = "RAW_TRANSCRIPTION:\n<<<RAW_TRANSCRIPTION"
    if section and isinstance(content, str) and content.count(marker) == 1:
        messages[-1]["content"] = content.replace(marker, section + "\n" + marker)


# ---------------------------------------------------------------- LLM (child process)

_llm: subprocess.Popen | None = None
_llm_port = 0
_llm_lock = asyncio.Lock()
_cleanup_system_prompt: str | None = None  # memory only; contains the user's vocabulary


def _llm_alive() -> bool:
    return _llm is not None and _llm.poll() is None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method: str, path: str, body: dict | None = None, timeout: float = 300) -> tuple[int, bytes, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{_llm_port}{path}", data=data, method=method,
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
    """Start mlx_lm.server if needed and wait until *it* (not some other server) answers."""
    global _llm, _llm_port
    async with _llm_lock:
        if _llm_alive():
            return
        t = time.perf_counter()
        _llm_port = LLM_PORT or _free_port()
        logf = open(HERE / "logs" / "llm.log", "ab")
        cmd = [str(Path(sys.executable).parent / "mlx_lm.server"), "--model", LLM_SOURCE,
               "--host", "127.0.0.1", "--port", str(_llm_port), "--max-tokens", "4096",
               "--chat-template-args", '{"enable_thinking":false}',
               "--allowed-origins", "none"]  # no web page may call the model server (default would be "*")
        if LLM_ADAPTER:
            cmd += ["--adapter-path", LLM_ADAPTER]
        _llm = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=HERE)
        while True:
            if not _llm_alive():
                log(f"llm failed to start on port {_llm_port}; see logs/llm.log")
                raise RuntimeError("mlx_lm.server exited during startup; see logs/llm.log")
            status, body, _ = await asyncio.to_thread(_http, "GET", "/v1/models", None, 2)
            if status == 200 and json.dumps(LLM_SOURCE)[1:-1].encode() in body:
                break
            if time.perf_counter() - t > 60:
                _stop_llm()
                log("llm did not become ready within 60s; stopped it")
                raise RuntimeError("mlx_lm.server did not become ready within 60s")
            await asyncio.sleep(0.2)
        log(f"llm process up on port {_llm_port} in {time.perf_counter() - t:.1f}s (pid {_llm.pid})")
        asyncio.create_task(_prefill())  # load weights + warm the cleanup prompt cache


async def _prefill() -> None:
    t = time.perf_counter()
    system = _cleanup_system_prompt or "You clean up dictated text."
    status, _, _ = await asyncio.to_thread(_http, "POST", "/v1/chat/completions", {
        "model": LLM_REQUEST_MODEL, "temperature": 0, "max_tokens": 1,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": "Instructions: Clean up RAW_TRANSCRIPTION."}]})
    log(f"llm weights + prompt cache warm in {time.perf_counter() - t:.1f}s"
        f"{'' if _cleanup_system_prompt else ' (no cleanup prompt seen yet)'}"
        f"{'' if status == 200 else f' FAILED (HTTP {status})'}")


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
    if _asr is None and _ane is None:
        loop.run_in_executor(_mlx, _load_asr)
    if not _llm_alive():
        asyncio.create_task(_ensure_llm())


def _should_unload_asr() -> bool:
    """On idle: unload the speech model only if it is loaded and KEEP_ASR_LOADED is off."""
    return not KEEP_ASR_LOADED and (_asr is not None or _ane is not None)


async def _idle_watch() -> None:
    while True:
        await asyncio.sleep(15)
        if IDLE_UNLOAD_SECONDS <= 0 or _inflight:
            continue
        if time.monotonic() - _last_activity < IDLE_UNLOAD_SECONDS:
            continue
        unload_asr = _should_unload_asr()
        if _llm is not None or unload_asr:
            log(f"idle for {IDLE_UNLOAD_SECONDS:.0f}s: unloading" + ("" if unload_asr else " (speech model kept loaded)"))
            async with _llm_lock:
                _stop_llm()
            if unload_asr:
                await asyncio.get_running_loop().run_in_executor(_mlx, _unload_asr)
            await asyncio.get_running_loop().run_in_executor(_ocr_pool, _close_ocr)


@asynccontextmanager
async def lifespan(_app):
    (HERE / "logs").mkdir(exist_ok=True)
    watcher = asyncio.create_task(_idle_watch())
    if IDLE_UNLOAD_SECONDS <= 0:  # always-warm mode: load everything now
        _wake()
    elif KEEP_ASR_LOADED:  # load the speech model now; the cleanup LLM still loads on demand
        asyncio.get_running_loop().run_in_executor(_mlx, _load_asr)
    log(f"ready (idle unload after {IDLE_UNLOAD_SECONDS:.0f}s; 0 = never"
        + ("; speech model always loaded)" if KEEP_ASR_LOADED else ")"))
    yield
    watcher.cancel()
    _stop_llm()
    _close_ocr()


app = FastAPI(lifespan=lifespan)


def _from_web_page(headers) -> bool:
    """Browsers label requests made by web pages with Origin (and Sec-Fetch-Site); FreeFlow, curl and the backend's
    own calls don't. Without this check, any website you visit could make the backend transcribe or run the model."""
    return bool(headers.get("origin")) or headers.get("sec-fetch-site", "none") != "none"


@app.middleware("http")
async def refuse_web_pages(request: Request, call_next):
    if _from_web_page(request.headers):
        return JSONResponse({"error": {"message": "Requests from web pages are not accepted."}}, status_code=403)
    return await call_next(request)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [
        {"id": ASR_MODEL, "object": "model", "owned_by": "local"},
        {"id": LLM_MODEL, "object": "model", "owned_by": "local"},
    ]}


@app.get("/health")
async def health():
    return {"ok": True, "asr_loaded": _asr is not None or _ane is not None,
            "asr_backend": "ane" if _ane is not None else ("mlx" if _asr is not None else ASR_BACKEND),
            "llm_running": _llm_alive(),
            "screen_words": SCREEN_WORDS and not _ocr_failed and OCR_HELPER.exists(),
            "asr_boost": ASR_BOOST,
            "idle_seconds": round(time.monotonic() - _last_activity),
            "idle_unload_after": IDLE_UNLOAD_SECONDS, "asr_model": ASR_MODEL, "llm_model": LLM_MODEL}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Forward to mlx_lm.server, pinning the model so FreeFlow's model fields don't matter."""
    global _cleanup_system_prompt
    async with _Activity():
        _wake()  # recording just started (context request) or cleanup is due
        payload = await request.json()
        payload["model"] = LLM_REQUEST_MODEL
        payload["stream"] = False
        for hosted_only in ("reasoning_effort", "include_reasoning"):  # Groq/OpenAI options
            payload.pop(hosted_only, None)
        messages = payload.get("messages", [])
        _start_screen_words(messages)  # before the image is dropped below; OCR runs while the user speaks
        for message in messages:
            # mlx_lm can't read images for this model and reports that as a 404, which FreeFlow
            # treats as fatal (no text-only retry). Keep only text so context uses metadata.
            if isinstance(message.get("content"), list):
                message["content"] = [p for p in message["content"] if p.get("type") == "text"]
        raw = None
        if (len(messages) >= 2 and messages[0].get("role") == "system"
                and "RAW_TRANSCRIPTION" in str(messages[-1].get("content", ""))):
            _use_prompt_style(messages)
            _cleanup_system_prompt = messages[0]["content"]
            m = _RAW_BLOCK.search(str(messages[-1].get("content", "")))
            raw = m.group(1).strip() if m else None
        screen_task = _take_screen_task() if raw is not None else None
        await _ensure_llm()
        await _with_screen_words(messages, screen_task, raw or "")
        status, data, ctype = await asyncio.to_thread(_http, "POST", "/v1/chat/completions", payload)
        if status == 200 and raw:
            data = _keep_spoken_words(data, raw)
        return Response(content=data, status_code=status, media_type=ctype)


_RAW_BLOCK = re.compile(r"<<<RAW_TRANSCRIPTION\n(.*?)\nRAW_TRANSCRIPTION", re.S)
_prompt_note_logged = False


def _use_prompt_style(messages: list) -> None:
    """With LLM_PROMPT=compact, give the fine-tuned model the prompt it was trained on (vocabulary etc. kept)."""
    global _prompt_note_logged
    if LLM_PROMPT != "compact" or not isinstance(messages[0].get("content"), str):
        return
    compact = compact_prompt.compact_system_prompt(messages[0]["content"])
    if compact is not None:
        messages[0]["content"] = compact
    if not _prompt_note_logged:  # once per run; says which path, never the prompt itself
        log("cleanup prompt: compact" if compact is not None else "cleanup prompt: custom, passed through unchanged")
        _prompt_note_logged = True


def _keep_spoken_words(data: bytes, raw: str) -> bytes:
    """If cleanup answers EMPTY for a transcript with real words (e.g. "Yeah."), return the transcript.

    Fillers are already stripped after transcription, so anything left is something the user said.
    """
    try:
        body = json.loads(data)
        message = body["choices"][0]["message"]
        if message.get("content", "").strip().strip('"').upper() != "EMPTY" or not re.search(r"\w", raw):
            return data
        message["content"] = raw
        log("cleanup returned EMPTY for a non-empty transcript; kept the transcript")
        return json.dumps(body).encode()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return data


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
            boost = boost_terms(vocabulary_terms(_cleanup_system_prompt), _peek_screen_words()) if ASR_BOOST else None
            text = await asyncio.get_running_loop().run_in_executor(_mlx, _transcribe, path, boost)
            log(f"asr {(time.perf_counter() - t) * 1000:.0f}ms chars={len(text)}")
        finally:
            os.unlink(path)

    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text})
