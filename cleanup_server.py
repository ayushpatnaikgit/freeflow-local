"""A small OpenAI-style chat server for the cleanup model, with copy-ahead (prompt-lookup) decoding.

Drop-in for the subset of `mlx_lm.server` that freeflow-local uses: the same command-line flags, GET /v1/models and
non-streaming POST /v1/chat/completions. Decoding is greedy. Each step also checks up to K tokens copied from earlier
in the prompt (cleanup output mostly repeats the transcript) in the same forward pass and keeps the ones the model
agrees with, so the text is what plain greedy decoding gives, only faster: about 40% on an M4 (median 163 -> 97 ms on
the cleanup test suites). Like mlx_lm.server it keeps the previous request's KV cache and reuses the longest common
prefix, so the fixed instructions are not read again.

Privacy: binds to the given host (127.0.0.1), answers no CORS preflight and rejects requests that carry an Origin
header, so web pages cannot call it. It logs timings and token counts only, never prompts or outputs.
"""

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

DRAFT_TOKENS = 4  # measured best of 4, 8, 16 on the cleanup suites


def lookup_draft(ctx: list, k: int, max_n: int = 4) -> list:
    """Tokens that followed the most recent earlier occurrence of ctx's last n tokens (longest n first)."""
    for n in range(max_n, 0, -1):
        if len(ctx) <= n:
            continue
        tail = ctx[-n:]
        for i in range(len(ctx) - n - 1, -1, -1):
            if ctx[i:i + n] == tail:
                return ctx[i + n:i + n + k]
    return []


def lookahead_decode(predict, trim, first: int, ctx: list, eos: set, max_tokens: int, k: int) -> tuple[list, bool]:
    """Greedy decoding with copy-ahead drafts.

    predict(tokens) feeds tokens to the model (appending them to its cache) and returns the greedy next token after
    each one; trim(n) drops the last n tokens from the cache. `first` is the greedy token after the prompt and `ctx`
    the prompt tokens. Returns (output tokens without the end token, whether it stopped at max_tokens).
    """
    out, ctx = [], list(ctx)
    y = first
    while True:
        if y in eos:
            return out, False
        if len(out) >= max_tokens:
            return out, True
        out.append(y)
        ctx.append(y)
        draft = lookup_draft(ctx, k) if k else []
        preds = predict([y] + draft)
        a = 0
        while a < len(draft) and preds[a] == draft[a]:
            if draft[a] in eos:
                trim(len(draft) - a)  # keep y and draft[:a]; the end token itself is not needed in the cache
                return out, False
            if len(out) >= max_tokens:
                trim(len(draft) - a)
                return out, True
            out.append(draft[a])
            ctx.append(draft[a])
            a += 1
        if len(draft) - a:
            trim(len(draft) - a)
        y = preds[a]


class Model:
    def __init__(self, path: str, adapter: str | None, template_args: dict):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

        self.mx, self.make_cache, self.trim_cache = mx, make_prompt_cache, trim_prompt_cache
        self.model, self.tok = load(path, adapter_path=adapter)
        self.eos = set(self.tok.eos_token_ids)
        self.template_args = template_args
        self.cache, self.cached = None, []
        self.lock = threading.Lock()

    def complete(self, messages: list, max_tokens: int) -> tuple[str, int, int, bool]:
        mx = self.mx
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, **self.template_args)
        with self.lock:
            n = 0
            if self.cache is not None:
                while n < min(len(ids) - 1, len(self.cached)) and ids[n] == self.cached[n]:
                    n += 1
                self.trim_cache(self.cache, len(self.cached) - n)
            else:
                self.cache = self.make_cache(self.model)
            cache = self.cache
            fed = list(ids[:n])

            def predict(tokens):
                logits = self.model(mx.array(tokens)[None], cache=cache)
                fed.extend(tokens)
                return mx.argmax(logits[0], axis=-1).tolist()

            def trim(count):
                self.trim_cache(cache, count)
                del fed[len(fed) - count:]

            first = predict(ids[n:])[-1]
            try:
                out, hit_limit = lookahead_decode(predict, trim, first, ids, self.eos, max_tokens, DRAFT_TOKENS)
            finally:
                self.cached = fed  # exactly what the cache now holds
            return self.tok.decode(out), len(ids), len(out), hit_limit


def make_handler(model: Model, model_id: str, default_max_tokens: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # no request logging: paths only, but keep the log to timings
            pass

        def _send(self, status: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _from_web_page(self) -> bool:
            if self.headers.get("Origin"):
                self._send(403, {"error": "requests from web pages are not allowed"})
                return True
            return False

        def do_GET(self):
            if self._from_web_page():
                return
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": model_id, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self._from_web_page():
                return
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                messages = req["messages"]
                max_tokens = int(req.get("max_tokens") or default_max_tokens)
            except (ValueError, KeyError, TypeError):
                self._send(400, {"error": "bad request"})
                return
            t = time.perf_counter()
            text, n_prompt, n_out, hit_limit = model.complete(messages, max_tokens)
            ms = (time.perf_counter() - t) * 1000
            print(f"chat: prompt {n_prompt} tokens, output {n_out} tokens, {ms:.0f} ms", flush=True)
            self._send(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                "model": req.get("model", model_id),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "length" if hit_limit else "stop"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out},
            })

        def do_OPTIONS(self):  # no CORS preflight: browsers will not send cross-origin JSON requests
            self._send(405, {"error": "not allowed"})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--chat-template-args", default="{}")
    ap.add_argument("--allowed-origins", default="none")  # accepted for compatibility; web origins are always refused
    args = ap.parse_args()
    model = Model(args.model, args.adapter_path, json.loads(args.chat_template_args))
    server = HTTPServer((args.host, args.port), make_handler(model, args.model, args.max_tokens))
    print(f"cleanup server (copy-ahead decoding) on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
