"""Parity + speed: stock parakeet-mlx transcribe vs fast_tdt, on WAV copies of OpenWhispr clips.

Usage: uv run python bench_fast.py <wav_dir> [model_id]
"""

import sys
import time
from pathlib import Path

from parakeet_mlx import from_pretrained

import fast_tdt

wavs = sorted(Path(sys.argv[1]).glob("*.wav"))
model = from_pretrained(sys.argv[2] if len(sys.argv) > 2 else "itsreidar/orukeet-mlx")
model.transcribe(str(wavs[0]))
fast_tdt.transcribe(model, wavs[0])

stock_t, fast_t, mismatches = [], [], []
for w in wavs:
    t = time.perf_counter(); a = model.transcribe(str(w)).text.strip(); stock_t.append(time.perf_counter() - t)
    t = time.perf_counter(); b = fast_tdt.transcribe(model, w); fast_t.append(time.perf_counter() - t)
    if a != b:
        mismatches.append((w.name, a, b))


def stats(x):
    x = sorted(x)
    return f"median={x[len(x)//2]*1000:.0f}ms p95={x[int(len(x)*.95)]*1000:.0f}ms max={x[-1]*1000:.0f}ms"


print(f"files={len(wavs)} identical text: {len(wavs) - len(mismatches)}/{len(wavs)}")
print("stock:", stats(stock_t))
print("fast :", stats(fast_t))
for n, a, b in mismatches[:5]:
    print(f"  {n}\n    stock: {a[:120]}\n    fast : {b[:120]}")
