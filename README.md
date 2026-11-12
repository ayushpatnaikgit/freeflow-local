# freeflow-local

Private, offline dictation on your Mac. This is a local backend for [FreeFlow](https://github.com/zachlatta/freeflow),
the free and open-source dictation app. Your audio and text never leave your computer.

- **Speech-to-text:** [Orukeet](https://huggingface.co/oruk/orukeet), a multilingual fine-tune of NVIDIA Parakeet TDT 0.6B v3
- **Cleanup** (filler words, punctuation, names from your vocabulary): Google's Gemma 4 E4B
- **Runs on** Apple's [MLX](https://github.com/ml-explore/mlx), on the Mac's GPU

## Speed: slow first, then fast

The models are only loaded while you are using them, so they don't hold memory when you aren't dictating.

- **First dictation after a break is slower.** This is a cold start. Loading begins the moment you start
  recording, so much of it happens while you speak, but you still wait a few seconds after you stop.
- **Every dictation for the next 10 minutes takes about 1 second.** The models stay loaded, and each dictation
  resets the 10-minute timer.
- **After 10 minutes without dictating,** the models unload and memory use drops back to almost nothing.
- **The very first dictation after installing or restarting your Mac is the slowest,** around 10 seconds, because
  the backend also has to process FreeFlow's cleanup instructions for the first time. It keeps them in memory
  afterwards.

Measured on an M4 Mac with 24 GB of RAM, from when you stop speaking:

| Situation | Wait |
|---|---|
| **Warm** (dictated in the last 10 minutes) | **~1 s** (speech-to-text 0.1–0.4 s, cleanup ~0.7 s) |
| After a 10-minute break (5 s of speech) | ~3–5 s |
| First dictation after install or restart | ~8–10 s |

Longer dictations hide more of the loading, so cold starts feel shorter when you speak for longer.

Prefer it fast every time? Install with `IDLE_UNLOAD_SECONDS=0` to keep the models loaded permanently, at the cost
of about 6.3 GB of RAM.

## Requirements

- Apple Silicon Mac (M1 or later). 16 GB of RAM or more recommended.
- About 6 GB of free disk space for the models.
- [Homebrew](https://brew.sh) or [uv](https://docs.astral.sh/uv/).

## Install

1. Install [FreeFlow](https://github.com/zachlatta/freeflow/releases/latest/download/FreeFlow.dmg) and move it to Applications.
2. Install this backend:

   ```sh
   git clone https://github.com/ayushpatnaikgit/freeflow-local.git
   cd freeflow-local
   ./install.sh
   ```

   This downloads the models, starts the backend at every login, and points FreeFlow at it.
3. Open FreeFlow and finish its setup:
   - Allow **Microphone** and **Accessibility**. Screen Recording is optional; the local model does not use screenshots.
   - If it asks for an API key, type any text, for example `local`.
   - Keep **context capture** on (the default). It is what starts loading the models as soon as you start recording.

To update, run `git pull && ./install.sh` again.

## Commands

The backend lives in `~/Library/Application Support/FreeFlowLocal`:

```sh
cd ~/Library/Application\ Support/FreeFlowLocal
./freeflowctl status      # is it running, which models are loaded
./freeflowctl restart
./freeflowctl logs
./freeflowctl configure   # point FreeFlow at the backend again
./freeflowctl uninstall   # stop it and remove it from login
```

Change the idle timeout, for example to 15 minutes: `IDLE_UNLOAD_SECONDS=900 ./freeflowctl install`.

## Optional: speech-to-text on the Neural Engine

```sh
~/Library/Application\ Support/FreeFlowLocal/freeflowctl ane on    # back to the GPU: freeflowctl ane off
```

This runs Orukeet with Core ML on the Apple Neural Engine instead of MLX on the GPU, using Oruk AI's official
Core ML build (6-bit weights). It needs the Xcode Command Line Tools (`xcode-select --install`), downloads about
470 MB, and compiles the model for your Mac once (about 15 s).

Measured on an M4: speech-to-text takes about 60–80 ms instead of 150–250 ms, loads in 0.16 s instead of 1–7 s,
and leaves the GPU free for the cleanup model. Text matched the MLX version on 99% of words (924 words, 24
recordings, up to 2 minutes long); the differences were number spellings ("8" vs "eight"), apostrophes and one
misheard word. If the Neural Engine helper fails, the backend falls back to MLX automatically.

## How it works

FreeFlow talks to any OpenAI-compatible API. This backend serves one on `http://127.0.0.1:8081/v1`:

- `/v1/audio/transcriptions` runs Orukeet in-process with [parakeet-mlx](https://github.com/senstella/parakeet-mlx) and a
  faster greedy decoder (`fast_tdt.py`, ~30% faster with identical output).
- Hesitation sounds (uh, um, uhm, hmm) are removed right after transcription, so they disappear even when
  FreeFlow's "preserve exact wording" mode is on or cleanup is skipped. Real words such as "er", "ah" and "mm"
  are kept. Set `STRIP_FILLERS=0` in the environment to turn this off.
- `/v1/chat/completions` is forwarded to an `mlx_lm.server` process that the backend starts and stops on demand. The
  model name FreeFlow sends is ignored, so FreeFlow's model settings don't matter.

When you start recording, FreeFlow immediately asks what app you're in. That request wakes the backend: both models
start loading, and FreeFlow's cleanup instructions are processed into the model's cache while you speak. By the time
your audio arrives, most of the work is done.

## Troubleshooting

- **"No internet" or "Submission failed" in FreeFlow:** the backend isn't running. Run `./freeflowctl status`, then
  `./freeflowctl restart`.
- **The dictation key does nothing, but FreeFlow's Dictate button works:** check that FreeFlow has Accessibility
  permission. On non-Apple keyboards the key labels may not match macOS's Option and Command keys, so try the key
  next to the one you expect, or choose a different shortcut in FreeFlow.
- **The first dictation is slow:** expected after 10 idle minutes; see *Speed* above.

## Uninstall

```sh
~/Library/Application\ Support/FreeFlowLocal/freeflowctl uninstall
rm -rf ~/Library/Application\ Support/FreeFlowLocal
```

The models stay in `~/.cache/huggingface/hub` until you delete them.

## Credits and licences

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights are downloaded from Hugging Face and
keep their own licences: Orukeet is CC BY-SA 4.0 (Oruk AI), and Gemma 4 is Apache 2.0 (Google). This project is not
affiliated with FreeFlow, Oruk AI, NVIDIA or Google.
