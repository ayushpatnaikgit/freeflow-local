# freeflow-local

[Website](https://ayushpatnaikgit.github.io/freeflow-local/) · [Latest release](https://github.com/ayushpatnaikgit/freeflow-local/releases/latest)

**Private, offline dictation for your Mac.** Hold a key, speak, and clean, punctuated text appears wherever you are
typing. Speech recognition and cleanup both run on your Mac's own chips, so your audio, your words and your screen
never leave the computer, and it works without an internet connection.

freeflow-local is the engine behind [FreeFlow](https://github.com/zachlatta/freeflow), the free and open-source
dictation app for macOS. FreeFlow normally sends your voice to a cloud service; this replaces that with models
running locally on Apple Silicon.

## What you get

- **Accurate speech-to-text:** [Orukeet](https://huggingface.co/oruk/orukeet), a multilingual fine-tune of NVIDIA
  Parakeet TDT 0.6B v3. It can run on the Apple Neural Engine, taking about 0.1–0.2 s per dictation.
- **Cleanup that reads like you typed it:** Google's Gemma 4 E4B removes filler words and false starts, fixes
  punctuation and capitalisation, formats emails, and uses the spellings from your FreeFlow vocabulary.
- **Spells what's on your screen:** names and technical terms visible in the window you're dictating into are spelled
  correctly, even when the speech model mishears them ("Laura" becomes "LoRA"). See [Screen words](#screen-words).
- **Light on memory:** models load when you start dictating and unload after 10 idle minutes.
- **Open source** (Apache 2.0), built on Apple's [MLX](https://github.com/ml-explore/mlx).

## Requirements

- An Apple Silicon Mac (M1 or later) on macOS 14 (Sonoma) or later, with 16 GB of RAM or more.
- About 6 GB of free disk space for the models.

## Install

Paste this into Terminal:

```sh
curl -fsSL https://raw.githubusercontent.com/ayushpatnaikgit/freeflow-local/main/get.sh | sh
```

That one command sets up everything:

- **FreeFlow**, the dictation app (hotkey, microphone, pasting), if you don't have it yet. It downloads the official
  release from [github.com/zachlatta/freeflow](https://github.com/zachlatta/freeflow) and installs it only after
  checking that it is signed by FreeFlow's developer and notarized by Apple.
- **freeflow-local**, the on-device engine that FreeFlow talks to instead of a cloud service: the latest release,
  checked against its SHA-256, plus [uv](https://docs.astral.sh/uv/) (the Python package manager) if you don't have
  it, and the models (about 6 GB the first time). It starts at every login, and FreeFlow is pointed at it. Nothing
  needs compiling.

FreeFlow then opens. Finish its setup:

- Allow **Microphone** and **Accessibility**.
- Allow **Screen Recording** if you want on-screen names and terms spelled correctly. It is optional, and the
  screenshot never leaves your Mac.
- If FreeFlow asks for an API key, type any text, for example `local`.
- Keep **context capture** on (the default). It is what starts loading the models as soon as you start recording.

Already have FreeFlow? The command leaves it as it is. To skip it entirely, use `sh -s -- --no-freeflow` instead of
`sh`.

To update, run the same command again.

**From source** (to change the code): `git clone https://github.com/ayushpatnaikgit/freeflow-local.git`, then run
`./install.sh` in the folder. Building from source needs the Xcode Command Line Tools (`xcode-select --install`) for
the two small Swift helpers; to update, `git pull && ./install.sh`.

## Speed

Loading starts the moment you start recording, so most of it happens while you speak. Times are from when you stop
speaking, on an M4 Mac with 24 GB of RAM, 6 s of speech:

| Situation | Gemma 4 E4B (default) | Gemma 4 E2B |
|---|---|---|
| Warm (you dictated in the last 10 minutes) | ~0.9 s | ~0.5 s |
| First dictation after a 10-minute break | ~1.8 s | ~0.6 s |

- **The first dictation after restarting your Mac (or the backend) is the slowest,** a few seconds more, because
  FreeFlow's cleanup instructions are processed for the first time. They stay cached afterwards. Every dictation,
  including that one, gets full cleanup.
- **Long dictations take longer,** because the cleanup model rewrites the whole text: in our tests, 170–250 words
  (a minute or more of speech) took 7–14 s.
- **Want it fast every time?** Install with `IDLE_UNLOAD_SECONDS=0` to keep the models loaded permanently (about
  6 GB of memory).

## Choosing a cleanup model

```sh
~/Library/Application\ Support/FreeFlowLocal/freeflowctl model e4b   # Gemma 4 E4B (default)
~/Library/Application\ Support/FreeFlowLocal/freeflowctl model e2b   # Gemma 4 E2B: faster, smaller
~/Library/Application\ Support/FreeFlowLocal/freeflowctl model mlx-community/<model>   # any MLX chat model
```

The command downloads the model if needed and restarts the backend.

- **Gemma 4 E4B** is the default and the most accurate of the models we tested. It uses about 5 GB while loaded.
- **Gemma 4 E2B** is about twice as fast and uses about 3.5 GB, but follows FreeFlow's instructions less well: in our
  tests it used custom vocabulary and app context less often, and did not lay out emails, turn spoken lists into
  lists, or punctuate long run-on dictations.
- **Other models** work if they are MLX chat models, but FreeFlow's cleanup instructions are long and detailed.
  Small general-purpose models (we tried Qwen3 0.6B and 1.7B) mostly returned the transcript unchanged.

## Screen words

Speech recognition turns rare words into common ones: "LoRA" becomes "Laura", "Orukeet" becomes "or a keet". If
FreeFlow has Screen Recording permission, it sends a screenshot of the active window when you start recording. The
backend reads the text in it with Apple's on-device text recognition while you speak, and picks out the unusual
words: mixed case like LoRA, acronyms like MLX, words with digits like Qwen3, and names that aren't in the macOS
dictionary. Only the words that sound like something you said are given to the cleanup model, which is told to use
them only to fix a mishearing and never to add words you didn't say. Most dictations match none and are cleaned up
exactly as before.

- **Private:** the screenshot, the recognised text and the words stay in memory, are never written to disk or
  logged (the log only records how many words were found and matched), and are discarded after that dictation.
- **Speed:** reading the screenshot takes about 0.4–1 s and happens while you talk. Cleanup waits at most 0.5 s for
  it.
- **Accuracy:** on 16 invented dictations with a misheard on-screen name or term, cleanup fixed 13 with screen words
  and 7 without. On 77 other test dictations the results were unchanged.
- **Limits:** it helps most with names and terms that are misheard as other words. Code-like tokens
  (`freeflowctl` heard as "FreeFlow CTL") and misheard everyday words ("green words" for "screen words") usually
  stay as heard, and names spelled far from how they sound ("voy check" for Wojciech) aren't matched. For terms you
  use often, add them to FreeFlow's custom vocabulary instead; the cleanup model treats that list as authoritative.
- **Turn it off** with `freeflowctl screen-words off`, or don't give FreeFlow Screen Recording permission.

## Speech-to-text on the Neural Engine (optional)

```sh
~/Library/Application\ Support/FreeFlowLocal/freeflowctl ane on    # back to the GPU: freeflowctl ane off
```

This runs Orukeet with Core ML on the Apple Neural Engine instead of MLX on the GPU, using Oruk AI's official
Core ML build (6-bit weights). It downloads about 470 MB and compiles the model for your Mac once (about 15 s).

Measured on an M4: speech-to-text takes about 60–80 ms instead of 150–250 ms, loads in 0.16 s instead of 1–7 s,
and leaves the GPU free for the cleanup model. Text matched the MLX version on 99% of words (924 words, 24
recordings, up to 2 minutes long); the differences were number spellings ("8" vs "eight"), apostrophes and one
misheard word. If the Neural Engine helper fails, the backend falls back to MLX automatically.

## Where this is going

- **Use the Neural Engine for as much as possible.** Speech-to-text already runs there, which keeps the GPU free and
  uses little power. Next are the screenshot reading and, as Apple's on-device tools for language models mature, the
  cleanup model itself.
- **Much faster cleanup as better small models arrive.** The cleanup model is a single setting
  (`freeflowctl model`), and every candidate is measured on the same set of test dictations before it becomes the
  default. Long dictations are the main target: cleaning a long dictation's sentences in parallel made it about twice
  as fast in testing.
- **Fine-tuning for dictation.** A small adapter (LoRA) trained on FreeFlow's exact requests (self-corrections,
  custom vocabulary, app context and screen words) so that a smaller, faster model can match today's quality.
  Training data is synthetic or public; nothing from anyone's dictations.

## Privacy

- Audio, transcripts, screenshots and cleaned text are processed on your Mac and are not sent anywhere.
- The backend listens only on `127.0.0.1` (your own machine).
- Logs contain timings and counts, not what you said or what was on screen.
- Model files are downloaded once from Hugging Face; after that the backend runs offline.

## Commands

The backend lives in `~/Library/Application Support/FreeFlowLocal`:

```sh
cd ~/Library/Application\ Support/FreeFlowLocal
./freeflowctl status                # is it running, which models are loaded
./freeflowctl restart
./freeflowctl logs
./freeflowctl model e4b|e2b|<id>    # change the cleanup model
./freeflowctl ane on|off            # speech-to-text on the Neural Engine or the GPU
./freeflowctl screen-words on|off   # spell on-screen words (on by default)
./freeflowctl asr-warm on|off       # keep the speech model loaded when idle (off by default)
./freeflowctl boost on|off          # experimental: steer speech-to-text toward your vocabulary (off)
./freeflowctl configure             # point FreeFlow at the backend again
./freeflowctl uninstall             # stop it and remove it from login
```

Change the idle timeout, for example to 15 minutes: `IDLE_UNLOAD_SECONDS=900 ./freeflowctl install`.

## How it works

FreeFlow talks to any OpenAI-compatible API. This backend serves one on `http://127.0.0.1:8081/v1`:

- `/v1/audio/transcriptions` runs Orukeet, on the Neural Engine through a small Core ML helper or in-process with
  [parakeet-mlx](https://github.com/senstella/parakeet-mlx) and a faster greedy decoder (`fast_tdt.py`, ~30% faster
  with identical output).
- Hesitation sounds (uh, um, uhm, hmm) are removed right after transcription, so they disappear even when
  FreeFlow's "preserve exact wording" mode is on or cleanup is skipped. Real words such as "er", "ah" and "mm" are
  kept. Set `STRIP_FILLERS=0` in the environment to turn this off.
- `/v1/chat/completions` is forwarded to an `mlx_lm.server` process that the backend starts and stops on demand.
  The model name FreeFlow sends is ignored, so FreeFlow's model settings don't matter.
- When you start recording, FreeFlow asks what app you're in. That request wakes the backend: both models start
  loading, FreeFlow's cleanup instructions are processed into the model's cache, and the screenshot is read, all
  while you speak.

## Troubleshooting

- **"No internet" or "Submission failed" in FreeFlow:** the backend isn't running. Run `./freeflowctl status`, then
  `./freeflowctl restart`.
- **The dictation key does nothing, but FreeFlow's Dictate button works:** check that FreeFlow has Accessibility
  permission. On non-Apple keyboards the key labels may not match macOS's Option and Command keys, so try the key
  next to the one you expect, or choose a different shortcut in FreeFlow.
- **The first dictation is slow:** expected after 10 idle minutes or a restart; see [Speed](#speed).
- **A name or term on screen is still misspelled:** check that FreeFlow has Screen Recording permission and that the
  word is visible in the window you're dictating into, or add it to FreeFlow's custom vocabulary.

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
