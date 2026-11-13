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
- **Cleanup that reads like you typed it:** a small model fine-tuned for exactly this job
  ([S1-mini FreeFlow cleanup](https://huggingface.co/ayushpatnaik/S1-mini-FreeFlow-cleanup-MLX-6bit), built on
  Superwhisper's S1-mini) removes filler words and false starts, applies "no wait, I mean" corrections, fixes
  punctuation and capitalisation, formats emails, and uses the spellings from your FreeFlow vocabulary. It never
  finishes a sentence you left unfinished.
- **Fast:** about 0.2 s from the end of a short sentence to cleaned text, and 0.4 s for a two-line message (see
  [Speed](#speed)).
- **Spells what's on your screen:** names and technical terms visible in the window you're dictating into are spelled
  correctly, even when the speech model mishears them ("Laura" becomes "LoRA"). See [Screen words](#screen-words).
- **Light on memory:** about 3 GB while you dictate (about 1.5 GB with speech-to-text on the Neural Engine). Models
  load when you start dictating and the cleanup model unloads after 10 idle minutes.
- **Open source** (Apache 2.0), built on Apple's [MLX](https://github.com/ml-explore/mlx).

## Requirements

- An Apple Silicon Mac (M1 or later) on macOS 14 (Sonoma) or later. 8 GB of RAM is enough for the default models.
- About 2 GB of free disk space for the models (about 6 GB if you switch to Gemma 4 E4B).

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
  it, and the models (about 2 GB the first time). It starts at every login, and FreeFlow is pointed at it. Nothing
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

Measured on an M4 Mac from the end of speech to cleaned text, warm (you dictated in the last 10 minutes), with
speech-to-text on the Neural Engine (`freeflowctl ane on`):

| Dictation | Speech-to-text | Cleanup | Total |
|---|---|---|---|
| One sentence (~3 s of speech) | 0.08 s | 0.15 s | **0.22 s** |
| Two lines (~8 s) | 0.13 s | 0.26 s | **0.40 s** |
| A paragraph (~20 s) | 0.15 s | 0.52 s | **0.67 s** |

- **The first dictation after a 10-minute break** reloads the cleanup model, which adds about a second. Loading starts
  the moment you start recording, so much of it happens while you speak.
- **Want it fast every time?** Install with `IDLE_UNLOAD_SECONDS=0` to keep the models loaded permanently.
- With Gemma 4 E4B (`freeflowctl model e4b`) cleanup takes about 0.9 s for a short dictation and 7–14 s for 200 words.

## Choosing a cleanup model

```sh
~/Library/Application\ Support/FreeFlowLocal/freeflowctl model s1    # S1-mini FreeFlow cleanup (default)
~/Library/Application\ Support/FreeFlowLocal/freeflowctl model e4b   # Gemma 4 E4B
~/Library/Application\ Support/FreeFlowLocal/freeflowctl model e2b   # Gemma 4 E2B
```

The command downloads the model if needed and restarts the backend.

- **S1-mini FreeFlow cleanup** (default, 473 MB) is Superwhisper's S1-mini (0.6B) fine-tuned with LoRA on about 5,000
  synthetic and public cleanup examples, then quantized to 6-bit. It was trained on a compact version of FreeFlow's
  instructions, which the backend substitutes for FreeFlow's default prompt; your vocabulary and language settings
  are kept. On our test dictations it scored 73/79 against Gemma 4 E4B's 66/79, at about a fifth of the latency.
  Training data and results are on its [model page](https://huggingface.co/ayushpatnaik/S1-mini-FreeFlow-cleanup-MLX-6bit).
- **Gemma 4 E4B** (about 5 GB loaded) follows free-form instructions. Use it if you have written your own cleanup prompt
  in FreeFlow or use FreeFlow's edit mode; the fine-tuned model only handles FreeFlow's default cleanup, and a custom
  prompt is passed to it unchanged.
- **Gemma 4 E2B** is faster than E4B but used vocabulary and app context less reliably in our tests.

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

## Experimental features

- **Vocabulary boosting** (`freeflowctl boost on`, needs `ane on`): steers the speech model toward the words in your
  FreeFlow vocabulary and on your screen while it decodes, accepting only whole terms. On a synthetic speech benchmark
  it raised hard names and terms spelled right from 18 to 40 of 120, with no change on ordinary sentences.
- **Learning from your corrections** (`freeflowctl learn on`): after a paste, watches the field for up to 30 s and,
  when you fix a misheard name or term, remembers it (`freeflowctl vocab list`). It needs Accessibility permission for
  a small helper, and stores only the learned term and how it was misheard.

## Where this is going

- **Learning from corrections inside FreeFlow itself,** which already has the permission, instead of a separate helper.
- **Fewer misheard names at the source:** confidence-aware cleanup that only lets hints replace words the speech model
  was unsure about.
- **Better small models as they arrive.** Every candidate is measured on the same test dictations before it becomes
  the default.

## Privacy

- Audio, transcripts, screenshots and cleaned text are processed on your Mac and are not sent anywhere.
- The backend listens only on `127.0.0.1` (your own machine).
- Logs contain timings and counts, not what you said or what was on screen.
- The fine-tuned cleanup model was trained only on synthetic and public data, never on anyone's dictations.
- Model files are downloaded once from Hugging Face; after that the backend runs offline.

## Commands

The backend lives in `~/Library/Application Support/FreeFlowLocal`:

```sh
cd ~/Library/Application\ Support/FreeFlowLocal
./freeflowctl status                # is it running, which models are loaded
./freeflowctl restart
./freeflowctl logs
./freeflowctl model s1|e4b|e2b|<id>  # change the cleanup model
./freeflowctl ane on|off            # speech-to-text on the Neural Engine or the GPU
./freeflowctl screen-words on|off   # spell on-screen words (on by default)
./freeflowctl asr-warm on|off       # keep the speech model loaded when idle (off by default)
./freeflowctl boost on|off          # experimental: steer speech-to-text toward your vocabulary (off)
./freeflowctl learn on|off          # learn names/terms from your corrections (off; asks for Accessibility)
./freeflowctl vocab list            # what has been learned (vocab forget <term>, vocab clear)
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
keep their own licences: Orukeet is CC BY-SA 4.0 (Oruk AI); the cleanup model is built on S1-mini by Superwhisper and
keeps its license (Apache 2.0 plus a naming clause); Gemma 4 is Apache 2.0 (Google). This project is not affiliated
with FreeFlow, Oruk AI, Superwhisper, NVIDIA or Google.
