# freeflow-local

Private, offline dictation on your Mac. This is a local backend for [FreeFlow](https://github.com/zachlatta/freeflow),
the free and open-source dictation app. Your audio and text never leave your computer.

- **Speech-to-text:** [Orukeet](https://huggingface.co/oruk/orukeet), a multilingual fine-tune of NVIDIA Parakeet TDT 0.6B v3
- **Cleanup** (filler words, punctuation, names from your vocabulary): Google's Gemma 4 E4B
- **Runs on** Apple's [MLX](https://github.com/ml-explore/mlx), on the Mac's GPU

## Speed

The models are only loaded while you are using them, so they don't hold memory when you aren't dictating.
Loading starts the moment you start recording, so it happens while you speak.

- **Warm (you dictated in the last 10 minutes): about 1 second** from when you stop speaking.
- **First dictation after a break: about 2 seconds,** with full cleanup: the cleanup model loads while you talk.
- **First dictation after restarting your Mac (or the backend) is the slowest,** a few seconds more, because
  FreeFlow's cleanup instructions are also processed for the first time. They stay cached afterwards. Every
  dictation, including this one, gets full cleanup: the backend waits for the cleanup model rather than pasting
  raw text.
- **After 10 minutes without dictating,** the models unload and memory use drops back to almost nothing.

Measured on an M4 Mac with 24 GB of RAM, 6 s of speech, Neural Engine speech-to-text:

| Situation | Gemma 4 E4B (default) | Gemma 4 E2B |
|---|---|---|
| Warm | ~0.9 s | ~0.5 s |
| First dictation after a 10-minute break | ~1.8 s | ~0.6 s |

E2B is faster and smaller (~3.5 GB vs ~5 GB while loaded) but formats less well: in our tests it did not lay
out emails, turn spoken lists into lists, or punctuate long run-on dictations. Switch with
`freeflowctl model e2b` (and back with `freeflowctl model e4b`).

Prefer it fast every time? Install with `IDLE_UNLOAD_SECONDS=0` to keep the models loaded permanently.

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
   - Allow **Microphone** and **Accessibility**. **Screen Recording** is optional: with it, names and technical terms
     visible on screen are spelled correctly (see *Screen words* below). Screenshots never leave your Mac.
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
./freeflowctl screen-words on|off   # use words read from FreeFlow's screenshot (on by default)
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

## Screen words: spell what's on screen

Speech recognition turns rare words into common ones: "LoRA" becomes "Laura", "Orukeet" becomes "or a keet". If
FreeFlow has Screen Recording permission, it sends a screenshot of the active window when you start recording. The
backend reads it with Apple's on-device text recognition (Vision) while you speak, picks out the unusual words
(mixed case like LoRA, acronyms like MLX, words with digits like Qwen3, and capitalised words that aren't in the
macOS dictionary, such as names). Only the words that sound like something in your transcript are passed to the
cleanup model ("Laura" matches LoRA, "or a keet" matches Orukeet); most dictations match none and are cleaned up
exactly as before. The model is told to use a word only to fix something that was misheard, and never to add words
you didn't say.

- **Private:** the screenshot, the recognised text and the word list stay in memory, are never written to disk or
  logged (the log only says how many words were found), and are discarded after that one dictation.
- **Speed:** reading the screenshot takes about 0.4–0.8 s on an M4 and happens while you are still talking. If it
  hasn't finished when cleanup starts, cleanup waits at most 0.5 s and then goes ahead without it.
- **Accuracy:** FreeFlow sends a 1024-pixel screenshot, so editor text is only 8–9 pixels tall. On synthetic windows
  (dark and light themes, 11–14 pt text) the recogniser found 95% of the planted words; a word must be read the same
  way at least twice before it is used, which filters out most misreads of tiny text. On 16 invented dictations with
  a misheard on-screen word, cleanup got 13 right with screen words and 7 without; on 77 other test dictations the
  results were unchanged.
- **Limits:** names spelled far from how they sound ("voy check" for Wojciech) aren't matched, and when the
  recogniser writes a different real name ("Siddharth" with "Siddhant" on screen) it is trusted.
- **Needs** the Xcode Command Line Tools (`xcode-select --install`) to build the helper during `./install.sh`.
- **Turn it off** with `freeflowctl screen-words off` (or don't give FreeFlow Screen Recording permission).

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
