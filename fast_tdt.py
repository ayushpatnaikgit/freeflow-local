"""Faster greedy TDT decoding for parakeet-mlx models (Parakeet / Orukeet).

Same decoding rules as ParakeetTDT.decode_greedy, minus redundant work:
  * encoder side of the joint network is projected once for all frames
  * the LSTM prediction network only re-runs after a non-blank emission
    (on blanks its inputs are unchanged, so its output is identical)
  * one GPU->CPU sync per step instead of three; entropy confidence skipped
WAV input is read directly instead of spawning ffmpeg.
"""

import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from parakeet_mlx import tokenizer
from parakeet_mlx.alignment import AlignedToken, sentences_to_result, tokens_to_sentences
from parakeet_mlx.audio import get_logmel, load_audio
from parakeet_mlx.parakeet import DecodingConfig


def read_audio(path: Path, sample_rate: int) -> mx.array:
    try:
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == sample_rate:
                pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
                return mx.array(pcm).astype(mx.float32) / 32768.0
    except (wave.Error, EOFError):
        pass
    return load_audio(path, sample_rate)  # ffmpeg fallback: resampling / other formats


def _decode(model, features: mx.array) -> list[AlignedToken]:
    V = len(model.vocabulary)  # blank id
    durations = model.durations
    dtype = features.dtype
    feat = features[0]
    length = feat.shape[0]

    enc_proj = model.joint.enc(feat)  # [S, H]
    act, ident, out = model.joint.joint_net

    def predict(token, hc):
        y = mx.array([[token]]) if token is not None else None
        d_out, (h, c) = model.decoder(y, hc)
        return model.joint.pred(d_out.astype(dtype)[0, 0]), (h.astype(dtype), c.astype(dtype))

    def joint_step(enc_t, pred_t):
        logits = out(ident(act(enc_t + pred_t)))
        return mx.stack([mx.argmax(logits[: V + 1]), mx.argmax(logits[V + 1 :])])

    hidden = None
    pred_t, next_hidden = predict(None, None)
    tokens: list[AlignedToken] = []
    step = new_symbols = 0
    while step < length:
        token, decision = joint_step(enc_proj[step], pred_t).tolist()
        duration = durations[decision]
        if token != V:
            tokens.append(AlignedToken(
                token,
                start=step * model.time_ratio,
                duration=duration * model.time_ratio,
                confidence=1.0,
                text=tokenizer.decode([token], model.vocabulary),
            ))
            hidden = next_hidden
            pred_t, next_hidden = predict(token, hidden)  # lazy; evaluated with next step
        step += duration
        new_symbols += 1
        if duration != 0:
            new_symbols = 0
        elif model.max_symbols is not None and model.max_symbols <= new_symbols:
            step += 1
            new_symbols = 0
    return tokens


def transcribe(model, path, chunk_seconds: float = 120.0, overlap_seconds: float = 15.0) -> str:
    """Fast path for clips up to chunk_seconds; longer audio uses the library's chunked path."""
    cfg = model.preprocessor_config
    audio = read_audio(Path(path), cfg.sample_rate)
    if audio.shape[0] < cfg.hop_length:
        return ""
    if audio.shape[0] > chunk_seconds * cfg.sample_rate:
        return model.transcribe(path, chunk_duration=chunk_seconds, overlap_duration=overlap_seconds).text.strip()

    mel = get_logmel(audio, cfg)  # float32 in, as in BaseParakeet.transcribe
    features, _ = model.encoder(mel[None] if mel.ndim == 2 else mel)
    mx.eval(features)
    tokens = _decode(model, features)
    return sentences_to_result(tokens_to_sentences(tokens, DecodingConfig().sentence)).text.strip()
