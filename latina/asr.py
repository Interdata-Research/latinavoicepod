"""Whisper speech-to-text, for `POST /transcribe`.

miniclosedai's voice-backend contract has two halves: TTS (`/voices`,
`/speak/stream`) and ASR (`/transcribe`, used by push-to-talk). This module is
the ASR half, ported from miniclosedai-voice's `asr.py` so the two services
answer `/transcribe` with the same shape: `{text, language, segments}`.

Two deliberate differences from that reference:

* It is language-selectable. The caller's `language` field picks a model per
  language (`LATINA_ASR_MODELS`, default `es=large-v3,en=large-v3-turbo`) and
  forces that language; no language means `LATINA_ASR_MODEL` (turbo) with
  auto-detect. miniclosedai sends the bot's `voice_settings.asr_language`.
* It lives in the same process as VoxCPM2 and loads lazily in a background
  thread after the TTS model, so it never delays `/health` becoming ok. A
  request that arrives before it is loaded waits for the load instead of
  failing. `LATINA_ASR=0` leaves it out entirely (`/transcribe` then 503s).
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, Optional

import numpy as np

from . import config

# Low-signal outputs Whisper produces on near-silence. Same list as
# miniclosedai-voice, plus the Spanish equivalents it hallucinates.
_HALLUCINATIONS = frozenset({
    "thank you.", "thanks for watching.", "thanks for watching!",
    "you", "thank you", "bye.", "bye!", ".", "!", "?", "...",
    "okay.", "ok.", "please subscribe.", "subscribe.",
    "gracias.", "¡gracias!", "gracias por ver el video.",
    "subtítulos realizados por la comunidad de amara.org",
})


def _ffmpeg_decode(audio: bytes, sr: int = 16000) -> np.ndarray:
    """Any container the browser uploads (WebM/Opus, OGG, MP4, WAV…) → mono
    float32 at `sr`. ffmpeg rather than PyAV because MediaRecorder's WebM has
    no duration header and PyAV's demuxer refuses it."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        input=audio, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise ValueError("ffmpeg could not decode the audio: "
                         + proc.stderr.decode(errors="replace")[:300])
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _hf_id(name: str) -> str:
    return name if "/" in name else f"openai/whisper-{name}"


def _base_lang(language: Optional[str]) -> Optional[str]:
    """'es-MX' / 'es_419' / 'ES' → 'es'; '' / 'auto' / None → None."""
    lang = (language or "").strip().lower().replace("_", "-").split("-")[0]
    return None if lang in ("", "auto") else lang


class ASR:
    """One Whisper pipeline per distinct model in config.ASR_MODELS plus the
    auto-detect model. `transcribe(language=...)` picks the language's model and
    forces that language; no language means the auto model and Whisper's own
    language ID."""

    def __init__(self) -> None:
        self.auto_model = _hf_id(config.ASR_MODEL)
        self.by_lang = {lang: _hf_id(m) for lang, m in config.ASR_MODELS.items()}
        self.pipes: dict[str, Any] = {}          # hf model id -> pipeline
        self.errors: dict[str, str] = {}
        self._loaded = threading.Event()
        self._run_lock = threading.Lock()        # one Whisper call at a time

    # Kept for /health and older callers.
    @property
    def model_id(self) -> str:
        return self.auto_model

    @property
    def error(self) -> Optional[str]:
        return "; ".join(f"{m}: {e}" for m, e in self.errors.items()) or None

    @property
    def is_ready(self) -> bool:
        return self._loaded.is_set() and bool(self.pipes)

    def options(self) -> list[dict]:
        """What a client can pick: auto plus each configured language."""
        out = [{"language": "auto", "model": self.auto_model,
                "ready": self.auto_model in self.pipes}]
        out += [{"language": lang, "model": m, "ready": m in self.pipes}
                for lang, m in sorted(self.by_lang.items())]
        return out

    def load(self) -> None:
        """Blocking. Never raises — a broken ASR must not take TTS down. A model
        that fails to load is skipped; its languages fall back to the auto
        model (or any loaded one)."""
        try:
            import torch
            from transformers import pipeline

            cuda = torch.cuda.is_available()
            wanted = [self.auto_model] + [m for m in self.by_lang.values()
                                          if m != self.auto_model]
            for model_id in dict.fromkeys(wanted):
                t0 = time.perf_counter()
                try:
                    pipe = pipeline(
                        task="automatic-speech-recognition",
                        model=model_id,
                        device=0 if cuda else -1,
                        dtype=torch.float16 if cuda else torch.float32,
                    )
                    # One throwaway pass: the first real call otherwise pays
                    # ~4 s of CUDA warmup (measured 4.4-4.9 s vs ~1 s after).
                    pipe({"sampling_rate": 16000,
                          "raw": np.zeros(16000, dtype=np.float32)},
                         return_timestamps=True)
                    self.pipes[model_id] = pipe
                    langs = [l for l, m in self.by_lang.items() if m == model_id]
                    if model_id == self.auto_model:
                        langs.insert(0, "auto")
                    print(f"[latina] asr {model_id} ready in "
                          f"{time.perf_counter()-t0:.0f}s for {langs}", flush=True)
                except Exception as e:
                    self.errors[model_id] = f"{type(e).__name__}: {e}"
                    print(f"[latina] asr {model_id} failed: {self.errors[model_id]}",
                          flush=True)
        except Exception as e:
            self.errors["import"] = f"{type(e).__name__}: {e}"
            print(f"[latina] asr disabled: {self.errors['import']}", flush=True)
        finally:
            self._loaded.set()

    def _pick(self, lang: Optional[str]):
        """(model_id, pipeline) for a base language code, with fallbacks."""
        for model_id in (self.by_lang.get(lang or ""), self.auto_model, *self.pipes):
            if model_id and model_id in self.pipes:
                return model_id, self.pipes[model_id]
        raise RuntimeError(f"ASR unavailable: {self.error}")

    def transcribe(self, audio: bytes, language: Optional[str] = None) -> dict[str, Any]:
        self._loaded.wait()
        lang = _base_lang(language)
        model_id, pipe = self._pick(lang)
        pcm = _ffmpeg_decode(audio)
        kwargs: dict[str, Any] = {}
        # Forcing the language is the point of choosing one. English-only
        # checkpoints (*.en) reject the flag, so they just get the audio.
        if lang and not model_id.endswith(".en"):
            kwargs["language"] = lang
        with self._run_lock:
            # transformers 5 iterates generate_kwargs unconditionally, so
            # `generate_kwargs=None` is a TypeError — omit it when empty.
            extra = {"generate_kwargs": kwargs} if kwargs else {}
            result = pipe(
                {"sampling_rate": 16000, "raw": pcm},
                return_timestamps=True, **extra,
            )
        text = (result.get("text") or "").strip()
        if text.lower() in _HALLUCINATIONS:
            text = ""
        return {
            "text": text,
            "language": lang or ("en" if model_id.endswith(".en") else None),
            "model": model_id,
            "segments": [
                {"start": (c.get("timestamp") or (0, 0))[0],
                 "end": (c.get("timestamp") or (0, 0))[1],
                 "text": c.get("text", "")}
                for c in (result.get("chunks") or [])
            ],
        }


asr = ASR()
