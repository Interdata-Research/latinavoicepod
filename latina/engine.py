"""VoxCPM2 wrapper — loads the model once, keeps it in memory, synthesizes.

Deliberately NOT a port of the dealership pipeline it came from. The original
wraps English brand names in quotes so VoxCPM2 pronounces "Honda Civic" in
English inside a Spanish sentence, and it builds that brand list from a vehicle
database. That is car-specific vocabulary; none of it is here. What is here is
the voice.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from . import config


class Voice:
    """A reference clip on disk: `<id>.wav`, plus an optional `<id>.txt`
    transcript and an optional `<id>.json` sidecar.

    The sidecar is the same convention miniclosedai-voice uses —
    `{"name": ..., "language": "en", "gender": "F"}` — and it is what lets one
    instance serve voices in more than one language: `/voices` buckets each
    clip under its `language`. No sidecar means `LATINA_LANGUAGE` (Spanish) and
    a name derived from the id, which is exactly what `/voices` returned before
    sidecars existed. A malformed sidecar is ignored rather than hiding the
    voice.
    """

    def __init__(self, wav: Path):
        self.id = wav.stem
        self.wav = wav
        txt = wav.with_suffix(".txt")
        self.text: Optional[str] = (
            txt.read_text(encoding="utf-8").strip() if txt.is_file() else None
        )
        meta: dict = {}
        side = wav.with_suffix(".json")
        if side.is_file():
            try:
                meta = json.loads(side.read_text(encoding="utf-8")) or {}
            except (OSError, ValueError):
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
        self.name: str = str(meta.get("name") or self.id.replace("_", " "))
        self.language: str = str(
            meta.get("language") or config.DEFAULT_LANGUAGE).lower()
        self.gender: Optional[str] = meta.get("gender") or None

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "language": self.language,
                "gender": self.gender, "reference_text": self.text}


class LatinaVoiceEngine:
    """VoxCPM2, loaded once and held in memory for the process's lifetime.

    Generation is synchronous and single-threaded on the GPU, so a lock
    serializes calls: two concurrent requests would otherwise interleave on the
    same CUDA context and both come out slower (or wrong).
    """

    def __init__(self) -> None:
        self.model = None
        self.sample_rate: int = 48000     # VoxCPM2's audiovae_v2 output rate
        self.voices: dict[str, Voice] = {}
        self.loaded_at: Optional[float] = None
        self._lock = threading.Lock()
        # Guards runtime re-scans only. Deliberately NOT `self._lock`: a rescan
        # must never queue behind a 10-second generation, nor delay one.
        self._voices_lock = threading.Lock()
        # voice id -> ((path, mtime_ns, size), prompt_cache). See _prompt_cache().
        self._prompt_caches: dict[str, tuple] = {}
        self._pc_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def scan_voices(self) -> dict[str, Voice]:
        d = Path(config.VOICES_DIR)
        if not d.is_dir():
            raise RuntimeError(f"voices directory not found: {d}")
        found = {w.stem: Voice(w) for w in sorted(d.glob("*.wav"))}
        if not found:
            raise RuntimeError(f"no .wav reference clips in {d}")
        return found

    def load(self) -> None:
        """Pull the model onto the GPU. Slow (seconds to minutes on first run,
        because the weights download); called once at startup."""
        from voxcpm import VoxCPM

        t0 = time.perf_counter()
        self.voices = self.scan_voices()
        print(f"[latina] loading {config.MODEL_ID} "
              f"(optimize={config.OPTIMIZE}, denoiser={config.LOAD_DENOISER})…",
              flush=True)
        self.model = VoxCPM.from_pretrained(
            config.MODEL_ID,
            load_denoiser=config.LOAD_DENOISER,
            optimize=config.OPTIMIZE,
        )
        self.loaded_at = time.time()
        print(f"[latina] ready in {time.perf_counter()-t0:.0f}s · "
              f"{self.sample_rate} Hz · voices: {list(self.voices)}", flush=True)

        self.warm_prompt_caches()

        if config.WARMUP:
            t = time.perf_counter()
            try:
                self.synthesize("Hola, buenas tardes.")
                print(f"[latina] warmed up in {time.perf_counter()-t:.1f}s", flush=True)
            except Exception as e:                      # non-fatal
                print(f"[latina] warmup failed ({type(e).__name__}: {e})", flush=True)

    def rescan_voices(self) -> dict[str, Voice]:
        """Pick up voices added or removed since startup.

        Unlike `scan_voices()` this never raises and never empties the registry
        — those are the two ways a refresh could take down a live service. The
        new dict is built first and then rebound in one statement, so a
        concurrent `/voices` handler iterates a consistent snapshot instead of
        hitting "dictionary changed size during iteration".
        """
        with self._voices_lock:
            d = Path(config.VOICES_DIR)
            if not d.is_dir():
                return self.voices
            try:
                found = {w.stem: Voice(w) for w in sorted(d.glob("*.wav"))}
            except OSError:
                return self.voices
            if not found:
                return self.voices          # keep the last good set
            self.voices = found             # single rebind = atomic swap
        # Drop cached references for voices that no longer exist. Entries for
        # voices that merely changed re-key themselves on (mtime, size).
        with self._pc_lock:
            for gone in set(self._prompt_caches) - set(found):
                self._prompt_caches.pop(gone, None)
        return self.voices

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def resolve(self, voice_id: Optional[str]) -> Voice:
        v = self.voices.get(voice_id or config.DEFAULT_VOICE)
        if v is None:
            raise KeyError(
                f"unknown voice {voice_id!r}; available: {sorted(self.voices)}")
        return v

    # ------------------------------------------------------------------ #
    def _prompt_cache(self, v: Voice):
        """VoxCPM2's encoded reference clip, cached per voice.

        `model.generate()` calls `build_prompt_cache()` on every single request,
        which re-reads the clip from disk, resamples it to 16 kHz through
        librosa and re-runs the audio VAE encoder — work that depends on
        nothing but the file, sitting directly in the time-to-first-chunk path.

        Reusing one cache across requests is safe on both counts that matter:

        * Generation only ever *reads* it. `_generate_with_prompt_cache` passes
          `ref_audio_feat` into `torch.cat`, which allocates new tensors and
          never writes back (verified by comparing the tensor before and after
          a full generation).
        * The encoder is not a sampler — `AudioVAE.encode` returns the
          posterior mean (`["mu"]`), not a draw from it. Repeated builds are
          not bit-identical, but only because the conv kernels reduce in a
          non-deterministic order: measured max drift 7e-4 on values spanning
          ±5.8 (~0.01%), and the encoder is equally non-reproducible when
          handed the very same input tensor. Caching freezes one draw of that
          float noise, which if anything makes successive requests *more*
          consistent with each other.

        Keyed on (path, mtime, size), so re-uploading a clip through the studio
        rebuilds the entry on its own — no invalidation to remember. Returns
        None when the fast path is unavailable, and the caller falls back.
        """
        if not config.PROMPT_CACHE:
            return None
        tts = getattr(self.model, "tts_model", None)
        if tts is None or not hasattr(tts, "generate_with_prompt_cache_streaming"):
            return None                      # not a VoxCPM2 model — old path
        try:
            st = v.wav.stat()
        except OSError:
            return None
        key = (str(v.wav), st.st_mtime_ns, st.st_size)

        with self._pc_lock:
            hit = self._prompt_caches.get(v.id)
        if hit is not None and hit[0] == key:
            return hit[1]

        try:
            cache = tts.build_prompt_cache(reference_wav_path=str(v.wav))
        except Exception as e:               # never fail a request over a cache
            print(f"[latina] prompt cache unavailable for {v.id!r} "
                  f"({type(e).__name__}: {e}); using the per-request path",
                  flush=True)
            return None
        with self._pc_lock:
            self._prompt_caches[v.id] = (key, cache)
        return cache

    def warm_prompt_caches(self) -> None:
        """Build every voice's reference cache up front, so the first request
        for a voice is not the one that pays for it."""
        for v in list(self.voices.values()):
            self._prompt_cache(v)

    @staticmethod
    def _clean(text: str) -> str:
        """The same normalisation `voxcpm.core._generate` applies before
        tokenising. Replicated because the fast path bypasses that function."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")
        return re.sub(r"\s+", " ", text.replace("\n", " "))

    # ------------------------------------------------------------------ #
    def synthesize(self, text: str, voice: Optional[str] = None,
                   language: Optional[str] = None) -> np.ndarray:
        """Whole utterance as mono float32 at `self.sample_rate`."""
        if not self.is_ready:
            raise RuntimeError("engine not loaded")
        v = self.resolve(voice)
        cache = self._prompt_cache(v)
        with self._lock:
            if cache is not None:
                wav, _, _ = self.model.tts_model.generate_with_prompt_cache(
                    target_text=self._clean(text),
                    prompt_cache=cache,
                    min_len=2,
                    max_len=4096,          # the value voxcpm.core passes
                    cfg_value=config.CFG_VALUE,
                    inference_timesteps=config.INFERENCE_TIMESTEPS,
                    retry_badcase=config.RETRY_BADCASE,
                )
                audio = wav.squeeze(0).cpu().numpy()
            else:
                audio = self.model.generate(
                    text=text,
                    reference_wav_path=str(v.wav),
                    normalize=False,
                    denoise=False,
                    cfg_value=config.CFG_VALUE,
                    inference_timesteps=config.INFERENCE_TIMESTEPS,
                    retry_badcase=config.RETRY_BADCASE,
                )
        return np.asarray(audio, dtype=np.float32).squeeze()

    def synthesize_streaming(
        self, text: str, voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Generator[np.ndarray, None, None]:
        """Chunks as they are produced, so the caller can start playing before
        the whole utterance exists. This is the path a phone call wants."""
        if not self.is_ready:
            raise RuntimeError("engine not loaded")
        v = self.resolve(voice)
        # Built OUTSIDE the lock: encoding the reference does not touch
        # generation state, so it need not hold up another request's audio.
        cache = self._prompt_cache(v)
        with self._lock:
            if cache is not None:
                gen = self.model.tts_model.generate_with_prompt_cache_streaming(
                    target_text=self._clean(text),
                    prompt_cache=cache,
                    min_len=2,
                    max_len=4096,          # the value voxcpm.core passes
                    cfg_value=config.CFG_VALUE,
                    inference_timesteps=config.INFERENCE_TIMESTEPS,
                    # A retry would discard chunks already streamed to the
                    # caller, so it is doubly wrong on this path. (VoxCPM2
                    # force-disables it in streaming mode anyway.)
                    retry_badcase=False,
                )
                try:
                    for wav, _, _ in gen:
                        yield np.asarray(wav.squeeze(0).cpu().numpy(),
                                         dtype=np.float32).squeeze()
                finally:
                    gen.close()
            else:
                for chunk in self.model.generate_streaming(
                    text=text,
                    reference_wav_path=str(v.wav),
                    normalize=False,
                    denoise=False,
                    cfg_value=config.CFG_VALUE,
                    inference_timesteps=config.INFERENCE_TIMESTEPS,
                    retry_badcase=config.RETRY_BADCASE,
                ):
                    yield np.asarray(chunk, dtype=np.float32).squeeze()


# One engine per process.
engine = LatinaVoiceEngine()


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def to_pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1,1] → little-endian int16, which is what browsers and
    telephony stacks actually want to receive."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Minimal 16-bit mono WAV. Avoids a soundfile/libsndfile dependency just
    to write a 44-byte header."""
    import struct
    pcm = to_pcm16(audio)
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE" + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data" + struct.pack("<I", n) + pcm)
