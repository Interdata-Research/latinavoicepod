"""Configuration, all of it overridable by environment variable.

Nothing here reaches outside this folder — the reference clips ship in
`voices/`, so the project is a `zip -r` away from running on another box.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = Path(os.getenv("LATINA_VOICES_DIR", ROOT / "voices"))

# openbmb/VoxCPM2 — voice-cloning TTS with native ES/EN/FR. It is the engine
# behind the Spanish voice this project exists to serve.
MODEL_ID = os.getenv("LATINA_MODEL_ID", "openbmb/VoxCPM2")

# The voice served by default. `es_f_19` is a Latin-American female with a warm,
# receptionist-appropriate timbre — picked by listen test, not by accident.
DEFAULT_VOICE = os.getenv("LATINA_VOICE", "es_f_19")
DEFAULT_LANGUAGE = os.getenv("LATINA_LANGUAGE", "es")

# Generation settings, tuned for STREAMING. The library defaults (2.0 / 10) are
# what we use: measured RTF 1.03 against 1.52 for the 3.0 / 20 "quality" preset.
# That gap decides whether streaming works at all — above RTF 1.0 the player
# runs out of audio between chunks and the speech stutters, so quality settings
# that push past realtime make the voice worse, not better.
# Set LATINA_CFG=3.0 / LATINA_TIMESTEPS=20 for offline batch work where the
# extra prosodic steadiness is free.
CFG_VALUE = float(os.getenv("LATINA_CFG", "2.0"))
INFERENCE_TIMESTEPS = int(os.getenv("LATINA_TIMESTEPS", "10"))

# VoxCPM2 re-generates the WHOLE utterance when it dislikes the audio-to-text
# ratio — up to 3 times, silently. It is on by default in the library and it is
# a latency disaster for short conversational lines: measured "Sí, con él."
# (1.1 s of audio) taking 9.1 s, RTF 8.12, against 1.4 for a normal sentence.
# It also defeats streaming, since a retry throws away everything already sent.
# Off here; set LATINA_RETRY=1 if you would rather have the occasional retry
# than the occasional bad clip.
RETRY_BADCASE = os.getenv("LATINA_RETRY", "0").lower() in ("1", "true", "yes")

# torch.compile. OFF by default: on some GPUs (notably the GB10 / DGX Spark,
# which reports "Not enough SMs to use max_autotune_gemm") warmup compiles
# forever and never yields. On a normal datacenter GPU — A100, L40S, 4090 —
# turning it on is usually a real speedup, so try LATINA_OPTIMIZE=1 there and
# keep it off if the first request never returns.
OPTIMIZE = os.getenv("LATINA_OPTIMIZE", "0").lower() in ("1", "true", "yes")

# VoxCPM2's denoiser cleans noisy reference clips. Ours is clean, and loading it
# costs memory and startup time.
LOAD_DENOISER = os.getenv("LATINA_DENOISER", "0").lower() in ("1", "true", "yes")

# Cache VoxCPM2's encoded reference clip between requests. Without it every
# request re-reads the clip from disk, resamples it to 16 kHz through librosa
# and re-runs the audio VAE encoder — identical work every time, since the
# result is a pure function of the file. Measured on an RTX A6000: it is the
# single largest avoidable slice of time-to-first-chunk. Set to 0 to fall back
# to the library's per-request path (useful for A/B measurement).
PROMPT_CACHE = os.getenv("LATINA_PROMPT_CACHE", "1").lower() in ("1", "true", "yes")

HOST = os.getenv("LATINA_HOST", "0.0.0.0")
PORT = int(os.getenv("LATINA_PORT", "8000"))

# Optional shared secret. When set, every request needs
# `Authorization: Bearer <key>`. Leave empty on a private network.
API_KEY = os.getenv("LATINA_API_KEY", "")

# Warm the model at startup so the first real request is not the slow one.
WARMUP = os.getenv("LATINA_WARMUP", "1").lower() in ("1", "true", "yes")

# --- voice studio (the web GUI) -------------------------------------------- #
# The studio is a detachable module: `latina/api.py` imports it inside a
# try/except, so a missing dependency disables the GUI instead of killing the
# TTS service. LATINA_STUDIO=0 turns it off outright.
STUDIO_ENABLED = os.getenv("LATINA_STUDIO", "1").lower() in ("1", "true", "yes")
STUDIO_STATIC_DIR = Path(os.getenv("LATINA_STUDIO_STATIC", ROOT / "static"))

# Uploads. Starlette spools an UploadFile over 1 MB to disk, so the cap is
# enforced while reading, not after — otherwise one request can fill the volume.
STUDIO_MAX_UPLOAD_MB = int(os.getenv("LATINA_STUDIO_MAX_UPLOAD_MB", "20"))
STUDIO_MAX_VOICES = int(os.getenv("LATINA_STUDIO_MAX_VOICES", "50"))
STUDIO_FFMPEG_TIMEOUT = int(os.getenv("LATINA_STUDIO_FFMPEG_TIMEOUT", "20"))

# Reference clips are stored at this rate. VoxCPM2 resamples every reference to
# 16 kHz before encoding (audio_vae.sample_rate), so the model only ever reads
# 0-8 kHz; 24 kHz keeps all of that with headroom at half the size of 48 kHz.
STUDIO_CANONICAL_SR = int(os.getenv("LATINA_STUDIO_SR", "24000"))

# --- speech-to-text (POST /transcribe) ------------------------------------- #
# The ASR half of miniclosedai's voice-backend contract (push-to-talk). Loads
# in the background after VoxCPM2, so it never delays /health. Multilingual by
# default because the callers speak Spanish; `medium.en` etc. also work, and
# any HF id with a slash is used verbatim. ~1.6 GB of VRAM in fp16.
ASR_ENABLED = os.getenv("LATINA_ASR", "1").lower() in ("1", "true", "yes")
# Model for auto-detect (no `language` on the request, or one not listed below).
ASR_MODEL = os.getenv("LATINA_ASR_MODEL", "large-v3-turbo")
# Per-language models, "lang=model,lang=model". A request with `language=es`
# uses the `es` model AND forces Spanish decoding (no guessing — short clips like
# "sí" are where language ID goes wrong). Spanish gets full large-v3, the most
# accurate open Whisper on Spanish; English stays on turbo, which is within
# noise of large-v3 on English at a fraction of the latency. Models shared by
# several entries load once. large-v3 is ~3 GB of VRAM in fp16, turbo ~1.6 GB.
ASR_MODELS = {
    k.strip().lower(): v.strip()
    for k, _, v in (item.partition("=") for item in
                    os.getenv("LATINA_ASR_MODELS", "es=large-v3,en=large-v3-turbo").split(","))
    if k.strip() and v.strip()
}
