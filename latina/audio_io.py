"""Decode, measure and store reference clips.

Pure functions only — nothing here imports FastAPI, so this module can be
unit-tested on its own and can never break the app at import time.

The measurements exist because of one fact about VoxCPM2: every reference clip
is resampled to 16 kHz mono before it is encoded
(`voxcpm/model/voxcpm2.py:412`, `audio_vae.sample_rate == 16000`; the 48 kHz you
see elsewhere is the *output* rate). The model therefore only ever reads
0-8 kHz, so what decides whether a clone sounds clear is **how much of that band
the clip actually fills** — not the sample rate written in its header. The
shipped `es_f_19.wav` fills about a third of it (99% of its energy below
~2.8 kHz) and that, not any generation parameter, is why it sounds muffled.
"""

from __future__ import annotations

import io
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# soundfile and librosa are declared dependencies of voxcpm, so they are present
# wherever the engine runs.
import soundfile as sf

CANONICAL_SR = 24000
HARD_MIN_S, HARD_MAX_S = 1.5, 30.0
SOFT_MIN_S, SOFT_MAX_S = 4.0, 15.0

# The band VoxCPM2 actually encodes from.
MODEL_BAND_HZ = 8000


class AudioRejected(ValueError):
    """The clip cannot be stored. `reasons` is a list of human-readable strings."""

    def __init__(self, *reasons: str):
        self.reasons = list(reasons)
        super().__init__("; ".join(reasons))


# --------------------------------------------------------------------------- #
# ids
# --------------------------------------------------------------------------- #

_SLUG_STRIP = re.compile(r"[^a-z0-9_-]+")
_SLUG_COLLAPSE = re.compile(r"[_-]{2,}")


def slugify_voice_id(raw: str, fallback: str = "") -> str:
    """Reduce arbitrary text to a safe voice id.

    User text is never joined to a path before passing through here, so path
    traversal is impossible by construction rather than by filtering.
    """
    s = (raw or "").strip().lower().replace(" ", "_")
    s = _SLUG_STRIP.sub("", s)
    s = _SLUG_COLLAPSE.sub("_", s).strip("_-")
    if not s and fallback:
        return slugify_voice_id(fallback)
    if not s or s in (".", "..") or len(s) < 2:
        raise AudioRejected(
            "voice id must be at least 2 characters of letters, digits, _ or -")
    return s[:48]


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #

def _decode_ffmpeg(raw: bytes, timeout: int) -> tuple[np.ndarray, int]:
    """webm/opus and m4a/aac, which libsndfile cannot read.

    That is exactly what a browser's MediaRecorder produces, so this path is
    not an edge case — it is how every in-browser recording arrives.
    """
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            fh.write(raw)
            tmp = fh.name
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", tmp,
             "-map", "0:a:0", "-f", "wav", "-c:a", "pcm_s16le", "-"],
            capture_output=True, timeout=timeout,
        )
        if proc.returncode != 0 or not proc.stdout:
            msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            raise AudioRejected(
                "could not decode this file — is it really audio? "
                + (msg[-1] if msg else ""))
        y, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
        return y, sr
    except subprocess.TimeoutExpired:
        raise AudioRejected("decoding timed out — the file is too long or corrupt")
    except FileNotFoundError:
        raise AudioRejected("this format needs ffmpeg, which is not installed")
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def decode_upload(raw: bytes, filename: str = "",
                  ffmpeg_timeout: int = 20) -> tuple[np.ndarray, int, str]:
    """bytes -> (float32 [frames, channels], sample_rate, decoder_name)."""
    if not raw:
        raise AudioRejected("the uploaded file is empty")
    try:
        y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        if y.size:
            return y, sr, "soundfile"
    except Exception:
        pass
    y, sr = _decode_ffmpeg(raw, ffmpeg_timeout)
    if not y.size:
        raise AudioRejected("the file decoded to no audio at all")
    return y, sr, "ffmpeg"


def to_canonical(y: np.ndarray, sr: int, target_sr: int = CANONICAL_SR,
                 normalize: bool = False) -> np.ndarray:
    """Downmix to mono, resample to `target_sr`, optionally peak-normalise."""
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr,
                             res_type="soxr_hq").astype(np.float32)
    if normalize:
        peak = float(np.abs(y).max())
        if peak > 1e-6:
            y = (y * (10 ** (-3 / 20) / peak)).astype(np.float32)
    return y


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def _dbfs(x: float) -> float:
    return -120.0 if x <= 1e-12 else round(20 * math.log10(x), 1)


def analyze(y: np.ndarray, sr: int) -> dict:
    """Measure the things that decide whether a clip clones well."""
    y = np.asarray(y, dtype=np.float32).squeeze()
    n = len(y)
    dur = n / sr if sr else 0.0
    peak = float(np.abs(y).max()) if n else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if n else 0.0

    # Mean power spectrum over 2048-sample Hann frames.
    win = min(2048, max(256, 1 << int(math.log2(max(n, 256)))))
    hop = max(1, win // 4)
    if n >= win:
        idx = range(0, n - win + 1, hop)
        w = np.hanning(win).astype(np.float32)
        acc = np.zeros(win // 2 + 1, dtype=np.float64)
        count = 0
        for i in idx:
            acc += np.abs(np.fft.rfft(y[i:i + win] * w)) ** 2
            count += 1
        psd = acc / max(count, 1)
    else:
        psd = np.abs(np.fft.rfft(y, n=win)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / sr)
    total = float(psd.sum()) or 1.0
    cdf = np.cumsum(psd) / total

    def rolloff(p: float) -> float:
        return round(float(freqs[min(int(np.searchsorted(cdf, p)), len(freqs) - 1)]), 1)

    above = lambda hz: round(100.0 * float(psd[freqs >= hz].sum()) / total, 2)

    # Frame RMS spread as a crude SNR: quiet frames are noise floor, loud
    # frames are speech.
    fr = max(1, int(0.02 * sr))
    if n >= fr * 4:
        frames = y[:n // fr * fr].reshape(-1, fr)
        r = np.sqrt((frames ** 2).mean(axis=1))
        p10, p90 = float(np.percentile(r, 10)), float(np.percentile(r, 90))
        if p90 <= 1e-6:
            snr = 0.0                      # nothing here to have a ratio with
        elif p10 > 1e-9:
            snr = round(20 * math.log10(p90 / p10), 1)
        else:
            snr = 60.0                     # noise floor below float resolution
        thresh = p90 * (10 ** (-40 / 20))
        silence = round(100.0 * float((r < thresh).mean()), 1)
    else:
        snr, silence = 0.0, 0.0

    return {
        "duration_s": round(dur, 2),
        "sample_rate": int(sr),
        "samples": int(n),
        "peak": round(peak, 4),
        "peak_dbfs": _dbfs(peak),
        "rms_dbfs": _dbfs(rms),
        "dc_offset": round(float(y.mean()), 5) if n else 0.0,
        "clipped_pct": round(100.0 * float((np.abs(y) >= 0.999).mean()), 3) if n else 0.0,
        "rolloff_85_hz": rolloff(0.85),
        "rolloff_95_hz": rolloff(0.95),
        "rolloff_99_hz": rolloff(0.99),
        "energy_above_4k_pct": above(4000),
        "energy_above_8k_pct": above(8000),
        "snr_db": snr,
        "silence_pct": silence,
        "model_band_hz": MODEL_BAND_HZ,
        "spectrum": _spectrum_bands(psd, freqs, sr),
        "waveform": _waveform_peaks(y),
    }


def _spectrum_bands(psd: np.ndarray, freqs: np.ndarray, sr: int,
                    bands: int = 64) -> dict:
    """Log-spaced band energies in dB, for the spectrum strip in the GUI."""
    lo, hi = 80.0, max(sr / 2, 1000.0)
    edges = np.geomspace(lo, hi, bands + 1)
    ref = float(psd.max()) or 1.0
    out = []
    for i in range(bands):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        v = float(psd[m].mean()) if m.any() else 0.0
        out.append(round(max(-90.0, 10 * math.log10(v / ref)) if v > 0 else -90.0, 1))
    return {"freqs": [round(float(f), 1) for f in edges[:-1]], "db": out}


def _waveform_peaks(y: np.ndarray, buckets: int = 400) -> list[float]:
    if not len(y):
        return []
    step = max(1, len(y) // buckets)
    trimmed = y[:len(y) // step * step].reshape(-1, step)
    return [round(float(v), 3) for v in np.abs(trimmed).max(axis=1)[:buckets]]


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #

def _check(cid, level, label, detail, why, fix, value=None):
    return {"id": cid, "level": level, "label": label, "detail": detail,
            "why": why, "fix": fix, "value": value}


def verdict(a: dict) -> tuple[str, list[dict], list[dict]]:
    """-> (grade in {good, fair, poor}, errors, warnings).

    Errors reject the upload. They are the clip shapes CLAUDE.md blames for
    generation that never terminates, plus the degenerate ones.
    """
    errors: list[dict] = []
    warnings: list[dict] = []
    d, roll, hi4 = a["duration_s"], a["rolloff_99_hz"], a["energy_above_4k_pct"]

    if d < HARD_MIN_S:
        errors.append(_check(
            "duration", "error", "Too short", f"{d:.1f} s",
            "There is not enough voice here to clone from.",
            f"Record {SOFT_MIN_S:.0f}-{SOFT_MAX_S:.0f} seconds of continuous speech.", d))
    elif d > HARD_MAX_S:
        errors.append(_check(
            "duration", "error", "Too long", f"{d:.1f} s",
            "Long references slow every request and can make generation run away.",
            f"Trim to {SOFT_MIN_S:.0f}-{SOFT_MAX_S:.0f} seconds.", d))
    elif not (SOFT_MIN_S <= d <= SOFT_MAX_S):
        warnings.append(_check(
            "duration", "warn", "Unusual length", f"{d:.1f} s",
            f"{SOFT_MIN_S:.0f}-{SOFT_MAX_S:.0f} s clones most reliably.",
            "Trim or re-record to that range.", d))

    if a["peak"] < 0.01:
        errors.append(_check(
            "silent", "error", "No audible audio", f"peak {a['peak_dbfs']:.0f} dBFS",
            "This clip is silent or so quiet there is no voice to clone.",
            "Check you recorded the right input, then record again closer to the mic.",
            a["peak"]))

    if a["silence_pct"] > 60:
        errors.append(_check(
            "silence", "error", "Mostly silence", f"{a['silence_pct']:.0f}%",
            "Most of this clip has no speech in it.",
            "Trim the silence, or record again closer to the mic.", a["silence_pct"]))
    elif a["silence_pct"] > 35:
        warnings.append(_check(
            "silence", "warn", "A lot of silence", f"{a['silence_pct']:.0f}%",
            "Silence is wasted reference material.",
            "Trim the gaps at the start and end.", a["silence_pct"]))

    if a["clipped_pct"] > 1.0:
        errors.append(_check(
            "clipping", "error", "Badly clipped", f"{a['clipped_pct']:.2f}%",
            "The waveform is squared off; the distortion gets cloned too.",
            "Re-record with the input gain lower.", a["clipped_pct"]))
    elif a["clipped_pct"] > 0.05:
        warnings.append(_check(
            "clipping", "warn", "Some clipping", f"{a['clipped_pct']:.2f}%",
            "Clipped peaks add harshness the clone inherits.",
            "Lower the input gain a few dB.", a["clipped_pct"]))

    if a["snr_db"] < 6:
        errors.append(_check(
            "snr", "error", "Very noisy", f"{a['snr_db']:.0f} dB",
            "The background is nearly as loud as the voice.",
            "Record somewhere quieter, closer to the mic.", a["snr_db"]))
    elif a["snr_db"] < 18:
        warnings.append(_check(
            "snr", "warn", "Noisy background", f"{a['snr_db']:.0f} dB",
            "Background noise is cloned along with the voice.",
            "Record somewhere quieter, or move closer to the mic.", a["snr_db"]))

    # The bandwidth checks — the reason this analyser exists.
    if roll < 3500:
        warnings.append(_check(
            "bandwidth", "warn", "Telephone-band", f"99% below {roll:.0f} Hz",
            "The model reads 0-8 kHz from a reference and can only reproduce "
            "frequencies that are actually in it. This clip fills roughly the "
            "bottom third, so every sentence will sound muffled.",
            "Record with a real microphone, not a phone call, a voice note or a "
            "video-call recording. Do not just re-save at a higher sample rate — "
            "that adds no detail.", roll))
    elif roll < 5000:
        warnings.append(_check(
            "bandwidth", "warn", "Band-limited", f"99% below {roll:.0f} Hz",
            "The model reads up to 8 kHz; this clip stops well short, so the "
            "clone will sound dull.",
            "A cleaner, closer recording will carry more high end.", roll))

    if hi4 < 3.0:
        warnings.append(_check(
            "brightness", "warn", "Little high-frequency detail",
            f"{hi4:.2f}% above 4 kHz",
            "Consonants and air live above 4 kHz. Without them speech sounds "
            "blurry and less intelligible.",
            "Use a better microphone and avoid aggressive noise suppression.", hi4))

    if a["peak_dbfs"] < -20:
        warnings.append(_check(
            "level", "warn", "Very quiet", f"{a['peak_dbfs']:.0f} dBFS",
            "Quiet recordings carry more relative noise.",
            "Record closer to the mic, or enable normalisation on upload.",
            a["peak_dbfs"]))
    elif a["peak_dbfs"] > -1:
        warnings.append(_check(
            "level", "warn", "Close to clipping", f"{a['peak_dbfs']:.0f} dBFS",
            "There is no headroom left.",
            "Lower the input gain a few dB.", a["peak_dbfs"]))

    if errors:
        grade = "poor"
    elif any(w["id"] in ("bandwidth", "brightness") for w in warnings):
        grade = "poor"          # the failure mode this tool exists to catch
    elif warnings:
        grade = "fair"
    else:
        grade = "good"
    return grade, errors, warnings


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #

def write_voice_atomic(voices_dir: Path, voice_id: str, y: np.ndarray, sr: int,
                       transcript: Optional[str] = None) -> Path:
    """Write `<id>.txt` then `<id>.wav`, both atomically.

    The temp file is `<id>.wav.part`, deliberately NOT a dotfile: pathlib's
    `glob("*.wav")` DOES match `.hidden.wav`, so a dot-prefixed temp name would
    let `scan_voices()` pick up a half-written clip. `synthesize` opens the wav
    by path at generation time, so a torn file would reach the model.
    """
    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    wav = voices_dir / f"{voice_id}.wav"
    txt = voices_dir / f"{voice_id}.txt"

    if transcript is not None:
        text = transcript.strip()
        if text:
            tmp_txt = txt.with_suffix(".txt.part")
            tmp_txt.write_text(text + "\n", encoding="utf-8")
            os.replace(tmp_txt, txt)
        elif txt.exists():
            txt.unlink()

    tmp_wav = voices_dir / f"{voice_id}.wav.part"
    sf.write(str(tmp_wav), np.asarray(y, dtype=np.float32), sr,
             subtype="PCM_16", format="WAV")
    os.replace(tmp_wav, wav)          # same directory, so this is atomic
    return wav


def trash_voice(voices_dir: Path, voice_id: str) -> list[str]:
    """Move a voice out of the library instead of deleting it.

    `.trash/` is a subdirectory and `glob("*.wav")` is not recursive, so trashed
    clips are invisible to the scanner but recoverable by hand.
    """
    import time
    voices_dir = Path(voices_dir)
    trash = voices_dir / ".trash"
    trash.mkdir(exist_ok=True)
    stamp = int(time.time())
    moved = []
    for suffix in (".wav", ".txt", ".json"):
        src = voices_dir / f"{voice_id}{suffix}"
        if src.exists():
            dst = trash / f"{voice_id}-{stamp}{suffix}"
            os.replace(src, dst)
            moved.append(dst.name)
    return moved
