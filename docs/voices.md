# Voices and reference clips

A voice is a file pair in `voices/`:

```
<id>.wav      the reference clip — this IS the voice
<id>.txt      what the clip says, verbatim (optional, but do write it)
```

The transcript is optional and **materially improves cloning**. Write what the
clip actually says, word for word, including the false starts if there are any.

## Adding one

**By hand** — drop the pair into `voices/` and restart. It appears in `/voices`
and can be requested with `{"voice": "<id>"}`.

**Through the studio** — open `/studio/`, upload or record, read the quality
verdict, save. No restart: `engine.rescan_voices()` rebuilds the registry as
part of the upload.

**Through the API** — see [`POST /api/voices/upload`](api.md#post-apivoicesupload).

```bash
curl -F 'file=@clip.wav' -F 'voice_id=maria' \
     -F 'transcript=Buenas tardes, le llamo de la oficina.' \
     localhost:8000/api/voices/upload
```

Ids are slugified: lowercased, spaces to underscores, anything outside
`[a-z0-9_-]` stripped, runs collapsed, truncated to 48 characters, minimum 2.
User text is never joined to a path before passing through the slugifier, so
traversal is impossible by construction rather than by filtering.

Uploads are decoded with libsndfile first (wav, flac, ogg) and fall back to
ffmpeg (webm/opus, m4a/aac). That fallback is not an edge case — it is how every
in-browser recording arrives, because that is what `MediaRecorder` produces.

Stored clips are downmixed to mono and resampled to `LATINA_STUDIO_SR`
(24 kHz), optionally peak-normalised to −3 dBFS.

## What makes a good reference clip

Clean, 4–15 seconds, one speaker, no music, recorded on a real microphone.

A bad clip is not merely lower quality — **it can hang generation**. VoxCPM2
generation is open-ended, and a noisy, telephone-band or badly transcribed
reference can send it into a loop that does not terminate, pinning the GPU and
slowing every later request. See
[troubleshooting](troubleshooting.md#a-request-never-returns-and-the-gpu-stays-busy).

## Why bandwidth, not sample rate

This is the single most misunderstood thing in the project, so the analyser is
built around it.

VoxCPM2 resamples **every** reference clip to 16 kHz mono before encoding
(`voxcpm/model/voxcpm2.py:412`; `audio_vae.sample_rate == 16000`). The 48 kHz
elsewhere in this project is the *output* rate. So the model only ever reads
**0–8 kHz**, and what decides clone quality is **how much of that band the clip
actually fills** — not what its container claims.

Consequences:

- Upsampling a telephone-band clip to 48 kHz changes nothing. There is no
  detail to recover; you are storing interpolated zeros.
- A 16 kHz clip from a good microphone beats a 48 kHz clip from a phone call.
- The analyser therefore grades **bandwidth**, and its warning copy must never
  tell anyone to "raise the sample rate".

## What the analyser measures

`GET /api/voices/{id}/analysis` and `POST /api/voices/analyze` return
`{analysis, grade, errors, warnings, checks}`.

| Field | Meaning |
|---|---|
| `duration_s` | clip length |
| `peak`, `peak_dbfs`, `rms_dbfs` | level |
| `dc_offset` | mean sample value |
| `clipped_pct` | share of samples at or above 0.999 |
| `rolloff_85_hz`, `rolloff_95_hz`, `rolloff_99_hz` | frequency below which that share of energy sits |
| `energy_above_4k_pct`, `energy_above_8k_pct` | share of energy above those points |
| `snr_db` | 90th vs 10th percentile frame RMS — a crude speech-vs-noise-floor ratio |
| `silence_pct` | share of frames more than 40 dB below the loud frames |
| `model_band_hz` | `8000` — what the model can read |
| `spectrum`, `waveform` | 64 log-spaced band energies and 400 peak buckets, for the GUI |

The spectrum comes from a mean power spectrum over 2048-sample Hann frames.

## Grading

**Errors reject the upload** (HTTP 422). They are the degenerate cases plus the
clip shapes blamed for generation that never terminates.

| Check | Error | Warning |
|---|---|---|
| duration | `< 1.5 s` or `> 30 s` | outside 4–15 s |
| audible audio | `peak < 0.01` | — |
| silence | `> 60%` | `> 35%` |
| clipping | `> 1.0%` | `> 0.05%` |
| SNR | `< 6 dB` | `< 18 dB` |
| bandwidth | — | 99% rolloff `< 3500 Hz` (telephone-band) or `< 5000 Hz` (band-limited) |
| brightness | — | `< 3.0%` of energy above 4 kHz |
| level | — | peak `< −20 dBFS` or `> −1 dBFS` |

Grade:

- **poor** — any error, **or** any bandwidth/brightness warning
- **fair** — any other warning
- **good** — nothing flagged

Bandwidth and brightness warnings force `poor` even though they do not block the
upload, because that is the failure mode this tool exists to catch: it is the
one a listener notices and nobody diagnoses.

Each check carries `label`, `detail`, `why` and `fix`, so the GUI explains
itself rather than showing a bare number.

## The shipped clip is a deliberate self-test

`es_f_19.wav` measures **99% rolloff at 2848 Hz** and **0.44% of energy above
4 kHz**, and the analyser grades it **poor**.

That is on purpose. It is a real telephone-band clip in the library, and it is
the regression test for the thresholds: **if a change ever makes `es_f_19` pass,
the thresholds are wrong.**

## Languages and the English voice

VoxCPM2 speaks English as well as Spanish, and it takes the language from the
*text*, not from a flag — so one instance serves both. What decides where a
voice appears in `/voices` is its optional sidecar:

```json
// voices/<id>.json
{"name": "Romina", "language": "es", "gender": "F"}
```

No sidecar → `LATINA_LANGUAGE` (`es`). A malformed sidecar is ignored rather
than hiding the voice. Studio deletes move the sidecar to `.trash/` with the clip.

`default` (`voices/default.wav` + `default.json`, `language: "en"`) is the
reference clip from **miniclosedai-voice** — the voice that service serves as
`default` with Chatterbox Turbo. Here VoxCPM2 clones it instead: same speaker,
different engine, so it will sound close to but not identical with the
Chatterbox rendering. The id and catalog entry (`Default voice`, `F`) are the
same as miniclosedai-voice's on purpose, so a bot that was set to `default` on
that backend keeps working when pointed here, and miniclosedai's fallback
("first English voice on the backend") lands on it.

## Voices in this repo

Measured with the analyser (`LATINA_STUDIO_SR` = 24 kHz):

| id | duration | 99% rolloff | energy > 4 kHz | SNR | grade | flagged |
|---|---|---|---|---|---|---|
| `es_f_19` | 5.1 s | 2848 Hz | 0.44% | 28.6 dB | `poor` | bandwidth, brightness |
| `carla` | 17.3 s | 5766 Hz | 1.49% | 23.8 dB | `poor` | duration, brightness, level |
| `romina` | 16.1 s | 1898 Hz | 0.12% | 55.9 dB | `poor` | duration, bandwidth, brightness |
| `default` (en) | 9.6 s | — | — | — | *not graded* | from miniclosedai-voice, 22.05 kHz |

`es_f_19` is the default (`LATINA_VOICE`) — a Latin-American female with a warm
receptionist timbre, picked by listen test. `carla` and `romina` were added
through the studio while building it, and share one transcript.

**All three grade `poor`, and none of them is a model reference clip.** `carla`
is the least band-limited but is over the recommended 15 s and short on high
end; `romina` is more telephone-band than the shipped voice. If you want to hear
what the analyser is complaining about, they are useful; if you want a good
clone, record something new against
[the criteria above](#what-makes-a-good-reference-clip).

Reproduce these numbers against a running service:

```bash
for v in es_f_19 carla romina; do
  echo "== $v"; curl -s localhost:8000/api/voices/$v/analysis \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); a=d["analysis"]; \
print(d["grade"], a["rolloff_99_hz"], "Hz", a["energy_above_4k_pct"], "% >4k")'
done
```

## Deleting

The studio's delete moves the pair to `voices/.trash/<id>-<timestamp>.wav`
rather than unlinking it. `glob` is not recursive, so trashed clips vanish from
the registry but stay on disk.

Two deletes are refused with `409`: the configured `LATINA_VOICE`, and the last
remaining voice.

`voices/.trash/` is gitignored.
