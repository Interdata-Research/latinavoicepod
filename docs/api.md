# HTTP API reference

Base URL is `http://<host>:8000` unless you changed `LATINA_PORT`. Everything
below is served by one process holding one model.

Two surfaces live here and they have different stability guarantees:

- **Frozen** — `/health`, `/voices`, `/voices/detail`, `/speak`, `/speak/stream`,
  `/speak.wav`. External consumers depend on these exact shapes. See
  [Wire protocol](#wire-protocol-what-must-not-change).
- **Studio** — everything under `/api/…`, plus the `/studio/` page. Added with
  the GUI, kept off the frozen namespace on purpose, and disabled entirely when
  `LATINA_STUDIO=0`.

## Authentication

`LATINA_API_KEY` is empty by default and then **no endpoint requires auth**.
Set it and every protected endpoint demands an exact header:

```
Authorization: Bearer <LATINA_API_KEY>
```

| Endpoint | Protected when a key is set |
|---|---|
| `GET /health`, `GET /voices`, `GET /voices/detail` | no — always open |
| `GET /api/studio` | no — the page reads it to learn whether to ask for a key |
| `POST /speak`, `/speak/stream`, `/speak.wav` | yes |
| every other `/api/…` route | yes |

A wrong or missing token returns `401` with `{"detail": "bad or missing bearer token"}`.
The comparison is a plain string equality against `Bearer <key>`.

CORS is wide open (`allow_origins=["*"]`) so a browser page on any origin can
call the service.

---

## `GET /health`

Liveness *and* readiness. The model loads in a background thread at startup, so
the socket answers long before generation is possible.

```json
{
  "ok": true,
  "model": "openbmb/VoxCPM2",
  "sample_rate": 48000,
  "voices": ["carla", "default", "es_f_19", "romina"],
  "default_voice": "es_f_19",
  "optimize": false,
  "loaded_at": 1757174400.12,
  "tts_model": "openbmb/VoxCPM2",
  "asr_model": "openai/whisper-large-v3-turbo",
  "asr_ready": true,
  "device": "cuda",
  "voices_loaded": true,
  "languages": ["en", "es"],
  "asr_options": [
    {"language": "auto", "model": "openai/whisper-large-v3-turbo", "ready": true},
    {"language": "en",   "model": "openai/whisper-large-v3-turbo", "ready": true},
    {"language": "es",   "model": "openai/whisper-large-v3",       "ready": true}
  ],
  "relay_capable": false
}
```

`ok` is `false` while the weights are still loading, and every `/speak*` call
returns `503` until it flips. **`ok:true` is the only trustworthy readiness
signal** — see the warning in [deployment](deployment.md#waiting-for-readiness).

`tts_model`, `asr_model`, `device` and `voices_loaded` are the fields
miniclosedai's `voice.py` documents for `/health`; `asr_model` is the
auto-detect model and `asr_options` the full list (same as [`GET /asr`](#get-asr)).
`asr_ready` turns true some seconds after `ok` — the Whisper models load (and
warm up) in the background *after* VoxCPM2, so they never delay readiness; a
`/transcribe` that arrives early waits for them.

`relay_capable` is always `false` — there is no `/call/*` or `/webrtc/offer`
(WebRTC call mode needs direct UDP, which the RunPod HTTP proxy cannot carry
anyway), so miniclosedai must never route this service into call mode.
Push-to-talk (`/transcribe` + `/speak/stream`) is fully supported.

## `GET /api/connect-info`

Self-description for miniclosedai's Settings → Add endpoint "paste this URL"
flow — mirrors miniclosedai-voice's own `/api/connect-info`.

```json
{
  "kind": "voice",
  "base_url": "https://<pod-id>-8000.proxy.runpod.net",
  "alt_base_url": "http://host.docker.internal:8000",
  "auth_required": false
}
```

## `GET /voices`

Catalog **keyed by language**, because that is the shape miniclosedai's
voice-backend client registers against.

```json
{"es": [{"id": "carla", "name": "Carla", "gender": "F"}, …],
 "en": [{"id": "default", "name": "Default voice", "gender": "F"}]}
```

Each clip is bucketed by the `language` in its `voices/<id>.json` sidecar —
the same `{name, language, gender}` convention miniclosedai-voice uses — and
`name`/`gender` come from there too. A clip with no sidecar lands under
`LATINA_LANGUAGE` (`es`) with `name` = id with underscores as spaces and
`gender: null`, which is exactly what this endpoint returned before sidecars.
See [voices](voices.md#languages-and-the-english-voice).

## `GET /voices/detail`

The human-readable version, including each clip's reference transcript.

```json
{
  "voices": [{"id": "es_f_19", "reference_text": "promover y apoyar el empleo…"}],
  "default": "es_f_19",
  "language": "es",
  "sample_rate": 48000
}
```

## `GET /asr`

The selectable speech-recognition options — `auto` plus one per language in
`LATINA_ASR_MODELS` — and the Whisper model behind each:

```json
{"enabled": true, "options": [
  {"language": "auto", "model": "openai/whisper-large-v3-turbo", "ready": true},
  {"language": "en",   "model": "openai/whisper-large-v3-turbo", "ready": true},
  {"language": "es",   "model": "openai/whisper-large-v3",       "ready": true}]}
```

## `POST /transcribe`

Speech-to-text, the ASR half of miniclosedai's voice contract (its push-to-talk
path calls it via `voice.transcribe()`). Multipart form: `audio` (any container
ffmpeg decodes — WAV, WebM/Opus from `MediaRecorder`, OGG, MP4) and an optional
`language` that **selects the ASR**:

| `language` | Model (default config) | Decoding |
|---|---|---|
| `es` (or `es-MX`, `es_419`…) | `whisper-large-v3` — most accurate on Spanish | forced Spanish |
| `en` (or `en-US`…) | `whisper-large-v3-turbo` | forced English |
| absent / `auto` | `whisper-large-v3-turbo` | Whisper detects the language per clip |
| any other code | the auto model | forced to that code |

miniclosedai sends the bot's `voice_settings.asr_language` here (its "ASR:
Auto / English / Spanish" picker).

> **Pick the language the caller actually speaks.** Forcing the *wrong* one
> makes Whisper **translate** rather than transcribe — measured: English audio
> with `language=es` came back as a Spanish translation, and Spanish audio with
> `language=en` as English. Use `auto` when callers may speak either.

```bash
curl -F audio=@clip.wav -F language=es localhost:8088/transcribe
```
```json
{"text": "Hola, gracias por llamar.", "language": "es",
 "model": "openai/whisper-large-v3",
 "segments": [{"start": 0.0, "end": 2.8, "text": " Hola, gracias por llamar."}]}
```

`language` is the base code that was forced (`null` for auto-detect); `model`
is the Whisper that ran. Measured on the RTX A6000 through the RunPod proxy:
~0.6–1.2 s for `es` (large-v3), ~0.2–0.7 s for `en`/`auto` (turbo) on 3–5 s
clips. If a language's model failed to load, the request falls back to the
auto model rather than failing. `400` for a missing or undecodable upload,
`503` when `LATINA_ASR=0` or no model loaded. Protected by `LATINA_API_KEY`
like the speak endpoints.

## `POST /speak/stream`

What miniclosedai calls for chat-reply TTS playback. Streams Server-Sent
Events as audio is produced.

`POST /speak` and `POST /speak.wav` are a separate, one-shot handler — see
below — matching the miniclosedai-voice reference contract that Voice
Studio's "Sample"/"Test…" buttons expect (a playable `audio/wav` blob, not
an SSE stream).

Request body (same shape for all three routes):

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | 1–4000 characters |
| `voice` | string | no | any id from `/voices`; defaults to `LATINA_VOICE` |
| `language` | string | no | accepted, currently unused by the engine |
| `speed` | number | no | **accepted and ignored** — miniclosedai sends it and VoxCPM2 has no speed control; rejecting it would fail the request |

```bash
curl -N -X POST localhost:8000/speak/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"Buenas tardes, ¿hablo con el señor Benítez?"}'
```

Audio frames, one per generated chunk:

```
data: {"chunk_b64":"…","sample_rate":48000,"chunk":1}
```

Terminal frame, exactly once, carrying **both** terminator keys:

```
data: {"done":true,"end":true,"chunks":7,"ms":3120}
```

Error frame — generation failed mid-stream, after HTTP 200 was already sent:

```
data: {"error":"RuntimeError: …"}
```

`chunk_b64` is base64 of **little-endian signed 16-bit PCM, mono**, at
`sample_rate` Hz. Read the rate from the frame rather than hardcoding it; it is
48000 today because that is VoxCPM2's `audiovae_v2` output rate.

The response sets `Cache-Control: no-cache` and `X-Accel-Buffering: no`, which
asks an nginx-style proxy not to buffer. Not every proxy honours it — see
[the RunPod proxy note](deployment.md#the-runpod-http-proxy-buffers-sse).

Decoding a stream in Python:

```python
import base64, json, urllib.request
req = urllib.request.Request(
    "http://localhost:8000/speak",
    data=json.dumps({"text": "Hola."}).encode(),
    headers={"Content-Type": "application/json"})
pcm = b""
for raw in urllib.request.urlopen(req):
    line = raw.decode().strip()
    if not line.startswith("data:"):
        continue
    ev = json.loads(line[5:])
    if ev.get("chunk_b64"):
        pcm += base64.b64decode(ev["chunk_b64"])
    elif ev.get("done"):
        break
```

## `POST /speak` · `POST /speak.wav`

Same handler mounted on both paths. `/speak` is the miniclosedai-voice
reference contract's one-shot endpoint (what Voice Studio's "Sample"/"Test…"
buttons call, expecting a directly-playable blob back); `/speak.wav` is the
plainer curl-friendly name. Same request body, one WAV file back (`audio/wav`,
16-bit mono). Blocking: the whole utterance is generated before the first
byte is sent, so it has no streaming advantage — but it is honest about that,
and it survives a buffering proxy unchanged.

```bash
curl -X POST localhost:8000/speak.wav \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hola."}' --output hola.wav
```

## Status codes on the speak endpoints

| Code | Meaning |
|---|---|
| `200` | streaming (SSE) or the WAV body |
| `401` | `LATINA_API_KEY` set and the bearer token did not match |
| `404` | `{"error": "unknown voice 'x'; available: [...]"}` |
| `422` | body failed validation (empty `text`, over 4000 chars) |
| `503` | `{"error": "model still loading"}` |

Note that a failure *during* generation cannot change the status code — the
stream has already started, so it arrives as an `error` frame instead.

---

## Studio API

All under `/api/…`, all requiring the bearer token when one is set, except
`GET /api/studio`. Present only when the studio loaded; see
[architecture](architecture.md#the-studio-is-detachable-on-purpose).

### `GET /api/studio`

Configuration for the page — limits, bounds and whether to prompt for a key.

```json
{
  "auth_required": false,
  "sample_rate": 48000,
  "default_voice": "es_f_19",
  "language": "es",
  "max_upload_bytes": 20971520,
  "max_voices": 50,
  "max_text_chars": 4000,
  "duration_bounds": [1.5, 30.0],
  "recommended_duration": [4.0, 15.0],
  "model_band_hz": 8000,
  "stored_sample_rate": 24000
}
```

### `GET /api/studio/auth`

Returns `{"ok": true}` if the supplied key is valid. The page uses it to
validate a key before storing it.

### `GET /api/studio/voices`

```json
{"voices": [{"id": "es_f_19", "reference_text": "…", "is_default": true}],
 "default": "es_f_19"}
```

### `POST /api/voices/analyze`

Multipart. Measures a clip **without saving it**, so the verdict lands before
the commit.

| Field | Type | Notes |
|---|---|---|
| `file` | file | wav/flac/ogg via libsndfile; webm/opus, m4a/aac via ffmpeg |
| `normalize` | bool | peak-normalise to −3 dBFS before measuring |

Returns `{analysis, grade, errors, warnings, checks}` — see
[voices](voices.md#what-the-analyser-measures) for every field and threshold.

### `POST /api/voices/upload`

Multipart, `201` on success. Same fields as analyze, plus:

| Field | Type | Notes |
|---|---|---|
| `voice_id` | string | slugified; falls back to the filename stem |
| `transcript` | string | the reference text — optional but improves cloning |
| `overwrite` | bool | required to replace an existing id |

Returns the analysis payload plus `id` and the refreshed `voices` list. The
registry is rebuilt immediately, so the new voice appears in `/voices` **without
a restart**.

| Code | Meaning |
|---|---|
| `413` | larger than `LATINA_STUDIO_MAX_UPLOAD_MB` |
| `409` | id exists and `overwrite` was not set |
| `507` | `LATINA_STUDIO_MAX_VOICES` reached |
| `422` | undecodable, or the analyser returned hard errors |

### `GET /api/voices/{id}/audio`

The stored reference clip as `audio/wav`.

### `GET /api/voices/{id}/analysis`

Grades a clip already in the library. Cached on `(path, mtime, size)`, so
repeated calls are free until the file changes.

### `PUT /api/voices/{id}/transcript`

```json
{"reference_text": "what the clip actually says, verbatim"}
```

Writes `<id>.txt` atomically; an empty string deletes it. Returns
`{"id": …, "reference_text": …}`.

### `DELETE /api/voices/{id}`

Moves `<id>.wav` and `<id>.txt` into `voices/.trash/` with a timestamp suffix
rather than unlinking them. Returns `{deleted, moved_to_trash, voices}`.

Refused with `409` in two cases: the id is the configured `LATINA_VOICE`, or it
is the last remaining voice.

---

## Wire protocol: what must not change

Two consumers depend on the exact shapes above, and both fail *silently* —
by hanging, not by erroring — if they change:

- **miniclosedai** (`voice.py`) calls `GET /voices` expecting a dict keyed by
  language, then `POST /speak/stream`, stopping when it sees `{"done": true}`.
- **The Mozart demo's page** stops on `{"end": true}`.

So the terminal frame carries both keys. Emit only one and the other client
waits forever for a terminator that never comes. `/voices` stays language-keyed
because that is what registration reads; `/voices/detail` exists to hold the
human-readable extras that would otherwise tempt someone to reshape `/voices`.
