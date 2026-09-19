# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Running or deploying this project? Read `AGENTS.md` first** — it is the
> step-by-step runbook, with the expected output at each step and a table of
> failure modes. This file covers design and the reasoning behind the code.
> `docs/` holds the reference documentation: [api](docs/api.md),
> [configuration](docs/configuration.md), [voices](docs/voices.md),
> [deployment](docs/deployment.md), [architecture](docs/architecture.md),
> [troubleshooting](docs/troubleshooting.md), [development](docs/development.md).
> When you change behaviour, update the matching page in `docs/` too.

## What this is

A single-purpose FastAPI service: load VoxCPM2 into GPU memory once, serve a
Latin-American Spanish voice over HTTP — plus an English voice and Whisper
`/transcribe`, so it is a complete miniclosedai `kind=voice` backend. Designed
to be copied onto a GPU box and run. There is no database, no queue, no framework beyond FastAPI.

```
server.py        entry point — uvicorn, one worker
handler.py       RunPod serverless entry point (same engine, different transport)
latina/config.py all settings, every one an env var
latina/engine.py VoxCPM2 wrapper + audio helpers
latina/api.py    the endpoints
latina/asr.py    Whisper for /transcribe — one model per language (es=large-v3,
                 en=turbo, auto=turbo); loads + warms after VoxCPM2, in background
voices/          reference clips: <id>.wav (+ optional <id>.txt transcript,
                 + optional <id>.json {name, language, gender})
```

## Running and verifying

```bash
./start.sh                  # venv + CUDA-matched torch + deps + serve on :8000
python smoke_test.py        # times first-audio latency, writes out.wav
```

`start.sh` picks the torch index from `nvidia-smi` (cu130 for CUDA 13, else
cu124) and falls back to CPU torch with a warning rather than failing at import.

There are no unit tests, and they would not tell you much: the thing that can
break is audio quality and latency, which `smoke_test.py` measures end to end.
When changing generation parameters, run it before and after and compare **RTF
and time-to-first-chunk**, not just "did it produce a file".

## Things that will bite you

**One worker, one model.** `server.py` hardcodes `workers=1` and the engine
holds a `threading.Lock` around generation. Do not raise the worker count to get
throughput — you would load a second copy of the model onto the same GPU. Scale
by running more containers.

**`torch.compile` hangs on some GPUs.** `LATINA_OPTIMIZE` is off by default
because on a GB10 / DGX Spark the warmup compiles forever ("Not enough SMs to
use max_autotune_gemm") and the service never becomes ready. It is a real
speedup on A100 / L40S / 4090. If someone reports "it never finishes loading",
this is the first thing to check.

**A bad reference clip can hang generation.** VoxCPM2 generation is open-ended.
A noisy, telephone-band, or badly transcribed reference can send it into a loop
that does not terminate. Reference clips should be clean, 4–15 s, single
speaker (the analyser's recommended band). If you add a timeout, remember `asyncio.wait_for` around
`asyncio.to_thread` stops *waiting* but cannot cancel the thread — the runaway
keeps consuming the GPU and slows every later request. The real fix is a
generation cap or a subprocess you can kill.

**Sample rate is 48 kHz**, from VoxCPM2's audiovae_v2. Do not assume 22050 or
24000; read it from `/health` or the `sample_rate` field on each chunk.

**Streaming is easy to break and the failure is silent.** `/speak/stream` uses
a plain daemon thread writing to a plain `queue.Queue`, consumed with
`await asyncio.to_thread(q.get)`. Two more obvious approaches were measured and
both delivered every chunk at the end instead of as produced — draining the
generator with `list(...)` inside `asyncio.to_thread`, and bridging onto a
bounded `asyncio.Queue` via `run_coroutine_threadsafe`. Symptom: time-to-first-
chunk equals total time (9.6 s instead of 0.27 s), and nothing errors. If you
touch this function, re-run `smoke_test.py` and check that **first audio is far
below total** — that is the only thing separating a streaming endpoint from a
slow blocking one.

## Wire protocol — do not casually change it

Two consumers depend on the exact shapes:

- **miniclosedai** (`voice.py`) calls `GET /voices` expecting a dict keyed by
  language, and `POST /speak/stream`, stopping when it sees `{"done": true}`.
  Push-to-talk also calls `POST /transcribe` (multipart `audio`) and expects
  `{text, language, segments}`.
- **The Mozart demo's page** stops on `{"end": true}`.

So `/speak` and `/speak/stream` are the same handler, and the terminal frame
carries **both** `done` and `end`. Emit only one and the other client hangs
waiting for a terminator that never comes. `/voices` is language-keyed for
registration; `/voices/detail` holds the human-readable extras.

`speed` is accepted and ignored — miniclosedai sends it, VoxCPM2 has no speed
control, and rejecting an unknown field would fail the request.

The language key comes from each voice's `<id>.json` sidecar (default
`LATINA_LANGUAGE`). `voices/default.*` is miniclosedai-voice's English
reference clip under its original id — keep the id, bots reference it.

**ASR is selected by language, and forcing the wrong one translates.**
`/transcribe`'s `language` (miniclosedai sends the bot's
`voice_settings.asr_language`) picks the model from `LATINA_ASR_MODELS` and
forces decoding in that language. Whisper then *translates* audio in another
language instead of transcribing it — English audio + `es` came back as Spanish.
That is Whisper, not a bug; `auto` is the answer for mixed callers.

**transformers 5 quirk in `asr.py`:** passing `generate_kwargs=None` to the
ASR pipeline is a `TypeError` (it iterates it). Omit the argument when empty.

`smoke_test.py` now also checks the `en`+`es` catalog and round-trips a clip
through every `/asr` option (`SMOKE_ASR=0` skips it). Run it after touching
`asr.py` as well as `api.py`.

## The voice studio (web GUI)

`/studio/` is a single-page GUI for cloning, auditioning and managing voices;
`/` redirects to it. It lives in `latina/studio.py` + `latina/audio_io.py` +
`static/`, and is **deliberately detachable**:

```python
if config.STUDIO_ENABLED:
    try:
        from .studio import install as _install_studio
        _install_studio(app, require_key)
    except Exception as e:
        print(f"[latina] voice studio disabled: ...")
```

That guard is not decoration. FastAPI runs `ensure_multipart_is_installed()` at
**decorator time** for any route taking a file, so on a box whose venv predates
`python-multipart` in `requirements.txt` — and `start.sh` only pip-installs when
`.venv` is *absent* — importing the studio raises and would otherwise take
`/speak` down with it. `LATINA_STUDIO=0` disables it outright.

All studio endpoints live under `/api/…` so they cannot shadow the frozen
`/voices` and `/speak` surfaces. `/health`, `/voices` and `/voices/detail` are
byte-identical to before the GUI existed.

**Uploads change `/voices` at runtime.** `engine.rescan_voices()` rebuilds the
registry after an upload or delete. It uses its own `_voices_lock`, never
`engine._lock` — a rescan must not queue behind a 10-second generation — and it
never raises and never empties the registry, the two ways `scan_voices()` can
kill a live service. Clips are written `<id>.wav.part` then `os.replace`d;
`.part` is deliberate because `Path.glob("*.wav")` **does** match dotfiles, so a
`.hidden.wav` temp name would be picked up half-written.

Deletes move the pair to `voices/.trash/` rather than unlinking (`glob` is
non-recursive, so trashed clips are invisible but recoverable). The configured
`LATINA_VOICE` cannot be deleted, and neither can the last remaining voice.

## What the quality analyser is actually measuring

VoxCPM2 resamples **every** reference clip to 16 kHz mono before encoding
(`voxcpm/model/voxcpm2.py:412`; `audio_vae.sample_rate == 16000`, and the 48 kHz
elsewhere is the *output* rate). So the model only ever reads **0–8 kHz**, and
what decides clone quality is how much of that band the clip fills — **not** its
container sample rate. Upsampling a telephone-band clip to 48 kHz changes
nothing. The analyser therefore grades bandwidth (`rolloff_99_hz`,
`energy_above_4k_pct`) and its warning copy must never tell anyone to "raise the
sample rate".

The shipped `es_f_19.wav` measures 99% rolloff at 2848 Hz and 0.44% of energy
above 4 kHz, and the analyser grades it **poor**. That is a deliberate
self-test: if a change ever makes `es_f_19` pass, the thresholds are wrong.

## Deliberate omissions

This voice came out of a car-dealership assistant. That pipeline wrapped English
brand names in quotes so VoxCPM2 would pronounce "Honda Civic" in English inside
a Spanish sentence, building the brand list from a vehicle database and a
dealership vocabulary file. **That is intentionally absent.** If someone asks
why brand names sound Spanish, that is the reason — and re-adding it means
porting `vehicle_db.py` and its data files, which makes this project
domain-specific again.

Also absent: RVC voice conversion. The source project optionally converted
output to a specific person's timbre. It costs roughly 10 points of WER on
Spanish and about a second per sentence, and it is an identity decision rather
than a quality one.

## Changing the voice

Drop a clip in `voices/`. `LATINA_VOICE` picks the default; callers override per
request with `"voice": "<id>"`. The `<id>.txt` transcript is optional but
materially improves cloning — write what the clip actually says, verbatim.
