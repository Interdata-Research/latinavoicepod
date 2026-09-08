# Architecture

A single-purpose service: load VoxCPM2 into GPU memory once, serve a
Latin-American Spanish voice over HTTP. No database, no queue, no framework
beyond FastAPI. The whole thing is about 1,300 lines of Python, plus roughly 1,000 lines of dependency-free HTML/CSS/JS for the studio page.

```
server.py            uvicorn entry point — one worker, on purpose
handler.py           RunPod serverless entry point — same engine, different transport
latina/config.py     every setting, every one an env var
latina/engine.py     VoxCPM2 wrapper, voice registry, audio helpers
latina/api.py        the HTTP endpoints
latina/studio.py     the studio's /api routes — detachable
latina/audio_io.py   decode, resample, analyse, store reference clips
static/              the studio page (vanilla JS, no build step)
voices/              reference clips: <id>.wav + optional <id>.txt
```

## Request paths

```
                    ┌──────────────────────────────────────────┐
 HTTP  ──> api.py ──┤  /speak/stream           → SSE frames     │
                    │  /speak, /speak.wav     → one WAV body   │──> engine ──> GPU
 RunPod ─> handler ─┤  yielded dicts                           │
                    └──────────────────────────────────────────┘
```

`handler.py` exists so the same engine can run as a RunPod serverless worker
without an HTTP server in front of it. It imports `latina.engine` directly,
calls `engine.load()` at import time (so cold start pays for it once per worker
rather than once per job), and yields dicts that RunPod streams back. The frame
shape mirrors SSE: `{"audio_chunk", "sample_rate", "chunk"}`, terminated by
`{"done": true, "chunks", "ms"}`.

## One worker, one model, one lock

`server.py` pins `workers=1`. Raising it does not buy throughput — it loads a
second copy of the model onto the same GPU.

Inside the process, `LatinaVoiceEngine` holds a `threading.Lock` around
generation. Generation is synchronous and single-threaded on the GPU; two
concurrent requests would interleave on the same CUDA context and both come out
slower. So requests serialize, by design.

**Scale by running more containers**, not more workers and not more threads.

There is a second, separate lock — `_voices_lock` — used only by
`rescan_voices()`. It is deliberately *not* the generation lock: a registry
refresh after an upload must never queue behind a 10-second generation, nor
delay one.

## Streaming is the fragile part

`/speak` is the reason this project exists in this shape, and it is easy to
break in a way that raises no error at all.

`engine.synthesize_streaming()` is a **blocking generator** that genuinely
yields progressively, roughly 250 ms apart. Getting those chunks out to the
client *as they arrive* — rather than all at once at the end — took three
attempts:

| Approach | Result |
|---|---|
| `list(generator)` inside `asyncio.to_thread` | every chunk delivered at the end |
| bridge onto a bounded `asyncio.Queue` via `run_coroutine_threadsafe` | every chunk delivered at the end (measured: 47 chunks all landing at 11.8 s) |
| **a thread writing to a plain `queue.Queue`, consumed with `await asyncio.to_thread(q.get)`** | **interleaves correctly** |

The third works because each `q.get` parks in the executor and leaves the event
loop free to flush the chunk it was just handed.

The producer runs on a **reused** single-slot `ThreadPoolExecutor`
(`_GEN_POOL`), not a fresh thread per request: spawning a thread per request
cost about 26 ms of extra time-to-first-chunk (median 181 ms new-thread vs
155 ms reused), because each new thread pays CUDA per-thread setup.
`max_workers=1` adds no new contention — the engine lock already serializes.

**The failure is silent.** When streaming breaks, nothing errors; the endpoint
just becomes a slow blocking one, with time-to-first-chunk equal to total time
(9.6 s instead of 0.27 s). If you touch `speak()`, re-run `smoke_test.py` and
confirm **first audio is far below total**. That single comparison is the only
thing separating a streaming endpoint from a slow blocking one.

The same trap exists outside the code: a buffering proxy in front of the service
produces *exactly* the same symptom. See
[the RunPod proxy note](deployment.md#the-runpod-http-proxy-buffers-sse).

## What happens between the POST and the first chunk

Time-to-first-chunk is the product's real latency number, so it is worth
knowing exactly what sits in that path:

1. **encode the reference clip** — read from disk, resample to 16 kHz through
   librosa, run the audio VAE encoder. **Cached** per voice (see below).
2. **tokenize** the text and assemble the reference prefix, text tokens and
   masks into one sequence.
3. **prefill** — one forward pass of `base_lm` over that sequence, then
   `residual_lm`.
4. **one diffusion step** through `feat_decoder` (`inference_timesteps` inner
   steps, always on a doubled batch — see below) producing the first latent
   patch.
5. **VAE streaming decode** of that patch, `.cpu()`, base64, SSE flush.

Only step 1 was avoidable work, and it was being repeated on every request.

### The reference clip is encoded once, not per request

`VoxCPM.generate()` calls `build_prompt_cache()` every time it is invoked, and
that function re-reads the clip, resamples it and re-runs the VAE encoder to
produce a tensor that depends on nothing but the file. `LatinaVoiceEngine`
therefore builds it once per voice and calls
`generate_with_prompt_cache_streaming()` directly.

Reuse is safe on both counts that matter. Generation only *reads* the cache —
`_generate_with_prompt_cache` passes `ref_audio_feat` into `torch.cat`, which
allocates new tensors and never writes back. And the encoder is not a sampler:
`AudioVAE.encode` returns the posterior mean (`["mu"]`), so caching is not
freezing one random draw of the voice.

Repeated builds are nonetheless not bit-identical — the conv kernels reduce in
a non-deterministic order, giving a measured max drift of 7e-4 on values
spanning ±5.8 (about 0.01%), and the encoder is equally non-reproducible when
handed the identical input tensor. Caching pins one draw of that float noise,
which makes successive requests marginally *more* consistent than before.

The entry is keyed on `(path, mtime, size)`, so a clip replaced through the
studio rebuilds its own entry — there is no invalidation step to forget.
`rescan_voices()` drops entries for deleted voices, and `warm_prompt_caches()`
builds them all at startup so no request pays for a cold one. The whole thing
is behind `LATINA_PROMPT_CACHE`, and turning it off restores the library call
unchanged.

### CFG is not a latency knob

`solve_euler` in `unified_cfm.py` always builds a `2 * b` batch and runs the
estimator on it — there is no `cfg_value == 1.0` branch. Raising `LATINA_CFG`
costs nothing; lowering it saves nothing. `LATINA_TIMESTEPS` is the parameter
that actually changes how much work step 4 does.

## Startup and readiness

`api.py`'s startup handler runs `engine.load()` in a thread, so the event loop
stays free and `/health` answers `ok:false` instead of hanging the socket while
weights load. Loading includes an optional warmup synthesis
(`LATINA_WARMUP=1`), so the first real request is not the cold one.

**Do not wait on the `warmed up` log line.** It is printed by the startup
handler, which completes *before* uvicorn begins accepting connections — wait on
it and the very next request gets "connection refused". Poll `/health` for
`"ok":true`. `make bg` does this correctly.

## The voice registry

A `Voice` is just a file pair on disk: `<id>.wav` plus an optional `<id>.txt`
holding what the clip says. `scan_voices()` builds the registry at startup and
raises if the directory is missing or empty — the right behaviour at boot.

`rescan_voices()` is the runtime counterpart, called after an upload or delete,
and it has the opposite contract: it **never raises and never empties the
registry**, because those are the two ways a refresh could take down a live
service. It builds the new dict first and rebinds in a single statement, so a
concurrent `/voices` handler iterates a consistent snapshot rather than hitting
"dictionary changed size during iteration".

Two details in the storage layer defend the same invariant:

- New clips are written `<id>.wav.part` and then `os.replace`d into position.
  The `.part` suffix is deliberate — `Path.glob("*.wav")` **does** match
  dotfiles, so a `.hidden.wav` temp name would let the scanner pick up a
  half-written clip, and `synthesize` opens the wav by path at generation time.
- Deletes move the pair into `voices/.trash/` instead of unlinking. `glob` is
  not recursive, so trashed clips are invisible to the scanner but recoverable
  by hand.

## The studio is detachable on purpose

```python
if config.STUDIO_ENABLED:
    try:
        from .studio import install as _install_studio
        _install_studio(app, require_key)
    except Exception as e:
        print(f"[latina] voice studio disabled: …")
```

That guard is not decoration. FastAPI runs `ensure_multipart_is_installed()` at
**decorator time** for any route taking a file. On a box whose venv predates
`python-multipart` in `requirements.txt` — and `start.sh` only pip-installs when
`.venv` is *absent* — importing the studio raises, and without the guard that
exception would propagate out of `latina.api` and take `/speak` down with it.

With the guard, the worst case is one log line and a missing GUI. The router is
also built inside `install()` rather than at module import, so the decorator-time
check happens *inside* the try/except.

Everything the studio adds lives under `/api/…` and `/studio/`, so it cannot
shadow the frozen `/voices` and `/speak` surfaces. `/health`, `/voices` and
`/voices/detail` are byte-identical to what they were before the GUI existed.

## Audio format, end to end

| Stage | Rate | Format |
|---|---|---|
| Reference clip on disk | 24 kHz (studio-stored) or whatever you dropped in | WAV |
| What VoxCPM2 actually reads | **16 kHz mono** — it resamples every reference | — |
| Model output | **48 kHz** (`audiovae_v2`) | float32 |
| On the wire | 48 kHz | base64 little-endian int16, mono |

Do not assume 22050 or 24000 for output. Read it from `/health` or from the
`sample_rate` field on each chunk.

`to_wav()` writes a 44-byte RIFF header by hand rather than pulling in
soundfile/libsndfile just for that.

## Deliberate omissions

This voice came out of a car-dealership assistant, and two pieces of that
pipeline are intentionally absent.

**English brand-name handling.** The original wrapped English brand names in
quotes so VoxCPM2 would pronounce "Honda Civic" in English inside a Spanish
sentence, building the brand list from a vehicle database and a dealership
vocabulary file. If someone asks why brand names sound Spanish, that is the
reason — and re-adding it means porting `vehicle_db.py` and its data files,
which makes this project domain-specific again.

**RVC voice conversion.** The source project optionally converted output to a
specific person's timbre. It costs roughly 10 points of WER on Spanish and about
a second per sentence, and it is an identity decision rather than a quality one.
