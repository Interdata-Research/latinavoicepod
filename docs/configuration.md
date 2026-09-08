# Configuration

Every setting lives in `latina/config.py` and every one is an environment
variable. There is no config file format, no CLI flags. `start.sh` sources
`.env` if it exists (`.env.example` is the annotated template), so the usual
workflow is:

```bash
cp .env.example .env
$EDITOR .env
./start.sh
```

Booleans accept `1`, `true` or `yes` (case-insensitive); anything else is false.

## Voice

| Variable | Default | What it does |
|---|---|---|
| `LATINA_VOICE` | `es_f_19` | The default voice — any `<id>.wav` in the voices directory. Callers override per request with `"voice"`. This voice cannot be deleted through the studio. |
| `LATINA_LANGUAGE` | `es` | The key `/voices` is grouped under. Registration reads this. |
| `LATINA_VOICES_DIR` | `./voices` | Where reference clips live. |

## Model and generation

| Variable | Default | What it does |
|---|---|---|
| `LATINA_MODEL_ID` | `openbmb/VoxCPM2` | Hugging Face model id. |
| `LATINA_CFG` | `2.0` | Classifier-free guidance strength. |
| `LATINA_TIMESTEPS` | `10` | Diffusion inference steps. |
| `LATINA_RETRY` | `0` | VoxCPM2's `retry_badcase`. |
| `LATINA_OPTIMIZE` | `0` | `torch.compile` the model at load. |
| `LATINA_DENOISER` | `0` | Load VoxCPM2's denoiser for noisy reference clips. |
| `LATINA_WARMUP` | `1` | Synthesize one throwaway line at startup so the first real request is not the cold one. |
| `LATINA_PROMPT_CACHE` | `1` | Cache VoxCPM2's encoded reference clip between requests. `0` restores the library's per-request path. |

Three of these have non-obvious reasoning behind the default.

### `LATINA_RETRY` is off, and that is deliberate

VoxCPM2 re-generates the **whole utterance** — up to three times, silently —
when it dislikes the audio-to-text ratio. It is on by default in the library and
it is a latency disaster on short conversational lines. Measured: `"Sí, con él."`
(1.1 s of audio) took **9.1 s, RTF 8.12**, against 1.4 for a normal sentence.
Turning it off took the same line to **177 ms to first chunk, RTF 1.08**.

It also defeats streaming outright: a retry discards everything already sent to
the caller, and there is no way to un-send it.

Set `LATINA_RETRY=1` only if you would rather have an occasional retry than an
occasional bad clip — offline batch work, not live calls.

### `LATINA_OPTIMIZE` is off because it hangs on some GPUs

`torch.compile` is a real speedup on datacenter GPUs, and it is off anyway,
because the failure mode when it does not work is *silent and total*: on a
GB10 / DGX Spark the warmup compiles forever (`Not enough SMs to use
max_autotune_gemm`) and the service never becomes ready.

Measured on an **RTX A6000**: safe, and a clear win — RTF 0.76 → 0.65, at the
cost of roughly 70 s of extra startup while it compiles.

So: try `LATINA_OPTIMIZE=1` on an A100, L40S, 4090 or A6000, measure it, and set
it back to `0` if the first request never returns. If someone reports "it never
finishes loading", this is the first thing to check.

### `LATINA_PROMPT_CACHE` cuts time-to-first-chunk

`VoxCPM.generate()` calls `build_prompt_cache()` on **every** request, which
re-reads the reference clip from disk, resamples it to 16 kHz through librosa
and re-runs the audio VAE encoder. That result depends on nothing but the file,
generation only ever reads it, and the encoder returns the posterior mean
rather than a sample — so doing the work once per clip instead of once per
request does not change what the model is conditioned on.

With it on, the engine calls `generate_with_prompt_cache_streaming()` directly
and keeps one cache per voice, keyed on `(path, mtime, size)` — so a clip
re-uploaded through the studio rebuilds its own entry with nothing to
invalidate by hand. Caches are built for every voice at startup.

Set it to `0` to A/B the difference on your own hardware; the fallback path is
the unmodified library call.

### `LATINA_CFG` / `LATINA_TIMESTEPS` — measure before you trust the folklore

The defaults `2.0` / `10` are VoxCPM2's own, chosen here because a preset that
pushes past realtime makes the voice *worse*: above RTF 1.0 the player runs out
of audio between chunks and the speech stutters. The slower "quality" preset
(`3.0` / `20`) measured RTF 1.52 against 1.03.

That said, the two knobs are **not** symmetric, and a later measurement on an
A6000 found:

- **`cfg 3.0` is free.** The CFG solver always runs a doubled batch regardless
  of the value (`unified_cfm.py`, `solve_euler`), so 3.0 costs the same as 2.0 —
  81 vs 82 ms to first chunk — and is steadier and brighter on Spanish: 99% of
  output energy reaches 7285 Hz at cfg 3.0 against 4312 Hz at cfg 2.0.
- **Timesteps are not free.** Lowering them mainly buys RTF, not
  time-to-first-chunk, which is prefill-bound — and it costs audible quality.

If you raise either, re-run `smoke_test.py` and compare **time-to-first-chunk
and RTF**, not just "did it produce a file".

## Serving

| Variable | Default | What it does |
|---|---|---|
| `LATINA_HOST` | `0.0.0.0` | Bind address. |
| `LATINA_PORT` | `8000` | Port. `make bg`, `make stop` and the Makefile's health poll all read this too. |
| `LATINA_API_KEY` | *(empty)* | When set, protected endpoints require `Authorization: Bearer <key>`. Leave empty on a private network. See [the auth table](api.md#authentication). |

The worker count is **not** configurable, on purpose. `server.py` hardcodes
`workers=1`: each uvicorn worker would load its own copy of the model onto the
same GPU. Scale by running more containers.

## Voice studio

| Variable | Default | What it does |
|---|---|---|
| `LATINA_STUDIO` | `1` | `0` disables the GUI and its `/api/…` routes outright. |
| `LATINA_STUDIO_STATIC` | `./static` | Directory served at `/studio/`. Missing directory = studio disabled, service unaffected. |
| `LATINA_STUDIO_MAX_UPLOAD_MB` | `20` | Upload ceiling, enforced *while reading* — Starlette spools an upload over 1 MB to disk, so a post-hoc check would let one request fill the volume. |
| `LATINA_STUDIO_MAX_VOICES` | `50` | Refuses new ids past this count (`507`). |
| `LATINA_STUDIO_FFMPEG_TIMEOUT` | `20` | Seconds before an ffmpeg decode is abandoned. |
| `LATINA_STUDIO_SR` | `24000` | Sample rate uploaded clips are **stored** at. |

`LATINA_STUDIO_SR` is not an output rate and raising it does not improve
anything: VoxCPM2 resamples every reference clip to 16 kHz before encoding, so
24 kHz already keeps the whole band the model can read, at half the size of
48 kHz. See [voices](voices.md#why-bandwidth-not-sample-rate).

## Environment variables that are not ours

| Variable | Read by | Why you care |
|---|---|---|
| `HF_HOME` | Hugging Face | Where the ~5 GB of weights are cached. **Point this at a persistent volume** on a cloud box (`HF_HOME=/workspace/hf-cache`) or every restart re-downloads them. |
| `TORCH_INDEX` | `start.sh` | Override the auto-detected torch wheel index, e.g. `TORCH_INDEX=https://download.pytorch.org/whl/cu130 ./start.sh`. |
| `PY` | `start.sh` | Interpreter used to build the venv. Defaults to `python3`. |
| `URL` | `Makefile` | `make test URL=https://host:8000` points the smoke test elsewhere. |
