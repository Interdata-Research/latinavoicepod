# Latina Voice TTS

A self-contained FastAPI service that serves a **Latin-American Spanish voice**
with [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2). Copy the folder to any
Linux box with an NVIDIA GPU, run one script, and you have a TTS endpoint.

The voice is `es_f_19` — a Latin-American female with a warm, receptionist
timbre, picked by listen test rather than by accident. The reference clip ships
in `voices/`, so the voice travels with the code.

## Run it

```bash
./start.sh
```

That creates a venv, picks the torch wheel that matches the host's CUDA,
installs everything, and serves on `:8000`. First start downloads the model
(~5 GB) and takes several minutes; after that it is seconds.

It is ready when `/health` reports `"ok":true` — not when the log says
`warmed up`, which is printed before uvicorn accepts connections.

Check it:

```bash
curl localhost:8000/health
python smoke_test.py           # synthesizes, times it, writes out.wav
```

## Documentation

| | |
|---|---|
| [docs/api.md](docs/api.md) | every endpoint, SSE frame shapes, auth, status codes |
| [docs/configuration.md](docs/configuration.md) | every environment variable and the measurement behind its default |
| [docs/voices.md](docs/voices.md) | reference clips, the quality analyser, why bandwidth beats sample rate |
| [docs/deployment.md](docs/deployment.md) | local, Docker, RunPod pod, RunPod serverless, miniclosedai |
| [docs/architecture.md](docs/architecture.md) | how it fits together and why streaming is written that way |
| [docs/troubleshooting.md](docs/troubleshooting.md) | symptoms, causes, fixes |
| [docs/development.md](docs/development.md) | layout, testing, and the rules for changing it |

`AGENTS.md` is the operational runbook; `CLAUDE.md` holds design notes for AI
coding agents.

## API

| | |
|---|---|
| `GET /health` | is the model in memory yet, which voices, what sample rate |
| `GET /voices` | the reference clips this instance serves, keyed by language |
| `GET /voices/detail` | the same, with reference transcripts, for humans |
| `POST /speak` | one WAV file back (miniclosedai-voice reference contract) |
| `POST /speak/stream` | SSE stream of base64 int16 PCM chunks — what miniclosedai calls for chat-reply playback |
| `POST /speak.wav` | same as `/speak`, kept as the plainer curl-friendly name |
| `POST /transcribe` | speech-to-text (Whisper); `language=es` / `en` picks the Spanish / English model and forces it, none = auto-detect |
| `GET /asr` | the selectable ASR options and the model behind each |
| `GET /api/connect-info` | self-description for miniclosedai's Settings "paste this URL" flow |
| `GET /studio/` | the voice studio GUI (`/` redirects here) |

Voice management lives under `/api/…` — upload, analyse, delete. Full reference
in [docs/api.md](docs/api.md).

```bash
curl -X POST localhost:8000/speak.wav \
  -H 'Content-Type: application/json' \
  -d '{"text":"Buenas tardes, ¿hablo con el señor Benítez?"}' \
  --output hola.wav
```

`POST /speak/stream` frames look like this — chunks arrive as they are
generated, so a caller can start playing before the sentence is finished:

```
data: {"chunk_b64":"…","sample_rate":48000,"chunk":1}
data: {"end":true,"chunks":7,"ms":3120}
```

Audio is **48 kHz mono, little-endian int16**.

## Deploy to RunPod

**As a pod** — copy the folder, `./start.sh`, expose 8000. Simplest thing that
works, and you keep an always-warm model. Running as root (the RunPod default)
it installs its own system packages. An RTX A5000 24 GB is plenty; the service
uses ~6.3 GB. Set `HF_HOME=/workspace/hf-cache` so a restart does not
re-download the weights. Full step-by-step in `AGENTS.md` §5.

**As a serverless worker** — build the image and set the start command to
`python handler.py`:

```bash
docker build -t your-registry/latina-voice:1.0 .
docker push  your-registry/latina-voice:1.0
```

The handler loads the model at import, so cold start pays for it once per
worker rather than once per job. Jobs take `{"input": {"text": "...", "voice":
"es_f_19"}}` and stream `{"audio_chunk": "...", "sample_rate": 48000}` items,
ending with `{"done": true}`.

Cold start is a real cost — model download plus load. Keep a worker warm if
latency matters.

## Connect it to miniclosedai (and therefore to the Mozart bot)

This service speaks miniclosedai's voice-backend protocol — `GET /voices`
keyed by language, `POST /speak/stream` ending on `{"done": true}` — so it
registers directly and then appears in every bot's voice picker, the Mozart
call bot included. No code change on the miniclosedai side.

Settings → Add endpoint → kind **Voice**, base URL = your RunPod URL. Or:

```bash
curl -k -X POST https://localhost:8095/api/backends \
  -H 'Content-Type: application/json' \
  -d '{"name":"Latina Voice","kind":"voice",
       "base_url":"https://<pod-id>-8000.proxy.runpod.net",
       "api_key":"<LATINA_API_KEY, if you set one>","enabled":true}'
```

Verified end to end against miniclosedai's own client: catalog lists `es_f_19`,
`speak_stream` returns 18 chunks and terminates on `done`, first audio 304 ms.
Through the Mozart demo the same call measured 279 ms.

> **The RunPod HTTP proxy buffers SSE.** The BCP docs hit this and say so
> outright: *"the pod proxy buffers SSE so `/stream/{job_id}` never arrives at
> the client."* Through `https://<pod-id>-8000.proxy.runpod.net` you will most
> likely get every chunk at the end — first audio equal to total time, which
> throws away the entire streaming advantage.
>
> Two ways around it. **Expose a direct TCP port** on the pod (RunPod gives you
> `<ip>:<port>` that bypasses the HTTP proxy) and register that URL — streaming
> then behaves as it does locally. Or accept the buffering and use
> `/speak.wav`, which is honest about being one blocking response.
>
> Measure it, do not assume: run `smoke_test.py` against the RunPod URL and
> compare **first audio vs total**. If they are equal, you are being proxied.

## Or pair it behind miniclosedai-voice (English + Spanish on one URL)

[miniclosedai-voice](https://github.com/edantonio505/miniclosedai-voice) is the
sibling service: Chatterbox Turbo (English), Whisper, and WebRTC **call mode**,
which this project does not have. It can treat this service as its **Spanish
engine**, so there is one URL to register and both languages — including on
calls:

```bash
# on the miniclosedai-voice box
VOICE_UPSTREAM_URL=http://127.0.0.1:8088 ./start.sh -d
```

It then merges our `/voices` into its own catalog and forwards `/speak`,
`/speak/stream` and (when its own Whisper is English-only) Spanish
`/transcribe` here. Nothing changes on this side — it is the same protocol the
section above describes, so this service neither knows nor cares whether its
caller is miniclosedai itself or miniclosedai-voice in front of it.

Two consequences worth knowing:

- Our English `default` voice is **shadowed** in that setup: the front service
  keeps its own `default` (Chatterbox) on an id clash. Our English voice is
  still there for anyone calling this service directly.
- Registering *both* services in miniclosedai instead is a valid layout, and
  the voices then show up grouped per backend — but do not do both at once, or
  every Spanish voice appears twice.

Standing the pair up on a fresh box: **[DEPLOY.md](https://github.com/edantonio505/miniclosedai-voice/blob/main/DEPLOY.md)**.

## The voice studio

`http://localhost:8000/studio/` (`/` redirects there) is a single-page GUI for
cloning, auditioning and managing voices. Upload or record a reference clip, get
a quality verdict **before** saving, then synthesize and listen. Uploaded voices
appear in `/voices` immediately — no restart.

It is deliberately detachable: if it fails to import, the service logs one line
and keeps serving `/speak`. `LATINA_STUDIO=0` disables it outright.

Recording needs a secure context, so over a plain remote `http://` URL the
browser withholds the microphone — use `localhost` or an SSH tunnel.

## Adding a voice

Drop `<name>.wav` into `voices/`, ideally with a `<name>.txt` alongside holding
its transcript — VoxCPM2 clones noticeably better when it knows what the
reference says. Restart. It appears in `/voices` and can be requested by name.
Through the studio or `POST /api/voices/upload`, no restart is needed.

Reference clips want to be clean, 4–15 seconds, one speaker, no music. A noisy
or telephone-band clip can send generation into a loop that never terminates.

**The clip's bandwidth is what matters, not its sample rate.** VoxCPM2 resamples
every reference to 16 kHz before encoding, so it only ever reads 0–8 kHz;
re-saving a telephone-band clip at 48 kHz adds nothing. The shipped `es_f_19` is
itself telephone-band and the analyser grades it `poor` on purpose — it is the
regression test for the thresholds. See [docs/voices.md](docs/voices.md).

## Why the defaults are what they are

Two settings decide whether streaming works at all, and both defaults were
chosen by measurement:

**`retry_badcase` is OFF.** VoxCPM2 re-generates the whole utterance up to three
times when it dislikes the audio-to-text ratio, silently, and it is ON in the
library. On short conversational lines that is catastrophic: `"Sí, con él."` —
1.1 s of audio — measured **9.1 s, RTF 8.12**, against 1.4 for a normal
sentence. It also defeats streaming outright, since a retry discards everything
already sent to the caller. Turning it off took the same line to **177 ms to
first chunk, RTF 1.08**. Set `LATINA_RETRY=1` if you would rather have an
occasional retry than an occasional bad clip.

**The generation preset is the fast one** (`cfg 2.0 / 10 timesteps`), not the
slower "quality" preset (3.0 / 20). Measured RTF 1.03 against 1.52. Above RTF
1.0 the player runs out of audio between chunks and the speech stutters — so a
quality setting that pushes past realtime makes the voice *worse*, not better.
Use 3.0 / 20 for offline batch work, where the extra steadiness is free.

## Tuning

Everything is an environment variable; see `.env.example`. The two that matter:

- **`LATINA_OPTIMIZE`** — `torch.compile`. Off by default because on a GB10 /
  DGX Spark the warmup compiles forever and never yields ("Not enough SMs to
  use max_autotune_gemm"). On an A100, L40S or 4090 it usually helps: set it to
  `1`, and set it back to `0` if the first request never returns.
- **`LATINA_CFG` / `LATINA_TIMESTEPS`** — `2.0` / `10` by default, the fast
  preset, because a setting that pushes RTF past 1.0 makes the voice *worse*.
  `3.0` / `20` is the steadier "quality" preset for offline batch work. Note
  that the two knobs are not symmetric: on an A6000, `cfg 3.0` measured free
  (the CFG solver runs a doubled batch either way) while timesteps are not.
  Details in [docs/configuration.md](docs/configuration.md).

## What this is not

The dealership pipeline this voice came from wrapped English brand names in
quotes so "Honda Civic" would be pronounced in English inside a Spanish
sentence, driven by a vehicle database and a dealership vocabulary. **None of
that is here.** This project is the voice and the serving, nothing
domain-specific.

## Performance, honestly

Measured on a GB10 (DGX Spark), `optimize=0`, via `smoke_test.py`:

| | |
|---|---|
| Model load | ~18 s (weights cached), plus ~6 s warmup |
| **Time to first audio** | **~180–210 ms** |
| RTF | ~1.08 |
| Intelligibility | 0.0% WER round-tripped through Whisper large-v3-turbo |

The number that matters for a live call is **time to first audio**, not RTF:
chunks stream out as they are generated, so the caller hears the start of the
sentence while the rest is still being produced. That lands inside the
150–300 ms that a conversational turn budget typically allots to synthesis.

RTF above 1 does mean that on long replies the tail eventually falls behind
playback. Split long replies into sentences and issue one request per sentence
if you hit that.

On a datacenter GPU, try `LATINA_OPTIMIZE=1` — measure with `smoke_test.py`
rather than trusting this table.

### On an RTX A6000 (RunPod pod), `optimize=1`, `cfg 3.0 / 10 timesteps`

Measured with `bench_ttfc.py`, 5 runs per phrase after a warmup, median:

| | on the box | through the RunPod HTTPS proxy |
|---|---|---|
| **Time to first chunk** | **61 ms** | 302 ms |
| Total, 2.2 s utterance | 1.6 s | 1.7 s |
| RTF | 0.63 | 0.68 |
| One-shot `/speak` (WAV) | 1.49 s | 1.56 s |

`LATINA_PROMPT_CACHE=1` (the default) is worth 39% of the on-box figure:

| | median | p95 |
|---|---|---|
| `LATINA_PROMPT_CACHE=0` | 100 ms | 156 ms |
| `LATINA_PROMPT_CACHE=1` | **61 ms** | **68 ms** |

It removes the variance as well as the mean, because the work it skips is a
disk read and a librosa resample.

**The proxy column is network, not compute.** Streaming genuinely streams
through the RunPod proxy on this pod — 302 ms to first chunk against 1.7 s
total — but the round trip swamps the server-side saving. What an external
caller sees depends on their own RTT to the pod; measure from where your
client actually runs.

Startup on this box: ~63–106 s to `ready` with `optimize=1` (the `torch.compile`
warmup), then under a second to warm up.

## License

No license has been chosen yet, so default copyright applies — all rights
reserved. Add a `LICENSE` file before publishing this anywhere public.

Note that the model (`openbmb/VoxCPM2`) and any reference clip you add carry
their own terms, independent of whatever you pick for this code.
