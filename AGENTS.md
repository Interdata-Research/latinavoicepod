# AGENTS.md — runbook for an AI coding agent

You are working in **latina_voice_tts**: a FastAPI service that loads VoxCPM2
into GPU memory and speaks Latin-American Spanish. This file is the operating
manual. Follow it top to bottom; every step says how to tell whether it worked.

`CLAUDE.md` holds the same guidance plus deeper design notes. `README.md` is
written for a human. `docs/` is the reference documentation — [api](docs/api.md),
[configuration](docs/configuration.md), [voices](docs/voices.md),
[deployment](docs/deployment.md), [architecture](docs/architecture.md),
[troubleshooting](docs/troubleshooting.md), [development](docs/development.md).
If they disagree with this file, this file is newer.

Ports below assume the default `:8000`; `LATINA_PORT` moves everything.

---

## 0. Preconditions — check before doing anything

```bash
nvidia-smi                 # must print a GPU. No GPU → see §6 "No GPU".
python3 --version          # 3.10 or newer
df -h .                    # need ~15 GB free: 5 GB weights + venv + torch
```

The project must contain `voices/es_f_19.wav`. That file *is* the voice — if it
is missing the service starts and then fails every request. Verify:

```bash
ls -la voices/             # expect es_f_19.wav (~163 KB) and es_f_19.txt
```

## 1. Start it

```bash
./start.sh
```

One command. It creates `.venv`, detects CUDA from `nvidia-smi`, installs the
matching torch wheel, installs the rest, and serves on `:8000`.

**First run takes several minutes** — it downloads ~5 GB of model weights.
Do not interrupt it and do not conclude it is hung before 10 minutes.

**Success looks like this** in the output:

```
[latina] loading openbmb/VoxCPM2 (optimize=False, denoiser=False)…
[latina] ready in 18s · 48000 Hz · voices: ['es_f_19']
[latina] warmed up in 5.7s
```

`/health` returns `ok:false` while loading and every `/speak` returns HTTP 503
until it flips. In the foreground, `warmed up` tells you the load succeeded;
to wait for it *programmatically*, poll `/health` instead — see the warning
below.

To run it in the background and wait correctly, **poll `/health`** — or just
use `make bg`, which does exactly this:

```bash
rm -f latina.log                      # a stale log makes the wait match instantly
nohup ./start.sh > latina.log 2>&1 &
until curl -s --max-time 3 localhost:8000/health | grep -q '"ok":true'; do sleep 5; done
echo ready
```

**Do not wait on the `warmed up` log line.** It is printed by the startup
handler, which completes *before* uvicorn begins accepting connections — wait on
it and your very next request gets "connection refused". Measured, not
theoretical: it cost a debugging round while writing this. `/health` returning
`"ok":true` is the only trustworthy readiness signal.

## 2. Verify

```bash
curl localhost:8000/health
```

Expect `"ok":true`, `"sample_rate":48000`, `"voices":["es_f_19"]`.

Then the real check, which synthesizes and times it:

```bash
.venv/bin/python smoke_test.py
```

Expect something close to:

```
  health: ok=True voices=['es_f_19'] sr=48000
  [1] first audio    267 ms · total   3551 ms ·  2.2s audio · RTF 1.59
      wrote out.wav — listen to it
  OK
```

**The number that matters is `first audio`, and it must be far below `total`.**
If they are equal, streaming is broken — see §6.

## 2b. The web GUI

Open **`http://localhost:8000/studio/`** (`/` redirects there). Upload or record
a reference clip, see a quality verdict before saving, then synthesize and
listen. Uploaded voices appear in `/voices` immediately — no restart.

```bash
curl -s localhost:8000/api/studio                      # config + auth_required
curl -s localhost:8000/api/voices/es_f_19/analysis     # grade the shipped clip
curl -s -F 'file=@clip.wav' -F 'voice_id=maria' localhost:8000/api/voices/upload
curl -s -X DELETE localhost:8000/api/voices/maria      # -> voices/.trash/
```

**Recording and streaming cannot both work over the RunPod proxy.**
`getUserMedia` needs a secure context, so plain `http://<pod-ip>:8000` has no
record button; the HTTPS proxy allows recording but buffers SSE, so playback
arrives all at once. The only setup where both work is an SSH tunnel:

```bash
ssh -p <tcp-port> root@<pod-ip> -L 8000:localhost:8000
# then open http://localhost:8000/studio/
```

The page detects the insecure-context case and says so; it also warns when it
sees every chunk arrive at once, which is the proxy-buffering symptom.

If the GUI is missing, the service will have logged one line saying why —
`[latina] voice studio disabled: ...` — and kept serving `/speak` normally. The
usual cause on an existing box is `python-multipart` not being installed
(`start.sh` only pip-installs when `.venv` is absent):

```bash
.venv/bin/pip install python-multipart
```

## 3. Use it

```bash
# One WAV file (simplest; no streaming)
curl -X POST localhost:8000/speak.wav \
  -H 'Content-Type: application/json' \
  -d '{"text":"Buenas tardes, ¿hablo con el señor Benítez?"}' \
  --output hola.wav

# Streaming SSE: base64 int16 PCM chunks as they are generated
curl -N -X POST localhost:8000/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hola."}'
```

Audio is **48 kHz, mono, little-endian int16**. Read the rate from the
`sample_rate` field on each chunk rather than hardcoding it.

Endpoints: `GET /health`, `GET /voices`, `GET /voices/detail`,
`POST /speak`, `POST /speak/stream` (same handler), `POST /speak.wav`.

## 4. Register it with miniclosedai

This service already speaks miniclosedai's voice-backend protocol. Registering
it makes the voice appear in every bot's picker there.

```bash
curl -k -X POST https://localhost:8095/api/backends \
  -H 'Content-Type: application/json' \
  -d '{"name":"Latina Voice","kind":"voice",
       "base_url":"http://localhost:8000","enabled":true}'
```

Verify it took:

```bash
curl -sk https://localhost:8095/api/voices | grep -o '"backend_name":"[^"]*"' | sort -u
```

Remove it with `DELETE /api/backends/<id>` if you registered a URL that no
longer exists — a dead voice backend makes miniclosedai's catalog slow, because
it waits for that backend to time out on every listing.

## 5. Deploy to RunPod

### Pod — the recommended path, start to finish

1. Create a pod. **RTX A5000 24 GB** on Community Cloud is enough (the service
   uses ~6.3 GB). Pick a **PyTorch** template, not a bare Ubuntu one.
2. Expose port **8000**. For streaming to actually stream, expose it as a
   **direct TCP port**, not only the HTTP proxy — see the proxy warning below.
3. Get the zip onto the pod, whichever way is available:
   ```bash
   # from your machine
   scp latina_voice_tts.zip root@<pod-ip>:/workspace/
   # then, on the pod
   cd /workspace && unzip latina_voice_tts.zip && cd latina_voice_tts
   ```
4. Run it:
   ```bash
   ./start.sh
   ```
   As root — which is the norm on a RunPod pod — this installs any missing
   system packages itself (`libsndfile1`, `ffmpeg`, `python3-venv`), builds the
   venv, picks the torch wheel matching the pod's CUDA, and serves.
5. Wait for `warmed up` in the output. First run pulls ~2–3 GB of torch and
   ~5 GB of model weights, so give it 5–15 minutes depending on the pod.
6. Verify from the pod:
   ```bash
   .venv/bin/python smoke_test.py
   ```
   Then from outside, against the pod URL, to check the proxy is not buffering:
   ```bash
   python smoke_test.py https://<pod-id>-8000.proxy.runpod.net
   ```

Nothing needs to be edited to get this far. Everything the service needs —
including the voice itself — is in the zip; the only downloads are the model
weights and Python packages.

**Persist the weights.** Put the project on the pod's persistent volume
(`/workspace`) and set `HF_HOME=/workspace/hf-cache` so a pod restart does not
re-download 5 GB. Add it to `.env`:
```bash
echo "HF_HOME=/workspace/hf-cache" >> .env
```

**Serverless.** Build the image, set the start command to `python handler.py`:

```bash
docker build -t <registry>/latina-voice:1.0 .
docker push  <registry>/latina-voice:1.0
```

Cold start is model download plus load — tens of seconds. Do not put a live
phone call behind a worker that scales to zero.

**The RunPod HTTP proxy buffers SSE.** Through
`https://<pod-id>-8000.proxy.runpod.net` you will likely receive every chunk at
the end, making `first audio == total`. Expose a **direct TCP port** instead
(RunPod gives `<ip>:<port>` that bypasses the proxy), or use `/speak.wav` and
accept that it is one blocking response. Always confirm with `smoke_test.py`
against the deployed URL.

## 6. Failure modes, and what each one means

| Symptom | Cause | Fix |
|---|---|---|
| Startup never prints `ready`, GPU busy | `torch.compile` on an unsupported GPU. Look for `Not enough SMs to use max_autotune_gemm` | `LATINA_OPTIMIZE=0` (already the default). Never set it to 1 without testing. |
| `first audio == total` in smoke test | Streaming buffered — either the code in `latina/api.py` or a proxy in front | Locally: see the streaming note in `CLAUDE.md`. On RunPod: it is the HTTP proxy, use a direct TCP port. |
| HTTP 503 on `/speak` | Model still loading | Wait for `warmed up` in the log. |
| A request never returns, GPU pinned | Bad reference clip → open-ended generation that does not stop | Use a clean 5–15 s single-speaker clip. Noisy or telephone-band audio causes this. |
| `Address already in use`, process exits 3 | An old instance holds the port | `kill -9 $(ss -ltnp \| grep ':8000' \| grep -oP 'pid=\K[0-9]+')` — matching on a process-name pattern often misses it, because the command line is just `server.py`. |
| Import error on torch / CUDA mismatch | Wrong wheel for the host CUDA | `rm -rf .venv && TORCH_INDEX=https://download.pytorch.org/whl/cu124 ./start.sh` (or `cu130`). |
| **No GPU** | CPU torch fallback | It will run and be unusably slow (RTF ~10). This project needs a GPU; do not benchmark on CPU and report the result as the project's performance. |

## 7. Rules for changing this project

- **Do not raise the uvicorn worker count.** `server.py` pins `workers=1` on
  purpose: each worker loads its own copy of the model onto the same GPU. Scale
  with more containers.
- **Do not add the dealership vocabulary back.** The voice came from a
  car-dealership bot that wrapped English brand names for pronunciation, driven
  by a vehicle database. It is deliberately absent; re-adding it makes this
  project domain-specific again.
- **After touching `latina/api.py`, re-run `smoke_test.py`** and check first-vs-
  total. Streaming breaks silently — nothing errors, it just stops streaming.
- **Do not change the terminal SSE frame.** It carries both `done` (miniclosedai
  stops on this) and `end` (the Mozart page stops on this). Drop either and that
  client hangs forever.
- Settings live in `latina/config.py`, all env-overridable. Add new ones there,
  not inline in the code, and document them in `.env.example`.

## 8. Where things are

```
server.py          HTTP entry point (uvicorn, one worker)
handler.py         RunPod serverless entry point — same engine, different transport
start.sh           venv + CUDA-matched torch + deps + run
smoke_test.py      stdlib-only; times first-audio and writes out.wav
bench_ttfc.py      stdlib-only; median/p95 TTFC, for comparing two builds
Makefile           make setup | run | bg | test | stop | docker | clean
latina/config.py   every setting, every one an env var
latina/engine.py   VoxCPM2 wrapper, voice registry, audio helpers
latina/api.py      the endpoints
latina/studio.py   voice studio /api routes — detachable, fails open
latina/audio_io.py decode, resample, analyse, store reference clips
static/            the studio page — vanilla JS, no build step
voices/            <id>.wav + optional <id>.txt transcript
docs/              reference documentation (api, config, voices, deploy, …)
```

Measured on a GB10 (DGX Spark): load 18 s, warmup 6 s, **first audio 267 ms**,
RTF ~1.6, 0.0% WER round-tripped through Whisper large-v3-turbo.
