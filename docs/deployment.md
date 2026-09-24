# Deployment

Four ways to run this, in increasing order of ceremony. All of them serve the
same engine.

## Local / any GPU box

```bash
./start.sh
```

`start.sh` is safe to run repeatedly — it only does setup work once. On the
first run it:

1. installs missing system packages (`libsndfile1`, `ffmpeg`, `python3-venv`)
   when running as root;
2. creates `.venv`;
3. reads `CUDA Version` out of `nvidia-smi` and installs the matching torch
   wheel — `cu130` for CUDA 13, `cu124` otherwise, CPU torch with a warning if
   there is no GPU at all;
4. installs `requirements.txt` and verifies `import torch, voxcpm` before going
   further, so a broken install fails immediately rather than 40 s into the
   model load;
5. sources `.env` and execs `server.py`.

Override the wheel index if the detection guesses wrong:

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu130 ./start.sh
```

First start downloads roughly **5 GB** of model weights. Give it 5–15 minutes
and do not conclude it is hung.

Success looks like:

```
[latina] loading openbmb/VoxCPM2 (optimize=False, denoiser=False)…
[latina] ready in 18s · 48000 Hz · voices: ['es_f_19']
[latina] warmed up in 5.7s
```

`make` targets wrap the same thing: `make run`, `make bg`, `make test`,
`make stop`, `make docker`, `make clean`.

### Waiting for readiness

**Poll `/health` for `"ok":true`.** Do not wait on the `warmed up` log line: it
is printed by the startup handler, which finishes *before* uvicorn starts
accepting connections, so grepping for it races the socket and your next request
gets "connection refused". This cost a real debugging round.

```bash
rm -f latina.log                    # a stale log makes the wait match instantly
nohup ./start.sh > latina.log 2>&1 &
until curl -s --max-time 3 localhost:8000/health | grep -q '"ok":true'; do sleep 5; done
echo ready
```

`make bg` does exactly this, and additionally aborts early if it sees a
`Traceback` in the log.

## Docker

```bash
docker build -t latina-voice .
docker run --gpus all -p 8000:8000 latina-voice
```

The image is based on `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
i.e. **CUDA 12.4 torch**. On a CUDA 13 host, switch the base image to a cu130
one — VoxCPM2 does not care, torch does.

The build **bakes the model weights into the image** so cold start is load-only
rather than download-plus-load. That makes the image large; comment out the
prefetch `RUN` line to trade image size for a slower first boot. The prefetch is
tolerant — if it fails at build time it warns and the weights download on first
start instead.

Reference clips are copied in, so the voice ships with the image.

## RunPod pod — the recommended cloud path

1. Create a pod. **RTX A5000 24 GB** on Community Cloud is enough; the service
   uses about 6.3 GB. Pick a **PyTorch** template, not a bare Ubuntu one.
2. Expose port **8000** — and expose it as a **direct TCP port**, not only
   through the HTTP proxy. See the proxy warning below.
3. Get the project onto the pod's persistent volume:
   ```bash
   scp latina_voice_tts.zip root@<pod-ip>:/workspace/
   # on the pod
   cd /workspace && unzip latina_voice_tts.zip && cd latina_voice_tts
   ```
4. `./start.sh`. As root — the RunPod default — it installs its own system
   packages, builds the venv, picks the CUDA-matched torch wheel and serves.
5. Wait for `/health` to report `ok:true`.
6. Verify on the pod, then from outside:
   ```bash
   .venv/bin/python smoke_test.py
   python smoke_test.py https://<pod-id>-8000.proxy.runpod.net
   ```

Nothing needs editing to get this far. Everything the service needs, the voice
included, is in the archive; the only downloads are weights and Python packages.

**Persist the weights.** Keep the project on `/workspace` and point `HF_HOME` at
the same volume, or a pod restart re-downloads 5 GB:

```bash
echo "HF_HOME=/workspace/hf-cache" >> .env
```

## RunPod serverless

```bash
docker build -t <registry>/latina-voice:1.0 .
docker push  <registry>/latina-voice:1.0
```

Set the container start command to `python handler.py`. Jobs take:

```json
{"input": {"text": "Buenas tardes.", "voice": "es_f_19"}}
```

and stream `{"audio_chunk": "<base64 int16 PCM>", "sample_rate": 48000,
"chunk": 1}` items, ending with `{"done": true, "chunks": 7, "ms": 3120}`.

`handler.py` calls `engine.load()` at import, so cold start pays for the model
once per worker rather than once per job. Cold start is still model load —
tens of seconds, or minutes if the weights are not baked into the image. **Do
not put a live phone call behind a worker that scales to zero.**

## The RunPod HTTP proxy buffers SSE

Through `https://<pod-id>-8000.proxy.runpod.net` you will most likely receive
every chunk at the end. RunPod's own BCP docs say so outright: *"the pod proxy
buffers SSE so `/stream/{job_id}` never arrives at the client."*

The symptom is exactly the same as broken streaming code: **first audio equals
total time**, and nothing errors. The entire streaming advantage is gone.

Two ways around it:

- **Expose a direct TCP port.** RunPod gives you an `<ip>:<port>` that bypasses
  the HTTP proxy; register that URL and streaming behaves as it does locally.
- **Use `/speak.wav`** and accept that it is one blocking response — honest,
  and immune to the problem.

Measure, do not assume: run `smoke_test.py` against the deployed URL and compare
**first audio vs total**. If they are equal, you are being proxied.

## Pairing behind miniclosedai-voice

Set `VOICE_UPSTREAM_URL=http://<this-host>:8088` on a
[miniclosedai-voice](https://github.com/Interdata-Research/miniclosedai-voice)
instance and it serves our voices as its Spanish half, under its own URL, with
call mode. Register **only** that service with miniclosedai in that layout.
Nothing changes here; if this service is down its voices simply disappear from
the front service's catalog. Full instructions: that repo's `DEPLOY.md`.

## Registering with miniclosedai

The service already speaks miniclosedai's voice-backend protocol — `GET /voices`
keyed by language, `POST /speak/stream` terminating on `{"done": true}` — so it
registers directly and then appears in every bot's voice picker, the Mozart call
bot included. No code change on the miniclosedai side.

Settings → Add endpoint → kind **Voice**, base URL = your pod URL. Or:

```bash
curl -k -X POST https://localhost:8095/api/backends \
  -H 'Content-Type: application/json' \
  -d '{"name":"Latina Voice","kind":"voice",
       "base_url":"https://<pod-id>-8000.proxy.runpod.net",
       "api_key":"<LATINA_API_KEY, if you set one>","enabled":true}'
```

Verify it took:

```bash
curl -sk https://localhost:8095/api/voices | grep -o '"backend_name":"[^"]*"' | sort -u
```

Remove a stale registration with `DELETE /api/backends/<id>`. A dead voice
backend makes miniclosedai's catalog slow, because it waits for that backend to
time out on every listing.

Verified end to end against miniclosedai's own client: the catalog lists
`es_f_19`, `speak_stream` returns 18 chunks and terminates on `done`, first
audio 304 ms. Through the Mozart demo the same call measured 279 ms.

## Reaching the studio through a proxy

Recording and streaming cannot both work over the RunPod HTTPS proxy.
`getUserMedia` needs a secure context, so plain `http://<pod-ip>:8000` shows no
record button; the HTTPS proxy allows recording but buffers SSE, so playback
arrives all at once.

The only setup where both work is an SSH tunnel:

```bash
ssh -p <tcp-port> root@<pod-ip> -L 8000:localhost:8000
# then open http://localhost:8000/studio/
```

The page detects the insecure-context case and says so, and warns when it sees
every chunk arrive at once — the proxy-buffering symptom.
