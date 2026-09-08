# Development

## Layout

```
server.py            uvicorn entry point — one worker
handler.py           RunPod serverless entry point
start.sh             venv + CUDA-matched torch + deps + serve
smoke_test.py        stdlib-only; times first-audio, writes out.wav
bench_ttfc.py        stdlib-only; median/p95 time-to-first-chunk, for A/B
Makefile             make setup | run | bg | test | stop | docker | clean
Dockerfile           CUDA 12.4 base, weights baked in
latina/config.py     every setting, every one an env var
latina/engine.py     VoxCPM2 wrapper, voice registry, audio helpers
latina/api.py        HTTP endpoints
latina/studio.py     studio /api routes — detachable
latina/audio_io.py   decode, resample, analyse, store clips
static/              studio page — vanilla JS, no build step
voices/              <id>.wav + optional <id>.txt
docs/                this documentation
```

`AGENTS.md` is the step-by-step runbook. `CLAUDE.md` covers design reasoning for
AI coding agents. This directory is the reference documentation for both humans
and agents.

## Testing

There are **no unit tests**, and they would not tell you much. What can break
here is audio quality and latency, and `smoke_test.py` measures both end to end:

```bash
make test                          # or: .venv/bin/python smoke_test.py
make test URL=https://host:8000    # against a deployed instance
```

```
  health: ok=True voices=['es_f_19'] sr=48000
  [1] first audio    267 ms · total   3551 ms ·  2.2s audio · RTF 1.59
      wrote out.wav — listen to it
  OK
```

Two numbers matter, and they are not the same number:

- **time to first audio**, which must be **far below total**. Equal means
  streaming is broken. This is the only thing separating a streaming endpoint
  from a slow blocking one.
- **RTF**, which must stay below 1.0 or playback stutters.

The smoke test is deliberately stdlib-only, so it runs against a deployed URL
from any machine without installing anything. It defaults to
`http://localhost:$LATINA_PORT`, falling back to 8000.

### Comparing two builds

`smoke_test.py` runs each phrase once, which is too noisy to judge a latency
change. `bench_ttfc.py` repeats and reports median/p95:

```bash
.venv/bin/python bench_ttfc.py                       # local, 5 runs/phrase
.venv/bin/python bench_ttfc.py https://host -n 10 --wav
```

```
short    n= 5  median    61.0 ms   min    59.5   p95    67.6   | total med  689 ms  RTF 0.61
```

Report the **median and p95 of time-to-first-chunk**, not a single run — the
work that dominates it (a disk read, a resample) is exactly the kind that
produces a long tail. It is also stdlib-only, and sends a `User-Agent`, because
RunPod's proxy 403s `Python-urllib`.

Then **listen to `out.wav`**. No metric here catches a voice that sounds wrong.

## Rules for changing this project

These are the ones that have already cost someone a debugging round.

**Do not raise the uvicorn worker count.** `server.py` pins `workers=1` because
each worker loads its own copy of the model onto the same GPU. Scale with more
containers.

**After touching `latina/api.py`, re-run `smoke_test.py`** and compare first
audio against total. Streaming breaks silently — nothing errors, it just stops
streaming.

**Do not change the terminal SSE frame.** It carries both `done` (miniclosedai
stops on this) and `end` (the Mozart page stops on this). Drop either and that
client hangs forever waiting for a terminator that never comes.

**Do not reshape `/voices`.** It is keyed by language because that is what
registration reads. Human-readable extras go in `/voices/detail`.

**Keep `speed` accepted and ignored.** miniclosedai sends it, VoxCPM2 has no
speed control, and rejecting an unknown field would fail the request.

**New settings go in `latina/config.py`**, env-overridable, and documented in
both `.env.example` and [docs/configuration.md](configuration.md) — not inline
in the code.

**Keep studio routes under `/api/…`.** The frozen surfaces must stay
unshadowable, and `/health`, `/voices` and `/voices/detail` byte-identical.

**Keep the studio import guarded.** The try/except in `latina/api.py` is what
turns a missing `python-multipart` into one log line instead of a dead `/speak`.

**Do not tell users to raise the sample rate** anywhere in the analyser's
warning copy. The model reads 0–8 kHz; upsampling adds nothing. See
[voices](voices.md#why-bandwidth-not-sample-rate).

**Do not make `es_f_19` pass the analyser.** It is graded `poor` on purpose as a
threshold regression test.

**Do not add the dealership vocabulary back.** It makes the project
domain-specific again.

## When changing generation parameters

Run `smoke_test.py` before and after, and compare **RTF and time-to-first-chunk**
— not just "did it produce a file". Record the numbers in the commit message;
that is how the existing defaults got their justification.

## Dependencies

`requirements.txt` deliberately does **not** pin torch, because the right wheel
depends on the host CUDA. `start.sh` installs it first from the matching index,
then the rest.

`soundfile` and `librosa` arrive transitively via `voxcpm`, but are declared
anyway — a hidden dependency on another package's requirements is how a working
box turns into a broken one after an unrelated upgrade.

Note that `start.sh` only pip-installs when `.venv` is **absent**. After adding
a dependency, either `rm -rf .venv` or install it by hand on existing boxes.

## Git hygiene

`.gitignore` covers `.venv/`, `.env`, `latina.log`, `out.wav`, `*.part` files
and `voices/.trash/`. Reference clips in `voices/` *are* tracked — the voice
ships with the code, which is what makes the project a `zip -r` away from
running on another box.
