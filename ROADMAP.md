# Notes for the next version

Written 2026-09-24. Context a future session needs before changing the parts
that other software now depends on.

## Who calls this service

1. **miniclosedai** directly, as a `kind=voice` backend.
2. **miniclosedai-voice**, which can run in front of this service and use it as
   its Spanish engine (`VOICE_UPSTREAM_URL`). It merges our `/voices` into its
   catalog and forwards `/speak`, `/speak/stream` and Spanish `/transcribe`. Its
   `upstream.py` is a thin client of the contract below; its `ROADMAP.md`
   documents that side.
3. The Mozart demo page.

**The wire contract is load-bearing for all three** and is spelled out in
`CLAUDE.md` under *Wire protocol*. The short version, none of which may change
casually: `/voices` is keyed by language; the terminal SSE frame carries **both**
`done` and `end`; `/speak` returns one WAV body while `/speak/stream` returns
SSE; audio is 48 kHz mono int16 and every frame carries its own `sample_rate`;
`speed` is accepted and ignored.

## Known limitations

| Limitation | Notes |
|---|---|
| No call mode | No `/call/*`, no `/webrtc/offer`; `/health` reports `relay_capable: false` so miniclosedai never routes a call here. Pair with miniclosedai-voice if you want calls with these voices. |
| One generation at a time | `engine._lock` serialises synthesis and `asr._run_lock` serialises Whisper. Concurrent callers queue. Scale by running more instances, not more workers — `server.py` hard-codes `workers=1` because each worker would load its own copy of the model. |
| A bad reference clip can hang generation | VoxCPM2 generation is open-ended; a noisy or badly transcribed clip can loop forever, and `asyncio.wait_for` cannot cancel the thread. A generation cap or a killable subprocess is the real fix. |
| Whisper translates on a wrong language | `/transcribe` with `language=en` on Spanish audio returns English prose, with no error. Intrinsic to Whisper; `auto` is the safe default for mixed callers. |
| ASR VRAM | `large-v3` + `large-v3-turbo` ≈ 4.5 GB on top of VoxCPM2. `LATINA_ASR_MODELS=es=large-v3-turbo` collapses it to one model, `LATINA_ASR=0` drops ASR entirely. |
| Studio uploads are local-only | When paired, a clip uploaded through the front service's GUI lands in *its* voices dir, not ours. Upload through **this** service's `/studio/` for Spanish voices. |

## Worth building next

1. **A generation timeout that actually cancels.** The single worst failure mode
   is a runaway generation holding the lock: every later request queues behind
   it and the service looks hung to the watchdog. Needs a subprocess (or a
   cooperative cancel inside the loop), not `asyncio.wait_for`.
2. **Per-request cfg / timesteps overrides.** `LATINA_CFG` and
   `LATINA_TIMESTEPS` are process-wide; auditioning quality settings means a
   restart. Accepting them per request (bounded) would make the studio far more
   useful.
3. **Voice-level defaults.** `<id>.json` carries `name`/`language`/`gender`; a
   per-voice `cfg`, `timesteps` or loudness target would let a dull clip be
   compensated without changing the global config.
4. **Loudness normalisation on output.** Clips differ in level (`romina` peaks
   near full scale, `es_f_19` is quiet), which is audible when a bot switches
   voices. Normalising the *output* to a target LUFS would fix it without
   touching the references.
5. **Expose the analyser verdict in `/voices/detail`.** The studio grades a clip
   at upload; that grade is not in the API, so a caller cannot warn "this voice
   is telephone-band" without re-running the analysis.

## Things that look like bugs and are not

- `es_f_19` grading **poor** is a deliberate self-test: if a change ever makes it
  pass, the analyser's thresholds are wrong.
- The "Unable to detect NVIDIA GPU" style warnings from dependencies are
  cosmetic; the real check is `cuda=True` in the startup log and `device` in
  `/health`.
- `speed` being ignored is intentional — miniclosedai sends it, VoxCPM2 has no
  speed control, and rejecting an unknown field would fail the request.
