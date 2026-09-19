#!/usr/bin/env python3
"""Smoke test. Run it against a live service:

    python smoke_test.py                       # http://localhost:8000
    python smoke_test.py https://host:8000     # anywhere else

Writes out.wav so you can actually listen, and prints the latency that matters
for a phone call: time to the FIRST audio chunk, not to the last.

Then checks what miniclosedai relies on beyond Spanish TTS: `/voices` has both
an `en` and an `es` bucket, the English voice speaks, and every ASR option in
`/asr` transcribes a clip in its own language (Spanish with `es` and `auto`,
English with `en`). Skip that part with SMOKE_ASR=0.
"""
import base64, json, os, struct, sys, time, urllib.request, uuid

# Default to the port the service is actually configured for, so this works
# without arguments on a box whose .env moves LATINA_PORT off 8000.
_PORT = os.getenv("LATINA_PORT", "8000")
BASE = (sys.argv[1] if len(sys.argv) > 1
        else f"http://localhost:{_PORT}").rstrip("/")
KEY = None
FRASES = [
    "Buenas tardes, ¿hablo con el señor Benítez?",
    "Tiene un vencimiento de hace cuarenta y cinco días por un millón "
    "ochocientos cincuenta mil guaraníes.",
    "Novecientos veinticinco mil cada una. La primera este mes, la segunda "
    "el mes que viene, misma fecha.",
]


def post(path, payload):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {KEY}"} if KEY else {})})
    return urllib.request.urlopen(req, timeout=300)


def wav(pcm, sr):
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36+n) + b"WAVE" + b"fmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr*2, 2, 16) +
            b"data" + struct.pack("<I", n) + pcm)


h = json.load(urllib.request.urlopen(f"{BASE}/health", timeout=30))
print(f"  health: ok={h['ok']} voices={h['voices']} sr={h['sample_rate']}")
if not h["ok"]:
    sys.exit("  model still loading — wait and retry")

for i, texto in enumerate(FRASES, 1):
    t0 = time.perf_counter(); first = None; pcm = b""; sr = h["sample_rate"]
    r = post("/speak/stream", {"text": texto})
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        ev = json.loads(line[5:])
        if ev.get("error"):
            sys.exit(f"  ERROR: {ev['error']}")
        if ev.get("chunk_b64"):
            if first is None:
                first = (time.perf_counter()-t0)*1000
            pcm += base64.b64decode(ev["chunk_b64"]); sr = ev["sample_rate"]
    total = (time.perf_counter()-t0)*1000
    dur = len(pcm)/2/sr
    print(f"  [{i}] first audio {first:6.0f} ms · total {total:6.0f} ms · "
          f"{dur:4.1f}s audio · RTF {(total/1000)/dur:.2f}")
    if i == 1:
        open("out.wav", "wb").write(wav(pcm, sr))
        print("      wrote out.wav — listen to it")
        es_clip = wav(pcm, sr)


def transcribe(audio, language):
    """multipart POST /transcribe, stdlib only."""
    b = uuid.uuid4().hex
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="audio"; '
             f'filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
             + audio + b"\r\n"]
    if language:
        parts.append(f'--{b}\r\nContent-Disposition: form-data; '
                     f'name="language"\r\n\r\n{language}\r\n'.encode())
    body = b"".join(parts) + f"--{b}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/transcribe", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={b}",
                 **({"Authorization": f"Bearer {KEY}"} if KEY else {})})
    return json.load(urllib.request.urlopen(req, timeout=300))


if os.getenv("SMOKE_ASR", "1") != "0":
    cat = json.load(urllib.request.urlopen(f"{BASE}/voices", timeout=30))
    if not ("en" in cat and "es" in cat):
        sys.exit(f"  ERROR: /voices lacks an en or es bucket: {sorted(cat)}")
    en_voice = cat["en"][0]["id"]
    print(f"  catalog: " + " · ".join(f"{k}: {len(v)}" for k, v in cat.items()))

    en_clip = post("/speak", {"text": "Hello, thank you for calling. How can I help you?",
                              "voice": en_voice}).read()
    opts = json.load(urllib.request.urlopen(f"{BASE}/asr", timeout=30))["options"]
    # Each option gets a clip in its own language, plus a word that must survive.
    cases = {"es": (es_clip, "tardes"), "auto": (es_clip, "tardes"),
             "en": (en_clip, "thank")}
    for o in opts:
        clip, word = cases.get(o["language"], (es_clip, "tardes"))
        t0 = time.perf_counter()
        r = transcribe(clip, None if o["language"] == "auto" else o["language"])
        ms = (time.perf_counter() - t0) * 1000
        ok = word in r["text"].lower()
        print(f"  asr {o['language']:>4} ({r.get('model', '?').split('/')[-1]}) "
              f"{ms:5.0f} ms · {r['text']!r}")
        if not ok:
            sys.exit(f"  ERROR: asr {o['language']} lost {word!r}")
print("  OK")
