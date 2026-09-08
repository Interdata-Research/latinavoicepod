#!/usr/bin/env python3
"""Measure time-to-first-chunk (TTFC): POST sent -> first audio bytes in hand.

    python bench_ttfc.py                          # http://localhost:8088
    python bench_ttfc.py https://host -n 10       # more runs
    python bench_ttfc.py --wav                    # also time the one-shot /speak

TTFC is the number that decides whether a phone call feels responsive. Total
time and RTF matter only for whether playback keeps up afterwards.

Stdlib only, so it runs against a deployed URL from any machine.
"""
import argparse, base64, json, statistics, sys, time, urllib.request

PHRASES = [
    ("short",  "Sí, con él."),
    ("medium", "Buenas tardes, ¿hablo con el señor Benítez?"),
    ("long",   "Tiene un vencimiento de hace cuarenta y cinco días por un "
               "millón ochocientos cincuenta mil guaraníes."),
]


# RunPod's HTTPS proxy 403s the default "Python-urllib/3.11" agent.
UA = "latina-bench/1.0"


def _get(url):
    return urllib.request.Request(url, headers={"User-Agent": UA})


def _req(base, path, payload, key):
    hdrs = {"Content-Type": "application/json", "User-Agent": UA}
    if key:
        hdrs["Authorization"] = f"Bearer {key}"
    return urllib.request.Request(f"{base}{path}",
                                  data=json.dumps(payload).encode(), headers=hdrs)


def stream_once(base, text, voice, key):
    """-> (ttfc_ms, total_ms, audio_s, chunks, sample_rate)"""
    t0 = time.perf_counter()
    first = None
    n = 0
    nbytes = 0
    sr = 48000
    r = urllib.request.urlopen(_req(base, "/speak/stream",
                                    {"text": text, **({"voice": voice} if voice else {})},
                                    key), timeout=300)
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        ev = json.loads(line[5:])
        if ev.get("error"):
            raise RuntimeError(ev["error"])
        if ev.get("chunk_b64"):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            n += 1
            nbytes += len(base64.b64decode(ev["chunk_b64"]))
            sr = ev.get("sample_rate", sr)
    total = (time.perf_counter() - t0) * 1000
    if first is None:
        raise RuntimeError("no audio chunks received")
    return first, total, nbytes / 2 / sr, n, sr


def wav_once(base, text, voice, key):
    t0 = time.perf_counter()
    r = urllib.request.urlopen(_req(base, "/speak",
                                    {"text": text, **({"voice": voice} if voice else {})},
                                    key), timeout=300)
    body = r.read()
    total = (time.perf_counter() - t0) * 1000
    ctype = r.headers.get("Content-Type", "")
    return total, len(body), ctype


def summarize(label, vals):
    vals = sorted(vals)
    med = statistics.median(vals)
    p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
    return (f"{label:8s} n={len(vals):2d}  median {med:7.1f} ms   "
            f"min {vals[0]:7.1f}   p95 {p95:7.1f}   max {vals[-1]:7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://localhost:8088")
    ap.add_argument("-n", "--runs", type=int, default=5)
    ap.add_argument("-v", "--voice", default=None)
    ap.add_argument("-k", "--key", default=None)
    ap.add_argument("--wav", action="store_true", help="also time one-shot /speak")
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()
    base = a.base.rstrip("/")

    h = json.load(urllib.request.urlopen(_get(f"{base}/health"), timeout=30))
    print(f"target {base}")
    print(f"health ok={h['ok']} sr={h['sample_rate']} optimize={h.get('optimize')} "
          f"voices={h['voices']} default={h.get('default_voice')}")
    if not h["ok"]:
        sys.exit("model still loading")

    for _ in range(a.warmup):
        stream_once(base, PHRASES[1][1], a.voice, a.key)

    print(f"\n/speak/stream — time to FIRST audio chunk ({a.runs} runs/phrase)")
    print("-" * 78)
    allttfc = []
    for name, text in PHRASES:
        ttfcs, totals, rtfs, chunks = [], [], [], 0
        for _ in range(a.runs):
            ttfc, total, dur, n, _sr = stream_once(base, text, a.voice, a.key)
            ttfcs.append(ttfc); totals.append(total); chunks = n
            rtfs.append((total / 1000) / dur if dur else 0)
        allttfc += ttfcs
        print(summarize(name, ttfcs)
              + f"   | total med {statistics.median(totals):6.0f} ms"
                f"  RTF {statistics.median(rtfs):.2f}  chunks {chunks}")
    print("-" * 78)
    print(summarize("ALL", allttfc))

    if a.wav:
        print(f"\n/speak — one-shot WAV, total time ({a.runs} runs/phrase)")
        print("-" * 78)
        for name, text in PHRASES:
            ts = []
            for _ in range(a.runs):
                t, nb, ct = wav_once(base, text, a.voice, a.key)
                ts.append(t)
            print(summarize(name, ts) + f"   | {nb/1024:.0f} KiB  {ct}")


if __name__ == "__main__":
    main()
