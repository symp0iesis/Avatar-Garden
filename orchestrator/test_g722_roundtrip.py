"""
Fase 6 step 1: G722 decoder round-trip test.

The encoder is already byte-exact vs ffmpeg (Fase 1). This validates the new
decoder: encode(pcm) -> decode() -> compare vs original PCM (SNR gate), plus
per-material breakdown and speed.
"""
import math
import struct
import sys
import time

import g722

RATE = 16000
FRAME_SAMPLES = 320


def synth_pcm():
    import random
    samples = []
    for i in range(int(RATE * 0.5)):
        samples.append(int(14000 * math.sin(2 * math.pi * 440 * i / RATE)))
    samples += [0] * int(RATE * 0.25)
    rng = 12345
    for _ in range(int(RATE * 0.5)):
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        samples.append((rng % 32000) - 16000)
    for i in range(int(RATE * 0.5)):
        f = 300 + 2700 * i / (RATE * 0.5)
        samples.append(int(9000 * math.sin(2 * math.pi * f * i / RATE)))
    return samples


def snr_db(ref, got, align=True):
    """SNR with codec-delay compensation: find the lag (0..48) that maximizes
    correlation, then measure error at that alignment."""
    n = min(len(ref), len(got))
    ref = ref[:n]; got = got[:n]
    best = None
    lags = range(0, 49) if align else [0]
    for lag in lags:
        if lag >= n: break
        err = sum((a - b) ** 2 for a, b in zip(ref[lag:], got))
        if best is None or err < best[1]:
            best = (lag, err)
    lag, err = best
    sig = sum(a * a for a in ref[lag:]) or 1
    return 10 * math.log10(sig / max(err, 1)), lag


def main():
    samples = synth_pcm()
    enc = g722.G722Encoder()
    dec = g722.G722Decoder()

    t0 = time.perf_counter()
    g722_bytes = enc.encode(samples)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded = dec.decode(g722_bytes)
    t_dec = time.perf_counter() - t0

    n = min(len(samples), len(decoded))
    ref, got = samples[:n], decoded[:n]

    overall = snr_db(ref, got)
    seg = lambda a, b: snr_db(ref[int(a * RATE):int(b * RATE)],
                              got[int(a * RATE):int(b * RATE)], align=False)[0]
    print(f"overall SNR: {overall[0]:.1f} dB (best alignment lag {overall[1]} samples)")
    print(f"  0.0-0.5s tone : {seg(0.0, 0.5):.1f} dB")
    print(f"  0.75-1.25s rnd: {seg(0.75, 1.25):.1f} dB (worst case: noise)")
    print(f"  1.25-1.75s swp: {seg(1.25, 1.75):.1f} dB")

    t_enc_ms = t_enc * 1000 / (len(samples) / FRAME_SAMPLES)
    t_dec_ms = t_dec * 1000 / (len(samples) / FRAME_SAMPLES)
    print(f"encode: {t_enc_ms:.2f} ms/frame | decode: {t_dec_ms:.2f} ms/frame "
          f"(budget 20 ms)")

    ok = overall[0] > 20.0
    print("ROUND-TRIP:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())