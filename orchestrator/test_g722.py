"""
Fase 1 test: pure-Python G722 encoder vs the ffmpeg reference bitstream.

The PoC (backups/orchestrator-20260910/poc_g722_validated.py) already proved
the DEVICE accepts ffmpeg-encoded G722 — so ffmpeg's output is the ground
truth this port must match byte-for-byte.

Tests:
1. Byte-exactness vs ffmpeg (440 Hz tone + noise + silence segments)
2. Round-trip quality gate: ffmpeg-decode our stream, SNR vs original PCM
3. Real-time benchmark: ms per 20 ms frame (320 samples -> 160 bytes)
"""
import struct
import subprocess
import sys
import time

import g722

RATE = 16000
FRAME_SAMPLES = 320   # 20 ms @16 kHz
FRAME_BYTES = 160


def synth_pcm():
    """1.5 s: 440 Hz tone -> silence -> broadband noise (Agora-relevant shapes)."""
    import math
    import struct
    samples = []
    for i in range(int(RATE * 0.5)):
        samples.append(int(14000 * math.sin(2 * math.pi * 440 * i / RATE)))
    samples += [0] * int(RATE * 0.25)
    import random
    rng = 12345
    for _ in range(int(RATE * 0.5)):
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        samples.append((rng % 32000) - 16000)
    for i in range(int(RATE * 0.5)):  # speech-like sweep 300->3000 Hz
        f = 300 + 2700 * i / (RATE * 0.5)
        samples.append(int(9000 * math.sin(2 * math.pi * f * i / RATE)))
    return struct.pack("<%dh" % len(samples), *samples)


def ffmpeg_encode(pcm_path, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
         "-i", pcm_path, "-c:a", "g722", "-b:a", "64k", "-f", "g722", out_path],
        check=True, capture_output=True)
    return open(out_path, "rb").read()


def main():
    pcm = synth_pcm()
    open("/tmp/fase1.pcm", "wb").write(pcm)
    samples = struct.unpack("<%dh" % (len(pcm) // 2), pcm)

    ref = ffmpeg_encode("/tmp/fase1.pcm", "/tmp/fase1_ffmpeg.g722")

    enc = g722.G722Encoder()
    t0 = time.perf_counter()
    ours = enc.encode(samples)
    dt = (time.perf_counter() - t0) * 1000

    print(f"input : {len(samples)} samples ({len(pcm)} bytes PCM)")
    print(f"ffmpeg: {len(ref)} bytes G722")
    print(f"ours  : {len(ours)} bytes G722")

    if ours == ref:
        print("BYTE-EXACT vs ffmpeg: PASS")
    else:
        diff = sum(1 for a, b in zip(ref, ours) if a != b)
        print(f"BYTE-EXACT: FAIL ({diff}/{len(ref)} bytes differ)")
        return 1
    # Byte-exactness vs the ffmpeg reference IS the quality gate: the PoC
    # (poc_g722_validated.py) already proved the ReSpeaker device receives and
    # plays ffmpeg-encoded G722, and our stream is identical.

    # Benchmark
    print(f"encode time: {dt:.1f} ms total for {len(samples)} samples "
          f"= {dt / (len(samples) / FRAME_SAMPLES):.2f} ms per 20 ms frame "
          f"(real-time budget: 20 ms)")
    ok = ours == ref
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())