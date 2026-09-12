"""
Fase 2 test: Cartesia TTS -> Deepgram streaming ASR -> transcript.

No human speaker needed: synthesize a pt sentence with marumbi's voice
(Cartesia sonic-3, the .ai domain), then stream the PCM to the
DeepgramStreamer in real time and verify:
1. the transcript comes back (contains 'cutia')
2. UtteranceEnd arrives ~500ms after speech ends (endpointing)
"""
import asyncio
import re
import struct
import sys
import time

import requests

import asr_deepgram

CARTESIA_TTS = "https://api.cartesia.ai/tts/bytes"
MARUMBI_VOICE = "9904416a-0831-44ea-b8ee-5f145e8f9bbf"
SENTENCE = "Hoje eu avistei uma cutia perto do rio Marumbi."
RATE = 16000


def load_env():
    env = {}
    for line in open("/root/AvatarGarden/backend/.env"):
        m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    return env


def cartesia_tts_pcm(api_key, text, voice_id):
    """Synthesize speech -> raw s16le 16 kHz mono PCM (the .ai domain works from BR)."""
    import json as _json
    import requests as _requests
    resp = _requests.post(
        CARTESIA_TTS,
        headers={"X-API-Key": api_key, "Content-Type": "application/json",
                 "Cartesia-Version": "2026-08-14"},
        json={
            "transcript": text,
            "model_id": "sonic-3",
            "voice": voice_id,
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": RATE},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


async def run_stream_test(dg_key, pcm, chunk_ms=20):
    events = []
    streamer = asr_deepgram.DeepgramStreamer(
        dg_key,
        on_utterance=lambda t: events.append(("utterance", t, time.monotonic())),
        on_interim=lambda t: events.append(("interim", t, time.monotonic())),
        language="multi",
        keyterms=["Marumbi"],
    )
    await streamer.connect()

    chunk_bytes = RATE * 2 * chunk_ms // 1000
    t0 = time.monotonic()
    first_transcript_at = None
    for off in range(0, len(pcm), chunk_bytes):
        await streamer.send_audio(pcm[off:off + chunk_bytes])
        if first_transcript_at is None and any(e[0] == "interim" for e in events):
            first_transcript_at = time.monotonic() - t0
        await asyncio.sleep(chunk_ms / 1000)  # real-time pacing

    # Trailing silence MUST be streamed: endpointing counts silence *in the
    # received audio*, not wall-clock. Continuous mic audio always contains it.
    silence = b"\x00\x00" * (RATE * 2 * chunk_ms // 1000 // 2)
    for _ in range(int(1500 / chunk_ms)):  # 1.5 s of ambient silence
        await streamer.send_audio(silence)
        if any(e[0] == "utterance" for e in events):
            break
        await asyncio.sleep(chunk_ms / 1000)
    await streamer.close()

    return events, t0, first_transcript_at


RATE = 16000


async def main():
    env = load_env()
    dg_key = env["DEEPGRAM_API_KEY"]
    cart_key = env["CARTESIA_API_KEY"]

    pcm = cartesia_tts_pcm(cart_key, SENTENCE, MARUMBI_VOICE)
    dur = len(pcm) / (RATE * 2)
    print(f"TTS: {len(pcm)} bytes PCM ({dur:.2f}s)")

    events, t0, first_at = await run_stream_test(dg_key, pcm)

    finals = [(k, t) for k, t in [(e[0], e[2] - t0) for e in events] if k == "utterance"]
    utterances = [e[1] for e in events if e[0] == "utterance"]
    interims = [e[1] for e in events if e[0] == "interim"]

    print(f"first interim transcript at: {first_at:.2f}s" if first_at else "no interim")
    print(f"interim samples: {interims[:2]}")
    print(f"utterances: {utterances}")
    if finals:
        t = finals[0][1]
        print(f"UtteranceEnd after speech end: {t - dur:.2f}s (endpointing target ~0.5s)")

    passed = any(("cutia" in u.lower() or "cotia" in u.lower()) for u in utterances)
    print("TRANSCRIPT CONTAINS 'cutia/cotia':", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))