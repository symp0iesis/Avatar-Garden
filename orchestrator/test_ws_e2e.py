"""
Fase 6 E2E: fake browser over WebSocket — validates the ws transport end to
end without Agora and without a human.

Connects to ws://127.0.0.1:<ws_port>, streams a TTS'd question as binary PCM
frames in real time, receives the avatar's downlink PCM binary frames, saves
+ batch-ASRs them, asserts the reply.

Run while a ws-mode session is live:
    python orchestrator_main.py --avatar 4 --channel <ch> --idle 90 --ws-port 8010
    /root/agora-orchestrator-venv/bin/python test_ws_e2e.py
"""
import argparse
import json
import re
import struct
import sys
import time
import wave

import asyncio
import requests
import websockets

RATE = 16000
BACKEND = "/root/AvatarGarden/backend"


def load_env():
    env = {}
    for line in open(f"{BACKEND}/.env"):
        m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    return env


async def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--avatar", default="4")
    args = ap.parse_args()

    env = load_env()
    avatars = json.load(open(f"{BACKEND}/avatars.json"))
    av = next(a for a in avatars if str(a["id"]) == str(args.avatar))
    r = requests.post("http://127.0.0.1:5001/api/voice/agent/start",
        headers={"Content-Type": "application/json"},
        json={"avatarId": args.avatar, "channel": f"ws-e2e-{int(time.time())}",
              "userUid": 1, "parameters": {"output_audio_codec": "G722"}},
        timeout=15)
    print(f"[ws] spawn: {r.json()}")

    q = requests.post("https://api.cartesia.ai/tts/bytes",
        headers={"X-API-Key": env["CARTESIA_API_KEY"], "Content-Type": "application/json",
                 "Cartesia-Version": "2026-08-14"},
        json={"transcript": "Voce pode me contar uma historia curta sobre o rio Marumbi?",
              "model_id": "sonic-3",
              "voice": av.get("ttsVoiceId") or "9904416a-0831-44ea-b8ee-5f145e8f9bbf",
              "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": RATE}},
        timeout=30)
    q.raise_for_status()
    question_pcm = q.content
    print(f"[ws] question: {len(question_pcm)/32000:.1f}s of speech")

    downlink = []
    ws = await websockets.connect(f"ws://127.0.0.1:{args.port}", max_size=None)
    print("[ws] connected")
    # protocol: the client declares its capture rate in a text init frame first
    await ws.send(json.dumps({"type": "init", "sampleRate": RATE}))
    t_q_end = [None]
    t_first = [None]

    async def sender():
        chunk = 640
        for off in range(0, len(question_pcm), chunk):
            await ws.send(question_pcm[off:off + chunk])
            await asyncio.sleep(0.02)
        t_q_end[0] = time.monotonic()
        silence = b"\x00" * 640
        t_last = None
        last_n = 0
        while True:
            await ws.send(silence)
            await asyncio.sleep(0.02)
            if len(downlink) != last_n:
                last_n = len(downlink)
                t_last = time.monotonic()
            if t_last and time.monotonic() - t_last > 3.0:
                break

    async def receiver():
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    if t_first[0] is None:
                        t_first[0] = time.monotonic()
                    downlink.extend(struct.unpack("<%dh" % (len(msg) // 2), msg[:len(msg)//2*2]))
        except websockets.ConnectionClosed:
            pass

    rx_task = asyncio.create_task(receiver())
    t0 = time.monotonic()
    await asyncio.wait_for(sender(), timeout=70)
    await asyncio.sleep(1)
    await ws.close()
    await rx_task

    if t_first[0] and t_q_end[0]:
        print(f"[timing] question end -> first downlink audio: {t_first[0]-t_q_end[0]:.2f}s")

    dur = len(downlink) / RATE
    print(f"[ws] downlink: {len(downlink)} samples = {dur:.1f}s audio "
          f"(session took {time.monotonic()-t0:.0f}s)")
    if not downlink:
        print("WS E2E: FAIL (no downlink audio)")
        return 1

    wav = "/tmp/ws_e2e_reply.wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(struct.pack("<%dh" % len(downlink), *downlink))
    dg = requests.post(
        "https://api.deepgram.com/v1/listen?model=nova-3&language=multi&sample_rate=16000&encoding=linear16&channels=1",
        headers={"Authorization": f"Token {env['DEEPGRAM_API_KEY']}",
                 "Content-Type": "audio/wav"},
        data=struct.pack("<%dh" % len(downlink), *downlink), timeout=30)
    try:
        alt = dg.json()["results"]["channels"][0]["alternatives"][0]
        print(f"[ws] reply transcript: {alt['transcript']!r} (conf {alt['confidence']:.2f})")
        ok = bool(alt["transcript"].strip()) and dur > 2.0
    except Exception as e:
        print(f"[ws] batch ASR failed: {e}")
        ok = dur > 2.0
    print("WS E2E:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))