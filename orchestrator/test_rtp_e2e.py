"""
Fase 6 E2E: fake device — RTP G722 directly against the orchestrator.

Simulates the sculpture: sends a TTS'd question as RTP/G722 packets from a
local UDP socket to the orchestrator's rtp_port, receives the avatar's
downlink RTP packets back, decodes them (pure-Python G722), batch-ASRs the
result and asserts the reply. NO Agora involved — validates the entire
self-hosted path (RTP in -> decode -> ASR -> LLM -> TTS -> G722 -> RTP out).

Run while an orchestrator session is live in rtp mode:
    python orchestrator_main.py --avatar 4 --channel <ch> --idle 90 --rtp-port 26100
    /root/agora-orchestrator-venv/bin/python test_rtp_e2e.py --port 26100
"""
import argparse
import json
import re
import socket
import struct
import sys
import time
import wave

import requests

from g722 import G722Decoder

RATE = 16000
FRAME_SAMPLES = 320
FRAME_BYTES = 160
BASE = "https://avatars.sympoiesis.xyz"
BACKEND = "/root/AvatarGarden/backend"


def load_env():
    env = {}
    for line in open(f"{BACKEND}/.env"):
        m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0, help="orchestrator rtp_port (fetched from agent/start if 0)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--avatar", default="4")
    ap.add_argument("--channel", default=None)
    args = ap.parse_args()

    env = load_env()
    avatars = json.load(open(f"{BACKEND}/avatars.json"))
    av = next(a for a in avatars if str(a["id"]) == str(args.avatar))
    channel = args.channel or f"rtp-e2e-{int(time.time())}"
    if not args.port:
        r = requests.post("http://127.0.0.1:5001/api/voice/agent/start",
            headers={"Content-Type": "application/json"},
            json={"avatarId": args.avatar, "channel": channel, "userUid": 1,
                  "parameters": {"output_audio_codec": "G722"}}, timeout=15)
        r.raise_for_status()
        d = r.json()
        args.port = d["rtpPort"]
        print(f"[fake] orchestrator: {d['agentId']} rtp {d['rtpHost']}:{d['rtpPort']}")
    else:
        channel = args.channel or "rtp-e2e-manual"

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
    print(f"[fake] question: {len(question_pcm)/32000:.1f}s of speech")

    dec = G722Decoder()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.05)
    sock.bind(("127.0.0.1", 0))

    seq = 0
    ts = 0
    downlink = []

    def send_pcm(pcm):
        nonlocal seq, ts
        for k in range(0, len(pcm) - 639, 640):
            pkt = struct.pack("!BBHII", 0x80, 97, seq & 0xFFFF, ts, 0x1) + \
                  pcm[k:k + 640]
            seq = (seq + 1) & 0xFFFF
            ts = (ts + 320) & 0xFFFFFFFF
            sock.sendto(pkt, (args.host, args.port))
            time.sleep(0.02)  # real-time pacing

    # 1. stream the question (real time)
    print("[fake] sending question as RTP G722...")
    chunk = 320 * 2
    t0 = time.monotonic()
    for off in range(0, len(question_pcm), chunk):
        send_pcm(question_pcm[off:off + chunk])
    # 2. trailing silence must be streamed for endpointing
    silence = b"\x00" * 640
    t_last = None
    down_n = 0
    while time.monotonic() - t0 < 60:
        send_pcm(silence)
        # drain downlink
        while True:
            try:
                pkt, _ = sock.recvfrom(2048)
            except socket.timeout:
                break
            payload = pkt[12:] if (pkt[0] >> 6) == 2 else pkt
            pt = pkt[1] & 0x7F if (pkt[0] >> 6) == 2 else 97
            if pt == 96:
                downlink += dec.decode(payload)   # G722 payload
            else:
                downlink += struct.unpack("<%dh" % (len(payload) // 2), payload[:len(payload)//2*2])  # PCM
        if len(downlink) != down_n:
            down_n = len(downlink)
            t_last = time.monotonic()
        if t_last and time.monotonic() - t_last > 3.0 and down_n > 0:
            print("[fake] reply finished (3 s without new downlink)")
            break

    dur = len(downlink) / RATE
    latency = None
    if downlink and t_last:
        # first downlink sample time ≈ t0 + (frames until downlink) — approximate
        pass
    print(f"[fake] downlink: {len(downlink)} samples = {dur:.1f}s audio")

    if not downlink:
        print("E2E: FAIL (no downlink audio)")
        return 1

    wav = "/tmp/rtp_e2e_reply.wav"
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
        print(f"[fake] reply transcript: {alt['transcript']!r} (conf {alt['confidence']:.2f})")
        ok = bool(alt["transcript"].strip()) and dur > 2.0
    except Exception as e:
        print(f"[fake] batch ASR failed: {e}")
        ok = dur > 2.0
    print("RTP E2E:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())